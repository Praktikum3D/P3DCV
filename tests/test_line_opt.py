"""Line optimization: the plumb line term in the training loss and the per scene refinement.

Both halves rest on `training/line_geometry.py`, whose own tests pin the residual itself
(`tests/test_line_infra.py`). What is new here, and what these tests are about, is everything
around it:

* the training term must be INVISIBLE when it is off, bit for bit, whether it is off because the
  weight is zero or because the loader never detected a line,
* it must read the lines of the INPUT views, the ones whose camera it is scoring, which means the
  gather with `input_idx` in the model has to be the same gather the images get; a misaligned
  gather is the classic way to build a loss that looks plausible and teaches nothing,
* a view or a batch without lines must contribute exactly zero and never a NaN,
* the refinement must actually move the camera back to the truth, through the real renderer, which
  is where the gradient path intrinsics -> rays -> image is tested end to end rather than in theory.

Run from the repository root:
    pytest tests/test_line_opt.py
"""

import inspect
import math
import os
import sys

import pytest
import torch
from easydict import EasyDict as edict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

cv2 = pytest.importorskip("cv2")

import train  # noqa: E402
from test_line_infra import FISHEYE, UCM, synthetic_line_pixels  # noqa: E402
from training.camera_refine import (  # noqa: E402
    CameraOffsets,
    RefineScene,
    detect_scene_lines,
    fov_degrees,
    refine_camera,
    render_views,
    renderer_for,
    scale_intrinsics,
)
from training.line_detection import PAD_VALUE  # noqa: E402

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA renderer")


# --------------------------------------------------------------------------------------------
# a result and a config shaped like the ones train.py passes around
# --------------------------------------------------------------------------------------------

def loss_config(camera_model="fisheye", line_loss_weight=None):
    config = edict(
        {
            "model": {"camera_model": camera_model, "renderer": "geer", "geer_tile_mode": "projection"},
            "dataset": {"camera_name": "lcam_fish"},
            "training": {"l2_loss_weight": 1.0},
        }
    )
    if line_loss_weight is not None:
        config.training.line_loss_weight = line_loss_weight
    return config


def loss_inputs(v=3, h=32, w=32, camera_model="fisheye", f=None, alpha=0.62):
    """A render, a target and a predicted camera, normalized intrinsics as the model produces them."""
    g = torch.Generator().manual_seed(0)
    render = torch.rand(1, v, 3, h, w, generator=g)
    target = torch.rand(1, v, 3, h, w, generator=g)
    if f is None:
        f = (w / 2) / math.radians(195.0 / 2)
    k = torch.tensor([[f, f, w / 2, h / 2, alpha]]).repeat(v, 1)
    if camera_model not in ("ucm", "eucm"):
        k = k[:, :4]
    normalized = k.clone()
    normalized[:, 0] /= w
    normalized[:, 1] /= h
    normalized[:, 2] /= w
    normalized[:, 3] /= h
    result = edict(
        render=render,
        fxfycxcy_target=k[None],
        target_image=target,
        fxfycxcy_input=normalized[None],
        image=target,
    )
    return result, {"image": target}


def padded(pixels, n_max=8):
    """One view's lines [n, K, 2] padded out to the batch width the loader uses, [n_max, K, 2]."""
    rows = torch.full((n_max, pixels.shape[1], 2), PAD_VALUE, dtype=torch.float32)
    rows[: pixels.shape[0]] = pixels.float()
    return rows


def lines_for(result, pixels_per_view):
    """Hang per view line tensors on a result the way `ERayZer.forward` does."""
    result.input_line_pixels = torch.stack(pixels_per_view)[None]
    result.input_line_count = torch.tensor([[int((p[:, 0, 0] >= 0).sum()) for p in pixels_per_view]])
    return result


def true_camera_lines(result, camera_model, v, n_max=8):
    """Lines that are exactly coplanar under the camera `result` predicts, one set per view.

    Built by projecting straight 3D lines with the very intrinsics the loss will be told about, so
    the correct answer is a residual of zero and any deviation is the camera moving, not noise.
    """
    k = result.fxfycxcy_input[0, 0].clone()
    k[0] *= result.image.shape[-1]
    k[1] *= result.image.shape[-2]
    k[2] *= result.image.shape[-1]
    k[3] *= result.image.shape[-2]
    pixels = synthetic_line_pixels(k.double(), camera_model)
    return lines_for(result, [padded(pixels, n_max) for _ in range(v)])


# --------------------------------------------------------------------------------------------
# off means off
# --------------------------------------------------------------------------------------------

def test_the_loss_is_untouched_without_a_weight():
    """No training.line_loss_weight at all is the state every existing run is in."""
    result, batch = loss_inputs()
    baseline = train.compute_loss(result, batch, loss_config(), edict()).loss
    with_zero = train.compute_loss(result, batch, loss_config(line_loss_weight=0.0), edict()).loss
    assert baseline.item() == with_zero.item()
    assert torch.equal(baseline, with_zero)


def test_lines_in_the_batch_change_nothing_while_the_weight_is_zero():
    """The dataset flag and the loss weight are independent switches; either one off means off."""
    result, batch = loss_inputs()
    baseline = train.compute_loss(result, batch, loss_config(), edict()).loss.clone()
    true_camera_lines(result, "fisheye", v=3)
    with_lines = train.compute_loss(result, batch, loss_config(line_loss_weight=0.0), edict()).loss
    assert torch.equal(baseline, with_lines)
    # and the residual is reported as an honest zero rather than silently omitted
    metrics = train.compute_loss(result, batch, loss_config(line_loss_weight=0.0), edict())
    assert float(metrics.line_residual) == 0.0


def test_a_weight_without_lines_changes_nothing():
    """A run that asks for the term but whose loader never detected a line must not crash or drift."""
    result, batch = loss_inputs()
    baseline = train.compute_loss(result, batch, loss_config(), edict()).loss.clone()
    weighted = train.compute_loss(result, batch, loss_config(line_loss_weight=10.0), edict()).loss
    assert torch.equal(baseline, weighted)
    assert train.plumbline_loss(result, loss_config(line_loss_weight=10.0)) is None


# --------------------------------------------------------------------------------------------
# on means on, and in the right direction
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("camera_model", ["fisheye", "ucm"])
def test_the_true_camera_scores_about_zero(camera_model):
    result, batch = loss_inputs(camera_model=camera_model)
    true_camera_lines(result, camera_model, v=3)
    residual = train.plumbline_loss(result, loss_config(camera_model, 1.0))
    # Not exactly zero: the pixels are projected in float64 and the residual is evaluated in the
    # float32 the model trains in, on a 32 pixel image. Four orders of magnitude below what a
    # camera off by a quarter costs below, which is the margin that matters.
    assert float(residual) < 1e-6, f"{camera_model}: {float(residual)}"


@pytest.mark.parametrize("camera_model", ["fisheye", "ucm"])
def test_a_wrong_camera_scores_more_and_the_gradient_points_back(camera_model):
    """The whole point of the term: a focal length that is off must cost, on both sides of the truth."""
    result, batch = loss_inputs(camera_model=camera_model)
    true_camera_lines(result, camera_model, v=3)
    truth = result.fxfycxcy_input.clone()
    config = loss_config(camera_model, 1.0)

    at_truth = float(train.plumbline_loss(result, config))
    for factor in (0.8, 1.25):
        result.fxfycxcy_input = truth.clone()
        result.fxfycxcy_input[..., :2] *= factor
        result.fxfycxcy_input.requires_grad_(True)
        residual = train.plumbline_loss(result, config)
        assert float(residual) > 100.0 * at_truth, f"{camera_model} x{factor}: residual did not grow"
        residual.backward()
        slope = float(result.fxfycxcy_input.grad[..., 0].sum())
        # Downhill in f has to be towards the true focal length: negative slope above it,
        # positive below it, which is the sign of -(factor - 1).
        assert slope * (factor - 1.0) > 0.0, f"{camera_model} x{factor}: gradient points away, {slope}"


def test_the_alpha_gradient_points_back_as_well():
    """f is not the only intrinsic the term constrains; alpha is the one the weekend is about."""
    result, batch = loss_inputs(camera_model="ucm", alpha=0.62)
    true_camera_lines(result, "ucm", v=3)
    truth = result.fxfycxcy_input.clone()
    config = loss_config("ucm", 1.0)
    for alpha in (0.45, 0.78):
        result.fxfycxcy_input = truth.clone()
        result.fxfycxcy_input[..., 4] = alpha
        result.fxfycxcy_input.requires_grad_(True)
        residual = train.plumbline_loss(result, config)
        residual.backward()
        slope = float(result.fxfycxcy_input.grad[..., 4].sum())
        assert slope * (alpha - 0.62) > 0.0, f"alpha {alpha}: gradient points away, {slope}"


def test_the_weight_scales_the_contribution_and_nothing_else():
    result, batch = loss_inputs()
    true_camera_lines(result, "fisheye", v=3)
    # a wrong camera, so the residual is not zero and the scaling is visible
    result.fxfycxcy_input = result.fxfycxcy_input * torch.tensor([1.3, 1.3, 1.0, 1.0])
    base = train.compute_loss(result, batch, loss_config(), edict()).loss.item()
    metrics = train.compute_loss(result, batch, loss_config("fisheye", 1.0), edict())
    doubled = train.compute_loss(result, batch, loss_config("fisheye", 2.0), edict())
    residual = float(metrics.line_residual)
    assert residual > 0.0
    assert metrics.loss.item() == pytest.approx(base + residual, rel=1e-6)
    assert doubled.loss.item() == pytest.approx(base + 2.0 * residual, rel=1e-6)


# --------------------------------------------------------------------------------------------
# degenerate batches
# --------------------------------------------------------------------------------------------

def test_a_batch_without_a_single_line_is_exactly_zero_and_finite():
    """The loader finds nothing on a dark frame now and then; a training step must survive it."""
    result, batch = loss_inputs()
    empty = torch.full((3, 8, 8, 2), PAD_VALUE)
    lines_for(result, list(empty))
    result.fxfycxcy_input = result.fxfycxcy_input.clone().requires_grad_(True)
    residual = train.plumbline_loss(result, loss_config("fisheye", 5.0))
    assert float(residual) == 0.0
    residual.backward()
    assert torch.isfinite(result.fxfycxcy_input.grad).all()
    assert float(result.fxfycxcy_input.grad.abs().sum()) == 0.0
    metrics = train.compute_loss(result, batch, loss_config("fisheye", 5.0), edict())
    assert torch.isfinite(metrics.loss)


def test_a_view_without_lines_does_not_drag_the_others_down():
    """Mixed batch: two views with lines, one empty. The empty one must simply not count."""
    result, batch = loss_inputs(v=3)
    k = result.fxfycxcy_input[0, 0].clone()
    k[:4] *= 32
    pixels = padded(synthetic_line_pixels(k.double(), "fisheye"))
    lines_for(result, [pixels, torch.full_like(pixels, PAD_VALUE), pixels])
    mixed = float(train.plumbline_loss(result, loss_config("fisheye", 1.0)))

    # The same two views without the empty one in between. Exactly the same number, because the
    # padded rows are weighted out before the trimmed mean and never enter its sort either.
    full = loss_inputs(v=2)[0]
    true_camera_lines(full, "fisheye", v=2)
    assert mixed == pytest.approx(float(train.plumbline_loss(full, loss_config("fisheye", 1.0))), abs=1e-12)


def test_a_duplicated_pixel_line_is_zero_rather_than_a_nan():
    """A line whose samples collapse onto one pixel constrains nothing and must say so quietly."""
    result, batch = loss_inputs()
    collapsed = torch.full((3, 8, 8, 2), 16.5)
    lines_for(result, list(collapsed))
    result.fxfycxcy_input = result.fxfycxcy_input.clone().requires_grad_(True)
    residual = train.plumbline_loss(result, loss_config("fisheye", 1.0))
    assert float(residual) == 0.0
    residual.backward()
    assert torch.isfinite(result.fxfycxcy_input.grad).all()


# --------------------------------------------------------------------------------------------
# the gather, which is where this loss would silently go wrong
# --------------------------------------------------------------------------------------------

def test_the_model_pads_and_gathers_the_lines_exactly_like_the_images():
    """Source level, because a full ERayZer forward needs a CUDA renderer and 200M parameters.

    The four properties: the tensors are padded together with the images (or input_idx would index
    past them on a sample with fewer than ten views), they are gathered with the same batch_idx and
    input_idx the images are gathered with, the gather happens before the result is assembled, and
    nothing of it runs when the batch carries no lines.
    """
    from erayzer_core.model.erayzer import ERayZer

    source = inspect.getsource(ERayZer.forward)
    assert 'data.get("line_pixels")' in source
    assert 'data.get("line_count")' in source
    assert source.index("line_pixels_all = torch.cat") < source.index("v_all = v")
    assert "line_pixels_all[batch_idx, input_idx]" in source
    assert "line_count_all[batch_idx, input_idx]" in source
    gather = source.index("line_pixels_input = line_pixels_all[batch_idx, input_idx]")
    assert gather < source.index("result.input_line_pixels = line_pixels_input")
    # the images use the very same pair of index tensors
    assert "img_tokens_all[batch_idx, input_idx]" in source
    # and everything is behind a None check, so a batch without lines takes none of it
    assert "if line_pixels_all is not None:" in source


def test_view_k_lines_reach_input_slot_k():
    """Adversarial permutation of the gather itself, with a marker per view.

    Every view carries a line tensor filled with its own view number, and the input views are a
    scattered subset in a scrambled order. Picking the first v_input views, or forgetting the batch
    index, or transposing the two indices all produce a different answer than this.
    """
    b, v_all, n_max, k = 2, 8, 4, 6
    marker = torch.arange(b * v_all, dtype=torch.float32).reshape(b, v_all, 1, 1, 1)
    line_pixels = marker.expand(b, v_all, n_max, k, 2).contiguous()
    line_count = torch.arange(b * v_all, dtype=torch.long).reshape(b, v_all)
    batch_idx = torch.arange(b).unsqueeze(1)
    input_idx = torch.tensor([[7, 0, 4], [2, 6, 1]])

    picked_pixels = line_pixels[batch_idx, input_idx]
    picked_count = line_count[batch_idx, input_idx]
    for sample in range(b):
        for slot, view in enumerate(input_idx[sample].tolist()):
            assert torch.equal(picked_pixels[sample, slot], line_pixels[sample, view])
            assert int(picked_count[sample, slot]) == int(line_count[sample, view])
    # the naive version a reader might expect, and why the test is not vacuous
    naive = line_pixels[:, : input_idx.shape[1]]
    assert not torch.equal(naive, picked_pixels)


def test_the_padding_of_a_short_sample_repeats_the_last_view():
    """With fewer than ten views the model repeats the last one; the lines must repeat with it."""
    b, v, pad, n_max, k = 1, 5, 5, 3, 4
    line_pixels = torch.randn(b, v, n_max, k, 2).abs()
    line_count = torch.randint(0, n_max, (b, v))
    padded_pixels = torch.cat([line_pixels, line_pixels[:, -1:].repeat(1, pad, 1, 1, 1)], dim=1)
    padded_count = torch.cat([line_count, line_count[:, -1:].repeat(1, pad)], dim=1)
    assert padded_pixels.shape[1] == 10 and padded_count.shape[1] == 10
    for view in range(v, 10):
        assert torch.equal(padded_pixels[:, view], line_pixels[:, -1])
        assert torch.equal(padded_count[:, view], line_count[:, -1])
    # an input index into the padded range is what makes this necessary at all
    assert torch.equal(padded_pixels[0, 9], line_pixels[0, -1])


def test_scoring_the_wrong_views_would_be_visible():
    """The test above checks the gather; this checks that a wrong gather would change the loss.

    Views whose lines belong to a different camera than the one being scored give a clearly larger
    residual, so the alignment is not a cosmetic property of the code.
    """
    result, batch = loss_inputs(v=3)
    k = result.fxfycxcy_input[0, 0].clone()
    k[:4] *= 32
    right = padded(synthetic_line_pixels(k.double(), "fisheye"))
    wrong_k = k.clone()
    wrong_k[:2] *= 1.4
    wrong = padded(synthetic_line_pixels(wrong_k.double(), "fisheye"))

    lines_for(result, [right, right, right])
    aligned = float(train.plumbline_loss(result, loss_config("fisheye", 1.0)))
    lines_for(result, [wrong, wrong, wrong])
    misaligned = float(train.plumbline_loss(result, loss_config("fisheye", 1.0)))
    assert misaligned > aligned + 1e-6


# --------------------------------------------------------------------------------------------
# the parametrization of the refinement
# --------------------------------------------------------------------------------------------

def test_camera_offsets_start_exactly_at_the_prediction():
    for camera_model, intrinsics in (("fisheye", FISHEYE), ("ucm", UCM)):
        k = torch.tensor(intrinsics, dtype=torch.float32).repeat(4, 1)
        offsets = CameraOffsets(k, camera_model)
        got = offsets()
        assert got.shape == k.shape
        assert torch.allclose(got, k, atol=1e-5), camera_model


def test_camera_offsets_keep_alpha_inside_its_range():
    """A sigmoid of an offset logit, so no step the optimizer takes can push alpha out of (0, 1)."""
    k = torch.tensor(UCM, dtype=torch.float32).repeat(2, 1)
    offsets = CameraOffsets(k, "ucm")
    for push in (8.0, -8.0, 50.0, -50.0):
        with torch.no_grad():
            offsets.alpha_offset.fill_(push)
        alpha = float(offsets()[0, 4])
        assert 0.0 <= alpha <= 1.0, push
        if abs(push) < 10.0:  # far out it saturates to exactly 0 or 1 in float32, which is still valid
            assert 0.0 < alpha < 1.0, push


def test_alpha_can_be_frozen():
    """Refining f alone is a supported mode, because f and alpha are coupled under few lines."""
    k = torch.tensor(UCM, dtype=torch.float32).repeat(2, 1)
    frozen = CameraOffsets(k, "ucm", learn_alpha=False)
    assert frozen.alpha_offset is None
    assert [name for name, _ in frozen.named_parameters()] == ["log_f"]
    assert float(frozen()[0, 4]) == float(k[0, 4])


def test_beta_moves_only_when_it_is_asked_to():
    k = torch.tensor([300.0, 300.0, 500.0, 500.0, 0.6, 1.0]).repeat(2, 1)
    frozen = CameraOffsets(k, "eucm", learn_beta=False)
    assert frozen.log_beta is None
    assert [name for name, _ in frozen.named_parameters()] == ["log_f", "alpha_offset"]
    learning = CameraOffsets(k, "eucm", learn_beta=True)
    assert "log_beta" in dict(learning.named_parameters())


def test_scaling_the_intrinsics_leaves_alpha_alone():
    k = torch.tensor(UCM, dtype=torch.float32)[None]
    scaled = scale_intrinsics(k, 4.0)
    assert torch.allclose(scaled[0, :4], k[0, :4] * 4.0)
    assert float(scaled[0, 4]) == float(k[0, 4])


def test_fov_degrees_matches_the_training_log():
    """The number the refinement reports has to be the one `train.camera_stats` logs."""
    from erayzer_core.model.utils_gaussian import camera_theta

    width = 256
    k = torch.tensor([[70.0, 70.0, 128.0, 128.0]])
    expected = math.degrees(2.0 * float(camera_theta(torch.tensor(0.5 / (70.0 / width)), "fisheye")))
    assert fov_degrees(k, "fisheye", width) == pytest.approx(expected, rel=1e-6)


# --------------------------------------------------------------------------------------------
# the refinement itself, on a tiny scene rendered by the real renderer
# --------------------------------------------------------------------------------------------

def tiny_scene(num_gaussians=48, views=3, size=64, fov_deg=150.0, seed=0, camera_model="fisheye"):
    """A handful of Gaussians in front of a few cameras, rendered by the renderer under test.

    The images are rendered at the TRUE camera and then the scene is handed back with that camera,
    so a caller can perturb the intrinsics and ask the refinement to find its way back. Small on
    purpose: it has to run on a six gigabyte card next to everything else.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    device = torch.device("cuda")
    xyz = torch.rand(num_gaussians, 3, generator=g) * torch.tensor([4.0, 4.0, 2.0]) - torch.tensor([2.0, 2.0, -1.5])
    features = torch.zeros(num_gaussians, 16, 3)
    features[:, 0] = torch.rand(num_gaussians, 3, generator=g) * 2.0 - 0.5  # the DC term carries the colour
    scaling = torch.full((num_gaussians, 3), math.log(0.08))
    rotation = torch.zeros(num_gaussians, 4)
    rotation[:, 0] = 1.0
    opacity = torch.full((num_gaussians, 1), 4.0)  # sigmoid(4) is nearly opaque

    c2w = torch.eye(4)[None].repeat(views, 1, 1)
    c2w[:, 0, 3] = torch.linspace(-0.3, 0.3, views)
    f = (size / 2) / math.radians(fov_deg / 2)
    k = torch.tensor([[f, f, size / 2, size / 2]]).repeat(views, 1)
    if camera_model in ("ucm", "eucm"):
        k = torch.cat([k, torch.full((views, 1), 0.62)], dim=-1)
    if camera_model == "eucm":
        k = torch.cat([k, torch.ones(views, 1)], dim=-1)

    scene = RefineScene(
        xyz=xyz, features=features, scaling=scaling, rotation=rotation, opacity=opacity,
        sh_degree=3, c2w=c2w, fxfycxcy=k, images=torch.zeros(views, 3, size, size),
        camera_model=camera_model, near_plane=0.01, tile_mode="projection", name="tiny",
    ).to(device)
    with torch.no_grad():
        scene.images = render_views(scene, renderer_for(scene), scene.fxfycxcy, size, size).clamp(0, 1)
    return scene


def perturbed(scene, factor=1.2, alpha=None):
    """The same scene with its camera moved off the truth, which is what a refinement starts from."""
    k = scene.fxfycxcy.clone()
    k[:, :2] *= factor
    if alpha is not None and k.shape[-1] > 4:
        k[:, 4] = alpha
    moved = RefineScene(**{**scene.__dict__, "fxfycxcy": k})
    return moved


@needs_cuda
def test_the_photometric_term_recovers_a_perturbed_focal_length():
    """f x1.2 back to within two percent, through the real renderer and its ray gradients.

    Nothing here is pixel aligned, so the images really do pin the camera down, which is the
    situation a refinement against observed input images is in.
    """
    scene = tiny_scene()
    truth = float(scene.fxfycxcy[0, 0])
    start = perturbed(scene, 1.2)
    result = refine_camera(start, steps=120, lr=0.02, line_weight=0.0)
    got = float(result.fxfycxcy[0, 0])
    assert abs(got - truth) / truth < 0.02, f"{got} against a true {truth}"
    assert result.last["image_loss"] < result.first["image_loss"]


@needs_cuda
def test_the_line_term_alone_recovers_a_perturbed_focal_length():
    """Same perturbation, no image term at all: the plumb lines carry the camera on their own."""
    scene = tiny_scene()
    truth = float(scene.fxfycxcy[0, 0])
    lines = torch.stack([
        padded(synthetic_line_pixels(scene.fxfycxcy[0].double().cpu(), "fisheye"))
        for _ in range(scene.c2w.shape[0])
    ]).to(scene.device)
    start = perturbed(scene, 1.2)
    result = refine_camera(start, line_pixels=lines, steps=120, lr=0.02, image_weight=0.0, line_weight=1.0)
    got = float(result.fxfycxcy[0, 0])
    assert abs(got - truth) / truth < 0.02, f"{got} against a true {truth}"
    assert result.last["residual"] < result.first["residual"]


@needs_cuda
def test_both_terms_together_recover_the_focal_length():
    """The combination the script runs, with the weight measured at step 0 instead of guessed."""
    scene = tiny_scene()
    truth = float(scene.fxfycxcy[0, 0])
    lines = torch.stack([
        padded(synthetic_line_pixels(scene.fxfycxcy[0].double().cpu(), "fisheye"))
        for _ in range(scene.c2w.shape[0])
    ]).to(scene.device)
    start = perturbed(scene, 1.2)
    result = refine_camera(start, line_pixels=lines, steps=120, lr=0.02, auto_lambda=True)
    got = float(result.fxfycxcy[0, 0])
    assert abs(got - truth) / truth < 0.02, f"{got} against a true {truth}"


@needs_cuda
def test_zero_steps_returns_the_prediction_untouched():
    scene = tiny_scene()
    start = perturbed(scene, 1.2)
    result = refine_camera(start, steps=0)
    assert torch.allclose(result.fxfycxcy, start.fxfycxcy, atol=1e-4)
    assert len(result.history) == 1


@needs_cuda
def test_a_refinement_without_lines_stays_finite():
    """No lines at all is a supported configuration, not a crash and not a NaN."""
    scene = tiny_scene()
    empty = torch.full((scene.c2w.shape[0], 4, 8, 2), PAD_VALUE, device=scene.device)
    result = refine_camera(perturbed(scene, 1.2), line_pixels=empty, steps=5, line_weight=1.0)
    assert result.num_lines == 0
    assert all(math.isfinite(row["total"]) for row in result.history)


@needs_cuda
def test_the_ucm_alpha_is_refined_too():
    """alpha is the quantity the weekend goal is about, so it has to move with f."""
    scene = tiny_scene(camera_model="ucm")
    truth = float(scene.fxfycxcy[0, 4])
    start = perturbed(scene, 1.0, alpha=0.3)
    result = refine_camera(start, steps=80, lr=0.05, line_weight=0.0)
    got = float(result.fxfycxcy[0, 4])
    assert abs(got - truth) < abs(0.3 - truth), f"alpha {got} did not move towards {truth}"


@needs_cuda
def test_detection_runs_on_the_scene_and_stays_in_the_working_resolution():
    """`detect_scene_lines` renders bigger, detects there and hands the pixels back at 1x."""
    scene = tiny_scene(num_gaussians=200, size=64)
    pixels = detect_scene_lines(scene, detect_size=128, max_count=4, num_samples=5, min_length=0.1)
    assert pixels.shape == (scene.c2w.shape[0], 4, 5, 2)
    real = pixels[pixels[..., 0, 0] >= 0]
    if real.numel():
        assert float(real.max()) <= 64.0 + 1.0, "pixels were left in the detection resolution"
