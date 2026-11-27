"""The evaluation options that make a cross model comparison on TartanAir fair.

Four models are to be scored on the same scenes: a pinhole model that was pretrained on pinhole data,
a pinhole fine-tune and two wide angle runs. Each of them was trained with its own loss mask, so the
default evaluation scores each on different pixels and the numbers cannot be compared. The circle
override puts every model on the same image circle, the unmasked numbers show the whole frame, and
SSIM comes from this repository because the container image has neither torchmetrics nor skimage.

These tests also pin the default: without the new options the mask and the reported numbers are the
ones the evaluation produced before, down to the format of the first lines and columns of its output.

Run from the repository root:
    pytest tests/test_quality_eval.py
"""

import csv
import math
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
from easydict import EasyDict as edict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import evaluate
from erayzer_core.model.utils_gaussian import loss_render_mask, render_mask
from erayzer_core.utils.image_metrics import ssim, ssim_map

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H = W = 256
THETA_MAX = math.radians(195.0 / 2)  # lcam_fish sees 195 degrees across its image circle
F_TRUE = (W / 2) / THETA_MAX  # equidistant fisheye, 0.29383 * W


def intrinsics(camera_model, f=F_TRUE, alpha=0.62, v=2):
    """fxfycxcy [v, 4], for UCM [v, 5] with alpha last, principal point in the image centre."""
    k = torch.tensor([[f, f, W / 2, H / 2]]).repeat(v, 1)
    if camera_model == "ucm":
        k = torch.cat([k, torch.full((v, 1), alpha)], dim=1)
    return k


def eval_config(camera_model, renderer="geer", tile_mode="projection", **evaluation):
    return edict(
        {
            "model": {"camera_model": camera_model, "renderer": renderer, "geer_tile_mode": tile_mode},
            "dataset": {"camera_name": "lcam_fish"},
            "evaluation": {"checkpoint_path": "none", "out_dir": "none", **evaluation},
        }
    )


# --- the default has to stay what it was -------------------------------------------------------


@pytest.mark.parametrize("camera_model", ["fisheye", "ucm"])
@pytest.mark.parametrize("renderer,tile_mode", [("gsplat", "pbf"), ("geer", "pbf"), ("geer", "projection")])
def test_without_an_override_the_mask_is_the_training_mask(camera_model, renderer, tile_mode):
    """The old evaluation masked with loss_render_mask, and it still does."""
    config = eval_config(camera_model, renderer, tile_mode)
    fxfycxcy = intrinsics(camera_model)
    expected = loss_render_mask(config, fxfycxcy, H, W)
    actual = evaluate.evaluation_mask(config, fxfycxcy, H, W)
    assert actual is not None
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("override", [None, "", "none", "unset"])
def test_without_an_override_a_pinhole_model_is_not_masked(override):
    """The old evaluation never masked a pinhole model, which is what the unmasked numbers now say."""
    config = eval_config("pinhole", **({} if override is None else {"mask_override": override}))
    assert evaluate.evaluation_mask(config, intrinsics("pinhole"), H, W) is None


def test_an_unknown_override_is_refused():
    config = eval_config("fisheye", mask_override="square")
    with pytest.raises(ValueError, match="mask_override"):
        evaluate.evaluation_mask(config, intrinsics("fisheye"), H, W)


@pytest.mark.parametrize("camera_model", ["pinhole", "fisheye", "ucm"])
def test_the_default_numbers_are_the_old_numbers(camera_model):
    """mse per image is computed exactly as before, masked for a wide angle model and plain for pinhole."""
    generator = torch.Generator().manual_seed(7)
    pred = torch.rand(1, 2, 3, H, W, generator=generator)
    target = torch.rand(1, 2, 3, H, W, generator=generator)
    config = eval_config(camera_model)
    fxfycxcy = intrinsics(camera_model)
    mask = evaluate.evaluation_mask(config, fxfycxcy, H, W)
    metrics = evaluate.image_metrics(pred, target, mask)

    if camera_model == "pinhole":  # the old else branch
        expected = (pred - target).pow(2).flatten(2).mean(dim=2)
    else:  # the old if branch, verbatim
        old_mask = loss_render_mask(config, fxfycxcy, H, W).reshape(1, 2, 1, H, W)
        expected = ((pred - target).pow(2) * old_mask).flatten(2).sum(2) / (
            old_mask.flatten(2).sum(2) * 3
        ).clamp_min(1.0)
    assert torch.equal(metrics["mse"], expected)


def test_the_summary_keeps_its_old_lines_and_columns(tmp_path):
    """New metrics are appended, the lines and columns that existed keep their place and format."""
    rows = [
        {
            "basename": "000000",
            "mse": 0.012345678,
            "psnr": 19.1234,
            "RPA@5": 0.5,
            "RPA@15": 0.75,
            "RPA@30": 0.875,
            "mse_unmasked": 0.02,
            "psnr_unmasked": 16.9897,
            "ssim": 0.6,
            "ssim_unmasked": 0.55,
            "mask_fraction": 0.785,
        }
    ]
    totals = {**rows[0], "num_images": 10, "mask": "model"}
    evaluate.write_summary(str(tmp_path), rows, totals)

    lines = (tmp_path / "metrics.txt").read_text(encoding="utf-8").splitlines()
    assert lines[:6] == [
        "mse: 0.01234568",
        "psnr: 19.1234",
        "RPA@5: 0.500000",
        "RPA@15: 0.750000",
        "RPA@30: 0.875000",
        "num_images: 10",
    ]
    assert "psnr_unmasked: 16.9897" in lines[6:]
    assert "ssim: 0.600000" in lines[6:]

    with open(tmp_path / "summary.csv", encoding="utf-8", newline="") as handle:
        content = handle.read()
    header = content.splitlines()[0]
    assert header.startswith("basename,mse,psnr,RPA@5,RPA@15,RPA@30"), header
    reader = list(csv.DictReader(content.replace("\n\n", "\n").splitlines()))
    assert [row["basename"] for row in reader] == ["000000", "average"]
    for row in reader:
        assert row["mse"] == "0.01234568"
        assert row["psnr"] == "19.1234"
        assert row["RPA@30"] == "0.875000"


# --- the shared image circle -------------------------------------------------------------------


@pytest.mark.parametrize("camera_model", ["pinhole", "fisheye", "ucm"])
def test_the_circle_override_is_the_exact_image_circle(camera_model):
    config = eval_config(camera_model, mask_override="circle")
    fxfycxcy = intrinsics(camera_model)
    expected = render_mask(fxfycxcy, H, W, camera_model, image_circle=True, exact=True)
    actual = evaluate.evaluation_mask(config, fxfycxcy, H, W)
    assert torch.equal(actual, expected)


def test_the_circle_override_scores_every_model_on_the_same_pixels():
    """The point of the override: one mask for the pinhole model and for the two wide angle ones.

    For a UCM camera render_mask also drops pixels outside the model's own image circle. At the
    alpha and focal length of the TartanAir fisheye that circle is wider than the image, so the three
    masks really are the same disc. A collapsed camera would break this, and the test would say so.
    """
    masks = {}
    for camera_model in ["pinhole", "fisheye", "ucm"]:
        config = eval_config(camera_model, mask_override="circle")
        masks[camera_model] = evaluate.evaluation_mask(config, intrinsics(camera_model), H, W)
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    radius = torch.sqrt((x + 0.5 - W / 2) ** 2 + (y + 0.5 - H / 2) ** 2)
    disc = (radius <= W / 2).float()[None, None].expand(2, 1, H, W)
    for camera_model, mask in masks.items():
        assert torch.equal(mask, disc), camera_model
    assert disc.mean().item() == pytest.approx(math.pi / 4, abs=0.002)


def test_the_circle_is_wider_than_the_mask_a_wide_angle_model_was_trained_with():
    """Guard against the override quietly doing nothing: it has to add the ring beyond 90 degrees.

    A gsplat run was trained on the pixels up to 90 degrees off axis, which is where its renderer
    stops, so for that model the circle really is the wider mask. A geer run in its projection tile
    mode was already trained on the whole circle, and there the override changes nothing, which is
    exactly why both have to be scored with it.
    """
    config = eval_config("fisheye", renderer="gsplat", tile_mode="pbf")
    fxfycxcy = intrinsics("fisheye")
    trained = evaluate.evaluation_mask(config, fxfycxcy, H, W)
    config.evaluation.mask_override = "circle"
    circle = evaluate.evaluation_mask(config, fxfycxcy, H, W)
    assert (circle >= trained).all()
    assert circle.mean().item() > trained.mean().item()


# --- the metrics ---------------------------------------------------------------------------------


def test_unmasked_mse_and_psnr_are_the_plain_full_frame_numbers():
    """A known constant error: the unmasked mse is that error squared and psnr follows from it."""
    target = torch.full((1, 1, 3, 16, 16), 0.5)
    pred = target + 0.1
    mask = torch.zeros(1, 1, 16, 16)
    mask[:, :, :4, :4] = 1.0  # a mask that covers a sixteenth of the image
    metrics = evaluate.image_metrics(pred, target, mask)
    assert metrics["mse_unmasked"].item() == pytest.approx(0.01, rel=1e-6)
    assert metrics["mse"].item() == pytest.approx(0.01, rel=1e-6)
    assert metrics["mask_fraction"].item() == pytest.approx(1.0 / 16.0)
    psnr = -10.0 * math.log10(metrics["mse_unmasked"].item())
    assert psnr == pytest.approx(20.0, rel=1e-6)


def test_the_mask_really_changes_the_masked_numbers():
    """Error only outside the mask: masked mse stays zero, the unmasked one does not.

    The mask stops three rows short of the error, because the 11 by 11 SSIM window reaches five
    pixels and would otherwise see the edge from inside the mask.
    """
    target = torch.zeros(1, 1, 3, 32, 32)
    pred = target.clone()
    pred[..., 16:, :] = 1.0
    mask = torch.zeros(1, 1, 32, 32)
    mask[:, :, :10, :] = 1.0
    metrics = evaluate.image_metrics(pred, target, mask)
    assert metrics["mse"].item() == pytest.approx(0.0, abs=1e-9)
    assert metrics["mse_unmasked"].item() == pytest.approx(0.5, rel=1e-6)
    assert metrics["ssim"].item() == pytest.approx(1.0, abs=1e-3)
    assert metrics["ssim_unmasked"].item() < 0.99
    assert metrics["mask_fraction"].item() == pytest.approx(10.0 / 32.0)


def numpy_ssim_map(pred, target, window_size=11, sigma=1.5, data_range=1.0):
    """An independent SSIM, written in numpy, with the settings of torchmetrics and skimage."""
    coords = np.arange(window_size) - (window_size - 1) / 2.0
    line = np.exp(-(coords**2) / (2.0 * sigma**2))
    line = line / line.sum()
    window = np.outer(line, line)
    pad = window_size // 2

    def blur(x):
        padded = np.pad(x, ((pad, pad), (pad, pad)), mode="reflect")
        out = np.zeros_like(x)
        for i in range(window_size):
            for j in range(window_size):
                out += window[i, j] * padded[i : i + x.shape[0], j : j + x.shape[1]]
        return out

    mu_a, mu_b = blur(pred), blur(target)
    var_a = blur(pred * pred) - mu_a * mu_a
    var_b = blur(target * target) - mu_b * mu_b
    cov = blur(pred * target) - mu_a * mu_b
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    return ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2))


def test_ssim_of_an_image_with_itself_is_one():
    generator = torch.Generator().manual_seed(1)
    image = torch.rand(2, 3, 64, 64, generator=generator)
    assert torch.allclose(ssim(image, image), torch.ones(2), atol=1e-5)


def test_ssim_of_two_constant_images_matches_the_closed_form():
    """Constant images have no variance, so SSIM collapses to (2ab + C1) / (a^2 + b^2 + C1)."""
    a, b = 0.3, 0.7
    c1 = 0.01**2
    expected = (2 * a * b + c1) / (a * a + b * b + c1)
    assert expected == pytest.approx(0.724185, abs=1e-6)  # the number that formula gives
    exact = ssim(
        torch.full((1, 1, 32, 32), a, dtype=torch.float64),
        torch.full((1, 1, 32, 32), b, dtype=torch.float64),
    ).item()
    assert exact == pytest.approx(expected, rel=1e-12)
    # In float32 the variances, which are exactly zero here, carry rounding noise of about 1e-8
    # against a C2 of 9e-4, so the value moves in the fourth decimal. The renderer works in float32.
    in_float32 = ssim(torch.full((1, 1, 32, 32), a), torch.full((1, 1, 32, 32), b)).item()
    assert in_float32 == pytest.approx(expected, abs=5e-4)


def test_ssim_matches_an_independent_numpy_implementation():
    generator = torch.Generator().manual_seed(3)
    pred = torch.rand(1, 1, 48, 48, generator=generator)
    target = (pred + 0.2 * torch.rand(1, 1, 48, 48, generator=generator)).clamp(0, 1)
    ours = ssim_map(pred, target)[0, 0].numpy()
    theirs = numpy_ssim_map(pred[0, 0].numpy().astype(np.float64), target[0, 0].numpy().astype(np.float64))
    assert np.abs(ours - theirs).max() < 1e-5
    assert ours.mean() == pytest.approx(theirs.mean(), abs=1e-6)


def test_ssim_of_a_shifted_pattern_is_what_the_numpy_implementation_says():
    """A checkerboard against itself shifted by one pixel, a value both implementations agree on."""
    base = np.indices((64, 64)).sum(axis=0) % 2
    pattern = base.astype(np.float64)
    shifted = np.roll(pattern, 1, axis=1)
    reference = float(numpy_ssim_map(pattern, shifted).mean())
    ours = ssim(
        torch.from_numpy(pattern)[None, None].float(),
        torch.from_numpy(shifted)[None, None].float(),
    ).item()
    assert ours == pytest.approx(reference, abs=1e-5)
    assert ours < 0.0, f"a one pixel shift of a checkerboard anticorrelates, got {ours}"


def test_a_mask_restricts_the_ssim_average():
    generator = torch.Generator().manual_seed(5)
    pred = torch.rand(1, 1, 32, 32, generator=generator)
    target = pred.clone()
    target[..., 16:, :] = torch.rand(1, 1, 16, 32, generator=generator)
    mask = torch.zeros(1, 1, 32, 32)
    mask[..., :10, :] = 1.0  # well clear of the disturbed half and of the window radius
    assert ssim(pred, target, mask).item() == pytest.approx(1.0, abs=1e-5)
    assert ssim(pred, target).item() < 0.9


# --- loading the released checkpoint ---------------------------------------------------------------


def synthetic_pretrained_state_dict(prefix=""):
    """A checkpoint shaped like erayzer_multi.pt: weights plus the VGG 19 of its perceptual loss."""
    state_dict = {
        f"{prefix}camera_token": torch.zeros(1, 1, 4),
        f"{prefix}register_token": torch.zeros(1, 4, 4),
        f"{prefix}image_tokenizer.1.weight": torch.zeros(4, 4),
    }
    for layer in range(1, 17):  # 16 convolutions with a weight and a bias, the 32 unexpected keys
        for part in ("weight", "bias"):
            state_dict[f"{prefix}loss_computer.perceptual_loss_module.Net.conv{layer}.{part}"] = torch.zeros(2)
    return state_dict


def test_the_released_checkpoint_loses_its_thirty_two_loss_tensors():
    checkpoint = {"model": synthetic_pretrained_state_dict(), "optimizer": {"state": {}}, "step": 152000}
    cleaned = evaluate.extract_state_dict(checkpoint)
    assert set(cleaned) == {"camera_token", "register_token", "image_tokenizer.1.weight"}
    assert len(checkpoint["model"]) - len(cleaned) == 32


def test_a_checkpoint_trained_under_ddp_loses_its_module_prefix():
    checkpoint = {"model": synthetic_pretrained_state_dict(prefix="module.")}
    assert set(evaluate.extract_state_dict(checkpoint)) == {
        "camera_token",
        "register_token",
        "image_tokenizer.1.weight",
    }


@pytest.mark.parametrize("wrapper", [None, "model", "state_dict", "model_state_dict"])
def test_both_checkpoint_layouts_load(wrapper):
    """A bare state dict and every wrapper the training code wrote give the same weights."""
    state_dict = synthetic_pretrained_state_dict()
    checkpoint = state_dict if wrapper is None else {wrapper: state_dict, "step": 1}
    assert set(evaluate.extract_state_dict(checkpoint)) == {
        "camera_token",
        "register_token",
        "image_tokenizer.1.weight",
    }


# --- the extra renders ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [(None, []), ("", []), ("0,2,4", [0, 2, 4]), ("0, 2 ,4", [0, 2, 4]), (2, [2]), ([1, 3], [1, 3])],
)
def test_sample_views_are_parsed(value, expected):
    assert evaluate.parse_view_indices(value) == expected


def input_poses(count=5, radius=2.0):
    """Cameras on a short arc, looking at the origin, as c2w matrices [count, 4, 4]."""
    poses = []
    for index in range(count):
        angle = math.radians(10.0 * index)
        forward = np.array([math.sin(angle), 0.0, math.cos(angle)])
        right = np.array([math.cos(angle), 0.0, -math.sin(angle)])
        up = np.cross(forward, right)
        pose = np.eye(4)
        pose[:3, 0], pose[:3, 1], pose[:3, 2] = right, up, forward
        pose[:3, 3] = -radius * forward
        poses.append(pose)
    return torch.from_numpy(np.stack(poses)).float()


def test_novel_poses_are_rotations_between_the_input_poses():
    poses = input_poses()
    novel = evaluate.novel_pose_c2ws(poses, count=3, lateral_offset=0.0)
    assert novel.shape == (3, 4, 4)
    for pose in novel:
        rotation = pose[:3, :3].double()
        assert torch.allclose(rotation @ rotation.T, torch.eye(3, dtype=torch.float64), atol=1e-6)
        assert torch.det(rotation).item() == pytest.approx(1.0, abs=1e-6)
    # no novel pose is one of the input poses, and all of them sit between them
    centres = novel[:, :3, 3]
    assert len({tuple(centre.tolist()) for centre in centres}) == 3
    for centre in centres:
        assert (poses[:, :3, 3] - centre).norm(dim=1).min().item() > 1e-3
        assert centre.norm().item() == pytest.approx(2.0, rel=0.02)


def test_the_lateral_offset_is_a_share_of_the_scene_radius():
    poses = input_poses()
    centres = poses[:, :3, 3].numpy()
    radius = float(np.linalg.norm(centres - centres.mean(axis=0, keepdims=True), axis=1).max())
    straight = evaluate.novel_pose_c2ws(poses, count=3, lateral_offset=0.0)
    shifted = evaluate.novel_pose_c2ws(poses, count=3, lateral_offset=0.3)
    for index in range(3):
        step = (shifted[index, :3, 3] - straight[index, :3, 3]).double()
        assert step.norm().item() == pytest.approx(0.3 * radius, rel=1e-5)
        # along the right axis of that very pose, alternating sides
        right = straight[index, :3, 0].double()
        assert torch.dot(step / step.norm(), right).item() == pytest.approx(1.0 if index % 2 == 0 else -1.0, abs=1e-5)
    assert torch.allclose(straight[:, :3, :3], shifted[:, :3, :3])


def test_a_single_input_pose_still_gives_poses():
    """The eval loader can hand over fewer poses than expected; it must not raise here."""
    novel = evaluate.novel_pose_c2ws(input_poses(count=1), count=3, lateral_offset=0.3)
    assert novel.shape == (3, 4, 4)
    assert torch.isfinite(novel).all()


def test_a_panel_puts_the_ground_truth_next_to_the_rendering(tmp_path):
    from PIL import Image

    target = torch.zeros(3, 8, 8)
    pred = torch.ones(3, 8, 8)
    path = str(tmp_path / "panel.png")
    evaluate.save_panel(target, pred, path, gap=4)
    array = np.asarray(Image.open(path))
    assert array.shape == (8, 20, 3)
    assert array[:, :8].max() == 0
    assert array[:, 12:].min() == 255


# --- the job script ----------------------------------------------------------------------------------


def test_the_slurm_script_parses():
    script = os.path.join(REPO_ROOT, "slurm", "evaluate_model.sh")
    assert os.path.exists(script)
    subprocess.run(["bash", "-n", script], check=True)


def test_the_slurm_script_follows_the_training_script():
    with open(os.path.join(REPO_ROOT, "slurm", "evaluate_model.sh"), encoding="utf-8") as handle:
        text = handle.read()
    assert "source \"${REPO_DIR:-/mnt/home/$USER/code/P3DCV}/slurm/common.sh\"" in text
    for name in ["RUN_NAME", "CKPT", "CONFIG", "ENV_NAME", "CAMERA_MODEL", "IMAGE", "in_container"]:
        assert name in text, name
    for key in ["evaluation.checkpoint_path", "evaluation.out_dir", "evaluation.sample_views",
                "evaluation.novel_views", "dataset.eval_path"]:
        assert key in text, key
