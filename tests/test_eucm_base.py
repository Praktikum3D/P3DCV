"""The Enhanced UCM as the base camera model, pinned to be the UCM as long as beta stays at 1.

EUCM projects a ray (x, y, z) to the normalized image radius

    rho = m / (alpha * d + (1 - alpha) * z),  m = sqrt(x^2 + y^2),  d = sqrt(beta * m^2 + z^2),

so beta = 1 makes d the length of the unit ray, d = 1, and the model is exactly the UCM. That is why
`eucm` can replace `ucm` as the base model today and grow a learnable beta later: the switch
`model.learn_beta` is off, the beta head is pinned to a constant 1.0, and nothing about the training
changes. These tests are the proof of that claim and the proof that the extra parameter is real.

The equivalence is checked bit for bit, not with a tolerance. It holds because beta enters the
unprojection only as the factor `beta * rho**2`, and multiplying by a float of exactly 1.0 is exact in
IEEE 754, while the forward projection evaluates d as sqrt(1 + (beta - 1) sin^2), which returns
exactly 1.0 for beta = 1 where sqrt(sin^2 + cos^2) would only return 1 up to rounding.

Run from the repository root:
    pytest tests/test_eucm_base.py
"""

import math
import os
import sys
import types

import pytest
import torch
from easydict import EasyDict as edict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train
from erayzer_core.model.gaussians_renderer import GaussianModel, render_opencv_cam_geer
from erayzer_core.model.utils_gaussian import (
    UCM_MODELS,
    Renderer,
    camera_rays,
    camera_rho,
    camera_theta,
    loss_render_mask,
    pixel_angles,
    render_mask,
    ucm_radial_coeffs,
)

H = W = 64
THETA_MAX = math.radians(195.0 / 2)  # lcam_fish sees 195 degrees across its image circle
F_TRUE = (W / 2) / THETA_MAX  # equidistant fisheye, 0.29383 * W

ALPHAS = [0.0, 0.2, 0.45, 0.5, 0.62, 0.8, 0.95]
FOCALS = [F_TRUE, 0.5 * W, 1.2 * W]
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def intrinsics(camera_model, f=F_TRUE, alpha=0.62, beta=1.0, v=2):
    """fxfycxcy [v, 4], UCM [v, 5] with alpha last, EUCM [v, 6] with beta after it."""
    k = torch.tensor([[f, f, W / 2, H / 2]]).repeat(v, 1)
    if camera_model in UCM_MODELS:
        k = torch.cat([k, torch.full((v, 1), alpha)], dim=1)
    if camera_model == "eucm":
        k = torch.cat([k, torch.full((v, 1), beta)], dim=1)
    return k


def both(f=F_TRUE, alpha=0.62, v=2, **kwargs):
    """The same camera written as ucm and as eucm with beta = 1."""
    return (intrinsics("ucm", f, alpha, v=v, **kwargs),
            intrinsics("eucm", f, alpha, beta=1.0, v=v, **kwargs))


def mask_config(camera_model, renderer="geer", tile_mode="projection"):
    return edict({"model": {"camera_model": camera_model, "renderer": renderer,
                            "geer_tile_mode": tile_mode},
                  "dataset": {"camera_name": "lcam_fish"},
                  "training": {"l2_loss_weight": 1.0}})


# ---------------------------------------------------------------------------------------------------
# 1. the math really is the Enhanced UCM, not the UCM under a new name


@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("alpha", [0.1, 0.4, 0.62, 0.85])
def test_the_eucm_unprojection_inverts_its_own_projection(beta, alpha):
    """Project an angle to a radius and unproject it again, in float64 and well inside the circle."""
    theta = torch.linspace(1e-6, 1.0, 401, dtype=torch.float64)
    a = torch.tensor(alpha, dtype=torch.float64)
    b = torch.tensor(beta, dtype=torch.float64)

    rho = camera_rho(theta, "eucm", a, b)
    assert torch.isfinite(rho).all() and (rho > 0).all()
    back = camera_theta(rho, "eucm", a, b)
    assert (back - theta).abs().max().item() < 1e-9


@pytest.mark.parametrize("alpha", [0.1, 0.4, 0.62, 0.85])
def test_beta_really_changes_the_camera(alpha):
    """A beta away from 1 bends the mapping, so the sixth intrinsic is not decoration.

    d = sqrt(beta m^2 + z^2) grows with beta, so the denominator of the projection grows with it and
    the same angle lands on a smaller radius. The whole image circle shrinks with beta accordingly.
    """
    theta = torch.linspace(0.05, 1.0, 64, dtype=torch.float64)
    a = torch.tensor(alpha, dtype=torch.float64)
    rho = [camera_rho(theta, "eucm", a, torch.tensor(b, dtype=torch.float64)) for b in (0.5, 1.0, 2.0)]

    assert (rho[0] > rho[1]).all() and (rho[1] > rho[2]).all()
    # a visible difference, not a rounding one
    assert (rho[0] - rho[1]).abs().max().item() > 1e-3
    assert (rho[1] - rho[2]).abs().max().item() > 1e-3


def test_eucm_with_beta_one_matches_the_ucm_formula_written_out_by_hand():
    """Against sin / (alpha + (1 - alpha) cos), the textbook UCM, independent of the implementation."""
    theta = torch.linspace(0.0, 1.4, 128, dtype=torch.float64)
    for alpha in (0.0, 0.3, 0.62, 0.9):
        a = torch.tensor(alpha, dtype=torch.float64)
        expected = torch.sin(theta) / (alpha + (1 - alpha) * torch.cos(theta))
        got = camera_rho(theta, "eucm", a, torch.ones_like(a))
        assert torch.allclose(got, expected, atol=1e-15, rtol=0)


def test_the_eucm_branch_insists_on_a_beta():
    """A missing beta is a programming error, not a silent fall back to the UCM."""
    rho = torch.linspace(0.0, 1.0, 8)
    alpha = torch.tensor(0.62)
    with pytest.raises(ValueError):
        camera_theta(rho, "eucm", alpha)
    with pytest.raises(ValueError):
        camera_rho(rho, "eucm", alpha)


# ---------------------------------------------------------------------------------------------------
# 2. at beta = 1 every piece of the camera math is the ucm one bit for bit


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("alpha", ALPHAS)
def test_camera_theta_is_bitwise_equal(alpha, dtype):
    """Over the whole radius range, including the part outside the image circle."""
    rho = torch.linspace(0.0, 4.0, 2001, dtype=dtype)
    a = torch.tensor(alpha, dtype=dtype)
    expected = camera_theta(rho, "ucm", a)
    got = camera_theta(rho, "eucm", a, torch.ones_like(a))
    assert torch.equal(expected, got)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("alpha", ALPHAS)
def test_camera_rho_is_bitwise_equal(alpha, dtype):
    theta = torch.linspace(0.0, 2.2, 2001, dtype=dtype)
    a = torch.tensor(alpha, dtype=dtype)
    expected = camera_rho(theta, "ucm", a)
    got = camera_rho(theta, "eucm", a, torch.ones_like(a))
    assert torch.equal(expected, got)


@pytest.mark.parametrize("f", FOCALS)
@pytest.mark.parametrize("alpha", ALPHAS)
def test_camera_rays_are_bitwise_equal(alpha, f):
    """The rays are what the plucker embedding and the 3DGEER renderer are built from."""
    k5, k6 = both(f, alpha, v=3)
    c2w = torch.eye(4)[None].repeat(3, 1, 1)
    c2w[1, :3, 3] = torch.tensor([0.3, -0.2, 1.5])
    c2w[2, :3, :3] = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])

    o5, d5 = camera_rays(c2w, k5, H, W, "ucm")
    o6, d6 = camera_rays(c2w, k6, H, W, "eucm")
    assert torch.equal(o5, o6) and torch.equal(d5, d6)
    assert torch.isfinite(d5).all()


@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("f", FOCALS)
@pytest.mark.parametrize("alpha", ALPHAS)
def test_pixel_angles_and_the_masks_are_bitwise_equal(alpha, f, exact):
    k5, k6 = both(f, alpha)
    for a, b in zip(pixel_angles(k5, H, W, "ucm", exact), pixel_angles(k6, H, W, "eucm", exact)):
        assert torch.equal(a, b)
    assert torch.equal(render_mask(k5, H, W, "ucm", True, exact),
                       render_mask(k6, H, W, "eucm", True, exact))


@pytest.mark.parametrize("renderer,tile_mode", [("gsplat", None), ("geer", "pbf"), ("geer", "projection")])
@pytest.mark.parametrize("alpha", ALPHAS)
def test_loss_render_mask_is_bitwise_equal(alpha, renderer, tile_mode):
    """The pixels the training loss and the evaluation score keep, in every renderer mode."""
    k5, k6 = both(alpha=alpha)
    expected = loss_render_mask(mask_config("ucm", renderer, tile_mode), k5, H, W)
    got = loss_render_mask(mask_config("eucm", renderer, tile_mode), k6, H, W)
    assert torch.equal(expected, got)
    assert expected.mean().item() > 0.1, "an empty mask would make this test vacuous"


@pytest.mark.parametrize("f", FOCALS)
@pytest.mark.parametrize("alpha", ALPHAS)
def test_the_kannala_brandt_fit_is_bitwise_equal(alpha, f):
    """3DGEER assigns its tiles with this fit, so it has to land on the same coefficients."""
    k5, k6 = both(f, alpha)
    expected = ucm_radial_coeffs(k5, H, W, "ucm")
    got = ucm_radial_coeffs(k6, H, W, "eucm")
    assert torch.equal(expected, got)
    assert torch.isfinite(expected).all()


def test_the_kannala_brandt_fit_survives_a_collapsed_eucm_camera():
    """The guard that keeps a broken camera from killing a run works for eucm as well."""
    bad = torch.tensor([[float("nan"), 1.0, W / 2, H / 2, 0.9, 1.0],
                        [1e-6, 1e-6, W / 2, H / 2, 0.99, 1.0],
                        [F_TRUE, F_TRUE, W / 2, H / 2, 0.62, float("inf")]])
    coeffs = ucm_radial_coeffs(bad, H, W, "eucm")
    assert torch.isfinite(coeffs).all()
    assert torch.equal(coeffs, torch.zeros_like(coeffs))


def test_the_fit_rejects_a_camera_model_that_is_not_unified():
    with pytest.raises(AssertionError):
        ucm_radial_coeffs(intrinsics("fisheye"), H, W, "fisheye")


# ---------------------------------------------------------------------------------------------------
# 3. the rendering itself, on the GPU


needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="3DGEER needs a GPU")


def tiny_scene(device, n=160, seed=3):
    """A cloud of Gaussians in front of the camera, big enough to cover a good part of the image."""
    g = torch.Generator().manual_seed(seed)
    xyz = torch.randn(n, 3, generator=g) * 0.6
    xyz[:, 2] += 1.5
    return GaussianModel(0).set_data(
        xyz.to(device),
        torch.randn(n, 1, 3, generator=g).to(device),
        torch.full((n, 3), -2.0).to(device),
        torch.randn(n, 4, generator=g).to(device),
        torch.full((n, 1), 2.0).to(device),
    )


@needs_gpu
@pytest.mark.parametrize("tile_mode", ["pbf", "projection"])
@pytest.mark.parametrize("alpha", [0.3, 0.62, 0.9])
def test_the_geer_rendering_is_bitwise_equal(alpha, tile_mode):
    """Same rays, same fitted Kannala Brandt tiles, so 3DGEER sees the very same float32 inputs."""
    device = "cuda"
    pc = tiny_scene(device)
    k5, k6 = both(f=0.22 * W, alpha=alpha, v=1)
    k5, k6 = k5.to(device), k6.to(device)
    c2w = torch.eye(4, device=device)[None]

    out5 = render_opencv_cam_geer(pc, H, W, c2w, k5, 0, camera_model="fisheye",
                                  radial_coeffs=ucm_radial_coeffs(k5, H, W, "ucm"),
                                  ray_camera_model="ucm", geer_tile_mode=tile_mode)["render"]
    out6 = render_opencv_cam_geer(pc, H, W, c2w, k6, 0, camera_model="fisheye",
                                  radial_coeffs=ucm_radial_coeffs(k6, H, W, "eucm"),
                                  ray_camera_model="eucm", geer_tile_mode=tile_mode)["render"]

    assert out5.std().item() > 0.01, "a blank image would make this test vacuous"
    assert torch.equal(out5, out6)


def renderer_config(camera_model, tile_mode="projection"):
    return edict({"model": {"camera_model": camera_model, "renderer": "geer",
                            "geer_tile_mode": tile_mode, "near_plane": 0.01,
                            "use_gsplat": True, "use_deferred_rendering": False,
                            "gaussians": {"sh_degree": 0}},
                  "dataset": {"camera_name": "lcam_fish"},
                  "training": {}})


@needs_gpu
@pytest.mark.parametrize("alpha", [0.3, 0.62, 0.9])
def test_the_renderer_module_renders_the_same_image(alpha):
    """Through Renderer.forward, which is what ERayZer calls and where the fit is wired up."""
    device = "cuda"
    g = torch.Generator().manual_seed(5)
    n = 160
    xyz = (torch.randn(n, 3, generator=g) * 0.6).to(device)
    xyz[:, 2] += 1.5
    args = (xyz[None], torch.randn(n, 1, 3, generator=g).to(device)[None],
            torch.full((n, 3), -2.0).to(device)[None], torch.randn(n, 4, generator=g).to(device)[None],
            torch.full((n, 1), 2.0).to(device)[None])
    c2w = torch.eye(4, device=device)[None, None]
    k5, k6 = both(f=0.22 * W, alpha=alpha, v=1)

    out5 = Renderer(renderer_config("ucm"))(*args, H, W, c2w, k5.to(device)[None]).render
    out6 = Renderer(renderer_config("eucm"))(*args, H, W, c2w, k6.to(device)[None]).render
    assert out5.std().item() > 0.01
    assert torch.equal(out5, out6)


# ---------------------------------------------------------------------------------------------------
# 4. the beta head of the model


class TorchAttention:
    """Stands in for xformers memory_efficient_attention.

    The real one only has a flash kernel here, which needs compute capability 8 and half precision;
    the development GPU is a 7.5 card. The replacement is plain scaled dot product attention, which
    is the same function, so the forward pass under test is the real ERayZer one everywhere else.
    """

    fmha = types.SimpleNamespace(flash=types.SimpleNamespace(FwOp=None, BwOp=None))

    @staticmethod
    def memory_efficient_attention(q, k, v, attn_bias=None, p=0.0, op=None):
        assert attn_bias is None, "the tiny test model uses no attention bias"
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # b l nh dh -> b nh l dh
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=p).transpose(1, 2)


def model_config(camera_model, **model):
    """The smallest ERayZer that can run a forward pass: two encoder layers, 16 by 16 images."""
    config = edict({
        "inference": False,
        "model": {
            "image_tokenizer": {"image_size": 16, "patch_size": 8, "in_channels": 3},
            "transformer": {"d": 16, "d_head": 8, "encoder_n_layer": 2, "encoder_geom_n_layer": 2,
                            "decoder_n_layer": 1, "neck_n_layer": 1},
            "pose_latent": {"length": 1, "canonical": "middle", "mode": "pairwise"},
            "camera_model": camera_model,
            "alpha_init": 0.62,
            "target_image": {"height": 16, "width": 16},
            "gaussians": {"n_gaussians": 1, "sh_degree": 3, "upsampler": {"upsample_factor": 1},
                          "range_setting": {"far": 8, "near": 0, "type": "linear_depth"}},
            "gs_per_pixel_sqrt": 1,
            "hard_pixelalign": True,
            "use_gsplat": True,
            "near_plane": 0.01,
            "renderer": "geer",
        },
        "training": {"num_views": 3, "num_input_views": 2, "num_target_views": 1,
                     "grad_checkpoint_every": 1},
    })
    config.model.update(model)
    return config


def build_model(config, monkeypatch, device="cuda"):
    pytest.importorskip("xformers", reason="erayzer.py imports xformers for its attention")
    from erayzer_core.model import utils_transformer
    from erayzer_core.model.erayzer import ERayZer

    monkeypatch.setattr(utils_transformer, "xops", TorchAttention, raising=False)
    torch.manual_seed(0)
    return ERayZer(config).to(device)


def sample_batch(device="cuda", seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"image": torch.rand(1, 3, 3, 16, 16, generator=g).to(device)}


@needs_gpu
def test_the_frozen_beta_head_gives_a_sixth_intrinsic_of_exactly_one(monkeypatch):
    """The predicted intrinsics carry beta, it is 1.0 to the bit, and no gradient reaches the head."""
    model = build_model(model_config("eucm"), monkeypatch)
    assert model.learn_beta is False
    assert {"beta_head.weight", "beta_head.bias"} <= set(dict(model.named_parameters()))

    result = model(sample_batch())
    beta = result.fxfycxcy[:, 5]
    assert result.fxfycxcy.shape[-1] == 6
    assert torch.equal(beta, torch.ones_like(beta))
    assert torch.equal(result.fxfycxcy_target[..., 5], torch.ones_like(result.fxfycxcy_target[..., 5]))

    (result.render.square().mean() + result.fxfycxcy.sum()).backward()
    for name in ("weight", "bias"):
        grad = getattr(model.beta_head, name).grad
        assert grad is None or torch.equal(grad, torch.zeros_like(grad)), name
    assert model.alpha_head.weight.grad.abs().sum().item() > 0, "alpha still has to learn"


@needs_gpu
def test_a_whole_eucm_forward_pass_equals_the_ucm_one(monkeypatch):
    """The end to end claim: the same weights, the same images, the same rendering.

    The eucm model is the ucm model plus a beta head that contributes a constant, so loading the ucm
    weights into it has to leave every output alone except for the extra intrinsic column.
    """
    ucm = build_model(model_config("ucm"), monkeypatch)
    eucm = build_model(model_config("eucm"), monkeypatch)
    missing, unexpected = eucm.load_state_dict(ucm.state_dict(), strict=False)
    assert sorted(missing) == ["beta_head.bias", "beta_head.weight"] and unexpected == []

    torch.manual_seed(1)
    out_ucm = ucm(sample_batch())
    torch.manual_seed(1)
    out_eucm = eucm(sample_batch())

    assert out_ucm.fxfycxcy.shape[-1] == 5 and out_eucm.fxfycxcy.shape[-1] == 6
    assert torch.equal(out_ucm.fxfycxcy, out_eucm.fxfycxcy[:, :5])
    assert out_ucm.render.std().item() > 0.001, "a blank rendering would make this test vacuous"
    assert torch.equal(out_ucm.render, out_eucm.render)


@needs_gpu
def test_the_learnable_beta_starts_at_one_and_stays_positive(monkeypatch):
    """With the switch on, beta is exp of the head, so it is positive by construction.

    The head is initialised to zero, so the learnable mode starts at exp(0) = 1, the UCM, and a step
    of a deliberately brutal learning rate must not take it out of range.
    """
    model = build_model(model_config("eucm", learn_beta=True), monkeypatch)
    assert model.learn_beta is True

    result = model(sample_batch())
    beta = result.fxfycxcy[:, 5]
    assert torch.equal(beta, torch.ones_like(beta)), "exp of a zero initialised head is exactly 1"

    (result.render.square().mean() + result.fxfycxcy[:, 5].sum()).backward()
    assert model.beta_head.bias.grad.abs().sum().item() > 0, "the switch has to let the head learn"

    optimizer = torch.optim.SGD(model.beta_head.parameters(), lr=1.0)
    optimizer.step()
    beta = model(sample_batch()).fxfycxcy[:, 5]
    assert torch.isfinite(beta).all() and (beta > 0).all()


@needs_gpu
def test_no_beta_head_without_the_eucm_camera_model(monkeypatch):
    """ucm, fisheye and pinhole runs keep the parameter set they had, so old checkpoints still load."""
    for camera_model in ("pinhole", "fisheye", "ucm"):
        model = build_model(model_config(camera_model), monkeypatch)
        assert not hasattr(model, "beta_head"), camera_model
        assert hasattr(model, "alpha_head") is (camera_model == "ucm")


# ---------------------------------------------------------------------------------------------------
# 5. the plumbing around the model


def test_the_camera_head_partition_includes_the_beta_head():
    """So beta gets the camera learning rate and its decay, like the focal length and alpha."""
    assert train.is_camera_parameter("module.beta_head.weight")
    assert "beta_head" in train.CAMERA_HEAD_NAMES

    model = torch.nn.Module()
    model.beta_head = torch.nn.Linear(8, 1)
    model.backbone = torch.nn.Linear(8, 8)
    optimizer = train.configure_optimizer(model, 0.05, 1e-4, (0.9, 0.95))
    camera_ids = {id(p) for group in optimizer.param_groups if group["name"].startswith("camera")
                  for p in group["params"]}
    assert camera_ids == {id(p) for p in model.beta_head.parameters()}


def test_camera_stats_reports_the_eucm_camera():
    """The FOV logged to wandb uses alpha and beta, and beta is logged next to alpha."""
    k6 = intrinsics("eucm", f=0.3 * W, alpha=0.62, v=4)
    k6[:, 0] /= W  # camera_stats works on the focal length normalized by the image width
    stats = train.camera_stats(edict(fxfycxcy=k6), "eucm")
    assert stats["camera/beta"] == 1.0
    assert stats["camera/alpha"] == pytest.approx(0.62)

    k5 = intrinsics("ucm", f=0.3 * W, alpha=0.62, v=4)
    k5[:, 0] /= W
    expected = train.camera_stats(edict(fxfycxcy=k5), "ucm")
    assert stats["camera/fov_deg"] == expected["camera/fov_deg"]
    assert "camera/beta" not in expected


def test_aug_stats_reads_alpha_and_beta_out_of_six_columns():
    """The augmentation scoring sees the same FOV for both spellings of the same camera."""
    batch = {"aug_fov_true": torch.tensor([150.0]), "aug_alpha_true": torch.tensor([0.62])}
    k6 = intrinsics("eucm", f=0.3 * W, alpha=0.62, v=2)
    k6[:, 0] /= W
    k5 = intrinsics("ucm", f=0.3 * W, alpha=0.62, v=2)
    k5[:, 0] /= W

    got = train.aug_stats(edict(fxfycxcy=k6), batch, "eucm")
    expected = train.aug_stats(edict(fxfycxcy=k5), batch, "ucm")
    assert got["aug/fov_err_deg"] == expected["aug/fov_err_deg"]
    assert got["aug/alpha_err"] == expected["aug/alpha_err"]


def test_aug_stats_falls_back_to_the_pinhole_fov_without_the_extra_intrinsics():
    """A four column vector carries no alpha and no beta, so camera_theta must not be asked for them."""
    batch = {"aug_fov_true": torch.tensor([150.0])}
    k4 = intrinsics("pinhole", f=0.3 * W, v=2)
    k4[:, 0] /= W
    for camera_model in ("ucm", "eucm"):
        stats = train.aug_stats(edict(fxfycxcy=k4), batch, camera_model)
        assert stats == train.aug_stats(edict(fxfycxcy=k4), batch, "pinhole")


def test_the_training_script_treats_eucm_like_the_other_wide_cameras():
    """F_SCALE and NEAR hang off a single `!= pinhole`, and LEARN_BETA reaches the config."""
    script = open(os.path.join(REPO, "slurm", "train_tartanair.sh")).read()
    assert '[[ "$CAMERA_MODEL" != pinhole ]]' in script
    assert 'F_SCALE="${F_SCALE:-0.26}" NEAR="${NEAR:-0.01}"' in script
    assert "-s model.learn_beta true" in script
    assert '[[ -n "${LEARN_BETA:-}" ]]' in script
