"""3DGEER tile assignment by projection, which renders a fisheye image out to its border.

The geer renderer used to stop at 90 degrees off axis because its bounding function works in tangent
space and culls every Gaussian centre behind the camera plane. Its new "projection" tile mode projects
the centres with the camera model instead and fills the whole image circle, so the loss mask may follow
it out there. These tests pin the switch (config to kernel) and the mask that depends on it, and check
on the GPU that the mode really draws content beyond 90 degrees while leaving nearer content alone.

Run from the repository root:
    pytest tests/test_geer_projection.py
"""

import inspect
import math
import os
import sys

import pytest
import torch
from easydict import EasyDict as edict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import evaluate
import train
from erayzer_core.model import gaussians_renderer
from erayzer_core.model.gaussians_renderer import GaussianModel, render_opencv_cam_geer
from erayzer_core.model.utils_gaussian import Renderer, loss_render_mask, pixel_angles, render_mask

H = W = 256
THETA_MAX = math.radians(195.0 / 2)  # lcam_fish sees 195 degrees across its image circle
F_TRUE = (W / 2) / THETA_MAX  # equidistant fisheye, 0.29383 * W


def intrinsics(camera_model, f=F_TRUE, alpha=0.6, v=2):
    """fxfycxcy [v, 4], for UCM [v, 5] with alpha last, principal point in the image centre."""
    k = torch.tensor([[f, f, W / 2, H / 2]]).repeat(v, 1)
    if camera_model == "ucm":
        k = torch.cat([k, torch.full((v, 1), alpha)], dim=1)
    return k


def config_of(camera_model, renderer, tile_mode=None, camera_name="lcam_fish"):
    """A config as train.py and evaluate.py see it, without geer_tile_mode unless one is given."""
    model = {"camera_model": camera_model, "renderer": renderer}
    if tile_mode is not None:
        model["geer_tile_mode"] = tile_mode
    return edict({"model": model, "dataset": {"camera_name": camera_name},
                  "training": {"l2_loss_weight": 1.0}})


def loss_inputs(fxfycxcy):
    """A render and a target for compute_loss, batch of 1 with as many views as fxfycxcy has."""
    v = fxfycxcy.shape[0]
    g = torch.Generator().manual_seed(0)
    render = torch.rand(1, v, 3, H, W, generator=g)
    target = torch.rand(1, v, 3, H, W, generator=g)
    result = edict(render=render, fxfycxcy_target=fxfycxcy[None], target_image=target)
    return result, {"image": target}


def training_mask_fraction(config, fxfycxcy):
    """The share of pixels the training loss keeps, straight out of compute_loss."""
    result, batch = loss_inputs(fxfycxcy)
    return train.compute_loss(result, batch, config, edict()).mask_fraction.item()


@pytest.mark.parametrize("camera_model", ["fisheye", "ucm"])
def test_projection_mode_masks_the_whole_image_circle(camera_model):
    """With the projection tile mode the mask is the camera model's own range, not the 90 degree cap."""
    fxfycxcy = intrinsics(camera_model)
    mask = loss_render_mask(config_of(camera_model, "geer", "projection"), fxfycxcy, H, W)
    assert torch.equal(mask, render_mask(fxfycxcy, H, W, camera_model, True, exact=True))
    capped = render_mask(fxfycxcy, H, W, camera_model, True)
    assert (mask >= capped).all(), "the 90 degree mask has to stay a subset"
    assert mask.mean().item() > capped.mean().item()


@pytest.mark.parametrize("camera_model", ["fisheye", "ucm"])
@pytest.mark.parametrize("renderer,tile_mode", [("gsplat", None), ("gsplat", "projection"),
                                                ("geer", None), ("geer", "pbf")])
def test_everything_but_geer_projection_keeps_the_90_degree_mask(camera_model, renderer, tile_mode):
    """Default and gsplat behaviour is untouched, including a tile mode the gsplat path ignores."""
    fxfycxcy = intrinsics(camera_model)
    old = render_mask(fxfycxcy, H, W, camera_model, True)  # the mask this branch had before
    assert torch.equal(loss_render_mask(config_of(camera_model, renderer, tile_mode), fxfycxcy, H, W), old)
    assert training_mask_fraction(config_of(camera_model, renderer, tile_mode), fxfycxcy) == pytest.approx(
        old.mean().item())


def test_the_gained_ring_is_the_fisheye_image_circle_beyond_90_degrees():
    """The pixels the two masks differ in are exactly the dead ring the 90 degree cap gave up.

    At the true focal length of a 195 degree lcam_fish image the whole disc is 0.785 of the image and
    the 90 degree mask 0.669 of it, so the projection tile mode brings back about 15 percent more pixels.
    """
    fxfycxcy = intrinsics("fisheye")
    capped = loss_render_mask(config_of("fisheye", "geer", "pbf"), fxfycxcy, H, W)
    full = loss_render_mask(config_of("fisheye", "geer", "projection"), fxfycxcy, H, W)

    assert capped.mean().item() == pytest.approx(0.669, abs=0.004)
    assert full.mean().item() == pytest.approx(math.pi / 4, abs=0.002)
    gained = (full - capped)[:, 0] > 0
    assert gained.float().mean().item() / full.mean().item() == pytest.approx(0.148, abs=0.01)

    _, _, radius, theta, _ = pixel_angles(fxfycxcy, H, W, "fisheye")
    assert (theta[gained] > math.pi / 2).all()
    assert (radius[gained] <= W / 2).all()


def test_pinhole_has_no_mask_at_all():
    """The projection tile mode does not reach beyond 90 degrees for pinhole, and pinhole has no mask."""
    fxfycxcy = intrinsics("pinhole", f=W / 2)
    config = config_of("pinhole", "geer", "projection")
    result, batch = loss_inputs(fxfycxcy)
    out = train.compute_loss(result, batch, config, edict())
    assert out.mask_fraction.item() == 1.0


@pytest.mark.parametrize("camera_model", ["fisheye", "ucm"])
@pytest.mark.parametrize("tile_mode", [None, "pbf", "projection"])
def test_training_and_evaluation_mask_the_same_pixels(camera_model, tile_mode):
    """Both read the mask out of the same function, so they cannot drift apart again."""
    fxfycxcy = intrinsics(camera_model)
    config = config_of(camera_model, "geer", tile_mode)
    expected = loss_render_mask(config, fxfycxcy, H, W).mean().item()
    assert training_mask_fraction(config, fxfycxcy) == pytest.approx(expected)
    for source in [inspect.getsource(train.compute_loss), inspect.getsource(evaluate.main)]:
        assert "loss_render_mask(config" in source
        assert source.count("render_mask(") == 1, "a second mask would be a second convention"


class FakeRasterization:
    """Stands in for gsplat.rasterization and records the keyword arguments it was called with."""

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append(kwargs)
        width, height = args[7], args[8]
        return torch.zeros(1, height, width, 3), None, None


def renderer_call(config, monkeypatch, camera_model="fisheye", h=16, w=16, v=1):
    """Run Renderer.forward on the CPU with the rasterizer replaced, and return its recorded calls."""
    fake = FakeRasterization()
    monkeypatch.setattr(gaussians_renderer, "rasterization", fake)
    fxfycxcy = intrinsics(camera_model, f=w / 2, v=v).clone()
    fxfycxcy[:, 2], fxfycxcy[:, 3] = w / 2, h / 2
    C2W = torch.eye(4)[None].repeat(v, 1, 1)
    n = 8
    g = torch.Generator().manual_seed(0)
    xyz = torch.randn(n, 3, generator=g)
    features = torch.randn(n, (config.model.gaussians.sh_degree + 1) ** 2, 3, generator=g)
    scaling, rotation = torch.full((n, 3), -3.0), torch.randn(n, 4, generator=g)
    opacity = torch.zeros(n, 1)
    Renderer(config)(*[t[None] for t in (xyz, features, scaling, rotation, opacity)],
                     h, w, C2W[None], fxfycxcy[None])
    return fake.calls


def render_config(renderer, tile_mode=None, camera_model="fisheye"):
    config = config_of(camera_model, renderer, tile_mode)
    config.model.update({"gaussians": {"sh_degree": 0}, "near_plane": 0.01,
                         "use_gsplat": True, "use_deferred_rendering": False})
    return config


@pytest.mark.parametrize("tile_mode,expected", [(None, "pbf"), ("pbf", "pbf"), ("projection", "projection")])
def test_the_configured_tile_mode_reaches_the_rasterizer(tile_mode, expected, monkeypatch):
    """The config key travels through Renderer.forward and render_opencv_cam_geer into gsplat."""
    calls = renderer_call(render_config("geer", tile_mode), monkeypatch, v=2)
    assert len(calls) == 2, "3DGEER renders one view per call"
    for kwargs in calls:
        assert kwargs["geer_tile_mode"] == expected


def test_the_gsplat_path_never_sees_the_tile_mode(monkeypatch):
    """gsplat has no tile mode, so the geer arguments must not leak into its call."""
    calls = renderer_call(render_config("gsplat", "projection"), monkeypatch)
    assert len(calls) == 1
    assert "geer_tile_mode" not in calls[0] and "with_geer" not in calls[0]


def test_render_opencv_cam_geer_defaults_to_pbf(monkeypatch):
    """Calling the renderer without the new argument keeps the behaviour it had before."""
    fake = FakeRasterization()
    monkeypatch.setattr(gaussians_renderer, "rasterization", fake)
    n = 4
    g = torch.Generator().manual_seed(0)
    pc = GaussianModel(0).set_data(torch.randn(n, 3, generator=g), torch.randn(n, 1, 3, generator=g),
                                   torch.full((n, 3), -3.0), torch.randn(n, 4, generator=g), torch.zeros(n, 1))
    fxfycxcy = intrinsics("fisheye", f=8.0, v=1)
    fxfycxcy[:, 2], fxfycxcy[:, 3] = 8.0, 8.0
    render_opencv_cam_geer(pc, 16, 16, torch.eye(4)[None], fxfycxcy, 0)
    assert fake.calls[0]["geer_tile_mode"] == "pbf"


# ---------------------------------------------------------------------------------------------------
# GPU: the rendering itself

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="3DGEER needs a GPU")
GH = GW = 64
GF = (GW / 2) / THETA_MAX  # the lcam_fish geometry again, 195 degrees across the image circle


def one_gaussian(device, theta_deg, distance=2.0, scale=0.25):
    """A single dark Gaussian at theta_deg off the optical axis, on the diagonal of the image."""
    theta = math.radians(theta_deg)
    direction = torch.tensor([math.sin(theta) / math.sqrt(2), math.sin(theta) / math.sqrt(2), math.cos(theta)])
    xyz = (direction * distance)[None].to(device)
    features = torch.full((1, 1, 3), -0.5 / 0.28209479177387814, device=device)  # black after eval_sh
    scaling = torch.full((1, 3), math.log(scale), device=device)
    rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    opacity = torch.full((1, 1), 8.0, device=device)  # sigmoid of 8 is opaque
    return GaussianModel(0).set_data(xyz, features, scaling, rotation, opacity)


def fisheye_camera(device, v=1):
    fxfycxcy = torch.tensor([[GF, GF, GW / 2, GH / 2]], device=device).repeat(v, 1)
    return torch.eye(4, device=device)[None].repeat(v, 1, 1), fxfycxcy


def geer_render(pc, C2W, fxfycxcy, tile_mode):
    radial = torch.zeros(fxfycxcy.shape[0], 4, device=fxfycxcy.device)  # zeros are the equidistant model
    with torch.no_grad():
        out = render_opencv_cam_geer(pc, GH, GW, C2W, fxfycxcy, 0, camera_model="fisheye",
                                     radial_coeffs=radial, ray_camera_model="fisheye",
                                     geer_tile_mode=tile_mode)
    return out["render"]


@gpu
@pytest.mark.parametrize("theta_deg", [100.0, 120.0])
def test_projection_mode_draws_beyond_90_degrees(theta_deg):
    """A Gaussian past the camera plane shows up with projection and stays invisible with pbf."""
    device = "cuda"
    pc = one_gaussian(device, theta_deg)
    C2W, fxfycxcy = fisheye_camera(device)
    radius = GF * math.radians(theta_deg)
    row = int(GH / 2 + radius / math.sqrt(2))
    col = int(GW / 2 + radius / math.sqrt(2))
    assert radius > GW / 2 and 0 <= row < GH and 0 <= col < GW, "the test pixel has to be in the image"

    pbf = geer_render(pc, C2W, fxfycxcy, "pbf")
    projection = geer_render(pc, C2W, fxfycxcy, "projection")
    assert torch.isfinite(pbf).all() and torch.isfinite(projection).all()
    assert pbf.min().item() > 0.99, f"pbf drew something at {theta_deg} degrees"
    assert projection[0, :, row, col].max().item() < 0.2, "the Gaussian is missing from its own pixel"
    dark = projection.amin(1)[0] < 0.5
    assert dark.sum().item() > 1, "a single pixel is too little for a Gaussian of this size"
    rows, cols = dark.nonzero(as_tuple=True)
    assert abs(rows.float().mean().item() - row) < 2 and abs(cols.float().mean().item() - col) < 2


@gpu
def test_the_two_modes_agree_in_front_of_the_camera():
    """Below 90 degrees the projection mode has to reproduce the image the pbf mode renders."""
    device = "cuda"
    C2W, fxfycxcy = fisheye_camera(device)
    for theta_deg in [0.0, 35.0, 70.0]:
        pc = one_gaussian(device, theta_deg)
        pbf = geer_render(pc, C2W, fxfycxcy, "pbf")
        projection = geer_render(pc, C2W, fxfycxcy, "projection")
        assert (pbf.amin(1) < 0.5).sum().item() > 1, f"nothing to compare at {theta_deg} degrees"
        diff = (pbf - projection).abs()
        assert diff.max().item() < 1e-3, f"{theta_deg} degrees: max abs diff {diff.max().item()}"


@gpu
def test_the_renderer_passes_the_tile_mode_on_the_gpu():
    """The config path end to end, with the real kernel behind it."""
    device = "cuda"
    pc = one_gaussian(device, 120.0)
    C2W, fxfycxcy = fisheye_camera(device)
    leaves = [pc._xyz, pc._features_dc, pc._scaling, pc._rotation, pc._opacity]
    images = {}
    for tile_mode in ["pbf", "projection"]:
        config = render_config("geer", tile_mode)
        with torch.no_grad():
            out = Renderer(config)(*[t[None] for t in leaves], GH, GW, C2W[None], fxfycxcy[None])
        images[tile_mode] = out.render
    assert images["pbf"].min().item() > 0.99
    assert images["projection"].min().item() < 0.2


@gpu
def test_gradients_still_flow_in_projection_mode():
    """The mask opens up pixels the loss now has to be able to train on."""
    device = "cuda"
    pc = one_gaussian(device, 120.0)
    xyz = pc._xyz.detach().requires_grad_()
    pc._xyz = xyz
    C2W, fxfycxcy = fisheye_camera(device)
    fxfycxcy = fxfycxcy.requires_grad_()
    radial = torch.zeros(1, 4, device=device)
    out = render_opencv_cam_geer(pc, GH, GW, C2W, fxfycxcy, 0, camera_model="fisheye", radial_coeffs=radial,
                                 ray_camera_model="fisheye", geer_tile_mode="projection")["render"]
    out.sum().backward()
    for name, grad in [("xyz", xyz.grad), ("fxfycxcy", fxfycxcy.grad)]:
        assert grad is not None and torch.isfinite(grad).all(), f"bad gradient for {name}"
        assert grad.abs().max().item() > 0.0, f"zero gradient for {name}"
