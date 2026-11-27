"""The viewer renders with the ray maths the model trains with, and nothing else.

The acceptance criterion of the viewer is that the rays it hands to 3DGEER are the very same rays
`cam_info_to_plucker` builds for the plucker embeddings of that camera. Both go through
`camera_rays`, so the test demands bit identity, not a tolerance. The rest pins the pieces that
surround those rays: the focal length derived from a field of view and its clamps, the projection
that draws the camera frusta, checkpoint discovery and the synthetic development scene.

Everything but the last test runs on the CPU. Run from the repository root:
    pytest tests/test_viewer_rays.py
"""

import math
import os
import sys

import numpy as np
import pytest
import torch
from easydict import EasyDict as edict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from erayzer_core.model.erayzer import cam_info_to_plucker
from erayzer_core.model.gaussians_renderer import build_rotation
from erayzer_core.model.utils_gaussian import camera_rays, camera_theta, pixel_angles
from viewer.camera import (
    estimated_camera_info,
    fov_to_fxfycxcy,
    frustum_segments,
    build_rays,
    project_points,
    theta_max,
)
from viewer.scene import list_checkpoints, load_scene, make_synthetic_scene, save_scene

H, W = 24, 32
MODELS = [("pinhole", None), ("fisheye", None), ("ucm", 0.2), ("ucm", 0.7)]


def random_c2w(seed: int) -> torch.Tensor:
    """A random pose as [1, 4, 4], rotation through the quaternion helper of the renderer."""
    generator = torch.Generator().manual_seed(seed)
    quat = torch.randn(1, 4, generator=generator)
    c2w = torch.eye(4, dtype=torch.float32)[None].clone()
    c2w[0, :3, :3] = build_rotation(quat)[0]
    c2w[0, :3, 3] = torch.randn(3, generator=generator)
    return c2w


def normalized_intrinsics(alpha):
    """fxfycxcy normalized by the image size, the form the model predicts, [1, 4] or [1, 5]."""
    values = [0.9, 0.95, 0.5, 0.5]
    if alpha is not None:
        values.append(alpha)
    return torch.tensor([values], dtype=torch.float32)


@pytest.mark.parametrize("camera_model, alpha", MODELS)
def test_viewer_rays_are_identical_to_plucker(camera_model, alpha):
    """The viewer rays and the plucker rays of the same camera agree bit for bit."""
    c2w = random_c2w(7)
    norm = normalized_intrinsics(alpha)

    reference = cam_info_to_plucker(
        c2w, norm, edict(height=H, width=W), normalized=True, return_moment=False, camera_model=camera_model
    )  # [1, 6, h, w], channels 0 to 2 are the origin and 3 to 5 the direction

    pixels = norm.clone()  # the denormalisation of cam_info_to_plucker, same order of operations
    pixels[:, 0] *= W
    pixels[:, 1] *= H
    pixels[:, 2] *= W
    pixels[:, 3] *= H
    ray_o, ray_d = build_rays(c2w, pixels, H, W, camera_model)

    assert torch.equal(ray_o.permute(0, 3, 1, 2), reference[:, :3])
    assert torch.equal(ray_d.permute(0, 3, 1, 2), reference[:, 3:])


@pytest.mark.parametrize("camera_model, alpha", MODELS)
@pytest.mark.parametrize("fov_deg", [20.0, 60.0, 120.0, 170.0, 300.0])
def test_fov_to_focal_length_round_trip(camera_model, alpha, fov_deg):
    """The focal length of a field of view gives that field of view back through `camera_theta`."""
    fxfycxcy, fov_used = fov_to_fxfycxcy(camera_model, fov_deg, alpha, W, H)
    assert fov_used <= fov_deg + 1e-9
    limit = theta_max(camera_model, alpha)
    assert math.radians(fov_used) / 2.0 <= limit + 1e-9

    rho = (W / 2.0) / fxfycxcy[0, 0]
    alpha_t = fxfycxcy[:, 4] if camera_model == "ucm" else None
    back = 2.0 * camera_theta(rho, camera_model, alpha_t)
    assert abs(float(back) - math.radians(fov_used)) < 1e-5


def test_fov_clamps_report_the_value_they_used():
    """Asking for more than the camera model can do returns the clamped field of view."""
    _, pinhole = fov_to_fxfycxcy("pinhole", 300.0, None, W, H)
    assert abs(pinhole - 170.0) < 1e-6

    _, ucm = fov_to_fxfycxcy("ucm", 300.0, 0.9, W, H)
    assert ucm < 300.0
    assert abs(math.radians(ucm) / 2.0 - (theta_max("ucm", 0.9) - math.radians(0.5))) < 1e-9


@pytest.mark.parametrize("alpha", [0.6, 0.7, 0.9])
def test_theta_max_matches_the_ucm_validity_of_pixel_angles(alpha):
    """Above alpha 0.5 the image radius peaks, and a pixel past that peak is the one `pixel_angles` drops."""
    limit = theta_max("ucm", alpha)
    rho_max = math.sin(limit) / (alpha + (1.0 - alpha) * math.cos(limit))
    # The peak radius is exactly the bound behind the validity test 1 - (2 alpha - 1) rho^2 >= 0.
    assert abs(rho_max - 1.0 / math.sqrt(2.0 * alpha - 1.0)) < 1e-9

    f = 100.0
    for factor, expected in ((0.999, True), (1.001, False)):
        radius = rho_max * factor * f
        # A 3 x 1 image whose last pixel centre sits exactly that far from the principal point.
        fxfycxcy = torch.tensor([[f, f, 2.5 - radius, 0.5, alpha]], dtype=torch.float64)
        _, _, _, _, valid = pixel_angles(fxfycxcy, 1, 3, "ucm", exact=True)
        assert bool(valid[0, 0, 2]) is expected


@pytest.mark.parametrize("alpha", [0.0, 0.2, 0.5])
def test_theta_max_is_the_pole_of_the_ucm_radius(alpha):
    """At and below alpha 0.5 the image radius runs to infinity instead, which is what the clamp avoids.

    `pixel_angles` marks nothing invalid there, because every finite radius still maps back to an
    angle below the limit. The limit matters anyway, because a field of view beyond it would ask for
    a zero or negative focal length.
    """
    limit = theta_max("ucm", alpha)
    denominator = alpha + (1.0 - alpha) * math.cos(limit)
    assert abs(denominator) < 1e-5  # `theta_max` keeps alpha a millionth away from 0 and 1
    inside = math.sin(limit - 1e-3) / (alpha + (1.0 - alpha) * math.cos(limit - 1e-3))
    assert inside > 10.0
    fxfycxcy, fov_used = fov_to_fxfycxcy("ucm", 360.0, alpha, W, H)
    assert fxfycxcy[0, 0] > 0.0
    assert math.radians(fov_used) / 2.0 < limit


@pytest.mark.parametrize("camera_model, alpha", MODELS)
def test_projection_inverts_the_rays(camera_model, alpha):
    """Points on the ray of a pixel project back onto that pixel."""
    c2w = random_c2w(11)
    fov = {"pinhole": 90.0, "fisheye": 195.0, "ucm": 150.0}[camera_model]
    fxfycxcy, _ = fov_to_fxfycxcy(camera_model, fov, alpha, W, H)
    ray_o, ray_d = camera_rays(c2w, fxfycxcy, H, W, camera_model)

    generator = torch.Generator().manual_seed(3)
    rows = torch.randint(0, H, (24,), generator=generator)
    cols = torch.randint(0, W, (24,), generator=generator)
    dist = torch.rand(24, generator=generator) * 9.5 + 0.5
    points = ray_o[0, rows, cols] + ray_d[0, rows, cols] * dist[:, None]

    uv, valid = project_points(points, c2w[0], fxfycxcy[0], camera_model)
    assert bool(valid.all())
    assert torch.allclose(uv[:, 0], cols.float(), atol=1e-3)
    assert torch.allclose(uv[:, 1], rows.float(), atol=1e-3)


def test_projection_validity_behind_the_camera():
    """A pinhole drops a point behind it, a fisheye still images one at 120 degrees off axis."""
    c2w = torch.eye(4, dtype=torch.float32)
    behind = torch.tensor([[0.1, 0.0, -2.0]], dtype=torch.float32)
    wide = torch.tensor([[math.sin(math.radians(120.0)), 0.0, math.cos(math.radians(120.0))]], dtype=torch.float32)

    pinhole, _ = fov_to_fxfycxcy("pinhole", 90.0, None, W, H)
    fisheye, _ = fov_to_fxfycxcy("fisheye", 195.0, None, W, H)
    assert not bool(project_points(behind, c2w, pinhole[0], "pinhole")[1][0])
    assert not bool(project_points(wide, c2w, pinhole[0], "pinhole")[1][0])
    assert bool(project_points(wide, c2w, fisheye[0], "fisheye")[1][0])
    assert bool(project_points(behind, c2w, fisheye[0], "fisheye")[1][0])


def test_frustum_segments_span_the_image_border():
    """The cone edges end on the image border of the camera they belong to."""
    c2w = random_c2w(5)
    fxfycxcy, _ = fov_to_fxfycxcy("pinhole", 60.0, None, 64, 64)
    segments = frustum_segments(c2w[0], fxfycxcy[0], 64, 64, "pinhole", depth=1.0)
    assert len(segments) == 5

    corners = torch.tensor(np.stack([segment[-1] for segment in segments[:4]]), dtype=torch.float32)
    uv, valid = project_points(corners, c2w[0], fxfycxcy[0], "pinhole")
    assert bool(valid.all())
    # The image corners of a 64 x 64 image sit at the pixel coordinates -0.5 and 63.5.
    expected = torch.tensor([[-0.5, -0.5], [63.5, -0.5], [63.5, 63.5], [-0.5, 63.5]])
    assert torch.allclose(uv, expected, atol=1e-2)

    apexes = torch.tensor(np.stack([segment[0] for segment in segments[:4]]), dtype=torch.float32)
    assert torch.allclose(apexes, c2w[0, :3, 3][None].expand(4, -1), atol=1e-5)


def test_estimated_camera_info_reports_the_training_formula():
    """The reported field of view is the one `train.py` logs, and alpha comes through for UCM."""
    norm = torch.tensor([[0.5, 0.5, 0.5, 0.5], [1.0, 1.0, 0.5, 0.5]], dtype=torch.float32)
    info = estimated_camera_info(norm, "pinhole")
    assert abs(info[0]["fov_deg"] - 90.0) < 1e-3
    assert abs(info[1]["fov_deg"] - 2.0 * math.degrees(math.atan(0.5))) < 1e-3

    ucm = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.62]], dtype=torch.float32)
    assert abs(estimated_camera_info(ucm, "ucm")[0]["alpha"] - 0.62) < 1e-6


# ------------------------------------------------------------------------------------------------
# checkpoints and scenes
# ------------------------------------------------------------------------------------------------


def write_run(root, name, files, camera_model=None):
    import yaml

    run = os.path.join(root, "runs", name)
    os.makedirs(run, exist_ok=True)
    for file_name in files:
        open(os.path.join(run, file_name), "wb").close()
    if camera_model is not None:
        config = {"model": {"camera_model": camera_model, "renderer": "geer", "geer_tile_mode": "projection"}}
        with open(os.path.join(run, "config.yaml"), "w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle)
    return run


def test_checkpoint_discovery_picks_the_newest_and_reads_the_camera(tmp_path):
    """`last.pt` wins, otherwise the highest step, a run without a checkpoint is skipped."""
    root = str(tmp_path)
    write_run(root, "a", [f"ckpt_{1000:016d}.pt", f"ckpt_{2000:016d}.pt", "last.pt"], "ucm")
    write_run(root, "b", [f"ckpt_{9:016d}.pt", f"ckpt_{30000:016d}.pt"], "fisheye")
    write_run(root, "c", [])
    write_run(root, "d", ["last.pt"])  # no config.yaml at all
    pretrained = os.path.join(root, "erayzer_multi.pt")
    open(pretrained, "wb").close()

    entries = {entry.id: entry for entry in list_checkpoints(pretrained, os.path.join(root, "runs"))}
    assert set(entries) == {"pretrained", "runs/a", "runs/b", "runs/d"}  # c has no checkpoint
    assert entries["runs/a"].ckpt_path.endswith("last.pt")
    assert entries["runs/b"].ckpt_path.endswith(f"ckpt_{30000:016d}.pt")
    assert entries["runs/a"].camera_settings["camera_model"] == "ucm"
    assert entries["runs/b"].camera_settings["camera_model"] == "fisheye"
    assert "ucm" in entries["runs/a"].label and "projection" in entries["runs/a"].label
    # Without a config the entry survives, says so, and falls back to the training defaults.
    assert entries["runs/d"].warning
    assert entries["runs/d"].camera_settings["camera_model"] == "pinhole"
    assert list_checkpoints(None, None) == []


def test_synthetic_scene_uses_raw_gaussian_values():
    """`set_data` takes raw values, so the activations have to land in the intended ranges."""
    scene = make_synthetic_scene(torch.device("cpu"))
    assert 15000 < scene.num_gaussians < 25000
    pc = scene.pc
    assert pc.get_xyz.shape == (scene.num_gaussians, 3)
    assert pc.get_features.shape == (scene.num_gaussians, 1, 3)  # sh_degree 0
    assert pc._rotation.shape == (scene.num_gaussians, 4)
    assert pc.get_xyz.dtype == torch.float32

    scaling = pc.get_scaling
    assert float(scaling.min()) > 0.01 and float(scaling.max()) < 0.2
    opacity = pc.get_opacity
    assert 0.8 < float(opacity.min()) <= float(opacity.max()) < 1.0
    assert torch.allclose(pc.get_rotation.norm(dim=-1), torch.ones(scene.num_gaussians), atol=1e-6)

    assert scene.c2w_input.shape == (5, 4, 4)
    assert scene.fxfycxcy_input.shape == (5, 4)
    # The stand in cameras look at the origin from a circle around it.
    forward = scene.c2w_input[:, :3, 2]
    position = scene.c2w_input[:, :3, 3]
    assert torch.allclose(forward, -position / position.norm(dim=-1, keepdim=True), atol=1e-5)
    determinant = torch.linalg.det(scene.c2w_input[:, :3, :3])
    assert torch.allclose(determinant, torch.ones(5), atol=1e-5)  # a rotation, not a mirror


def test_scene_survives_a_round_trip_through_a_file(tmp_path):
    """A scene written on the cluster has to come back identical on the development GPU."""
    scene = make_synthetic_scene(torch.device("cpu"))
    path = save_scene(scene, str(tmp_path / "scene.pt"))
    back = load_scene(path, torch.device("cpu"))
    assert back.num_gaussians == scene.num_gaussians
    assert back.sh_degree == scene.sh_degree
    assert back.camera_model == scene.camera_model
    assert back.tile_mode == scene.tile_mode
    assert back.input_hw == scene.input_hw
    assert torch.equal(back.pc.get_xyz, scene.pc.get_xyz)
    assert torch.equal(back.pc.get_features, scene.pc.get_features)
    assert torch.equal(back.pc.get_opacity, scene.pc.get_opacity)
    assert torch.equal(back.c2w_input, scene.c2w_input)


# ------------------------------------------------------------------------------------------------
# rendering, needs the gsplat-geer build on a GPU
# ------------------------------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="3DGEER needs a GPU")
class TestRendering:
    @staticmethod
    def scene():
        return make_synthetic_scene(torch.device("cuda"))

    @pytest.mark.parametrize("camera_model, alpha", [("pinhole", None), ("fisheye", None), ("ucm", 0.7)])
    @pytest.mark.parametrize("tile_mode", ["projection", "pbf"])
    def test_every_camera_model_renders(self, camera_model, alpha, tile_mode):
        from viewer.render import render_view

        scene = self.scene()
        fov = {"pinhole": 90.0, "fisheye": 195.0, "ucm": 160.0}[camera_model]
        c2w = torch.eye(4, device="cuda")
        image, info = render_view(scene, c2w, camera_model, fov, alpha, 128, 128, tile_mode=tile_mode,
                                  frusta=False)
        assert image.shape == (128, 128, 3) and image.dtype == np.uint8
        assert int(image.min()) < int(image.max())  # something was drawn, not one flat colour
        assert abs(info["fov_used"] - fov) < 1e-6

    def test_ucm_blacks_out_everything_past_its_image_circle(self):
        from viewer.render import render_view

        scene = self.scene()
        alpha, fov = 0.7, 160.0
        image, _ = render_view(scene, torch.eye(4, device="cuda"), "ucm", fov, alpha, 128, 128, frusta=False)
        fxfycxcy, _ = fov_to_fxfycxcy("ucm", fov, alpha, 128, 128)
        _, _, _, _, valid = pixel_angles(fxfycxcy, 128, 128, "ucm", exact=True)
        outside = ~valid[0].numpy()
        assert outside.any()  # the corners of a square UCM image really do fall outside
        assert image[outside].max() == 0
        assert image[valid[0].numpy()].max() > 0

    def test_alpha_view_stays_between_zero_and_one(self):
        from viewer.render import render_view

        scene = self.scene()
        image, _ = render_view(scene, torch.eye(4, device="cuda"), "fisheye", 195.0, None, 128, 128,
                               view="alpha", frusta=False)
        assert image.min() >= 0 and image.max() <= 255
        assert image.max() > 200  # the scene surrounds the camera, so some pixels are fully covered

    def test_projection_renders_past_ninety_degrees_and_pbf_does_not(self):
        """The ring of columns is only visible beyond 90 degrees in the projection tile mode.

        3DGEER in its pbf mode bounds the Gaussians in tangent space and drops everything behind the
        camera plane, so the part of a 195 degree image past 90 degrees stays background.
        """
        from viewer.render import render_view

        scene = self.scene()
        size, fov = 256, 195.0
        c2w = torch.eye(4, device="cuda")
        fxfycxcy, _ = fov_to_fxfycxcy("fisheye", fov, None, size, size)
        _, _, radius, theta, _ = pixel_angles(fxfycxcy, size, size, "fisheye", exact=True)
        beyond = (theta[0] > math.pi / 2).numpy()  # the corners, more than 90 degrees off axis
        assert beyond.sum() > 500

        painted = {}
        for tile_mode in ("projection", "pbf"):
            image, _ = render_view(scene, c2w, "fisheye", fov, None, size, size, tile_mode=tile_mode,
                                   frusta=False)
            # The background is white, so anything else is scene content.
            painted[tile_mode] = float((image[beyond].min(axis=-1) < 240).mean())
        assert painted["projection"] > 0.2
        assert painted["pbf"] < 0.01
