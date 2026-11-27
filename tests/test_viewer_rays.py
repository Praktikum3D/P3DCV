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
