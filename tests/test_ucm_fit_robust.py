"""The Kannala Brandt fit that stands in for UCM must never kill a training run.

3DGEER has no UCM, so ucm_radial_coeffs fits a Kannala Brandt fisheye to the learned UCM camera of every
view. The fit samples angles up to the largest angle the camera can still render. When the learned camera
collapses, no pixel is inside the UCM image circle any more, that largest angle is zero, the sample grid
collapses onto the origin and the four columns of the design matrix vanish together. The CUDA least
squares driver (gels, QR) assumes full rank and raised `_LinAlgError` on exactly that matrix, which ended
a 20000 step run (job 31792) at step 3930. These tests pin the guard and the rank safe solve that replace
it, and they pin that a healthy fit is numerically untouched.

Run from the repository root:
    pytest tests/test_ucm_fit_robust.py
"""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from erayzer_core.model.utils_gaussian import pixel_angles, ucm_radial_coeffs

H = W = 64
THETA_MAX = math.radians(195.0 / 2)  # lcam_fish sees 195 degrees across its image circle
F_TRUE = (W / 2) / THETA_MAX  # equidistant fisheye, 0.29383 * W


def intrinsics(f=F_TRUE, alpha=0.62, cx=W / 2, cy=H / 2, v=1):
    """fxfycxcy [v, 5] for UCM, alpha last, principal point in the image centre by default."""
    return torch.tensor([[f, f, cx, cy, alpha]]).repeat(v, 1)


def theta_max_of(fxfycxcy):
    """The largest angle the fit samples up to, the quantity that collapses to zero, [v]."""
    _, _, _, theta, valid = pixel_angles(fxfycxcy, H, W, "ucm", exact=True)
    return torch.where(valid, theta, torch.zeros_like(theta)).flatten(1).amax(1)


def old_ucm_radial_coeffs(fxfycxcy, h, w):
    """The implementation before the fix, copied verbatim, as the reference for the healthy case.

    On CPU torch.linalg.lstsq defaults to the rank revealing driver gelsy, so this does not raise here.
    It is CUDA that defaults to gels and raises, which is what job 31792 hit.
    """
    _, _, _, theta, valid = pixel_angles(fxfycxcy, h, w, "ucm", exact=True)
    theta_max = torch.where(valid, theta, torch.zeros_like(theta)).flatten(1).amax(1)
    th = theta_max[:, None] * torch.linspace(0.02, 1.0, 64, device=theta.device)[None]
    a = fxfycxcy[:, 4:5]
    target = torch.sin(th) / (a + (1 - a) * torch.cos(th)) / th - 1
    A = torch.stack([th**2, th**4, th**6, th**8], dim=-1)
    return torch.linalg.lstsq(A.double(), target.double()[..., None]).solution[..., 0].float()


@pytest.mark.parametrize("alpha", [0.3, 0.5, 0.62, 0.8])
@pytest.mark.parametrize("f_scale", [0.7, 1.0, 1.4])
def test_a_healthy_fit_is_numerically_unchanged(alpha, f_scale):
    """A camera that can render gets exactly the coefficients the old solver produced."""
    fxfycxcy = intrinsics(f=F_TRUE * f_scale, alpha=alpha, v=3)
    assert (theta_max_of(fxfycxcy) > 0.5).all(), "this case is meant to be a healthy fit"
    new = ucm_radial_coeffs(fxfycxcy, H, W)
    old = old_ucm_radial_coeffs(fxfycxcy, H, W)
    assert torch.allclose(new, old, atol=1e-8, rtol=1e-8), f"{new.tolist()} vs {old.tolist()}"


def test_a_healthy_fit_really_reproduces_ucm():
    """Guard against the test above passing on coefficients that fit nothing."""
    fxfycxcy = intrinsics()
    k = ucm_radial_coeffs(fxfycxcy, H, W).double()[0]
    assert (k != 0).any(), "a healthy camera must not fall into the degenerate branch"
    th = theta_max_of(fxfycxcy).double()[0] * torch.linspace(0.02, 1.0, 64, dtype=torch.double)
    a = float(fxfycxcy[0, 4])
    ucm = torch.sin(th) / (a + (1 - a) * torch.cos(th))  # theta_d, the UCM image radius over f
    kb = th * (1 + k[0] * th**2 + k[1] * th**4 + k[2] * th**6 + k[3] * th**8)
    pixels = (kb - ucm).abs().max().item() * float(fxfycxcy[0, 0])  # the radii are in units of f
    assert pixels < 0.01, pixels


def test_the_collapsed_camera_falls_back_to_equidistant():
    """The crash case: no pixel inside the image circle, so there is nothing to fit."""
    fxfycxcy = intrinsics(f=0.5, alpha=0.9, v=4)  # four views, as in the run that died
    _, _, _, _, valid = pixel_angles(fxfycxcy, H, W, "ucm", exact=True)
    assert valid.sum() == 0, "this case is meant to leave no valid pixel"
    assert (theta_max_of(fxfycxcy) == 0).all()
    coeffs = ucm_radial_coeffs(fxfycxcy, H, W)
    assert coeffs.shape == (4, 4)
    assert (coeffs == 0).all(), coeffs.tolist()  # zeros are the equidistant fisheye


def test_a_nearly_collapsed_camera_stays_finite():
    """alpha just above 0.5 with a focal length so long that the whole image is a few milliradians."""
    fxfycxcy = intrinsics(f=9000.0, alpha=0.52, v=2)
    theta_max = theta_max_of(fxfycxcy)
    assert (theta_max > 0).all() and (theta_max < 1e-2).all(), theta_max.tolist()
    coeffs = ucm_radial_coeffs(fxfycxcy, H, W)
    assert torch.isfinite(coeffs).all(), coeffs.tolist()
    assert (coeffs == 0).all(), "below the guard the fit is meaningless, so it is not attempted"


def test_a_small_but_usable_angle_is_still_fitted():
    """Just above the guard the solve itself has to cope with the badly scaled matrix."""
    fxfycxcy = intrinsics(f=2000.0, alpha=0.52, v=2)
    theta_max = theta_max_of(fxfycxcy)
    assert (theta_max > 1e-2).all(), theta_max.tolist()
    coeffs = ucm_radial_coeffs(fxfycxcy, H, W)
    assert torch.isfinite(coeffs).all(), coeffs.tolist()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("column", [0, 1, 2, 3, 4])
def test_non_finite_intrinsics_fall_back_to_equidistant(bad, column):
    """A camera that has already blown up to NaN or Inf must not take the run down with it."""
    fxfycxcy = intrinsics(v=2)
    fxfycxcy[:, column] = bad
    coeffs = ucm_radial_coeffs(fxfycxcy, H, W)
    assert torch.isfinite(coeffs).all(), coeffs.tolist()
    assert (coeffs == 0).all(), coeffs.tolist()


def test_a_degenerate_view_does_not_spoil_its_neighbours():
    """The solve is batched over views, so one collapsed view must not poison the healthy ones."""
    healthy = intrinsics(v=1)
    batch = torch.cat([healthy, intrinsics(f=0.5, alpha=0.9), healthy, intrinsics(f=float("nan"))])
    coeffs = ucm_radial_coeffs(batch, H, W)
    alone = ucm_radial_coeffs(healthy, H, W)[0]
    assert torch.equal(coeffs[0], alone) and torch.equal(coeffs[2], alone)
    assert (coeffs[1] == 0).all() and (coeffs[3] == 0).all()


def test_shape_dtype_and_device_are_unchanged():
    """The renderer takes the result as is, so the contract of the function must not move."""
    for fxfycxcy in [intrinsics(v=3), intrinsics(f=0.5, alpha=0.9, v=3)]:
        coeffs = ucm_radial_coeffs(fxfycxcy, H, W)
        assert coeffs.shape == (3, 4)
        assert coeffs.dtype == torch.float32
        assert coeffs.device == fxfycxcy.device
