"""The line infrastructure: detect lines, turn their pixels into a differentiable camera residual.

Three later ideas share this foundation, a focal length and alpha initialiser, a plumb line loss
during training and a per scene refinement afterwards, and all three stand or fall with the same
two claims, so those are what the tests pin:

* `plumbline_residual` is zero exactly at the camera that produced the pixels, grows away from it
  and its gradient points back towards it, for the focal length and for the UCM alpha, checked on
  both sides of the true value and against synthetic lines whose correct answer is known exactly,
* the detector returns pixel sets that really do belong to one 3D line, which on a wide angle image
  means linked curves and not the straight segments the detector reports first.

Plus the plumbing that has to be invisible when it is off, and the degenerate inputs that must come
back as an honest zero rather than as a NaN in somebody's training step.

Run from the repository root:
    pytest tests/test_line_infra.py
"""

import math
import os
import random as pyrandom
import sys
import tempfile
import time

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

cv2 = pytest.importorskip("cv2")

from erayzer_core.model.utils_gaussian import camera_rays, camera_rho  # noqa: E402
from training.line_detection import (  # noqa: E402
    DEFAULT_MAX_COUNT,
    DEFAULT_NUM_SAMPLES,
    PAD_VALUE,
    LineSegments,
    available_detectors,
    detect_line_segments,
    detect_lines_in_views,
    draw_segments,
    link_segments,
    polyline_length,
    resolve_min_length,
    sample_polyline,
    sample_segment_pixels,
    to_gray_uint8,
)
from training.line_geometry import (  # noqa: E402
    DEGENERATE_SPREAD,
    pixel_ray_directions,
    plumbline_residual,
    plumbline_residual_mean,
    valid_segment_mask,
)

FISH_SIZE = 1000
SAMPLE_FRAME = "/praktikum/data-sample/AmericanDiner/Data_hard/P000/image_lcam_fish/000000_lcam_fish_image.png"


# ------------------------------------------------------------------------------------------------
# Synthetic cameras and lines, where the right answer is known exactly.
# ------------------------------------------------------------------------------------------------


def project(points, fxfycxcy, camera_model):
    """Project camera space points [n, 3] to pixels [n, 2] through one of our camera models.

    The forward direction of `pixel_ray_directions`, built on the repository's own `camera_rho`, so
    that a test line is projected by exactly the camera the residual is later told about. Pixels
    come out in the repository convention, centre of pixel (col, row) at (col + 0.5, row + 0.5).
    """
    points = torch.as_tensor(points, dtype=torch.float64)
    fxfycxcy = torch.as_tensor(fxfycxcy, dtype=torch.float64)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    radial = torch.hypot(x, y)
    theta = torch.atan2(radial, z)
    phi = torch.atan2(y, x)
    alpha = fxfycxcy[4] if len(fxfycxcy) > 4 else None
    beta = fxfycxcy[5] if len(fxfycxcy) > 5 else None
    rho = camera_rho(theta, camera_model, alpha, beta)
    return torch.stack([fxfycxcy[2] + fxfycxcy[0] * rho * torch.cos(phi),
                        fxfycxcy[3] + fxfycxcy[1] * rho * torch.sin(phi)], dim=-1)


def line_points(start, direction, count, span=3.0):
    """`count` points spread along the 3D line through `start` with `direction`."""
    t = torch.linspace(0.0, span, count, dtype=torch.float64)[:, None]
    return torch.as_tensor(start, dtype=torch.float64) + t * torch.as_tensor(direction, dtype=torch.float64)


LINES_3D = [
    ((-2.0, 1.0, 4.0), (1.0, 0.3, 0.2)),
    ((-1.5, -2.0, 3.0), (0.4, 1.0, -0.1)),
    ((2.0, 2.0, 5.0), (-1.0, 0.1, 0.4)),
    ((0.0, -1.0, 2.0), (0.2, 0.5, 1.0)),
    ((-3.0, 0.5, 6.0), (1.0, -0.6, 0.3)),
]
"""Five 3D lines in camera space, all in front of the camera and spread over the field of view."""

SEPARATED_LINES_3D = [
    ((-2.8, 1.2, 1.9), (0.3, 0.9, -0.6)),
    ((0.0, -2.2, 3.9), (0.5, -0.2, -1.0)),
    ((-0.9, -0.9, 1.9), (0.1, 0.8, 0.1)),
    ((-1.6, -1.9, 1.5), (0.4, 0.8, -0.5)),
    ((1.1, 0.8, 2.2), (-0.2, 0.2, -0.9)),
]
"""Five 3D lines whose image curves stay at least 35 pixels apart under both test cameras.

The linking tests need that. Two curves that touch are a real situation and the greedy linking does
sometimes run from one into the other there, but a test of whether a curve is reassembled should
fail over the reassembly and not over a scene that the detector itself could not have resolved.
"""


def synthetic_line_pixels(fxfycxcy, camera_model, num_samples=DEFAULT_NUM_SAMPLES, dtype=torch.float32):
    """The pixels of `LINES_3D` seen by this camera, [n, K, 2].

    These are coplanar ray sets by construction, and exactly so: every pixel comes from a point of
    one straight 3D line, projected by the very camera the residual is asked about.
    """
    rows = [project(line_points(start, direction, num_samples), fxfycxcy, camera_model)
            for start, direction in LINES_3D]
    return torch.stack(rows).to(dtype)


FISHEYE = [300.0, 300.0, 500.0, 500.0]
UCM = [300.0, 300.0, 500.0, 500.0, 0.6]
EUCM = [300.0, 300.0, 500.0, 500.0, 0.6, 1.0]


# ------------------------------------------------------------------------------------------------
# The ray directions, where this module couples to the rest of the repository.
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("camera_model,intrinsics", [
    ("pinhole", FISHEYE), ("fisheye", FISHEYE), ("ucm", UCM), ("eucm", EUCM),
])
def test_pixel_ray_directions_match_camera_rays(camera_model, intrinsics):
    """The subpixel ray builder is `utils_gaussian.camera_rays` on a grid, bit for bit.

    This is the coupling that matters: if the two ever drift apart, the residual would constrain a
    different camera than the one the renderer and the plucker embeddings use, and nothing else in
    the suite would notice.
    """
    h, w = 7, 9
    fxfycxcy = torch.tensor([intrinsics], dtype=torch.float64)
    _, rays = camera_rays(torch.eye(4, dtype=torch.float64)[None], fxfycxcy, h, w, camera_model)

    y, x = torch.meshgrid(torch.arange(h, dtype=torch.float64),
                          torch.arange(w, dtype=torch.float64), indexing="ij")
    pixels = torch.stack([x + 0.5, y + 0.5], dim=-1).reshape(-1, 2)
    mine = pixel_ray_directions(fxfycxcy, camera_model, pixels)

    assert torch.equal(mine, rays.reshape(-1, 3)), f"{camera_model}: ray directions drifted apart"


def test_pixel_ray_directions_survive_the_principal_point():
    """A pixel sitting exactly on the principal point has no direction in the image, only a ray."""
    pixels = torch.tensor([[[500.0, 500.0]]], dtype=torch.float32)
    for camera_model, intrinsics in (("fisheye", FISHEYE), ("ucm", UCM)):
        d = pixel_ray_directions(torch.tensor(intrinsics), camera_model, pixels)
        assert torch.isfinite(d).all(), f"{camera_model}: the optical axis pixel is not finite"
        assert torch.allclose(d[0, 0], torch.tensor([0.0, 0.0, 1.0]), atol=1e-6)


# ------------------------------------------------------------------------------------------------
# The residual itself.
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("camera_model,intrinsics", [
    ("pinhole", FISHEYE), ("fisheye", FISHEYE), ("ucm", UCM), ("eucm", EUCM),
])
@pytest.mark.parametrize("normalize", ["inplane", "trace"])
def test_residual_is_zero_at_the_true_camera(camera_model, intrinsics, normalize):
    """Pixels of a real 3D line, projected by this camera, have coplanar rays under this camera."""
    pixels = synthetic_line_pixels(intrinsics, camera_model)
    residual = plumbline_residual(torch.tensor(intrinsics), camera_model, pixels, normalize=normalize)
    assert residual.shape == (len(LINES_3D),)
    assert float(residual.max()) < 1e-6, f"{camera_model}/{normalize}: {residual.tolist()}"


@pytest.mark.parametrize("camera_model,intrinsics", [("fisheye", FISHEYE), ("ucm", UCM)])
@pytest.mark.parametrize("factor", [0.8, 0.9, 1.1, 1.25])
def test_residual_grows_at_a_wrong_focal_length(camera_model, intrinsics, factor):
    """Perturb the focal length and the same pixels stop being coplanar, on both sides."""
    pixels = synthetic_line_pixels(intrinsics, camera_model)
    true = plumbline_residual(torch.tensor(intrinsics), camera_model, pixels)
    wrong = list(intrinsics)
    wrong[0] = wrong[1] = intrinsics[0] * factor
    perturbed = plumbline_residual(torch.tensor(wrong), camera_model, pixels)
    assert float(perturbed.mean()) > float(true.mean()) + 1e-8, (
        f"{camera_model} at f x {factor}: {float(perturbed.mean()):.3e} "
        f"is not above {float(true.mean()):.3e}"
    )


def test_residual_grows_at_a_wrong_alpha():
    """The same for the UCM alpha, the intrinsic this whole weekend is about."""
    pixels = synthetic_line_pixels(UCM, "ucm")
    true = float(plumbline_residual(torch.tensor(UCM), "ucm", pixels).mean())
    for alpha in (0.4, 0.5, 0.7, 0.8):
        wrong = list(UCM)
        wrong[4] = alpha
        value = float(plumbline_residual(torch.tensor(wrong), "ucm", pixels).mean())
        assert value > true + 1e-8, f"alpha {alpha}: {value:.3e} is not above {true:.3e}"


@pytest.mark.parametrize("camera_model,intrinsics,column,name", [
    ("fisheye", FISHEYE, 0, "f"),
    ("ucm", UCM, 0, "f"),
    ("ucm", UCM, 4, "alpha"),
    ("eucm", EUCM, 4, "alpha"),
])
def test_gradient_points_towards_the_true_value(camera_model, intrinsics, column, name):
    """From either side the gradient pushes the intrinsic back to the value that made the pixels.

    A sign check and not a magnitude check: the magnitude is what the consumers will tune, the sign
    is what decides whether a line loss helps or actively drives the camera away, which is the one
    failure mode that would be invisible in a training run until it has wasted a night.
    """
    pixels = synthetic_line_pixels(intrinsics, camera_model)
    truth = intrinsics[column]
    offsets = (-0.2, -0.1, 0.1, 0.2) if column == 0 else (-0.2, -0.1, 0.1, 0.2)
    for offset in offsets:
        wrong = list(intrinsics)
        value = truth * (1.0 + offset) if column == 0 else truth + offset
        wrong[column] = value
        if column == 0:
            wrong[1] = value
        parameters = torch.tensor(wrong, dtype=torch.float32, requires_grad=True)
        plumbline_residual(parameters, camera_model, pixels).mean().backward()
        gradient = float(parameters.grad[column])
        assert abs(gradient) > 0.0, f"{camera_model} {name} at {offset:+.2f}: the gradient vanished"
        # Descending the gradient must move the intrinsic back towards the truth, so the gradient
        # is negative below the true value and positive above it.
        assert gradient * offset > 0.0, (
            f"{camera_model} {name} at {offset:+.2f}: gradient {gradient:.3e} points the wrong way"
        )


def test_gradient_matches_central_differences():
    """The Rayleigh quotient really is the derivative, checked against the function it differentiates."""
    pixels = synthetic_line_pixels(FISHEYE, "fisheye").double()
    base = [FISHEYE[0] * 1.08, FISHEYE[1] * 1.08, 500.0, 500.0]
    parameters = torch.tensor(base, dtype=torch.float64, requires_grad=True)
    plumbline_residual(parameters, "fisheye", pixels).sum().backward()

    step = 1e-4
    for column in range(4):
        shifted = list(base)
        shifted[column] = base[column] + step
        high = float(plumbline_residual(torch.tensor(shifted), "fisheye", pixels).sum())
        shifted[column] = base[column] - step
        low = float(plumbline_residual(torch.tensor(shifted), "fisheye", pixels).sum())
        numeric = (high - low) / (2.0 * step)
        analytic = float(parameters.grad[column])
        assert abs(numeric - analytic) <= 1e-6 + 1e-4 * abs(numeric), (
            f"column {column}: analytic {analytic:.6e} against numeric {numeric:.6e}"
        )


def test_residual_matches_the_eigenvalues_it_claims_to_be():
    """Both normalisations agree with a plain `torch.linalg.eigvalsh` of the scatter matrix.

    The implementation avoids the backward of `eigvalsh`, not its values; this is the independent
    check that avoiding it did not change what is computed.
    """
    pixels = synthetic_line_pixels(UCM, "ucm", dtype=torch.float64)
    wrong = list(UCM)
    wrong[0] = wrong[1] = UCM[0] * 1.3
    intrinsics = torch.tensor(wrong, dtype=torch.float64)

    directions = pixel_ray_directions(intrinsics[None, None], "ucm", pixels)
    scatter = torch.einsum("nki,nkj->nij", directions, directions)
    eigenvalues = torch.linalg.eigvalsh(scatter)

    want_trace = eigenvalues[:, 0] / eigenvalues.sum(-1)
    want_inplane = eigenvalues[:, 0] / (eigenvalues[:, 0] + eigenvalues[:, 1])
    got_trace = plumbline_residual(intrinsics, "ucm", pixels, normalize="trace")
    got_inplane = plumbline_residual(intrinsics, "ucm", pixels, normalize="inplane")

    assert torch.allclose(got_trace, want_trace, atol=1e-12)
    assert torch.allclose(got_inplane, want_inplane, atol=1e-12)


def test_residuals_stay_inside_their_bounds():
    """[0, 1/3] for the trace form and [0, 1/2] for the in plane form, even on nonsense input."""
    generator = torch.Generator().manual_seed(7)
    pixels = torch.rand(64, 8, 2, generator=generator) * 1000.0
    trace = plumbline_residual(torch.tensor(FISHEYE), "fisheye", pixels, normalize="trace")
    inplane = plumbline_residual(torch.tensor(FISHEYE), "fisheye", pixels, normalize="inplane")
    assert float(trace.min()) >= 0.0 and float(trace.max()) <= 1.0 / 3.0 + 1e-6
    assert float(inplane.min()) >= 0.0 and float(inplane.max()) <= 0.5 + 1e-6


def test_the_trace_normalisation_prefers_an_infinitely_narrow_camera():
    """Why the default is the in plane form, pinned so nobody quietly switches the default back.

    Straight image segments, the thing a detector reports, are exactly coplanar under a pinhole, so
    their trace residual falls without end as the focal length grows. The in plane residual does
    not, because the in plane extent it divides by shrinks along with the error.
    """
    generator = torch.Generator().manual_seed(11)
    start = torch.rand(16, 1, 2, generator=generator) * 600.0 + 200.0
    end = torch.rand(16, 1, 2, generator=generator) * 600.0 + 200.0
    t = torch.linspace(0.0, 1.0, 8)[None, :, None]
    pixels = start + t * (end - start)

    factors = [1.0, 2.0, 4.0, 8.0, 16.0]
    trace, inplane = [], []
    for factor in factors:
        intrinsics = torch.tensor([300.0 * factor, 300.0 * factor, 500.0, 500.0])
        trace.append(float(plumbline_residual(intrinsics, "fisheye", pixels, normalize="trace").mean()))
        inplane.append(float(plumbline_residual(intrinsics, "fisheye", pixels, normalize="inplane").mean()))

    assert all(b < a for a, b in zip(trace, trace[1:])), f"trace did not fall monotonically: {trace}"
    assert trace[-1] < trace[0] / 100.0, f"trace barely moved: {trace}"
    assert inplane[-1] > trace[-1] * 100.0, (
        f"the in plane form collapsed the same way: {inplane}"
    )


# ------------------------------------------------------------------------------------------------
# Degenerate input: it has to be an honest zero, never a NaN and never a maximum.
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("normalize", ["inplane", "trace"])
@pytest.mark.parametrize("name,pixels", [
    ("one pixel", torch.tensor([[[417.0, 233.0]], [[500.0, 500.0]]])),
    ("identical pixels", torch.full((3, 8, 2), 311.0)),
    ("two identical pixels", torch.tensor([[[100.0, 120.0], [100.0, 120.0]]])),
    ("the padded value", torch.full((2, 8, 2), PAD_VALUE)),
])
def test_degenerate_lines_are_zero_and_differentiable(name, pixels, normalize):
    """No in plane extent means no constraint: zero value, zero gradient, no NaN in either."""
    intrinsics = torch.tensor(UCM, requires_grad=True)
    residual = plumbline_residual(intrinsics, "ucm", pixels, normalize=normalize)
    assert torch.isfinite(residual).all(), f"{name}/{normalize}: not finite"
    assert float(residual.abs().max()) < 1e-3, f"{name}/{normalize}: {residual.tolist()}"
    residual.sum().backward()
    assert torch.isfinite(intrinsics.grad).all(), f"{name}/{normalize}: the gradient is not finite"
    assert float(intrinsics.grad.abs().max()) < 1e-3, f"{name}/{normalize}: {intrinsics.grad.tolist()}"


def test_no_lines_at_all():
    """An empty batch comes back empty, and the padded mean comes back a differentiable zero."""
    empty = torch.zeros(0, 8, 2)
    assert plumbline_residual(torch.tensor(FISHEYE), "fisheye", empty).shape == (0,)

    intrinsics = torch.tensor(FISHEYE, requires_grad=True)
    value = plumbline_residual_mean(intrinsics, "fisheye", torch.full((4, 8, 2), PAD_VALUE))
    assert float(value) == 0.0
    value.backward()
    assert torch.isfinite(intrinsics.grad).all()


def test_degenerate_threshold_is_far_below_a_real_line():
    """The degeneracy cut must never bite a line the detector would actually report."""
    pixels = synthetic_line_pixels(FISHEYE, "fisheye", dtype=torch.float64)
    wrong = list(FISHEYE)
    wrong[0] = wrong[1] = FISHEYE[0] * 1.2
    directions = pixel_ray_directions(torch.tensor(wrong, dtype=torch.float64)[None, None], "fisheye", pixels)
    scatter = torch.einsum("nki,nkj->nij", directions, directions)
    eigenvalues = torch.linalg.eigvalsh(scatter)
    spread = eigenvalues[:, 1] / eigenvalues.sum(-1)
    assert float(spread.min()) > 100.0 * DEGENERATE_SPREAD, (
        f"the smallest in plane spread of a real line is {float(spread.min()):.2e}, "
        f"too close to the cut at {DEGENERATE_SPREAD:.1e}"
    )


# ------------------------------------------------------------------------------------------------
# Shapes, broadcasting, padding and the batched mean.
# ------------------------------------------------------------------------------------------------


def test_shapes_and_broadcasting():
    """One camera for all lines, one camera per line, and a leading view dimension."""
    pixels = synthetic_line_pixels(FISHEYE, "fisheye")  # [n, K, 2]
    n = pixels.shape[0]
    single = plumbline_residual(torch.tensor(FISHEYE), "fisheye", pixels)
    per_line = plumbline_residual(torch.tensor(FISHEYE).expand(n, 4), "fisheye", pixels)
    assert torch.allclose(single, per_line)

    views = pixels[None].expand(3, -1, -1, -1)
    per_view = plumbline_residual(torch.tensor(FISHEYE).expand(3, 1, 4), "fisheye", views)
    assert per_view.shape == (3, n)
    assert torch.allclose(per_view[0], single)


def test_normalized_intrinsics_are_scaled_back():
    """The head and the loader carry intrinsics as fractions of the image side."""
    pixels = synthetic_line_pixels(FISHEYE, "fisheye")
    want = plumbline_residual(torch.tensor(FISHEYE), "fisheye", pixels)
    normalized = torch.tensor(FISHEYE) / FISH_SIZE
    got = plumbline_residual(normalized, "fisheye", pixels, (FISH_SIZE, FISH_SIZE),
                             normalized_intrinsics=True)
    assert torch.allclose(got, want, atol=1e-7)

    with pytest.raises(ValueError):
        plumbline_residual(normalized, "fisheye", pixels, normalized_intrinsics=True)


def test_padded_lines_do_not_change_the_mean():
    """Padding is ignored, and the trimmed mean drops the worst lines rather than the padding."""
    pixels = synthetic_line_pixels(UCM, "ucm")
    wrong = list(UCM)
    wrong[0] = wrong[1] = UCM[0] * 1.2
    intrinsics = torch.tensor(wrong)

    padded = torch.cat([pixels, torch.full((7,) + pixels.shape[1:], PAD_VALUE)], dim=0)
    assert valid_segment_mask(padded).tolist() == [True] * len(pixels) + [False] * 7

    plain = plumbline_residual_mean(intrinsics, "ucm", pixels, trim=0.0)
    with_padding = plumbline_residual_mean(intrinsics, "ucm", padded, trim=0.0)
    # Not bit for bit: the padded rows make the batch the eigen decomposition runs over larger,
    # and a batched float32 solver does not return the identical eigenvector for the identical
    # matrix in a different batch.
    assert torch.allclose(plain, with_padding, rtol=2e-3, atol=1e-12), "padding leaked into the mean"

    trimmed = plumbline_residual_mean(intrinsics, "ucm", padded, trim=0.4)
    every = plumbline_residual(intrinsics, "ucm", pixels)
    keep = max(1, int(round(0.6 * len(every))))
    want = torch.sort(every).values[:keep].mean()
    assert torch.allclose(trimmed, want, rtol=2e-3, atol=1e-12), "the trimmed mean trimmed the wrong lines"


def test_the_mean_keeps_pointing_at_the_true_camera():
    """The aggregate the consumers will actually call, with padding and trimming, still works."""
    pixels = synthetic_line_pixels(UCM, "ucm")
    padded = torch.cat([pixels, torch.full((5,) + pixels.shape[1:], PAD_VALUE)], dim=0)
    for offset in (-0.15, 0.15):
        wrong = list(UCM)
        wrong[4] = UCM[4] + offset
        parameters = torch.tensor(wrong, requires_grad=True)
        plumbline_residual_mean(parameters, "ucm", padded).backward()
        assert float(parameters.grad[4]) * offset > 0.0, "the trimmed mean points the wrong way"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_cuda_agrees_with_cpu():
    """Small, because the development box has six gigabytes: the same lines on both devices."""
    pixels = synthetic_line_pixels(UCM, "ucm")
    wrong = list(UCM)
    wrong[0] = wrong[1] = UCM[0] * 1.15

    cpu = torch.tensor(wrong, requires_grad=True)
    plumbline_residual(cpu, "ucm", pixels).sum().backward()
    gpu = torch.tensor(wrong, device="cuda", requires_grad=True)
    gpu_value = plumbline_residual(gpu, "ucm", pixels.cuda())
    gpu_value.sum().backward()

    assert torch.allclose(gpu_value.cpu(), plumbline_residual(torch.tensor(wrong), "ucm", pixels), atol=1e-6)
    # float32 on two different eigen solvers, so same answer rather than same bits.
    assert torch.allclose(gpu.grad.cpu(), cpu.grad, rtol=1e-3, atol=1e-7)


# ------------------------------------------------------------------------------------------------
# Detection.
# ------------------------------------------------------------------------------------------------


def test_a_detector_is_available():
    """Reported rather than asserted in detail: which detector this OpenCV build gives us."""
    detectors = available_detectors()
    print(f"\nOpenCV {cv2.__version__}, detectors available: {detectors}")
    assert detectors, "no line detector at all in this OpenCV build"


def grid_image(size=256, spacing=32):
    """A grid of straight lines, the one image whose correct detection is obvious.

    The two families do not cross: a crossing cuts both lines into pieces with a gap the width of
    the other line, and relinking those is the chaining's job, not this test's. Here the detector
    should see whole lines and the chaining should have nothing to do.
    """
    image = np.full((size, size, 3), 30, dtype=np.uint8)
    half = size // 2
    for position in range(spacing, size - spacing + 1, spacing):
        cv2.line(image, (position, 8), (position, half - 8), (235, 235, 235), 2)
        cv2.line(image, (8, half + position // 2), (size - 8, half + position // 2), (235, 235, 235), 2)
    return image


@pytest.mark.parametrize("detector", ["lsd", "hough"])
def test_detection_finds_the_lines_of_a_grid(detector):
    """Both detector paths find the grid, keep to the image and respect the knobs."""
    if detector not in available_detectors():
        pytest.skip(f"{detector} is not available in this OpenCV build")
    image = grid_image()
    lines = detect_line_segments(image, detector=detector, min_length=0.3, max_count=12, num_samples=6)

    assert 4 <= len(lines) <= 12, f"{detector}: found {len(lines)} lines on a grid of 14"
    assert lines.pixels.shape == (len(lines), 6, 2)
    assert (lines.score >= 0.3 * 256).all(), f"{detector}: a line below the minimum length survived"
    assert np.all(lines.pixels >= 0.0) and np.all(lines.pixels <= 256.0)
    assert np.all(np.diff(lines.score) <= 1e-4), f"{detector}: not sorted longest first"

    # Every line of a grid is axis aligned, so one of its pixel coordinates must stay constant.
    for row in lines.pixels:
        spread = row.max(axis=0) - row.min(axis=0)
        assert float(spread.min()) < 2.0, f"{detector}: a line of the grid is not axis aligned: {spread}"
        assert float(spread.max()) > 0.3 * 256, f"{detector}: a line is shorter than it claims"


def test_detection_on_an_empty_image_returns_nothing():
    lines = detect_line_segments(np.zeros((64, 64, 3), dtype=np.uint8))
    assert len(lines) == 0
    assert lines.endpoints.shape == (0, 4) and lines.pixels.shape == (0, DEFAULT_NUM_SAMPLES, 2)


def test_detection_accepts_torch_and_numpy_alike():
    """The loader hands over a torch CHW float tensor, a script hands over numpy HWC uint8."""
    image = grid_image()
    tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    from_numpy = detect_line_segments(image, min_length=0.3)
    from_torch = detect_line_segments(tensor, min_length=0.3)
    assert len(from_numpy) == len(from_torch)
    assert np.allclose(from_numpy.endpoints, from_torch.endpoints, atol=1e-3)
    assert to_gray_uint8(tensor).shape == (256, 256)


def test_a_line_touching_invalid_pixels_is_dropped():
    """The data mask of an augmented camera must not become the longest line in the view."""
    image = grid_image()
    mask = np.ones((256, 256), dtype=bool)
    mask[:, 120:136] = False  # a vertical strip of padding straight across the grid

    without = detect_line_segments(image, min_length=0.3, max_count=64)
    with_mask = detect_line_segments(image, mask=mask, min_length=0.3, max_count=64)
    assert len(with_mask) < len(without), "the mask dropped nothing at all"
    for row in with_mask.pixels:
        columns = np.round(row[:, 0] - 0.5).astype(int)
        rows = np.round(row[:, 1] - 0.5).astype(int)
        assert mask[rows, columns].all(), "a line survived on invalid pixels"


def test_min_length_reads_as_a_fraction_or_as_pixels():
    assert resolve_min_length(0.2, (256, 256)) == pytest.approx(51.2)
    assert resolve_min_length(40.0, (256, 256)) == pytest.approx(40.0)
    assert resolve_min_length(0.2, (100, 400)) == pytest.approx(20.0)


def test_sampling_along_a_segment_and_along_a_polyline():
    endpoints = np.array([[0.0, 0.0, 10.0, 0.0]], dtype=np.float32)
    assert np.allclose(sample_segment_pixels(endpoints, 3)[0], [[0, 0], [5, 0], [10, 0]])
    assert sample_segment_pixels(endpoints, 1).shape == (1, 1, 2)

    # Evenly by arc length, not by vertex: a long piece then a short one.
    polyline = np.array([[0.0, 0.0], [8.0, 0.0], [8.0, 2.0]])
    assert polyline_length(polyline) == pytest.approx(10.0)
    sampled = sample_polyline(polyline, 6)
    steps = np.linalg.norm(np.diff(sampled, axis=0), axis=1)
    assert np.allclose(steps, 2.0, atol=1e-4), f"not evenly spaced: {steps}"
    assert np.allclose(sample_polyline(np.array([[3.0, 4.0]]), 4), [[3.0, 4.0]] * 4)


# ------------------------------------------------------------------------------------------------
# Chaining, the part that makes the residual usable on a wide angle image.
# ------------------------------------------------------------------------------------------------


def fisheye_curve_image(size=600, f=None, line_width=2):
    """A synthetic fisheye frame showing `LINES_3D`, each drawn as the curve this camera makes of it.

    Used where an image full of real curved lines is needed, as the loader fixture. NOT used to
    test the detector against the residual: on a clean drawn stroke the Line Segment Detector keeps
    a long chord in one piece instead of cutting it into the short pieces it produces on a real
    photograph, so the detection on this image is not representative. The detector against the
    residual is tested on constructed pieces (`chopped_curve`) and on a real frame.
    """
    f = f or size / math.radians(195.0)
    intrinsics = [f, f, size / 2.0, size / 2.0]
    image = np.full((size, size, 3), 20, dtype=np.uint8)
    for start, direction in LINES_3D:
        pixels = project(line_points(start, direction, 400), intrinsics, "fisheye").numpy()
        points = np.round(pixels - 0.5).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [points], False, (240, 240, 240), line_width, cv2.LINE_AA)
    return image, intrinsics


def chopped_curve(intrinsics, camera_model="fisheye", piece=12.0, gap=1.5, jitter=0.0, seed=0):
    """The image curve of every 3D line, cut into short straight chords with gaps between them.

    This is what a straight segment detector hands back on a wide angle image, reproduced exactly
    and without a detector in the way: short chords along the curve, a pixel or two of gap where
    the detector's region stopped, and optional endpoint jitter for its subpixel noise. Feeding
    these to `link_segments` is the honest test of whether the linking puts a curve back together.
    """
    generator = np.random.default_rng(seed)
    side = 2.0 * intrinsics[2]
    chords = []
    for start, direction in SEPARATED_LINES_3D:
        curve = project(line_points(start, direction, 4000), intrinsics, camera_model).numpy()
        # Only the stretch that is really imaged: outside the image, or outside the UCM image
        # circle where the projection folds back, there is no curve to detect.
        inside = np.all((curve >= 1.0) & (curve <= side - 1.0), axis=1)
        if camera_model in ("ucm", "eucm"):
            radius = np.hypot(curve[:, 0] - intrinsics[2], curve[:, 1] - intrinsics[3])
            inside &= radius <= 0.45 * side
        longest, run = (0, 0), None
        for index, flag in enumerate(np.append(inside, False)):
            if flag and run is None:
                run = index
            elif not flag and run is not None:
                if index - run > longest[1] - longest[0]:
                    longest = (run, index)
                run = None
        curve = curve[longest[0]:longest[1]]
        if len(curve) < 2:
            continue
        step = np.linalg.norm(np.diff(curve, axis=0), axis=1)
        walked = np.concatenate([[0.0], np.cumsum(step)])
        position = 0.0
        while position + piece <= walked[-1]:
            first = int(np.searchsorted(walked, position))
            last = int(np.searchsorted(walked, position + piece))
            chord = np.concatenate([curve[first], curve[min(last, len(curve) - 1)]])
            if jitter:
                chord = chord + generator.normal(0.0, jitter, 4)
            chords.append(chord)
            position += piece + gap
    return np.asarray(chords, dtype=np.float64)


def sweep(pixels, intrinsics, camera_model="fisheye", factors=(0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.25)):
    """The residual of these lines over a range of focal length factors, for the sweeps below."""
    values = []
    for factor in factors:
        camera = list(intrinsics)
        camera[0] = intrinsics[0] * factor
        camera[1] = intrinsics[1] * factor
        values.append(float(plumbline_residual_mean(
            torch.tensor(camera, dtype=torch.float32), camera_model, pixels)))
    return list(factors), values


@pytest.mark.parametrize("camera_model,intrinsics", [("fisheye", FISHEYE), ("ucm", UCM)])
def test_linking_puts_a_chopped_curve_back_together(camera_model, intrinsics):
    """The pieces of one curve come back as one long line whose rays are coplanar again.

    The chords themselves are straight and short; the chain through them follows the curve, which
    is the difference the whole module turns on.
    """
    chords = chopped_curve(intrinsics, camera_model, jitter=0.05)
    chains = link_segments(chords, max_gap=4.0)
    assert len(chains) == len(SEPARATED_LINES_3D), (
        f"{len(chains)} chains out of {len(SEPARATED_LINES_3D)} well separated curves"
    )

    lengths = sorted(polyline_length(chain) for chain in chains)
    assert lengths[0] > 5.0 * 12.0, f"the shortest chain is only {lengths[0]:.1f} px, nothing linked"

    pixels = torch.stack([torch.from_numpy(sample_polyline(chain, 16)) for chain in chains])
    residual = plumbline_residual(torch.tensor(intrinsics, dtype=torch.float32), camera_model, pixels)
    assert float(residual.max()) < 1e-5, f"a chain is not coplanar: {residual.tolist()}"


@pytest.mark.parametrize("camera_model,intrinsics", [("fisheye", FISHEYE), ("ucm", UCM)])
def test_the_chained_residual_finds_the_true_focal_length(camera_model, intrinsics):
    """The sieve criterion in miniature: a focal length sweep with its minimum at the true camera."""
    chords = chopped_curve(intrinsics, camera_model, jitter=0.05)
    chains = link_segments(chords, max_gap=4.0)
    pixels = torch.stack([torch.from_numpy(sample_polyline(chain, 16)) for chain in chains])

    factors, values = sweep(pixels, intrinsics, camera_model)
    best = factors[int(np.argmin(values))]
    print(f"\n{camera_model} chained sweep: " + " ".join(f"{f:.2f}:{v:.1e}" for f, v in zip(factors, values)))
    assert best == 1.0, f"the minimum sits at f x {best}, not at the true focal length"
    assert values[factors.index(1.25)] > 10.0 * min(values), "the minimum is not pronounced"
    assert values[factors.index(0.8)] > 10.0 * min(values), "the minimum is not pronounced"


def test_linking_keeps_every_segment_and_rejects_a_wiggle():
    """No segment is lost, and a chain that bends both ways falls apart into its pieces again."""
    straight = np.array([[0.0, 0.0, 10.0, 0.0], [10.5, 0.2, 20.0, 1.0], [20.3, 1.2, 30.0, 3.0]])
    chains = link_segments(straight, max_gap=2.0)
    assert len(chains) == 1 and len(chains[0]) == 4, f"the chain did not close: {chains}"
    assert polyline_length(chains[0]) > 28.0

    # Turns of +20, -20, +20 degrees: every single one is inside max_turn_deg, so the chain does
    # form, and only the consistency rule afterwards can see that it wiggles.
    angles = np.radians([0.0, 20.0, 0.0, 20.0])
    point, zigzag = np.array([0.0, 0.0]), []
    for angle in angles:
        step = np.array([math.cos(angle), math.sin(angle)]) * 10.0
        zigzag.append([point[0], point[1], point[0] + step[0], point[1] + step[1]])
        point = point + step + np.array([0.2, 0.0])
    zigzag = np.array(zigzag)
    assert len(link_segments(zigzag, max_gap=2.0, max_turn_deg=40.0, consistent_turn=False)) == 1
    pieces = link_segments(zigzag, max_gap=2.0, max_turn_deg=40.0)
    assert len(pieces) == 4, f"a wiggling chain survived: {pieces}"

    # A chain that curls more than max_total_turn_deg has clearly left its line.
    circle = np.array([[math.cos(a) * 20.0, math.sin(a) * 20.0] for a in np.radians(np.arange(0, 200, 10))])
    arc = np.stack([circle[:-1, 0], circle[:-1, 1], circle[1:, 0], circle[1:, 1]], axis=1)
    assert len(link_segments(arc, max_gap=1.0, max_total_turn_deg=360.0)) == 1
    assert len(link_segments(arc, max_gap=1.0, max_total_turn_deg=90.0)) == len(arc)

    far_apart = np.array([[0.0, 0.0, 10.0, 0.0], [40.0, 0.0, 50.0, 0.0]])
    assert len(link_segments(far_apart, max_gap=2.0)) == 2, "segments far apart were linked"
    assert link_segments(np.zeros((0, 4))) == []


def test_the_unlinked_pieces_alone_have_no_signal():
    """The measurement that motivates the linking, pinned so the finding does not get lost.

    The very same chords, used one by one instead of linked, prefer a camera that is far too
    narrow: each of them is straight in the image, and a straight image segment is exactly coplanar
    under a pinhole, so its residual keeps falling as the focal length grows.
    """
    chords = chopped_curve(FISHEYE, piece=12.0, jitter=0.05)
    pixels = torch.from_numpy(sample_segment_pixels(chords.astype(np.float32), 16))
    factors, values = sweep(pixels, FISHEYE, factors=(1.0, 1.5, 2.5, 4.0))
    print("\nunlinked pieces: " + " ".join(f"{f:.2f}:{v:.1e}" for f, v in zip(factors, values)))
    assert values[-1] < values[0] / 2.0, (
        f"unlinked pieces now have a signal ({values}); if that is real, the module docstring and "
        f"the chaining default need revisiting"
    )


def test_drawing_accepts_both_shapes():
    image = grid_image(64, 16)
    lines = detect_line_segments(image, min_length=0.3)
    assert draw_segments(image, lines.endpoints).shape == (64, 64, 3)
    assert draw_segments(image, lines.pixels).shape == (64, 64, 3)


@pytest.mark.skipif(not os.path.exists(SAMPLE_FRAME), reason="no lcam_fish sample frame")
def test_a_real_frame_knows_its_own_camera():
    """The headline claim on real data: the residual is lowest at the true lcam_fish camera.

    One frame only, so that the test stays quick; `scripts/line_sanity_check.py` runs the same
    comparison over as many frames as it is given.
    """
    from PIL import Image

    frame = np.asarray(Image.open(SAMPLE_FRAME).convert("RGB"))
    size = frame.shape[0]
    y, x = np.mgrid[0:size, 0:size]
    mask = np.hypot(x + 0.5 - size / 2.0, y + 0.5 - size / 2.0) <= size / 2.0 - 2.0

    started = time.perf_counter()
    lines = detect_line_segments(frame, mask=mask, min_length=0.2, max_count=32, num_samples=16)
    elapsed = (time.perf_counter() - started) * 1000.0
    assert len(lines) >= 3, f"only {len(lines)} lines on a real diner frame"

    f_true = size / math.radians(195.0)
    pixels = torch.from_numpy(lines.pixels)

    def residual(factor):
        camera = torch.tensor([f_true * factor, f_true * factor, size / 2.0, size / 2.0])
        return float(plumbline_residual_mean(camera, "fisheye", pixels))

    true, wide = residual(1.0), residual(1.4)
    print(f"\nreal frame: {len(lines)} lines in {elapsed:.0f} ms, "
          f"residual {true:.3e} at the true camera against {wide:.3e} at f x 1.4")
    assert wide > 2.0 * true, f"the true camera is not clearly better: {true:.3e} against {wide:.3e}"


# ------------------------------------------------------------------------------------------------
# Dataset plumbing.
# ------------------------------------------------------------------------------------------------


def write_trajectory(root, frames, camera="lcam_fish"):
    from PIL import Image

    directory = os.path.join(root, "Scene", "Data_hard", "P000", f"image_{camera}")
    os.makedirs(directory, exist_ok=True)
    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(os.path.join(directory, f"{i:06d}_{camera}_image.png"))
    poses = np.zeros((len(frames), 7), dtype=np.float64)
    poses[:, 0] = np.arange(len(frames))
    poses[:, 6] = 1.0
    np.savetxt(os.path.join(root, "Scene", "Data_hard", "P000", f"pose_{camera}.txt"), poses)
    return root


@pytest.fixture(scope="module")
def fisheye_root():
    """Three 1000 x 1000 frames with real curved lines on them, the size the augmentation needs."""
    image, _ = fisheye_curve_image(size=FISH_SIZE, line_width=3)
    frames = [np.roll(image, shift * 7, axis=1) for shift in range(3)]
    with tempfile.TemporaryDirectory() as directory:
        write_trajectory(directory, frames)
        yield directory


def build_dataset(root, **kwargs):
    from training.tartanair_dataset import TartanAirDataset

    options = dict(
        image_size=256,
        patch_size=16,
        num_views=2,
        num_input_views=1,
        view_selector={"type": "uniform"},
        camera_name="lcam_fish",
        normalize_intrinsics=True,
    )
    options.update(kwargs)
    return TartanAirDataset(root, **options)


def test_dataset_default_is_byte_identical(fisheye_root):
    """Off by default, and off means every tensor is the one the dataset produced before."""
    plain = build_dataset(fisheye_root)
    assert plain.detect_lines is False

    pyrandom.seed(99)
    before = plain[0]
    pyrandom.seed(99)
    after = build_dataset(fisheye_root, detect_lines=False, line_max_count=4)[0]

    assert list(before.keys()) == ["image", "fxfycxcy", "c2w", "index"], "the returned keys changed"
    assert "line_pixels" not in before and "line_count" not in before
    for key in before:
        assert torch.equal(before[key], after[key]), f"{key} moved with the flag off"


def test_dataset_adds_padded_line_tensors(fisheye_root):
    """With the flag on: fixed shapes, honest counts and -1 everywhere beyond them."""
    data = build_dataset(fisheye_root, detect_lines=True, line_max_count=6, line_samples=5,
                         line_min_length=0.15)
    sample = data[0]

    views = sample["image"].shape[0]
    assert sample["line_pixels"].shape == (views, 6, 5, 2)
    assert sample["line_count"].shape == (views,)
    assert sample["line_pixels"].dtype == torch.float32
    assert sample["line_count"].dtype == torch.long

    counts = sample["line_count"]
    assert int(counts.max()) > 0, "no lines found on a frame full of lines"
    for view in range(views):
        count = int(counts[view])
        assert 0 <= count <= 6
        real = sample["line_pixels"][view, :count]
        padding = sample["line_pixels"][view, count:]
        assert float(real.min()) >= 0.0, "a real line carries the padding value"
        assert float(real.max()) <= 256.0, "a line left the image"
        assert torch.equal(padding, torch.full_like(padding, PAD_VALUE)), "the padding is not -1"
        assert valid_segment_mask(sample["line_pixels"][view]).sum() == count


def test_dataset_lines_are_detected_after_the_remap(fisheye_root):
    """With camera augmentation the lines belong to the drawn camera, not to the source frame.

    Checked where it can be checked without knowing the drawn camera: the lines lie inside the
    sample's own data mask, which only exists after the remap, and they fit the remapped image.
    """
    data = build_dataset(fisheye_root, camera_augmentation=True, detect_lines=True,
                         line_max_count=8, line_samples=6, line_min_length=0.15)
    pyrandom.seed(5)
    sample = data[0]
    assert "data_mask" in sample and "line_pixels" in sample

    mask = sample["data_mask"][:, 0] > 0.5
    for view in range(sample["image"].shape[0]):
        count = int(sample["line_count"][view])
        for row in sample["line_pixels"][view, :count]:
            columns = torch.round(row[:, 0] - 0.5).long().clamp(0, 255)
            rows = torch.round(row[:, 1] - 0.5).long().clamp(0, 255)
            assert bool(mask[view][rows, columns].all()), "a line sits outside the data mask"


def test_views_helper_pads_every_view_to_the_same_width():
    """`detect_lines_in_views` is what the loader calls; its shapes are what the collate needs."""
    views = [grid_image(), np.zeros((256, 256, 3), dtype=np.uint8)]
    pixels, counts = detect_lines_in_views(views, min_length=0.3, max_count=5, num_samples=4)
    assert pixels.shape == (2, 5, 4, 2) and counts.shape == (2,)
    assert int(counts[1]) == 0, "the empty view found lines"
    assert torch.equal(pixels[1], torch.full_like(pixels[1], PAD_VALUE))
    assert int(counts[0]) > 0

    masks = [torch.zeros(1, 256, 256), torch.zeros(1, 256, 256)]
    pixels, counts = detect_lines_in_views(views, masks=masks, min_length=0.3, max_count=5)
    assert int(counts.sum()) == 0, "an all invalid mask still produced lines"

    with pytest.raises(ValueError):
        detect_lines_in_views(views, masks=masks[:1])


def test_config_knobs_reach_the_dataset(fisheye_root):
    """`build_dataset` from the config, the path a training run takes."""
    from easydict import EasyDict as edict

    from training.datasets import build_dataset as build_from_config

    config = edict({
        "dataset": {"name": "tartanair", "train_path": fisheye_root, "camera_name": "lcam_fish",
                    "detect_lines": True, "line_max_count": 9, "line_samples": 4},
        "training": {"num_views": 2, "num_input_views": 1, "view_selector": {"type": "uniform"}},
        "model": {"image_tokenizer": {"image_size": 256, "patch_size": 16}},
    })
    data = build_from_config(config, split="train")
    assert data.detect_lines is True
    assert data.line_max_count == 9 and data.line_samples == 4 and data.line_chain is True
    assert data[0]["line_pixels"].shape[1:] == (9, 4, 2)

    config.dataset.line_chain = False
    assert build_from_config(config, split="train").line_chain is False

    config.dataset.detect_lines = False
    assert build_from_config(config, split="train").detect_lines is False
    assert build_dataset(fisheye_root).line_max_count == DEFAULT_MAX_COUNT


def test_line_segments_dataclass_length():
    empty = LineSegments(np.zeros((0, 4), np.float32), np.zeros((0,), np.float32),
                         np.zeros((0, 8, 2), np.float32))
    assert len(empty) == 0
