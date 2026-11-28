"""The differentiable plumb line residual: how far the rays of one line are from one plane.

The geometry
------------
A central camera sees a 3D line L through its projection centre C. Every point of L, the centre C
and therefore every viewing ray of every pixel imaging L lie in one plane, the interpretation plane
of the line. The plane goes through C, so in camera coordinates it is described by a unit normal n
with

    n . d_i = 0   for the direction d_i of every pixel i that images L.

That is the plumb line constraint. It says nothing about the image being straight, which is why it
survives a fisheye or a (E)UCM camera where a 3D line images as a curve (see the module docstring of
`training/line_detection.py`). It is a pure statement about the camera model: the pixels come from
the detector and never move, the directions d_i come from the intrinsics, and the residual is zero
exactly when the intrinsics turn those pixels into coplanar rays. A focal length that is too large
bends the rays the wrong way and the residual grows, which is the signal all three line ideas use.

The residual
------------
For the K sampled pixels of one line, collect the unit directions into the scatter matrix

    S = sum_i d_i d_i^T   (3 x 3, symmetric, positive semi definite),

with eigenvalues lambda_1 <= lambda_2 <= lambda_3. The best fitting plane minimises the squared
distances of the rays from the plane,

    min over unit n of  sum_i (n . d_i)^2 = min over unit n of n^T S n = lambda_1,

attained at the eigenvector of the smallest eigenvalue, so lambda_1 is the raw residual in squared
sine units. The three eigenvalues read, for a tight bundle of rays around a mean direction:
lambda_3 is the bundle itself (about K), lambda_2 is how far the rays spread WITHIN the fitted
plane, that is the angular extent of the line, and lambda_1 is how far they stick out of it.

Normalisation, and why it is not the trace
------------------------------------------
The obvious normalisation is by the trace, r = lambda_1 / trace(S), which is the mean squared sine
and lies in [0, 1/3] since lambda_1 is at most a third of the sum. It must not be the default here,
because it has a degenerate minimum that points the wrong way. Raise the focal length and every
pixel maps to a ray closer to the optical axis: the whole bundle shrinks, lambda_1 and lambda_2
both fall like 1 / f^2, trace(S) stays at K, and so lambda_1 / trace falls towards zero for ANY set
of pixels whatsoever. Minimising it drives the focal length to infinity, the narrowest camera
possible, which is the very failure the line constraint is meant to fight. Measured on a real
`lcam_fish` frame with chained lines (`scripts/line_sanity_check.py`): the mean of lambda_1 / trace
keeps falling from 2.2e-4 at the true focal length to 2.8e-5 at 2.5 times it.

The in plane normalisation removes exactly that shrink, because lambda_2 shrinks with the same
1 / f^2 as lambda_1:

    r = lambda_1 / (lambda_1 + lambda_2),

the squared ratio of the out of plane deviation to the in plane extent of the line. It is invariant
to a uniform contraction of the bundle, so it measures only the SHAPE of the ray set, and it has a
genuine minimum at the true camera. lambda_1 <= lambda_2 gives

    r is in [0, 1/2],

zero for exactly coplanar rays and 1/2 for rays with as much spread out of their plane as in it.
A bounded loss term matters in training: no single bad line can dominate the batch.

Both forms are implemented, `normalize="inplane"` (the default) and `normalize="trace"`, so the
trace form stays available for comparison and for the tests that pin the eigenvalue math.

A line whose rays have no in plane extent at all (K = 1, K identical pixels, the padded rows of a
batch) has lambda_1 = lambda_2 = 0, where the in plane quotient would be 0/0 and float noise would
make it anything at all, usually the worst possible 1/2. Those lines are detected by their vanishing
lambda_2 and returned as a hard 0 with a zero gradient. The threshold, `DEGENERATE_SPREAD` times the
trace, sits five orders of magnitude below the in plane extent of any line long enough to be
detected at all, and the quotient is clamped underneath it, so no NaN can enter the backward pass
through the unused branch either.

The gradient
------------
`torch.linalg.eigvalsh` is differentiable, but its backward divides by the gaps between eigenvalues
and returns NaN as soon as two eigenvalues coincide. They coincide in exactly the cases this code
must survive: K = 1, K identical pixels, perfectly collinear rays, all of which give lambda_1 =
lambda_2 = 0. So the eigenvalues are evaluated as Rayleigh quotients with their eigenvectors held
fixed,

    lambda_j = n_j^T S n_j,   n_j = the j-th eigenvector of S, detached,

and the residual is assembled from those. This is not an approximation of the gradient, it is the
gradient wherever it exists. First order perturbation theory for a symmetric matrix with a simple
eigenvalue gives d lambda_j = n_j^T (dS) n_j, which is precisely the derivative of the Rayleigh
quotient at fixed n_j, because n_j is a stationary point of the quotient and the term through d n_j
vanishes. Where an eigenvalue is degenerate and the derivative does not exist, the quotient still
returns a finite value and a finite gradient instead of a NaN that would poison the training step.

The eigenvectors are taken with `torch.linalg.eigh` on the detached matrix in float64, so no
gradient ever flows through the decomposition; `torch.linalg.eigvalsh` reproduces the same values
and the test suite pins the agreement.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch

from erayzer_core.model.utils_gaussian import UCM_MODELS, camera_theta

RHO_EPS = 1e-12
"""Added inside the square root of the image radius, as in `utils_gaussian.camera_rays`, so that a
pixel sitting exactly on the principal point gives rho > 0 and sin(theta) / rho stays finite."""

DEGENERATE_SPREAD = 1e-5
"""Below this in plane spread, relative to the trace, a line carries no constraint and the residual
is a hard zero. A line of K pixels spanning the angle psi has lambda_2 of about K psi^2 / 12, so at
K = 8 the threshold is reached at psi of about 0.01 radian, half a degree, far below anything the
detector reports; repeated pixels and padded rows land well under it, at the float32 noise of the
scatter matrix."""

DEFAULT_TRIM = 0.3
"""Fraction of the worst lines `plumbline_residual_mean` drops before averaging.

A detected line is a guess, and a chain that joined two different objects is not coplanar under any
camera: it contributes a large residual that barely moves when the camera does, and it drowns the
lines that do carry the signal. Dropping the worst thirty percent is a plain trimmed mean and costs
a sort. Measured over eight `data-sample` frames, residual at 1.4 times the true focal length over
residual at the true one, higher is a sharper signal:

    aggregation     plain mean   trimmed mean   median
    1000 px                6.2           32.3     33.8
    256 px                 1.3            2.3      2.4

The median is marginally sharper still but passes gradient to one single line per step, so the
trimmed mean is the better loss. Set trim = 0 for the plain mean.
"""

NORMALIZATIONS = ("inplane", "trace")
"""How lambda_1 is normalised, see the module docstring. "inplane" is scale free and the default,
"trace" is the mean squared sine and prefers an infinite focal length."""


def _as_leading(fxfycxcy: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    """Broadcast intrinsics [d], [1, d], ... to `shape` + [d] and flatten the leading dims."""
    if fxfycxcy.dim() == 1:
        fxfycxcy = fxfycxcy.reshape(*([1] * len(shape)), -1)
    if fxfycxcy.dim() != len(shape) + 1:
        raise ValueError(
            f"fxfycxcy with shape {tuple(fxfycxcy.shape)} does not broadcast to {tuple(shape)} + (d,)"
        )
    expanded = fxfycxcy.expand(*shape, fxfycxcy.shape[-1])
    return expanded.reshape(-1, fxfycxcy.shape[-1])


def _denormalize(fxfycxcy: torch.Tensor, image_hw: Optional[Tuple[int, int]]) -> torch.Tensor:
    """Scale normalised intrinsics (fractions of the image side) back to pixels.

    The dataset and the camera head carry fx, fy, cx, cy divided by the image size; `erayzer.py`
    multiplies them back by width and height before rendering, and this does the same, leaving the
    alpha and beta columns alone because they are dimensionless.
    """
    if image_hw is None:
        raise ValueError("normalized_intrinsics needs image_hw to scale by")
    h, w = int(image_hw[0]), int(image_hw[1])
    scale = torch.tensor([w, h, w, h], dtype=fxfycxcy.dtype, device=fxfycxcy.device)
    if fxfycxcy.shape[-1] > 4:
        pad = torch.ones(fxfycxcy.shape[-1] - 4, dtype=fxfycxcy.dtype, device=fxfycxcy.device)
        scale = torch.cat([scale, pad])
    return fxfycxcy * scale


def pixel_ray_directions(
    fxfycxcy: torch.Tensor,
    camera_model: str,
    pixels: torch.Tensor,
) -> torch.Tensor:
    """Unit viewing directions in camera coordinates for arbitrary subpixel coordinates.

    * `fxfycxcy` [..., 4] in pixels, [..., 5] with alpha for "ucm", [..., 6] with beta for "eucm",
      broadcast against the leading dimensions of `pixels`,
    * `pixels` [..., 2], x and y in this repository's convention (centre of pixel (col, row) at
      (col + 0.5, row + 0.5)).

    Returns [..., 3]. This is the same math as `utils_gaussian.camera_rays` with an identity pose,
    and it calls the very same `camera_theta`, which is where the camera model lives. It exists
    separately only because `camera_rays` and `pixel_angles` build a full h x w pixel grid, while
    the detected line pixels are a scattered handful of subpixel positions. A test pins this
    function against `camera_rays` on a full grid, so the two cannot drift apart.
    """
    fx = fxfycxcy[..., 0]
    fy = fxfycxcy[..., 1]
    cx = fxfycxcy[..., 2]
    cy = fxfycxcy[..., 3]
    x = (pixels[..., 0] - cx) / fx
    y = (pixels[..., 1] - cy) / fy
    z = torch.ones_like(x)

    if camera_model != "pinhole":
        rho = torch.sqrt(x * x + y * y + RHO_EPS)
        alpha = fxfycxcy[..., 4] if camera_model in UCM_MODELS else None
        beta = fxfycxcy[..., 5] if camera_model == "eucm" else None
        theta = camera_theta(rho, camera_model, alpha, beta)
        sin_theta = torch.sin(theta)
        x, y, z = x * sin_theta / rho, y * sin_theta / rho, torch.cos(theta)

    direction = torch.stack([x, y, z], dim=-1)
    return direction / torch.norm(direction, dim=-1, keepdim=True).clamp_min(RHO_EPS)


def valid_segment_mask(pixels: torch.Tensor) -> torch.Tensor:
    """Which padded segments are real, [...] bool for `pixels` [..., K, 2].

    `line_detection.detect_lines_in_views` pads unused rows with a negative coordinate, which no
    real pixel can have, so a row is padding exactly when its first coordinate is negative. Use this
    together with `line_count` from the batch, or instead of it when only the tensor is at hand.
    """
    return pixels[..., 0, 0] >= 0


def plumbline_residual(
    fxfycxcy: torch.Tensor,
    camera_model: str,
    pixels: torch.Tensor,
    image_hw: Optional[Tuple[int, int]] = None,
    *,
    normalized_intrinsics: bool = False,
    normalize: str = "inplane",
) -> torch.Tensor:
    """How far the rays of each detected line are from lying in one plane through the camera.

    * `fxfycxcy` [4] / [5] / [6] for one camera, or [..., d] broadcast against the lines, in
      pixels; with `normalized_intrinsics` the first four columns are fractions of the image side
      and `image_hw` scales them back.
    * `camera_model` "pinhole", "fisheye", "ucm" or "eucm", as everywhere else in the repository.
    * `pixels` [..., K, 2], the K sampled pixels of each line (`line_detection` produces them).
    * `image_hw` (h, w) of the image the pixels were found on. Only needed for normalised
      intrinsics; the residual itself does not care how large the image is.
    * `normalize` "inplane" for the scale free quotient in [0, 1/2] (the default, the only one with
      a minimum at the true camera) or "trace" for the mean squared sine in [0, 1/3]. The module
      docstring derives both and says why the default is what it is.

    Returns [...], differentiable with respect to every intrinsic, f, cx, cy, alpha and beta alike.
    Degenerate lines (K = 1, repeated pixels, exactly collinear rays) give 0 with a zero gradient
    and never a NaN. An empty input gives an empty result.

    This one function is the whole geometric half of the line infrastructure: the training loss
    averages it over the detected lines, the per scene refinement minimises it over f and alpha,
    and the initialiser scores a grid of cameras with it.
    """
    if pixels.dim() < 2 or pixels.shape[-1] != 2:
        raise ValueError(f"pixels must be [..., K, 2], got {tuple(pixels.shape)}")
    if not isinstance(fxfycxcy, torch.Tensor):
        raise TypeError("fxfycxcy must be a tensor")
    if normalize not in NORMALIZATIONS:
        raise ValueError(f"unknown normalize {normalize!r}, expected one of {NORMALIZATIONS}")

    needed = {"ucm": 5, "eucm": 6}.get(camera_model, 4)
    if fxfycxcy.shape[-1] < needed:
        raise ValueError(f"camera model {camera_model!r} needs {needed} intrinsics, got {fxfycxcy.shape[-1]}")
    if normalized_intrinsics:
        fxfycxcy = _denormalize(fxfycxcy, image_hw)

    leading = pixels.shape[:-2]
    k = pixels.shape[-2]
    flat_pixels = pixels.reshape(-1, k, 2)
    if flat_pixels.shape[0] == 0:
        return pixels.new_zeros(leading)
    flat_intr = _as_leading(fxfycxcy, leading)[:, None, :]  # [n, 1, d]

    directions = pixel_ray_directions(flat_intr, camera_model, flat_pixels)  # [n, K, 3]
    scatter = torch.einsum("nki,nkj->nij", directions, directions)  # [n, 3, 3]
    trace = scatter.diagonal(dim1=-2, dim2=-1).sum(-1)  # [n], equals K for unit rays

    with torch.no_grad():
        # float64 and detached: the decomposition only supplies the directions, the gradient comes
        # from the Rayleigh quotients below. eigh returns the eigenvalues in ascending order, so
        # column 0 is the normal of the fitted plane and column 1 the in plane spread direction.
        vectors = torch.linalg.eigh(scatter.detach().double())[1].to(scatter.dtype)  # [n, 3, 3]
    out_of_plane = vectors[..., 0]  # [n, 3]
    in_plane = vectors[..., 1]  # [n, 3]

    # S is positive semi definite, so neither quotient is truly negative; the clamps only remove the
    # rounding noise around zero that a degenerate line produces (about 1e-16 in float64). At an
    # exact zero a clamp still passes the gradient through, and where one does bite, the gradient it
    # drops is that same rounding noise.
    lambda_1 = torch.einsum("ni,nij,nj->n", out_of_plane, scatter, out_of_plane).clamp_min(0.0)
    if normalize == "trace":
        residual = lambda_1 / trace.clamp_min(RHO_EPS)
    else:
        lambda_2 = torch.einsum("ni,nij,nj->n", in_plane, scatter, in_plane).clamp_min(0.0)
        spread = (lambda_1 + lambda_2).clamp_min(RHO_EPS)
        quotient = lambda_1 / spread
        # A line with no in plane extent constrains nothing; see DEGENERATE_SPREAD. The clamp above
        # already makes the quotient finite, so this `where` cannot pull a NaN in through its other
        # branch, it only replaces a meaningless value by an honest zero.
        degenerate = lambda_2.detach() <= DEGENERATE_SPREAD * trace.detach()
        residual = torch.where(degenerate, torch.zeros_like(quotient), quotient)
    return residual.reshape(leading)


def plumbline_residual_mean(
    fxfycxcy: torch.Tensor,
    camera_model: str,
    pixels: torch.Tensor,
    image_hw: Optional[Tuple[int, int]] = None,
    *,
    normalized_intrinsics: bool = False,
    normalize: str = "inplane",
    trim: float = DEFAULT_TRIM,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """`plumbline_residual` averaged over the real lines of a padded tensor, a scalar.

    `pixels` [..., n_max, K, 2] straight out of the batch, padding included: padded rows are found
    with `valid_segment_mask`, replaced by a repeated pixel so they come out as a degenerate zero
    and no NaN can enter through them, and left out of the average. `weights` [..., n_max]
    optionally weights the real lines (their detection score, say). With no real line at all the
    result is a differentiable zero, which keeps a training step that found no lines from crashing.

    `trim` drops that fraction of the worst lines before averaging, over all leading dimensions at
    once; see DEFAULT_TRIM for why it is on by default and what it buys.
    """
    valid = valid_segment_mask(pixels)
    safe_pixels = torch.where(valid[..., None, None], pixels, torch.zeros_like(pixels))
    residual = plumbline_residual(
        fxfycxcy, camera_model, safe_pixels, image_hw,
        normalized_intrinsics=normalized_intrinsics, normalize=normalize,
    )
    weight = valid.to(residual.dtype)
    if weights is not None:
        weight = weight * weights.to(residual.dtype)

    if trim > 0.0:
        # Which lines survive is decided on the detached values and then held fixed as a weight, so
        # the average stays a plain weighted mean and the gradient never runs through the sorting.
        flat = residual.detach().reshape(-1)
        flat_weight = weight.reshape(-1)
        count = int(flat_weight.gt(0).sum())
        keep = max(1, int(round((1.0 - float(trim)) * count)))
        if 0 < keep < count:
            # Dropped lines go to the end of the order by being pushed past every real residual.
            order = torch.argsort(torch.where(flat_weight > 0, flat, torch.full_like(flat, float("inf"))))
            cut = torch.zeros_like(flat_weight)
            cut[order[:keep]] = 1.0
            weight = weight * cut.reshape(weight.shape)

    total = weight.sum()
    return (residual * weight).sum() / total.clamp_min(1e-6)


def residual_to_degrees(residual: torch.Tensor) -> torch.Tensor:
    """A "trace" residual read as an angle: the rms angle of the rays off their fitted plane.

    Only for reporting, and only for `normalize="trace"`, where r really is a mean squared sine and
    the angle is asin(sqrt(r)). The default "inplane" residual is a ratio of two angles and has no
    such reading.
    """
    return torch.asin(torch.sqrt(residual.clamp(0.0, 1.0))) * (180.0 / math.pi)
