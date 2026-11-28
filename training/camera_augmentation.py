"""Resample TartanAir `lcam_fish` frames to randomly drawn cameras, one camera per sample.

`lcam_fish` is equidistant (r = f * theta) over 195 degrees at 1000 x 1000, so every camera that
looks forward and covers at most 195 degrees can be computed from it exactly. Training on a fresh
random camera per sample is what teaches the model to read focal length and distortion off the
image instead of memorising the one camera of the dataset.

The remap core (`camera_rays`, `fisheye_pixels`, `remap_fisheye`, the `FISH_*` constants) is ported
from `scripts/tartanair_remap.py` on branch `tartanair-cameras`, which is an offline script. It is
ported rather than imported because that file lives on another branch; the pinhole and fisheye
results here are identical to it, which `tests/test_camera_aug.py` pins with reference numbers.

Added here on top of the template:

* the unified camera model (UCM), rho_norm = sin(theta) / (alpha + (1 - alpha) * cos(theta)),
* `sample_camera` to draw (model, fov, alpha) for one training sample,
* `remap_views` to put all views of a sample through the same camera in one pass,
* `fit_ucm_alpha` to describe an equidistant camera by the closest UCM alpha, for logging only.

Pixel centre convention: the template counts pixel centres at integer coordinates, so its principal
point is (size - 1) / 2. The rest of this repository (`utils_gaussian.pixel_angles`, `camera_rays`)
counts them at x + 0.5, where the same physical centre is size / 2. `remap_views` returns the
repository convention; `remap_fisheye` keeps the template's so the two stay comparable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

FISH_SIZE = 1000
FISH_FOV_DEG = 195.0
FISH_F = FISH_SIZE / math.radians(FISH_FOV_DEG)  # pixels per radian, about 293.8
FISH_C = (FISH_SIZE - 1) / 2.0  # OpenCV pixel centres

MODEL_CODES: Dict[str, int] = {"pinhole": 0, "fisheye": 1, "ucm": 2}
"""Integer code of each camera model, so the model choice survives the default collate."""

DEFAULT_MIX: Dict[str, float] = {"pinhole": 0.25, "fisheye": 0.25, "ucm": 0.5}
"""Default share of each model. UCM carries the most weight because it is the training target."""

DEFAULT_SUPERSAMPLE = 2
"""Render at twice the output size and shrink with INTER_AREA.

The source is 1000 x 1000 and the output is 256 x 256, so a 512 x 512 intermediate already samples
the source denser than one output pixel in every direction; 4 only costs time. See the timing in
`tests/test_camera_aug.py::test_remap_speed`.
"""


@dataclass(frozen=True)
class AugCamera:
    """One drawn camera. `alpha` is the UCM alpha, `None` for the equidistant fisheye."""

    model: str
    fov_deg: float
    alpha: Optional[float] = None

    @property
    def code(self) -> int:
        return MODEL_CODES[self.model]


def ucm_rho_norm(theta: np.ndarray | float, alpha: float) -> np.ndarray | float:
    """Image radius divided by f for the angle `theta` to the optical axis, the UCM projection."""
    return np.sin(theta) / (alpha + (1.0 - alpha) * np.cos(theta))


def ucm_theta(rho: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse of `ucm_rho_norm`, plus which radii the model can invert at all.

    This is `erayzer_core.model.utils_gaussian.camera_theta` for "ucm" in numpy, deliberately the
    same expression: the remap and the renderer must agree on what a UCM camera is.
    """
    disc = 1.0 - (2.0 * alpha - 1.0) * rho**2
    invertible = disc >= 0.0
    mz = (1.0 - alpha**2 * rho**2) / (alpha * np.sqrt(np.clip(disc, 0.0, None)) + 1.0 - alpha)
    return np.arctan2(rho, mz), invertible


def ucm_theta_max(alpha: float) -> float:
    """Largest angle a UCM camera with this alpha can still image, in radians.

    For alpha < 0.5 the denominator of `ucm_rho_norm` vanishes at acos(-alpha / (1 - alpha)) and the
    radius runs to infinity. For alpha >= 0.5 the denominator stays positive but the radius peaks at
    acos(-(1 - alpha) / alpha) and folds back, so the projection is no longer injective beyond it.
    Both cases are acos(-min(alpha, 1 - alpha) / max(alpha, 1 - alpha)).
    """
    lo, hi = min(alpha, 1.0 - alpha), max(alpha, 1.0 - alpha)
    if hi <= 0.0:
        return math.pi
    return math.acos(max(-1.0, min(1.0, -lo / hi)))


def focal_for_fov(model: str, fov_deg: float, size: int, alpha: Optional[float] = None) -> float:
    """Focal length in pixels that puts the half field of view on the image half width."""
    half = math.radians(fov_deg) / 2.0
    if model == "pinhole":
        if fov_deg >= 180.0:
            raise ValueError("a pinhole camera covers less than 180 degrees")
        return (size / 2.0) / math.tan(half)
    if model == "fisheye":
        return (size / 2.0) / half
    if model == "ucm":
        if alpha is None:
            raise ValueError("the UCM needs an alpha")
        if half >= ucm_theta_max(alpha):
            raise ValueError(
                f"a UCM camera with alpha {alpha} cannot reach {fov_deg} degrees "
                f"(limit {2.0 * math.degrees(ucm_theta_max(alpha)):.2f})"
            )
        return (size / 2.0) / float(ucm_rho_norm(half, alpha))
    raise ValueError(f"unknown camera model {model!r}")


def camera_rays(
    model: str, fov_deg: float, size: int, alpha: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Unit rays of a square camera in OpenCV axes, its validity mask, f and c.

    Ported from `scripts/tartanair_remap.py`; the "ucm" branch is new.
    """
    half = math.radians(fov_deg) / 2.0
    c = (size - 1) / 2.0
    u, v = np.meshgrid(np.arange(size, dtype=np.float64), np.arange(size, dtype=np.float64))
    x, y = u - c, v - c

    if model == "pinhole":
        if fov_deg >= 180.0:
            raise ValueError("a pinhole camera covers less than 180 degrees")
        f = (size / 2.0) / math.tan(half)
        rays = np.stack([x / f, y / f, np.ones_like(x)], axis=-1)
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
        valid = np.ones(x.shape, dtype=bool)
    elif model == "fisheye":
        f = (size / 2.0) / half
        theta = np.hypot(x, y) / f
        phi = np.arctan2(y, x)
        rays = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1)
        valid = theta <= half
    elif model == "ucm":
        f = focal_for_fov("ucm", fov_deg, size, alpha)
        rho = np.hypot(x, y) / f
        phi = np.arctan2(y, x)
        theta, invertible = ucm_theta(rho, float(alpha))
        rays = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1)
        # Outside the image circle of the model the angle is meaningless, so drop it as well.
        valid = invertible & (theta <= half)
    else:
        raise ValueError(f"unknown camera model {model!r}")
    return rays, valid, f, c


IDENTITY_FISHEYE = AugCamera("fisheye", FISH_FOV_DEG, None)
"""The camera an unaugmented sample still has: `lcam_fish` itself, only resized.

`remap_views` with this camera and output side S would reproduce the source frame scaled to S, and
`map_fisheye_pixels` with it is exactly the uniform scale S / 1000. Having it as a camera means the
unaugmented and the augmented path share one piece of code instead of two that can drift.
"""


def map_fisheye_pixels(
    pixels: np.ndarray,
    camera: AugCamera,
    size: int,
    source_size: int = FISH_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Where pixels of an `lcam_fish` frame land in the image of a drawn camera, analytically.

    `pixels` [..., 2] in the source frame, in this repository's convention (centre of pixel
    (col, row) at (col + 0.5, row + 0.5), so the principal point of a square frame is at
    source_size / 2). `source_size` may be smaller than 1000 if the frame was resized first, which
    is a pure scale and changes nothing but the focal length. Returns the pixels in the output
    image of side `size`, same shape, and a boolean mask [...] of the ones the drawn camera images
    at all.

    Both cameras look along the same axis, so the azimuth phi is untouched and only the radius
    changes: unproject the source radius with the equidistant law theta = r / f_source, then
    project theta with the drawn camera's own law. That is `remap_views` run backwards, exactly,
    and it is what lets lines be detected on the sharp 1000 pixel source and still be expressed in
    the pixels of the 256 pixel view the model sees, which is where the residual needs them.
    """
    points = np.asarray(pixels, dtype=np.float64)
    source_f = FISH_F * (float(source_size) / FISH_SIZE)
    dx = points[..., 0] - float(source_size) / 2.0
    dy = points[..., 1] - float(source_size) / 2.0
    radius = np.hypot(dx, dy)
    theta = radius / source_f
    phi = np.arctan2(dy, dx)

    # The same ranges `camera_rays` gives the remap, so a pixel counts as imaged here exactly when
    # the remap filled it. A pinhole has no angular limit of its own, its field of view is only
    # measured across the image width and its corners reach further; the other two stop at fov / 2.
    half = math.radians(float(camera.fov_deg)) / 2.0
    if camera.model == "pinhole":
        valid = theta < math.pi / 2.0
        rho = np.tan(np.clip(theta, 0.0, 0.499 * math.pi))
    elif camera.model == "fisheye":
        valid = theta <= half
        rho = theta
    elif camera.model == "ucm":
        valid = (theta <= half) & (theta < ucm_theta_max(float(camera.alpha)))
        rho = ucm_rho_norm(np.clip(theta, 0.0, half), float(camera.alpha))
    else:
        raise ValueError(f"unknown camera model {camera.model!r}")

    f = focal_for_fov(camera.model, float(camera.fov_deg), int(size), camera.alpha)
    centre = float(size) / 2.0
    out = np.stack([centre + f * rho * np.cos(phi), centre + f * rho * np.sin(phi)], axis=-1)
    valid &= np.all((out >= 0.0) & (out <= float(size)), axis=-1)
    return out.astype(np.float32), valid


def fisheye_pixels(rays: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixel coordinates in `lcam_fish` for unit rays, plus a mask of rays it covers.

    Ported unchanged from `scripts/tartanair_remap.py`.
    """
    theta = np.arctan2(np.hypot(rays[..., 0], rays[..., 1]), rays[..., 2])
    phi = np.arctan2(rays[..., 1], rays[..., 0])
    r = FISH_F * theta
    inside = theta <= math.radians(FISH_FOV_DEG) / 2.0
    return FISH_C + r * np.cos(phi), FISH_C + r * np.sin(phi), inside


@lru_cache(maxsize=4)
def _remap_tables(
    model: str, fov_deg: float, size: int, supersample: int, alpha: Optional[float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """The cv2.remap tables and the valid mask of one camera, shared by every view of a sample.

    `remap_views` builds them once and reuses them for all views of its sample, which is where the
    saving is; the drawn field of view is a continuous number, so a training run practically never
    asks for the same camera twice. The cache is therefore small on purpose, it only serves repeated
    calls with a fixed camera such as `remap_fisheye` in a loop, and holding more would just cost
    every data loader worker a few megabytes for nothing.
    """
    big = size * supersample
    rays, valid, f_big, _ = camera_rays(model, fov_deg, big, alpha)
    map_x, map_y, inside = fisheye_pixels(rays)
    mask = valid & inside
    return map_x.astype(np.float32), map_y.astype(np.float32), mask, f_big / supersample


def _apply_remap(
    image: np.ndarray, map_x: np.ndarray, map_y: np.ndarray, mask: np.ndarray, size: int, supersample: int
) -> Tuple[np.ndarray, np.ndarray]:
    out = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    out[~mask] = 0
    small_mask = mask
    if supersample > 1:
        out = cv2.resize(out, (size, size), interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(mask.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA) > 0.999
    return out, small_mask


def remap_fisheye(
    image: np.ndarray,
    model: str,
    fov_deg: float,
    size: int,
    supersample: int = 4,
    alpha: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float, float]]:
    """Resample a `lcam_fish` image (H x W x 3, uint8) to another camera.

    Returns the image, a boolean mask of valid pixels and fx, fy, cx, cy in pixels of the output,
    with the principal point in the template's convention, (size - 1) / 2.

    Ported from `scripts/tartanair_remap.py`, extended by the `alpha` argument for "ucm".
    """
    if image.shape[:2] != (FISH_SIZE, FISH_SIZE):
        raise ValueError(f"expected a {FISH_SIZE} x {FISH_SIZE} lcam_fish image, got {image.shape[:2]}")
    map_x, map_y, mask, f = _remap_tables(model, float(fov_deg), int(size), int(supersample), alpha)
    out, small_mask = _apply_remap(image, map_x, map_y, mask, int(size), int(supersample))
    c = (size - 1) / 2.0
    return out, small_mask, (f, f, c, c)


def _fit_cost(alpha: float, half: float, thetas: np.ndarray, radii: np.ndarray) -> float:
    """Squared radius error of a UCM camera with this alpha against the given true radii.

    Both cameras are normalised to image radius 1 at `half`, so only the shape of the curve counts.
    """
    if half >= ucm_theta_max(alpha):
        return float("inf")
    scale = 1.0 / float(ucm_rho_norm(half, alpha))
    return float(np.sum((scale * ucm_rho_norm(thetas, alpha) - radii) ** 2))


@lru_cache(maxsize=64)
def fit_ucm_alpha(fov_deg: float, samples: int = 64) -> float:
    """The UCM alpha whose projection is closest to an equidistant camera of this field of view.

    Least squares over theta in (0, fov / 2] on radii normalised to 1 at the image edge. Used only
    to report a true alpha for equidistant samples (`aug_alpha_true`); nothing is trained on it.
    A coarse scan brackets the minimum, golden section refines it, so there is no scipy dependency
    and the result is deterministic.
    """
    half = math.radians(fov_deg) / 2.0
    thetas = np.linspace(half / samples, half, samples)
    radii = thetas / half  # equidistant, normalised to 1 at the edge

    grid = np.linspace(1e-4, 1.0 - 1e-4, 201)
    costs = [_fit_cost(float(a), half, thetas, radii) for a in grid]
    best = int(np.argmin(costs))
    if not math.isfinite(costs[best]):
        raise ValueError(f"no UCM camera reaches {fov_deg} degrees")
    lo = float(grid[max(best - 1, 0)])
    hi = float(grid[min(best + 1, len(grid) - 1)])

    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c_pt, d_pt = b - inv_phi * (b - a), a + inv_phi * (b - a)
    fc, fd = _fit_cost(c_pt, half, thetas, radii), _fit_cost(d_pt, half, thetas, radii)
    for _ in range(80):
        if fc < fd:
            b, d_pt, fd = d_pt, c_pt, fc
            c_pt = b - inv_phi * (b - a)
            fc = _fit_cost(c_pt, half, thetas, radii)
        else:
            a, c_pt, fc = c_pt, d_pt, fd
            d_pt = a + inv_phi * (b - a)
            fd = _fit_cost(d_pt, half, thetas, radii)
        if b - a < 1e-10:
            break
    return 0.5 * (a + b)


def true_alpha(camera: AugCamera) -> float:
    """The UCM alpha that describes this camera, for logging and evaluation only.

    Exact for pinhole (0) and UCM, a least squares fit for the equidistant fisheye.
    """
    if camera.model == "pinhole":
        return 0.0
    if camera.model == "ucm":
        return float(camera.alpha)
    return float(fit_ucm_alpha(float(camera.fov_deg)))


def normalize_mix(mix: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Check a model mix and scale it to sum to one."""
    chosen = dict(DEFAULT_MIX if mix is None else mix)
    unknown = set(chosen) - set(MODEL_CODES)
    if unknown:
        raise ValueError(f"unknown camera models in camera_aug_mix: {sorted(unknown)}")
    weights = {k: float(chosen.get(k, 0.0)) for k in MODEL_CODES}
    if any(w < 0.0 for w in weights.values()):
        raise ValueError(f"camera_aug_mix must not be negative: {weights}")
    total = sum(weights.values())
    if total <= 0.0:
        raise ValueError("camera_aug_mix must have a positive total weight")
    return {k: w / total for k, w in weights.items()}


def sample_camera(rng: np.random.Generator, mix: Optional[Dict[str, float]] = None) -> AugCamera:
    """Draw one camera: a model from `mix`, then its field of view and alpha.

    Ranges: pinhole 60 to 120 degrees, equidistant fisheye 120 to 195, UCM 120 to 195 with alpha
    0.3 to 0.9. A UCM draw whose alpha cannot reach its field of view is redrawn, which is why the
    alpha of a wide UCM camera is biased towards 0.5 and never close to 0.9.
    """
    weights = normalize_mix(mix)
    names = list(MODEL_CODES)
    model = str(rng.choice(names, p=[weights[n] for n in names]))

    if model == "pinhole":
        return AugCamera("pinhole", float(rng.uniform(60.0, 120.0)), None)
    if model == "fisheye":
        return AugCamera("fisheye", float(rng.uniform(120.0, 195.0)), None)

    for _ in range(1000):
        fov = float(rng.uniform(120.0, 195.0))
        alpha = float(rng.uniform(0.3, 0.9))
        if math.radians(fov) / 2.0 < ucm_theta_max(alpha):
            return AugCamera("ucm", fov, alpha)
    raise RuntimeError("could not draw a valid UCM camera")


def remap_views(
    frames: Sequence[np.ndarray],
    camera: AugCamera,
    size: int,
    supersample: int = DEFAULT_SUPERSAMPLE,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Put every view of one sample through the same camera.

    `frames` are `lcam_fish` images, 1000 x 1000 x 3 uint8. Returns

    * images [v, 3, size, size] float32 in [0, 1],
    * a valid mask [v, 1, size, size] float32, the same for every view,
    * fxfycxcy [4] in pixels of the output, principal point at size / 2 (this repository's
      convention, half a pixel away from what `remap_fisheye` reports),
    * the true alpha of the camera.
    """
    if not frames:
        raise ValueError("remap_views needs at least one frame")
    map_x, map_y, mask, f = _remap_tables(
        camera.model, float(camera.fov_deg), int(size), int(supersample), camera.alpha
    )

    images = np.empty((len(frames), size, size, 3), dtype=np.float32)
    small_mask: Optional[np.ndarray] = None
    for i, frame in enumerate(frames):
        if frame.shape[:2] != (FISH_SIZE, FISH_SIZE):
            raise ValueError(f"expected a {FISH_SIZE} x {FISH_SIZE} lcam_fish image, got {frame.shape[:2]}")
        out, small_mask = _apply_remap(frame, map_x, map_y, mask, int(size), int(supersample))
        images[i] = out

    image_t = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous() / 255.0
    mask_t = torch.from_numpy(np.ascontiguousarray(small_mask, dtype=np.float32))[None, None]
    mask_t = mask_t.expand(len(frames), -1, -1, -1).contiguous()
    centre = size / 2.0
    intr = torch.tensor([f, f, centre, centre], dtype=torch.float32)
    return image_t, mask_t, intr, true_alpha(camera)
