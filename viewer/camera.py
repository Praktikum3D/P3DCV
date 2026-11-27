"""Camera maths for the viewer: rays, focal lengths, projection and camera frusta.

The viewer renders with the very same ray maths the model trains with. `camera_rays`
(`erayzer_core/model/utils_gaussian.py`) is the single source of it: `cam_info_to_plucker`
builds the plucker embeddings from it, `render_opencv_cam_geer` hands its rays to 3DGEER,
and `build_rays` below is a plain pass through so that the viewer provably walks the same
code path. Everything here is pure torch and runs on the CPU.

Image coordinates follow `camera_rays`: the centre of pixel (i, j) sits at the integer
coordinate (j, i), because `camera_rays` offsets the pixel index by half a pixel itself.
`project_points` therefore returns pixel indices, not the half pixel shifted variant.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import torch

from erayzer_core.model.utils_gaussian import camera_rays, camera_theta

CAMERA_MODELS = ("pinhole", "fisheye", "ucm")

# A pinhole blows up at 90 degrees and a fisheye wraps onto itself at 180 degrees off axis,
# so both keep a little distance from their limit.
PINHOLE_MAX_THETA = math.radians(85.0)
FISHEYE_MAX_THETA = math.radians(179.5)
UCM_THETA_MARGIN = math.radians(0.5)


def build_rays(c2w: torch.Tensor, fxfycxcy: torch.Tensor, h: int, w: int, camera_model: str = "pinhole"):
    """Ray origins and directions in world space, both [b, h, w, 3]. Pass through to `camera_rays`.

    c2w [b, 4, 4], fxfycxcy [b, 4] in pixels, for UCM [b, 5] with alpha last. This wrapper exists so
    that the ray identity test exercises the function the viewer itself calls.
    """
    return camera_rays(c2w, fxfycxcy, h, w, camera_model)


def theta_max(camera_model: str, alpha: Optional[float] = None) -> float:
    """Largest angle off the optical axis the camera model can still image, in radians.

    For UCM the image radius rho(theta) = sin theta / (alpha + (1 - alpha) cos theta) stops being
    usable at arccos(-min(alpha, 1-alpha) / max(alpha, 1-alpha)). Above alpha = 0.5 that is where
    rho peaks and starts shrinking again, below it is where the denominator crosses zero and rho
    runs to infinity. Both give the same formula, and the peak radius 1/sqrt(2 alpha - 1) is exactly
    the bound `pixel_angles` tests with `1 - (2 alpha - 1) rho^2 >= 0`.
    """
    if camera_model == "pinhole":
        return PINHOLE_MAX_THETA
    if camera_model == "fisheye":
        return FISHEYE_MAX_THETA
    if camera_model == "ucm":
        if alpha is None:
            raise ValueError("UCM needs alpha")
        a = float(min(max(alpha, 1e-6), 1.0 - 1e-6))
        return math.acos(-min(a, 1.0 - a) / max(a, 1.0 - a))
    raise ValueError(f"unknown camera model {camera_model!r}")


def _rho_of_theta(camera_model: str, theta: float, alpha: Optional[float]) -> float:
    """Image radius divided by the focal length for the angle theta, the inverse of `camera_theta`."""
    if camera_model == "pinhole":
        return math.tan(theta)
    if camera_model == "fisheye":
        return theta
    if camera_model == "ucm":
        a = float(alpha)
        return math.sin(theta) / (a + (1.0 - a) * math.cos(theta))
    raise ValueError(f"unknown camera model {camera_model!r}")


def fov_to_fxfycxcy(
    camera_model: str,
    fov_deg: float,
    alpha: Optional[float],
    w: int,
    h: int,
    device=None,
    dtype=torch.float32,
) -> Tuple[torch.Tensor, float]:
    """Intrinsics in pixels for a field of view across the image width, plus the value actually used.

    Returns fxfycxcy [1, 4], for UCM [1, 5] with alpha last. The half angle is clamped to the range
    the camera model can represent, so the caller can report back what it got instead of what it asked.
    """
    if camera_model not in CAMERA_MODELS:
        raise ValueError(f"unknown camera model {camera_model!r}")
    limit = theta_max(camera_model, alpha)
    if camera_model == "ucm":
        limit -= UCM_THETA_MARGIN
    half = min(max(math.radians(float(fov_deg)) / 2.0, math.radians(1.0)), limit)
    f = (w / 2.0) / _rho_of_theta(camera_model, half, alpha)
    values = [f, f, w / 2.0, h / 2.0]
    if camera_model == "ucm":
        values.append(float(alpha))
    return torch.tensor([values], dtype=dtype, device=device), math.degrees(2.0 * half)


def projection_limit(camera_model: str, alpha: Optional[float] = None) -> float:
    """Largest angle off the optical axis that still has a finite image radius, in radians.

    This is the limit `project_points` tests, and it is wider than `theta_max`: `theta_max` keeps a
    safety margin because a focal length is derived from it, while a point merely has to land
    somewhere on the image plane. Only UCM shares the same hard limit for both.
    """
    if camera_model == "pinhole":
        return math.pi / 2.0
    if camera_model == "fisheye":
        return math.pi
    return theta_max(camera_model, alpha)


def project_points(
    points: torch.Tensor,
    c2w: torch.Tensor,
    fxfycxcy: torch.Tensor,
    camera_model: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """World points [n, 3] to pixel coordinates [n, 2] and a validity mask [n], the inverse of `camera_rays`.

    c2w [4, 4], fxfycxcy [4] in pixels, for UCM [5] with alpha last. A point is invalid when its angle
    to the optical axis is outside the range of the camera model, so a pinhole drops everything at or
    behind 90 degrees while a fisheye keeps it.
    """
    with torch.no_grad():
        points = points.to(dtype=c2w.dtype, device=c2w.device)
        cam = (points - c2w[:3, 3]) @ c2w[:3, :3]  # R^T (p - t), the camera frame of `camera_rays`
        x, y, z = cam[:, 0], cam[:, 1], cam[:, 2]
        r = torch.sqrt(x * x + y * y + 1e-24)
        theta = torch.atan2(r, z)
        alpha = float(fxfycxcy[4]) if camera_model == "ucm" else None
        limit = projection_limit(camera_model, alpha)
        valid = theta < limit
        safe = torch.clamp(theta, max=limit * 0.999)
        if camera_model == "pinhole":
            rho = torch.tan(safe)
        elif camera_model == "fisheye":
            rho = safe
        else:
            rho = torch.sin(safe) / (alpha + (1.0 - alpha) * torch.cos(safe))
        scale = rho / r
        u = fxfycxcy[2] + fxfycxcy[0] * x * scale - 0.5
        v = fxfycxcy[3] + fxfycxcy[1] * y * scale - 0.5
        uv = torch.stack([u, v], dim=-1)
        valid = valid & torch.isfinite(uv).all(dim=-1)
        return uv, valid


def _border_grid_intrinsics(fxfycxcy: torch.Tensor, h: int, w: int, k: int) -> torch.Tensor:
    """Intrinsics that make a k x k pixel grid land exactly on the image border of an h x w image.

    `camera_rays` samples pixel centres, so a plain k x k grid would stop half a pixel short of the
    border. Scaling the focal length by (k - 1) / w and shifting the principal point by the same
    factor maps grid index i onto image coordinate i * w / (k - 1), which hits both borders exactly.
    """
    scaled = fxfycxcy.clone()
    sx, sy = (k - 1) / w, (k - 1) / h
    scaled[:, 0] = fxfycxcy[:, 0] * sx
    scaled[:, 1] = fxfycxcy[:, 1] * sy
    scaled[:, 2] = 0.5 + fxfycxcy[:, 2] * sx
    scaled[:, 3] = 0.5 + fxfycxcy[:, 3] * sy
    return scaled


def frustum_segments(
    c2w: torch.Tensor,
    fxfycxcy: torch.Tensor,
    h: int,
    w: int,
    camera_model: str,
    depth: float,
    grid: int = 9,
) -> List[np.ndarray]:
    """World space polylines that draw the view cone of one camera, each [n, 3].

    The rays come from `camera_rays` on a small grid stretched onto the image border, so the cone is
    the one the camera model really sees. Returned are the four edges from the apex to the image
    corners and the ring along the image border. Under a fisheye or UCM viewer camera the projected
    lines come out curved, which is the point of drawing them through the camera model.
    """
    c2w = c2w[None] if c2w.dim() == 2 else c2w
    fxfycxcy = fxfycxcy[None] if fxfycxcy.dim() == 1 else fxfycxcy
    with torch.no_grad():
        scaled = _border_grid_intrinsics(fxfycxcy, h, w, grid)
        ray_o, ray_d = camera_rays(c2w, scaled, grid, grid, camera_model)
        apex = ray_o[0, 0, 0].cpu().numpy()
        far = (ray_o[0] + ray_d[0] * depth).cpu().numpy()  # [grid, grid, 3]

    corners = [far[0, 0], far[0, -1], far[-1, -1], far[-1, 0]]
    segments = [np.stack([apex + (corner - apex) * t for t in np.linspace(0.0, 1.0, 9)]) for corner in corners]
    ring = np.concatenate([far[0, :], far[1:, -1], far[-1, -2::-1], far[-2:0:-1, 0]], axis=0)
    segments.append(np.concatenate([ring, ring[:1]], axis=0))
    return segments


def estimated_camera_info(fxfycxcy_norm: torch.Tensor, camera_model: str) -> List[dict]:
    """Field of view, focal length and alpha of predicted cameras, one dict per view.

    fxfycxcy_norm [v, 4] normalized by the image size as the model predicts it, for UCM [v, 5]. The
    angle formula is the one `train.py` logs as `camera/fov_deg`.
    """
    with torch.no_grad():
        fx = fxfycxcy_norm[:, 0].detach().float().cpu()
        alpha = fxfycxcy_norm[:, 4].detach().float().cpu() if camera_model == "ucm" else None
        fov = 2.0 * camera_theta(0.5 / fx.clamp_min(1e-6), camera_model, alpha)
    out = []
    for i in range(fx.shape[0]):
        entry = {"fx_norm": float(fx[i]), "fov_deg": math.degrees(float(fov[i]))}
        if alpha is not None:
            entry["alpha"] = float(alpha[i])
        out.append(entry)
    return out
