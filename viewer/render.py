"""One rendered frame of the viewer: 3DGEER through the camera model, the mask and the frusta overlay.

The camera mapping below is the same one `Renderer.forward` uses for its geer branch, so a frame the
viewer shows is a frame the training would have produced for that camera. The only difference is the
mask: the viewer always takes the exact range of the camera model, because its default tile mode
renders out to the image border.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from erayzer_core.model.gaussians_renderer import render_opencv_cam_geer
from erayzer_core.model.utils_gaussian import render_mask, ucm_radial_coeffs

from .camera import fov_to_fxfycxcy, frustum_segments, project_points
from .scene import Scene

VIEWS = ("rgb", "alpha", "depth")
FRUSTUM_PALETTE = [
    (255, 96, 96), (96, 220, 96), (96, 160, 255), (255, 210, 80),
    (220, 110, 255), (80, 225, 225), (255, 150, 60), (190, 190, 190),
]


def geer_camera_arguments(camera_model: str, fxfycxcy: torch.Tensor, h: int, w: int) -> dict:
    """How 3DGEER is told about a camera, exactly as the geer branch of `Renderer.forward` tells it.

    3DGEER has no UCM, so a Kannala Brandt fisheye fitted to it stands in for the tile assignment,
    while the real rays stay UCM through `ray_camera_model`.
    """
    if camera_model == "pinhole":
        return {"camera_model": "pinhole", "radial_coeffs": None, "ray_camera_model": "pinhole"}
    if camera_model == "fisheye":
        zeros = torch.zeros(fxfycxcy.shape[0], 4, device=fxfycxcy.device)
        return {"camera_model": "fisheye", "radial_coeffs": zeros, "ray_camera_model": "fisheye"}
    if camera_model == "ucm":
        return {
            "camera_model": "fisheye",
            "radial_coeffs": ucm_radial_coeffs(fxfycxcy.detach(), h, w),
            "ray_camera_model": "ucm",
        }
    raise ValueError(f"unknown camera model {camera_model!r}")


def render_view(
    scene: Scene,
    c2w: torch.Tensor,
    camera_model: str,
    fov_deg: float,
    alpha: Optional[float],
    h: int,
    w: int,
    tile_mode: str = "projection",
    view: str = "rgb",
    frusta: bool = True,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """One frame as an RGB uint8 image [h, w, 3], plus what the renderer did.

    c2w [4, 4] in the OpenCV convention, `fov_deg` across the image width. `view` picks the colour
    image, the accumulated opacity or the expected depth.
    """
    if view not in VIEWS:
        raise ValueError(f"unknown view {view!r}")
    device = scene.pc.get_xyz.device
    fxfycxcy, fov_used = fov_to_fxfycxcy(camera_model, fov_deg, alpha, w, h, device=device)
    arguments = geer_camera_arguments(camera_model, fxfycxcy, h, w)

    started = time.perf_counter()
    with torch.no_grad():
        buffers = render_opencv_cam_geer(
            scene.pc,
            h,
            w,
            c2w[None].to(device).float(),
            fxfycxcy,
            scene.sh_degree,
            near_plane=scene.near_plane,
            geer_tile_mode=tile_mode,
            with_alpha=(view != "rgb"),
            render_mode="RGB+ED" if view == "depth" else "RGB",
            **arguments,
        )
        # The exact range of the camera model, because the projection tile mode draws to the border.
        mask = render_mask(fxfycxcy, h, w, camera_model, image_circle=False, exact=True)[0, 0]

        if view == "rgb":
            image = buffers["render"][0].permute(1, 2, 0)
        elif view == "alpha":
            image = buffers["alpha"][0, 0, :, :, None].expand(-1, -1, 3)
        else:
            image = _depth_to_color(buffers["depth"][0, 0], buffers["alpha"][0, 0])
        image = (image.clamp(0.0, 1.0) * mask[..., None]) * 255.0
        rgb = image.round().to(torch.uint8).cpu().numpy()
    if device.type == "cuda":
        torch.cuda.synchronize()
    render_ms = (time.perf_counter() - started) * 1000.0

    if frusta and scene.c2w_input is not None and scene.c2w_input.shape[0] > 0:
        rgb = draw_frusta(rgb, scene, c2w, fxfycxcy[0], camera_model)
    return rgb, {"render_ms": render_ms, "fov_used": fov_used, "width": w, "height": h}


def _depth_to_color(depth: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Expected depth to a grey image, scaled between the 2nd and 98th percentile of what was hit."""
    hit = depth[alpha > 0.1]
    if hit.numel() < 16:
        return torch.zeros(depth.shape + (3,), device=depth.device)
    low, high = torch.quantile(hit.float(), torch.tensor([0.02, 0.98], device=depth.device))
    normalized = ((depth - low) / (high - low).clamp_min(1e-6)).clamp(0.0, 1.0)
    return (1.0 - normalized)[..., None].expand(-1, -1, 3)


def draw_frusta(rgb: np.ndarray, scene: Scene, c2w: torch.Tensor, fxfycxcy: torch.Tensor, camera_model: str):
    """Draw the view cone of every input camera through the current viewer camera, in place.

    Under a fisheye or UCM viewer camera the straight edges of a cone project to curves, which is why
    every edge is sampled and drawn as a polyline instead of as a straight segment.
    """
    h, w = rgb.shape[:2]
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    margin = 2.0 * float(np.hypot(h, w))
    device = c2w.device
    input_h, input_w = scene.input_hw
    pixels = scene.fxfycxcy_input.clone().to(device)
    pixels[:, 0] *= input_w
    pixels[:, 1] *= input_h
    pixels[:, 2] *= input_w
    pixels[:, 3] *= input_h
    depth = 0.15 * scene.radius

    for index in range(scene.c2w_input.shape[0]):
        color = FRUSTUM_PALETTE[index % len(FRUSTUM_PALETTE)][::-1]  # cv2 wants BGR
        segments = frustum_segments(
            scene.c2w_input[index].to(device), pixels[index], input_h, input_w, scene.camera_model, depth
        )
        for segment in segments:
            uv, valid = project_points(torch.from_numpy(segment).to(device), c2w.to(device), fxfycxcy, camera_model)
            uv = uv.cpu().numpy()
            inside = valid.cpu().numpy() & (np.abs(uv[:, 0] + 0.5 - w / 2) < w / 2 + margin)
            inside &= np.abs(uv[:, 1] + 0.5 - h / 2) < h / 2 + margin
            for run in _runs(inside):
                if len(run) >= 2:
                    cv2.polylines(bgr, [np.round(uv[run]).astype(np.int32)], False, color, 1, cv2.LINE_AA)
        apex, apex_valid = project_points(
            scene.c2w_input[index, :3, 3][None].to(device), c2w.to(device), fxfycxcy, camera_model
        )
        if bool(apex_valid[0]):
            x, y = apex[0].cpu().numpy()
            if -margin < x < w + margin and -margin < y < h + margin:
                cv2.putText(bgr, str(index + 1), (int(x) + 4, int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
                            cv2.LINE_AA)
    return np.ascontiguousarray(bgr[:, :, ::-1])


def _runs(mask: np.ndarray) -> List[np.ndarray]:
    """Index runs of consecutive True entries, so a polyline breaks where it leaves the image."""
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(indices) != 1) + 1
    return np.split(indices, breaks)


def encode_jpeg(rgb: np.ndarray, quality: int = 85) -> bytes:
    ok, buffer = cv2.imencode(".jpg", rgb[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buffer.tobytes()


def encode_png(rgb: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", rgb[:, :, ::-1])
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buffer.tobytes()


def probe_depth(scene: Scene) -> bool:
    """Whether the depth view works for this build, tried once on a tiny frame.

    3DGEER renders any number of colour channels, but the expected depth of its eval3d path is not
    something the project has needed before, so the viewer asks instead of promising.
    """
    try:
        image, _ = render_view(scene, scene.c2w_input[0], "pinhole", 60.0, None, 16, 16, view="depth", frusta=False)
        return bool(np.isfinite(image).all())
    except Exception as error:  # any failure simply hides the depth view
        print(f"Depth view unavailable: {error}", flush=True)
        return False


class RenderStats:
    """Render times in milliseconds, kept per resolution so the log can name both of them."""

    def __init__(self, window: int = 400):
        self.window = window
        self.samples: Dict[int, deque] = {}
        self.total = 0

    def add(self, height: int, ms: float) -> None:
        self.samples.setdefault(height, deque(maxlen=self.window)).append(ms)
        self.total += 1

    def summary(self) -> Dict[str, Dict[str, float]]:
        out = {}
        for height, values in sorted(self.samples.items()):
            array = np.asarray(values, dtype=np.float64)
            out[str(height)] = {
                "count": int(array.size),
                "p50": float(np.percentile(array, 50)),
                "p90": float(np.percentile(array, 90)),
                "fps_p50": float(1000.0 / max(np.percentile(array, 50), 1e-6)),
            }
        return out

    def line(self) -> str:
        parts = [f"{height}p {values['p50']:.1f}/{values['p90']:.1f} ms" for height, values in self.summary().items()]
        return f"render times after {self.total} frames, p50/p90: " + ", ".join(parts)
