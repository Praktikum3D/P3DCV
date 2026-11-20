"""Resample TartanAir fisheye images to cameras with a smaller field of view.

`lcam_fish` is equidistant (r = f * theta) over 195 degrees at 1000 x 1000,
see docs/datasets/tartanair.md. Every camera that looks forward and covers at
most 195 degrees can be computed exactly from it, so one trajectory gives the
same footage for many cameras.

Example:
    python scripts/tartanair_remap.py --archive <image_lcam_fish.zip> \
        --trajectory P000 --frame 0 --camera pinhole:90 fisheye:150 --out remap
"""

from __future__ import annotations

import argparse
import io
import math
import zipfile
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

FISH_SIZE = 1000
FISH_FOV_DEG = 195.0
FISH_F = FISH_SIZE / math.radians(FISH_FOV_DEG)  # pixels per radian, about 293.8
FISH_C = (FISH_SIZE - 1) / 2.0  # OpenCV pixel centres


def camera_rays(model: str, fov_deg: float, size: int) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Unit rays of a square camera in OpenCV axes, its validity mask, f and c."""
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
    else:
        raise ValueError(f"unknown camera model {model!r}")
    return rays, valid, f, c


def fisheye_pixels(rays: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixel coordinates in `lcam_fish` for unit rays, plus a mask of rays it covers."""
    theta = np.arctan2(np.hypot(rays[..., 0], rays[..., 1]), rays[..., 2])
    phi = np.arctan2(rays[..., 1], rays[..., 0])
    r = FISH_F * theta
    inside = theta <= math.radians(FISH_FOV_DEG) / 2.0
    return FISH_C + r * np.cos(phi), FISH_C + r * np.sin(phi), inside


def remap_fisheye(
    image: np.ndarray, model: str, fov_deg: float, size: int, supersample: int = 4
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float, float]]:
    """Resample a `lcam_fish` image (H x W x 3, uint8) to another camera.

    Returns the image, a boolean mask of valid pixels and fx, fy, cx, cy in
    pixels of the output. Rendering at `supersample` times the size and
    shrinking with INTER_AREA avoids aliasing when going from 1000 to 256.
    """
    if image.shape[:2] != (FISH_SIZE, FISH_SIZE):
        raise ValueError(f"expected a {FISH_SIZE} x {FISH_SIZE} lcam_fish image, got {image.shape[:2]}")
    big = size * supersample
    rays, valid, f_big, _ = camera_rays(model, fov_deg, big)
    map_x, map_y, inside = fisheye_pixels(rays)
    out = cv2.remap(
        image,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    mask = valid & inside
    out[~mask] = 0
    if supersample > 1:
        out = cv2.resize(out, (size, size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA) > 0.999
    f = f_big / supersample
    c = (size - 1) / 2.0
    return out, mask, (f, f, c, c)


def read_frame(archive: Path, trajectory: str, frame: int) -> np.ndarray:
    with zipfile.ZipFile(archive) as zf:
        names = sorted(
            n for n in zf.namelist()
            if f"/{trajectory}/image_lcam_fish/" in f"/{n}" and n.endswith(".png")
        )
        if not names:
            raise FileNotFoundError(f"no lcam_fish images of {trajectory} in {archive}")
        return np.asarray(Image.open(io.BytesIO(zf.read(names[frame]))).convert("RGB"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--archive", type=Path, required=True, help="an image_lcam_fish.zip")
    parser.add_argument("--trajectory", default="P000")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument(
        "--camera", nargs="+", default=["pinhole:60", "pinhole:90", "pinhole:120", "fisheye:150", "fisheye:195"],
        help="model:fov_degrees, model is pinhole or fisheye",
    )
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    image = read_frame(args.archive, args.trajectory, args.frame)
    args.out.mkdir(parents=True, exist_ok=True)

    # Self check: remapping to the camera of the input must give the input back.
    same, _, _ = remap_fisheye(image, "fisheye", FISH_FOV_DEG, FISH_SIZE, supersample=1)
    err = np.abs(same.astype(np.int16) - image.astype(np.int16)).max()
    print(f"identity check  max abs difference {err} (expected 0)")

    tiles = []
    for spec in args.camera:
        model, fov = spec.split(":")
        out, mask, (fx, fy, cx, cy) = remap_fisheye(image, model, float(fov), args.size)
        name = f"{model}_{float(fov):g}"
        Image.fromarray(out).save(args.out / f"{name}.png")
        print(f"{name:14s} fx = fy = {fx:7.2f}  cx = cy = {cx:6.1f}  valid pixels {mask.mean():.3f}")
        tiles.append(out)
    Image.fromarray(np.concatenate(tiles, axis=1)).save(args.out / "strip.png")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
