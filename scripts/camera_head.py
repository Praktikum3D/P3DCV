"""Show what camera E-RayZer assumes for an image sequence.

The model always assumes a pinhole camera. `canonical_k_head` predicts one
focal length f per scene (fx = fy), cx and cy are fixed at the image centre,
and `cam_info_to_plucker` turns every pixel into the ray
(x, y, 1) = ((u - cx) / f, (v - cy) / f, 1). This script runs the pretrained
model on 10 frames of a TartanAir trajectory, prints the predicted f and the
field of view that follows from it, and measures the corner rays that
`cam_info_to_plucker` produces (diagonal field of view), next to the true
camera.

Example:
    python scripts/camera_head.py \
        --archive <Env>/Data_hard/image_lcam_front.zip <Env>/Data_hard/image_lcam_fish.zip
"""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch
from easydict import EasyDict as edict

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app_core.engine import ERayZerEngine  # noqa: E402
from erayzer_core.model.erayzer import cam_info_to_plucker  # noqa: E402

TRUE_FOV_DEG = {"lcam_front": 90.0, "lcam_fish": 195.0}


def frames_of(archive: Path, trajectory: str, count: int, out_dir: Path) -> list[str]:
    with zipfile.ZipFile(archive) as zf:
        names = sorted(n for n in zf.namelist() if f"/{trajectory}/" in f"/{n}" and n.endswith(".png"))
        if len(names) < count:
            raise ValueError(f"{trajectory} in {archive} has only {len(names)} frames")
        picks = [names[round(i)] for i in np.linspace(0, len(names) - 1, count)]
        return [zf.extract(name, out_dir) for name in picks]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--archive", type=Path, nargs="+", required=True, help="image_<camera>.zip files")
    parser.add_argument("--trajectory", default="P000")
    parser.add_argument("--ckpt", default="/mnt/projects/p3dcv/anycam/checkpoints/pretrained/erayzer_multi.pt")
    parser.add_argument("--config", default=str(REPO_ROOT / "config" / "erayzer_inference.yaml"))
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        engine = ERayZerEngine(args.config, args.ckpt, "auto", tmp)
        size = engine.image_size
        rows = []
        for archive in args.archive:
            camera = archive.stem.removeprefix("image_")
            files = frames_of(archive, args.trajectory, engine.num_views, Path(tmp) / camera)
            batch = engine._move_to_device(engine._prepare_batch(files))
            with torch.no_grad(), torch.autocast(
                engine.device.type, dtype=engine.amp_dtype, enabled=engine.amp_enabled
            ):
                result = engine.model(batch)

            # fxfycxcy is normalised by the image size, so 0.5 is the centre.
            fxfycxcy = result.fxfycxcy.float()[: batch["num_real_views"]]
            f = float(fxfycxcy[0, 0]) * size
            cx, cy = float(fxfycxcy[0, 2]) * size, float(fxfycxcy[0, 3]) * size
            fov = math.degrees(2 * math.atan(size / (2 * f)))

            # The rays the model builds from this f, in camera coordinates.
            rays = cam_info_to_plucker(
                torch.eye(4, device=fxfycxcy.device)[None],
                fxfycxcy[:1],
                edict(height=size, width=size),
                normalized=True,
                return_moment=False,
            )[0, 3:6].reshape(3, -1)
            widest = math.degrees(float(torch.arccos(rays[2].clamp(-1, 1)).max()))
            rows.append((camera, f, cx, cy, fov, 2 * widest, TRUE_FOV_DEG.get(camera)))

    print(f"\n{args.trajectory}, 10 views at {size} x {size}, {Path(args.ckpt).name}")
    print(f"{'camera':12s} {'f [px]':>8s} {'cx':>6s} {'cy':>6s} {'FOV from f':>11s} {'diagonal FOV':>13s} {'true FOV':>9s}")
    for camera, f, cx, cy, fov, widest, true in rows:
        true_text = f"{true:.0f}" if true else "?"
        print(f"{camera:12s} {f:8.1f} {cx:6.1f} {cy:6.1f} {fov:10.1f}° {widest:12.1f}° {true_text:>8s}°")


if __name__ == "__main__":
    main()
