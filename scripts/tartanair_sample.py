"""Inspect a TartanAir V2 image archive without unpacking it.

Prints the trajectories, frame counts, image size and pose range of the
archive and writes a contact sheet with evenly spaced frames of one
trajectory. Images are read directly from the zip file, which is how the
data loader will have to read them too (the cluster limits the number of
files, so archives stay packed).
"""

import argparse
import io
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def index_archive(archive: zipfile.ZipFile) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Return image paths and the pose file per trajectory, e.g. 'P003'."""
    frames: dict[str, list[str]] = defaultdict(list)
    poses: dict[str, str] = {}
    for name in archive.namelist():
        parts = Path(name).parts
        trajectory = next((p for p in parts if p.startswith("P") and p[1:].isdigit()), None)
        if trajectory is None:
            continue
        if name.endswith(".png"):
            frames[trajectory].append(name)
        elif Path(name).name.startswith("pose_") and name.endswith(".txt"):
            poses[trajectory] = name
    return {t: sorted(f) for t, f in sorted(frames.items())}, poses


def load_image(archive: zipfile.ZipFile, name: str) -> Image.Image:
    return Image.open(io.BytesIO(archive.read(name))).convert("RGB")


def load_poses(archive: zipfile.ZipFile, name: str) -> np.ndarray:
    """One row per frame: tx ty tz qx qy qz qw, in the NED frame."""
    return np.loadtxt(io.BytesIO(archive.read(name))).reshape(-1, 7)


def contact_sheet(images: list[Image.Image], tile: int) -> Image.Image:
    sheet = Image.new("RGB", (tile * len(images), tile))
    for i, image in enumerate(images):
        sheet.paste(image.resize((tile, tile), Image.BILINEAR), (i * tile, 0))
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="path to an image_*.zip")
    parser.add_argument("--out", type=Path, required=True, help="where to write the contact sheet")
    parser.add_argument("--trajectory", default=None, help="e.g. P003; default: the longest one")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--tile", type=int, default=256, help="tile size in pixels")
    args = parser.parse_args()

    with zipfile.ZipFile(args.archive) as archive:
        frames, poses = index_archive(archive)
        if not frames:
            raise SystemExit(f"no images found in {args.archive}")

        print(f"archive     {args.archive} ({args.archive.stat().st_size / 1024**3:.2f} GB)")
        for trajectory, names in frames.items():
            line = f"trajectory  {trajectory}: {len(names)} frames"
            if trajectory in poses:
                p = load_poses(archive, poses[trajectory])
                extent = np.ptp(p[:, :3], axis=0)
                line += f", {len(p)} poses, extent {extent[0]:.1f} x {extent[1]:.1f} x {extent[2]:.1f} m"
            print(line)

        trajectory = args.trajectory or max(frames, key=lambda t: len(frames[t]))
        names = frames[trajectory]
        picks = np.linspace(0, len(names) - 1, min(args.num_frames, len(names))).round().astype(int)
        images = [load_image(archive, names[i]) for i in picks]

    print(f"image size  {images[0].width} x {images[0].height}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    contact_sheet(images, args.tile).save(args.out)
    print(f"wrote       {args.out} ({trajectory}, frames {', '.join(str(i) for i in picks)})")


if __name__ == "__main__":
    main()
