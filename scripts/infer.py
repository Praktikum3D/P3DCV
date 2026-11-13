"""Run a pretrained E-RayZer checkpoint on a handful of views.

The views come either from a folder of images or from one trajectory of a
TartanAir archive. Writes predicted views, a Gaussian point cloud and a video
to a new time-stamped folder below --out and prints the estimated camera
intrinsics.

    python scripts/infer.py --images /path/to/folder --ckpt erayzer_multi.pt --out runs/inference
    python scripts/infer.py --archive image_lcam_fish.zip --trajectory P003 --ckpt ... --out ...
"""

import argparse
import sys
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app_core.engine import ERayZerEngine  # noqa: E402

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def evenly_spaced(items: list, count: int) -> list:
    if len(items) <= count:
        return items
    step = (len(items) - 1) / (count - 1)
    return [items[round(i * step)] for i in range(count)]


def frames_from_folder(folder: Path, count: int) -> list[str]:
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise SystemExit(f"no images found in {folder}")
    return [str(p) for p in evenly_spaced(files, count)]


def frames_from_archive(archive_path: Path, trajectory: str, count: int, stride: int, target: Path) -> list[str]:
    """Extract evenly spaced frames of one trajectory into `target`."""
    with zipfile.ZipFile(archive_path) as archive:
        names = sorted(n for n in archive.namelist() if f"/{trajectory}/" in n and n.endswith(".png"))
        if not names:
            raise SystemExit(f"trajectory {trajectory} not found in {archive_path}")
        picks = names[::stride][:count] if stride > 0 else evenly_spaced(names, count)
        files = []
        for name in picks:
            path = target / Path(name).name
            path.write_bytes(archive.read(name))
            files.append(str(path))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--images", type=Path, help="folder with images of one scene")
    source.add_argument("--archive", type=Path, help="TartanAir image_*.zip")
    parser.add_argument("--trajectory", default="P000", help="trajectory inside --archive")
    parser.add_argument("--stride", type=int, default=0,
                        help="take every n-th frame of the trajectory; 0 spreads the views over all of it")
    parser.add_argument("--num-views", type=int, default=10, help="the released model expects 10")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config" / "erayzer_inference.yaml")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        if args.images is not None:
            files = frames_from_folder(args.images, args.num_views)
        else:
            files = frames_from_archive(args.archive, args.trajectory, args.num_views, args.stride, Path(tmp))
        print(f"views       {len(files)}")
        for path in files:
            print(f"            {Path(path).name}")

        engine = ERayZerEngine(str(args.config), str(args.ckpt), args.device, str(args.out))
        gallery, archive, _, glb, video = engine.run(files)

    print(f"rendered    {len(gallery)} views")
    print(f"point cloud {glb}")
    print(f"video       {video}")
    print(f"archive     {archive}")


if __name__ == "__main__":
    main()
