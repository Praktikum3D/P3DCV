"""Download and convert the views of the DL3DV benchmark that the evaluation reads.

The E-RayZer evaluation protocol is deterministic: with FRAME_DIST=96 and 10
views it always reads frames 0, 10, 21, 32, 42, 53, 64, 74, 85 and 96 of each
scene. This script downloads only those images (about 9 MB per scene instead
of 310 MB), undistorts them with the upstream converter code and writes the
same layout as scripts/convert_dl3dv_opencv_list.py:

    <output>/manifests/test_full.txt
    <output>/scenes/<scene id>/scene.json
    <output>/scenes/<scene id>/images_undistort/frame_XXXXX.png

scene.json lists the first FRAME_DIST + 1 frames with their poses so that the
frame indices match the full dataset. Only the selected frames have an image.

DL3DV-Benchmark is gated on Hugging Face. Accept its terms with your account
and pass a token through --token-file or HF_TOKEN.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_dl3dv_opencv_list import normalize_transforms_frame  # noqa: E402
from convert_dl3dv_opencv_list import read_scene_ids  # noqa: E402
from convert_dl3dv_opencv_list import transforms_camera_params  # noqa: E402

REPO_ID = "DL3DV/DL3DV-Benchmark"


def selected_indices(frame_dist: int, num_views: int) -> list[int]:
    # Same expression as DL3DVDataset._select_two_frame in evaluation mode.
    return np.linspace(0, frame_dist, num_views, dtype=int).tolist()


def fetch(filename: str, raw_dir: Path, token: str | None) -> Path:
    last_error: Exception | None = None
    for _ in range(5):
        try:
            return Path(
                hf_hub_download(
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    filename=filename,
                    local_dir=raw_dir,
                    token=token,
                )
            )
        except Exception as error:  # network errors are retried
            last_error = error
    raise RuntimeError(f"Could not download {REPO_ID}/{filename}: {last_error}")


def prepare_scene(
    scene_id: str,
    *,
    raw_dir: Path,
    output_dir: Path,
    indices: list[int],
    token: str | None,
    workers: int,
) -> Path:
    base = f"{scene_id}/nerfstudio"
    camera_path = fetch(f"{base}/transforms.json", raw_dir, token)
    camera_json = json.loads(camera_path.read_text(encoding="utf-8"))
    frames = camera_json["frames"]
    if len(frames) <= indices[-1]:
        raise ValueError(f"{scene_id} has only {len(frames)} frames")
    frames = frames[: indices[-1] + 1]

    names = {index: Path(frames[index]["file_path"]).name for index in indices}
    with ThreadPoolExecutor(workers) as pool:
        paths = list(
            pool.map(
                lambda name: fetch(f"{base}/images_4/{name}", raw_dir, token),
                names.values(),
            )
        )
    sources = dict(zip(names, paths))

    camera_matrix, distortion, new_matrix, width, height = transforms_camera_params(
        camera_json=camera_json, image_path=sources[indices[0]]
    )
    scene_dir = output_dir / "scenes" / scene_id
    (scene_dir / "images_undistort").mkdir(parents=True, exist_ok=True)
    out_frames = []
    for index, frame in enumerate(frames):
        rel_path = f"images_undistort/{Path(frame['file_path']).name}"
        if index in sources:
            image = cv2.imread(sources[index].as_posix(), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (height, width):
                raise ValueError(f"Unreadable or mis-sized image {sources[index]}")
            undistorted = cv2.undistort(image, camera_matrix, distortion, None, new_matrix)
            cv2.imwrite((scene_dir / rel_path).as_posix(), undistorted)
        out_frames.append(
            normalize_transforms_frame(
                frame, rel_path=rel_path, camera_matrix=new_matrix, width=width, height=height
            )
        )

    scene_path = scene_dir / "scene.json"
    scene = {
        "scene_name": scene_id,
        "frames": out_frames,
        "frames_with_image": indices,
        "source_camera_file": camera_path.name,
    }
    scene_path.write_text(json.dumps(scene), encoding="utf-8")
    return scene_path


def write_archive(output_dir: Path, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    partial = archive.with_suffix(archive.suffix + ".part")
    # PNG is already compressed, so the files are only stored.
    with zipfile.ZipFile(partial, "w", zipfile.ZIP_STORED) as handle:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(output_dir).as_posix())
    os.replace(partial, archive)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scene-id-list", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path, help="scratch directory")
    parser.add_argument("--archive", type=Path, help="zip file to write the result to")
    parser.add_argument("--split-name", default="test_full")
    parser.add_argument("--frame-dist", type=int, default=96)
    parser.add_argument("--num-views", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="only the first N scenes")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--token-file", type=Path)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if args.token_file is not None:
        token = args.token_file.read_text(encoding="utf-8").strip()
    if not token:
        raise SystemExit("No Hugging Face token: pass --token-file or set HF_TOKEN")

    scene_ids = read_scene_ids(args.scene_id_list)
    if args.limit > 0:
        scene_ids = scene_ids[: args.limit]
    indices = selected_indices(args.frame_dist, args.num_views)
    raw_dir = args.work_dir.resolve() / "raw"
    output_dir = args.work_dir.resolve() / "release"
    print(f"scenes      {len(scene_ids)}")
    print(f"frames      {indices}")

    manifest = []
    for number, scene_id in enumerate(scene_ids, start=1):
        scene_path = prepare_scene(
            scene_id,
            raw_dir=raw_dir,
            output_dir=output_dir,
            indices=indices,
            token=token,
            workers=args.workers,
        )
        manifest.append(os.path.relpath(scene_path, output_dir / "manifests"))
        print(f"[{number}/{len(scene_ids)}] {scene_id}", flush=True)

    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    (manifests_dir / f"{args.split_name}.txt").write_text(
        "\n".join(manifest) + "\n", encoding="utf-8"
    )
    (manifests_dir / f"{args.split_name}_scene_ids.txt").write_text(
        "\n".join(scene_ids) + "\n", encoding="utf-8"
    )
    metadata = {
        "dataset": "dl3dv",
        "source": REPO_ID,
        "split_name": args.split_name,
        "sequences": len(scene_ids),
        "frame_dist": args.frame_dist,
        "frames_with_image": indices,
        "manifest": f"manifests/{args.split_name}.txt",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    files = [path for path in output_dir.rglob("*") if path.is_file()]
    size = sum(path.stat().st_size for path in files)
    print(f"converted   {len(scene_ids)} scenes, {len(files)} files, {size / 1e9:.2f} GB")
    if args.archive is not None:
        write_archive(output_dir, args.archive.resolve())
        print(f"archive     {args.archive} ({args.archive.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
