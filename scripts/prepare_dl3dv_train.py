"""Turn the unpacked DL3DV benchmark archive into a training set for train.py.

The archive (see docs/datasets/dl3dv.md) lists 97 frames per scene, but only
the 10 frames named in `frames_with_image` have an image. The DL3DV loader
would pick frames without an image, so this script writes a `scene_train.json`
next to every `scene.json` that keeps only the frames with an image, and
writes the manifests

    manifests/train.txt     scenes used for training
    manifests/holdout.txt   scenes kept out of training, for evaluation

into the unpacked folder. Run it on a copy on the local disk of the node,
never on the shared data folder. The holdout split is the same on every run
for the same --holdout and --seed.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import List


def read_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip() and not line.startswith("#")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", help="Unpacked benchmark archive (contains manifests/ and scenes/)")
    parser.add_argument("--holdout", type=int, default=20, help="Number of scenes kept out of training")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the holdout split")
    parser.add_argument("--holdout-list", help="Also write the holdout scene ids to this file")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    manifest_dir = os.path.join(root, "manifests")
    scene_paths = [
        os.path.normpath(os.path.join(manifest_dir, path))
        for path in read_list(os.path.join(manifest_dir, "test_full.txt"))
    ]

    scene_ids = []
    for scene_path in scene_paths:
        with open(scene_path, "r", encoding="utf-8") as handle:
            scene = json.load(handle)
        frames = scene["frames"]
        keep = scene.get("frames_with_image")
        if keep is None:
            keep = list(range(len(frames)))
        scene_dir = os.path.dirname(scene_path)
        kept = [frames[i] for i in keep]
        for frame in kept:
            image_path = os.path.join(scene_dir, frame["file_path"])
            if not os.path.isfile(image_path):
                raise FileNotFoundError(image_path)
        scene_id = str(scene.get("scene_name") or os.path.basename(scene_dir))
        train_scene = {
            "scene_name": scene_id,
            "frames": kept,
            "source_frame_indices": list(keep),
        }
        with open(os.path.join(scene_dir, "scene_train.json"), "w", encoding="utf-8") as handle:
            json.dump(train_scene, handle)
        scene_ids.append(scene_id)

    if not 0 <= args.holdout < len(scene_ids):
        raise ValueError(f"--holdout must be in [0, {len(scene_ids) - 1}], got {args.holdout}")
    holdout = set(random.Random(args.seed).sample(sorted(scene_ids), args.holdout))

    def write_manifest(name: str, ids: List[str]) -> None:
        with open(os.path.join(manifest_dir, name), "w", encoding="utf-8") as handle:
            for scene_id in ids:
                handle.write(f"../scenes/{scene_id}/scene_train.json\n")

    train_ids = [s for s in scene_ids if s not in holdout]
    holdout_ids = [s for s in scene_ids if s in holdout]
    write_manifest("train.txt", train_ids)
    write_manifest("holdout.txt", holdout_ids)
    if args.holdout_list:
        os.makedirs(os.path.dirname(os.path.abspath(args.holdout_list)), exist_ok=True)
        with open(args.holdout_list, "w", encoding="utf-8") as handle:
            handle.write("".join(f"{s}\n" for s in holdout_ids))

    print(f"Prepared {len(train_ids)} training and {len(holdout_ids)} holdout scenes in {root}")


if __name__ == "__main__":
    main()
