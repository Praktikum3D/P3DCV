"""Smoke test for the TartanAir V2 dataset and PyTorch DataLoader."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from PIL import Image
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from training.datasets import build_dataset  # noqa: E402


def _to_edict(value):
    if isinstance(value, dict):
        return edict({k: _to_edict(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_edict(v) for v in value]
    return value


def _save_batch_preview(images: torch.Tensor, out_path: Path) -> None:
    # images: [V, 3, H, W] in [0, 1]
    tiles = []
    for view in images:
        arr = (view.clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        tiles.append(arr)
    strip = np.concatenate(tiles, axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(strip).save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config" / "erayzer_train_tartanair.yaml",
    )
    parser.add_argument("--data-path", required=True, help="zip archive, directory, or manifest .txt")
    parser.add_argument("--camera", default=None, help="optional camera filter, e.g. lcam_fish or lcam_front")
    parser.add_argument("--split", choices=("train", "eval"), default="train")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--preview-out", type=Path, default=None, help="optional PNG path for one sampled sequence")
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as handle:
        cfg = _to_edict(yaml.safe_load(handle))

    cfg.dataset[f"{args.split}_path"] = str(Path(args.data_path).expanduser().resolve())
    if args.camera is not None:
        cfg.dataset.camera_name = args.camera
    cfg.data_repeat = 2

    dataset = build_dataset(cfg, split=args.split)
    print(f"trajectories : {len(dataset.trajectories)} (dataset len={len(dataset)})")
    for rec in dataset.trajectories[:5]:
        print(f"  - {rec.name}: {len(rec.frame_entries)} frames, pose={'yes' if rec.pose_entry else 'no'}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(args.split == "train"),
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

    batch = next(iter(loader))
    img = batch["image"]
    intr = batch["fxfycxcy"]
    c2w = batch["c2w"]
    idx = batch["index"]

    expected_v = int(cfg.training.num_views)
    expected_s = int(cfg.model.image_tokenizer.image_size)
    assert img.shape == (args.batch_size, expected_v, 3, expected_s, expected_s), f"bad image shape: {img.shape}"
    assert intr.shape == (args.batch_size, expected_v, 4), f"bad intrinsics shape: {intr.shape}"
    assert c2w.shape == (args.batch_size, expected_v, 4, 4), f"bad c2w shape: {c2w.shape}"
    assert idx.shape == (args.batch_size, expected_v, 2), f"bad index shape: {idx.shape}"
    assert img.min().item() >= 0.0 and img.max().item() <= 1.0, "image values out of [0, 1]"

    # Verify c2w rotation blocks are valid SO(3) matrices.
    rot = c2w[..., :3, :3]
    eye = torch.eye(3).expand_as(rot)
    ortho_err = (rot.transpose(-1, -2) @ rot - eye).abs().max().item()
    dets = torch.linalg.det(rot)
    assert ortho_err < 1e-4, f"non-orthogonal rotation: {ortho_err}"
    assert (dets - 1.0).abs().max().item() < 1e-4, f"bad det(R): {dets}"

    print(f"batch image  : {tuple(img.shape)} {img.dtype} range=[{img.min():.3f}, {img.max():.3f}]")
    print(f"batch intr   : {tuple(intr.shape)} sample={intr[0, 0].tolist()}")
    print(f"batch c2w    : {tuple(c2w.shape)} max_ortho_err={ortho_err:.2e}")
    print(f"batch index  : {tuple(idx.shape)} frames[0]={idx[0, :, 0].tolist()}")

    if args.preview_out is not None:
        _save_batch_preview(img[0], args.preview_out)
        print(f"wrote preview: {args.preview_out}")

    print("OK")


if __name__ == "__main__":
    main()
