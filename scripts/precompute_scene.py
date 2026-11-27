"""Reconstruct one TartanAir scene with E-RayZer and dump it as a viewer scene file.

The interactive 3DGEER viewer renders Gaussians, which needs no bf16, but reconstructing them does,
so the reconstruction happens once on the cluster and the result travels as a `.pt` file to whatever
GPU runs the viewer. This script is that step. It is deliberately self contained: it imports only
`erayzer_core` and `training`, so it runs from any P3DCV checkout, not just the viewer branch.

Usage inside the cluster container, from a P3DCV checkout as the working directory:

    srun --partition=24g --gres=gpu:1,ccc:86 --qos=students_opportunistic --pty \
        enroot start --mount /mnt:/mnt --env PYTHONPATH=$PWD <image>.sqsh \
        python scripts/precompute_scene.py \
            --ckpt /mnt/projects/p3dcv/anycam/checkpoints/runs/fish-ucm-geer/last.pt \
            --env HongKong \
            --data-root /mnt/projects/p3dcv/anycam/data/tartanair \
            --out /mnt/home/$USER/runs/viewer/hongkong_ucm.pt

`--config` is only needed when the checkpoint carries no config of its own, which the pretrained
weights do not. Every checkpoint written by `train.py` does, and that config is what decides whether
the run is pinhole, fisheye or UCM, so pass `--config` only when you have to. Without one, and
without a config in the checkpoint, the repository config `config/erayzer_train_tartanair.yaml` of
the working directory stands in.

The output is a flat dict of plain tensors and scalars, `torch.load` able without this file:

    xyz [n, 3]              raw Gaussian centres
    features [n, k, 3]      spherical harmonics, k = (sh_degree + 1) ** 2
    scaling [n, 3]          raw, the viewer applies exp
    rotation [n, 4]         raw quaternions
    opacity [n, 1]          raw, the viewer applies sigmoid
    sh_degree               int
    c2w_input [v, 4, 4]     predicted poses of the input views
    fxfycxcy_input [v, 4]   predicted intrinsics, normalized by the image size, UCM [v, 5] with alpha
    camera_model            "pinhole", "fisheye" or "ucm"
    near_plane              float
    tile_mode               "pbf" or "projection"
    alpha_init              float
    input_hw                [h, w] the intrinsics are normalized by
    name                    label for the viewer
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from contextlib import nullcontext
from typing import Any, Dict

import torch
import yaml
from easydict import EasyDict as edict
from torch.utils.data import DataLoader

FALLBACK_CONFIG = "config/erayzer_train_tartanair.yaml"


def to_edict(value: Any) -> Any:
    if isinstance(value, dict):
        return edict({k: to_edict(v) for k, v in value.items()})
    if isinstance(value, list):
        return [to_edict(v) for v in value]
    return value


def parse_value(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def set_nested(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = config
    parts = dotted_key.split(".")
    for key in parts[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[parts[-1]] = value


def load_config(args: argparse.Namespace, checkpoint: Dict[str, Any]) -> edict:
    if args.config:
        with open(args.config, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    elif isinstance(checkpoint.get("config"), dict):
        data = checkpoint["config"]
        print("Using the config stored in the checkpoint", flush=True)
    elif os.path.isfile(FALLBACK_CONFIG):
        # The pretrained weights carry no config of their own, so the repository config stands in.
        print(f"The checkpoint carries no config, falling back to {FALLBACK_CONFIG}", flush=True)
        with open(FALLBACK_CONFIG, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    else:
        raise SystemExit("the checkpoint carries no config, pass --config")

    data = dict(data)
    for key, value in args.set:
        set_nested(data, key, parse_value(value))

    config = to_edict(data)
    config.inference = False
    config.pop("evaluation", None)  # would switch the model into its video rendering path
    config.create_visual = False
    config.training.random_split = False  # the same input and target views every time
    config.training.num_workers = 0
    config.training.batch_size_per_gpu = 1
    config.data_repeat = 1

    camera = str(config.dataset.get("camera_name", "lcam_fish"))
    root = args.data_root or str(config.dataset.get("train_path", ""))
    config.dataset.eval_path = os.path.join(root, args.env, args.difficulty, f"image_{camera}.zip")
    print(f"Eval path {config.dataset.eval_path}", flush=True)
    return config


# Weights that only the original E-RayZer training carried. Its perceptual loss lived inside the
# model as `loss_computer.perceptual_loss_module`, a VGG 19 with sixteen convolutions, so
# `erayzer_multi.pt` holds 32 tensors that no model in this repository builds. Those are the
# "32 unexpected keys" the pretrained load used to report. The same list lives in
# `viewer/scene.py`; this script imports nothing of the viewer on purpose.
TRAINING_ONLY_PREFIXES = ("loss_computer.",)


def model_state_dict(checkpoint: Any):
    """The model weights of a checkpoint, ready for `load_state_dict`, plus the names dropped."""
    raw = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    state, dropped = {}, []
    for key, value in raw.items():
        name = key.removeprefix("module.")
        if name.startswith(TRAINING_ONLY_PREFIXES):
            dropped.append(name)
        else:
            state[name] = value
    return state, dropped


def build_model(config: edict, checkpoint: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    module_name, class_name = config.model.class_name.rsplit(".", 1)
    model = importlib.import_module(module_name).__dict__[class_name](config).to(device)
    state_dict, dropped = model_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    report = f"Loaded weights, {len(state_dict)} tensors"
    if dropped:
        report += f", {len(dropped)} training only weights skipped"
    if missing:
        report += f", MISSING {len(missing)} ({', '.join(missing[:4])}...)"
    if unexpected:
        report += f", UNEXPECTED {len(unexpected)} ({', '.join(unexpected[:4])}...)"
    print(report, flush=True)
    return model.eval()


def load_batch(config: edict, device: torch.device, index: int) -> Dict[str, Any]:
    from training.datasets import build_dataset

    dataset = build_dataset(config, split="eval")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    print(f"Eval set has {len(dataset)} samples", flush=True)
    for position, batch in enumerate(loader):
        if position == index:
            return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    raise SystemExit(f"--index {index} is past the end of the eval set")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True, help="checkpoint to reconstruct with")
    parser.add_argument("--config", default=None, help="config yaml, only when the checkpoint has none")
    parser.add_argument("--env", required=True, help="TartanAir environment, for example HongKongAlley")
    parser.add_argument("--data-root", default=None, help="TartanAir root, defaults to dataset.train_path")
    parser.add_argument("--difficulty", default="Data_hard")
    parser.add_argument("--index", type=int, default=0, help="which sample of the eval set to take")
    parser.add_argument("--out", required=True, help="scene file to write")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("-s", "--set", nargs=2, action="append", default=[], metavar=("KEY", "VALUE"),
                        help="dotted config override, the same keys train.py takes")
    args = parser.parse_args()

    sys.path.insert(0, os.getcwd())
    device = torch.device(args.device)
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    config = load_config(args, checkpoint)
    model = build_model(config, checkpoint, device)
    del checkpoint

    amp_name = str(config.training.get("amp_dtype", "bf16")).lower()
    amp_dtype = torch.bfloat16 if amp_name == "bf16" else torch.float16
    context = (
        torch.autocast(device_type=device.type, dtype=amp_dtype)
        if device.type == "cuda" and config.training.get("use_amp", True)
        else nullcontext()
    )

    batch = load_batch(config, device, args.index)
    with torch.no_grad(), context:
        result = model(batch)

    pc = result.gaussians[0]  # a detached float GaussianModel
    camera_model = config.model.get("camera_model", "pinhole")
    target_image = config.model.get("target_image", edict(height=256, width=256))
    payload = {
        "xyz": pc._xyz.detach().float().cpu(),
        "features": pc.get_features.detach().float().cpu(),
        "scaling": pc._scaling.detach().float().cpu(),
        "rotation": pc._rotation.detach().float().cpu(),
        "opacity": pc._opacity.detach().float().cpu(),
        "sh_degree": int(config.model.gaussians.sh_degree),
        "c2w_input": result.c2w_input[0].detach().float().cpu(),
        "fxfycxcy_input": result.fxfycxcy_input[0].detach().float().cpu(),
        "camera_model": camera_model,
        "near_plane": float(config.model.get("near_plane", 0.2)),
        "tile_mode": config.model.get("geer_tile_mode", "pbf"),
        "alpha_init": float(config.model.get("alpha_init", 0.5)),
        "input_hw": [int(target_image.get("height", 256)), int(target_image.get("width", 256))],
        "name": f"{args.env} #{args.index} ({camera_model})",
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    torch.save(payload, args.out)
    size_mb = os.path.getsize(args.out) / (1024.0 * 1024.0)
    print(
        f"Wrote {args.out}, {payload['xyz'].shape[0]} Gaussians, {payload['c2w_input'].shape[0]} input views, "
        f"camera_model {camera_model}, sh_degree {payload['sh_degree']}, {size_mb:.1f} MB",
        flush=True,
    )


if __name__ == "__main__":
    main()
