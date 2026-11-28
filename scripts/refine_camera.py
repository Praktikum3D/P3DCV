"""Refine the focal length and alpha of one scene after inference, and report what moved.

This is the test time half of the line optimization idea; `training/camera_refine.py` holds the
mechanism and explains the two terms. The script only decides where the scene comes from, and there
are two ways:

    # the real thing: reconstruct a scene from a checkpoint and refine against its input images
    python scripts/refine_camera.py --ckpt last.pt --env HongKong --data-root .../tartanair \
        -s dataset.detect_lines true -s dataset.line_native_size 0

    # a precomputed scene file (scripts/precompute_scene.py), for a GPU without bf16
    python scripts/refine_camera.py --scene ~/praktikum/scenes/hongkong_b90_fisheye.pt

The second one has a caveat worth stating plainly, because it changes what the numbers mean. A
scene file stores the Gaussians, the poses and the camera, but NOT the images the scene was built
from, so this script renders the input views once with the scene's own camera and treats those
renders as the observed images. In the pixel aligned limit that rendering IS the input image, which
is why it is a defensible stand in, but the photometric term then starts at exactly zero and
`--auto-lambda` has nothing to balance. With `--ckpt` it starts at the real reconstruction error
instead, so the number means something, which is the mode the sieve should quote. What it does NOT
change is the direction that term pulls in: the Gaussians are pixel aligned under the predicted
camera either way, so the photometric term anchors the refinement where the network left it. Run it
with `--image-weight 0` to move the camera and read the photometric number as a diagnostic.

What is reported is the field of view and alpha before and after, which is what the weekend's five
degree criterion is stated in.

What it is worth, measured twice
--------------------------------
Far from the answer it is excellent. On scene files dumped from checkpoints whose field of view had
COLLAPSED to around 90 degrees against a true 195, 200 steps each, lines only:

    scene                            lines   fov before   fov after   error after
    hongkong_b90_fisheye                 5        97.1       192.7        2.3 deg
    office_b90_fisheye                  23        94.0       188.9        6.1 deg
    hongkong_b90_ucm, alpha frozen       2        90.0       197.3        2.3 deg
    hongkong_b90_ucm, alpha refined      2        90.0       232.6       37.6 deg

Near the answer it is not. The same three scenes were re-dumped from the FINAL checkpoints while
this was being written, so they now start at the operating point the trained model really sits at,
about 191 degrees, and the same command does this:

    scene                            lines   fov before   fov after   error 191 -> after
    office_b90_fisheye                  26       190.5       194.2        4.5 -> 0.8 deg
    hongkong_b90_fisheye                 7       190.5       188.2        4.5 -> 6.8 deg
    hongkong_b90_ucm, alpha frozen      24       191.9       193.7        3.1 -> 1.3 deg
    hongkong_b90_ucm, alpha refined     24       191.9       181.7        3.1 -> 13.3 deg

(The first table cannot be reproduced any more, the files it used are gone. The second is what the
current files give.) Two of three improve with alpha frozen, one gets worse by two degrees, and the
average gain is small. That is the same picture the training side shows at this operating point: a
correction of a hundred degrees is far above the noise, a correction of four degrees is not clearly
above it.

So: alpha is refined only with `--learn-alpha`, because on both dumps letting it move turned the
best result into the worst one. Short lines hurt rather than help (`--min-line-length 0.1` gave 174
and 178 degrees on the collapsed fisheye scenes against 193 and 189 with the default 0.2), which is
the selection effect `training/line_detection.py` describes: a short piece of a curve is nearly
straight and nearly coplanar under any camera.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from contextlib import nullcontext
from typing import Any, Dict, Optional

import torch
import yaml
from easydict import EasyDict as edict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.camera_refine import (  # noqa: E402
    DEFAULT_DETECT_SIZE,
    DEFAULT_LINE_WEIGHT,
    DEFAULT_LOG_F_LIMIT,
    DEFAULT_LR,
    DEFAULT_MIN_LINES_FOR_ALPHA,
    DEFAULT_STEPS,
    RefineScene,
    detect_scene_lines,
    refine_camera,
    render_views,
    renderer_for,
)

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


def scene_from_file(path: str, device: torch.device) -> RefineScene:
    """Load a `scripts/precompute_scene.py` dump and stand in for its missing images.

    The intrinsics in the file are normalized by the image size, as the model predicts them; they
    are scaled to pixels here, because that is what the refinement and the residual work in.
    """
    payload = torch.load(path, map_location="cpu")
    height, width = (int(v) for v in payload.get("input_hw", [256, 256]))
    fxfycxcy = payload["fxfycxcy_input"].float().clone()
    fxfycxcy[:, 0] *= width
    fxfycxcy[:, 1] *= height
    fxfycxcy[:, 2] *= width
    fxfycxcy[:, 3] *= height
    views = fxfycxcy.shape[0]
    scene = RefineScene(
        xyz=payload["xyz"].float(),
        features=payload["features"].float(),
        scaling=payload["scaling"].float(),
        rotation=payload["rotation"].float(),
        opacity=payload["opacity"].float(),
        sh_degree=int(payload["sh_degree"]),
        c2w=payload["c2w_input"].float(),
        fxfycxcy=fxfycxcy,
        images=torch.zeros(views, 3, height, width),
        camera_model=str(payload.get("camera_model", "fisheye")),
        near_plane=float(payload.get("near_plane", 0.01)),
        tile_mode=str(payload.get("tile_mode", "pbf")),
        name=str(payload.get("name", os.path.basename(path))),
    ).to(device)
    with torch.no_grad():
        # The stand in for the observed images, see the module docstring.
        scene.images = render_views(scene, renderer_for(scene), scene.fxfycxcy, height, width).clamp(0, 1)
    return scene


def load_config(args: argparse.Namespace, checkpoint: Dict[str, Any]) -> edict:
    if args.config:
        with open(args.config, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    elif isinstance(checkpoint.get("config"), dict):
        data = checkpoint["config"]
        print("Using the config stored in the checkpoint", flush=True)
    elif os.path.isfile(FALLBACK_CONFIG):
        with open(FALLBACK_CONFIG, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    else:
        raise SystemExit("the checkpoint carries no config, pass --config")

    data = dict(data)
    for key, value in args.set:
        set_nested(data, key, parse_value(value))

    config = to_edict(data)
    config.inference = False
    config.pop("evaluation", None)
    config.create_visual = False
    config.training.random_split = False
    config.training.num_workers = 0
    config.training.batch_size_per_gpu = 1
    config.data_repeat = 1
    camera = str(config.dataset.get("camera_name", "lcam_fish"))
    root = args.data_root or str(config.dataset.get("train_path", ""))
    config.dataset.eval_path = os.path.join(root, args.env, args.difficulty, f"image_{camera}.zip")
    return config


def scene_from_checkpoint(args: argparse.Namespace, device: torch.device):
    """Run the model once on one sample and freeze everything it predicted except the camera."""
    from torch.utils.data import DataLoader

    from training.datasets import build_dataset

    checkpoint = torch.load(args.ckpt, map_location="cpu")
    config = load_config(args, checkpoint)
    module_name, class_name = config.model.class_name.rsplit(".", 1)
    model = importlib.import_module(module_name).__dict__[class_name](config).to(device)
    raw = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    state = {k.removeprefix("module."): v for k, v in raw.items() if not k.startswith("loss_computer.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded {len(state)} tensors, {len(missing)} missing, {len(unexpected)} unexpected", flush=True)
    model.eval()
    del checkpoint

    dataset = build_dataset(config, split="eval")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    batch = None
    for position, item in enumerate(loader):
        if position == args.index:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in item.items()}
            break
    if batch is None:
        raise SystemExit(f"--index {args.index} is past the end of the eval set")

    amp = str(config.training.get("amp_dtype", "bf16")).lower()
    context = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16 if amp == "bf16" else torch.float16)
        if device.type == "cuda" and config.training.get("use_amp", True)
        else nullcontext()
    )
    with torch.no_grad(), context:
        result = model(batch)

    target_image = config.model.get("target_image", edict(height=256, width=256))
    height, width = int(target_image.get("height", 256)), int(target_image.get("width", 256))
    fxfycxcy = result.fxfycxcy_input[0].detach().float().clone()
    fxfycxcy[:, 0] *= width
    fxfycxcy[:, 1] *= height
    fxfycxcy[:, 2] *= width
    fxfycxcy[:, 3] *= height
    pc = result.gaussians[0]
    batch_idx = torch.zeros(1, 1, dtype=torch.long, device=device)
    images = result.image[batch_idx, result.input_idx][0].detach().float()
    scene = RefineScene(
        xyz=pc._xyz.detach().float(),
        features=pc.get_features.detach().float(),
        scaling=pc._scaling.detach().float(),
        rotation=pc._rotation.detach().float(),
        opacity=pc._opacity.detach().float(),
        sh_degree=int(config.model.gaussians.sh_degree),
        c2w=result.c2w_input[0].detach().float(),
        fxfycxcy=fxfycxcy,
        images=images.clamp(0, 1),
        camera_model=str(config.model.get("camera_model", "pinhole")),
        near_plane=float(config.model.get("near_plane", 0.2)),
        tile_mode=str(config.model.get("geer_tile_mode", "pbf")),
        name=f"{args.env} #{args.index}",
    ).to(device)
    # The loader's own lines, already gathered to the input views by the model, are the sharp ones:
    # detected on the 1000 pixel source frame and mapped into the view. Nothing beats that here, so
    # they are used whenever dataset.detect_lines was on.
    lines = result.get("input_line_pixels")
    return scene, None if lines is None else lines[0].detach().float().to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--scene", help="a scene file written by scripts/precompute_scene.py")
    source.add_argument("--ckpt", help="a checkpoint to reconstruct the scene with")
    parser.add_argument("--config", default=None, help="config yaml, only when the checkpoint has none")
    parser.add_argument("--env", default="HongKong", help="TartanAir environment, with --ckpt")
    parser.add_argument("--data-root", default=None, help="TartanAir root, with --ckpt")
    parser.add_argument("--difficulty", default="Data_hard")
    parser.add_argument("--index", type=int, default=0, help="which sample of the eval set to take")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--line-weight", type=float, default=DEFAULT_LINE_WEIGHT)
    parser.add_argument("--image-weight", type=float, default=0.0,
                        help="weight of the photometric term, 0 (the default, and what the "
                             "measurement recommends) for a line only refinement; the term anchors "
                             "the camera where the network left it, see DEFAULT_LINE_WEIGHT")
    parser.add_argument("--auto-lambda", action="store_true",
                        help="set the line weight so both terms are equal at step 0")
    parser.add_argument("--no-lines", action="store_true", help="photometric term only, the control run")
    parser.add_argument("--detect-size", type=int, default=DEFAULT_DETECT_SIZE,
                        help="resolution the lines are detected at, 0 for the working resolution")
    parser.add_argument("--min-line-length", type=float, default=0.2,
                        help="shortest line kept, a fraction of the image side below 1 and pixels above")
    parser.add_argument("--learn-beta", action="store_true", help="refine the EUCM beta as well")
    parser.add_argument("--learn-alpha", action="store_true",
                        help="refine the (E)UCM alpha as well as the focal length; off by default "
                             "because f and alpha trade off against each other and the measured "
                             "result was worse, see refine_camera in training/camera_refine.py")
    parser.add_argument("--min-lines-for-alpha", type=int, default=DEFAULT_MIN_LINES_FOR_ALPHA,
                        help="below this many lines alpha is frozen automatically, 0 to never do it")
    parser.add_argument("--log-f-limit", type=float, default=DEFAULT_LOG_F_LIMIT,
                        help="trust region on the focal length in log units, 0 to switch it off")
    parser.add_argument("--true-fov", type=float, default=None,
                        help="the true field of view in degrees, to report the error against")
    parser.add_argument("--json", default=None, help="write the step history here")
    parser.add_argument("--progress", type=int, default=20, help="print every n steps, 0 for silent")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("-s", "--set", nargs=2, action="append", default=[], metavar=("KEY", "VALUE"),
                        help="dotted config override, the same keys train.py takes")
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset_lines: Optional[torch.Tensor] = None
    if args.scene:
        scene = scene_from_file(args.scene, device)
    else:
        scene, dataset_lines = scene_from_checkpoint(args, device)
    height, width = scene.image_hw
    print(
        f"Scene {scene.name}: {scene.xyz.shape[0]} Gaussians, {scene.c2w.shape[0]} input views, "
        f"{width}x{height}, camera_model {scene.camera_model}, tile mode {scene.tile_mode}",
        flush=True,
    )

    line_pixels = None
    if not args.no_lines:
        if dataset_lines is not None:
            line_pixels = dataset_lines
            print("Lines from the loader (detected on the source frame)", flush=True)
        else:
            line_pixels = detect_scene_lines(scene, detect_size=args.detect_size,
                                             min_length=args.min_line_length)
            print(f"Lines detected on renders at {args.detect_size or width} px", flush=True)

    result = refine_camera(
        scene,
        line_pixels=line_pixels,
        steps=args.steps,
        lr=args.lr,
        line_weight=args.line_weight,
        image_weight=args.image_weight,
        auto_lambda=args.auto_lambda,
        learn_alpha=args.learn_alpha,
        learn_beta=args.learn_beta,
        min_lines_for_alpha=args.min_lines_for_alpha,
        log_f_limit=args.log_f_limit,
        progress=args.progress or None,
    )
    print(f"{result.num_lines} lines used", flush=True)
    print(result.summary(), flush=True)
    if args.true_fov is not None:
        before = abs(result.first["fov_deg"] - args.true_fov)
        after = abs(result.last["fov_deg"] - args.true_fov)
        print(f"fov error {before:.2f} -> {after:.2f} deg against a true {args.true_fov:.2f}", flush=True)

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "scene": scene.name,
                    "camera_model": scene.camera_model,
                    "num_lines": result.num_lines,
                    "steps": args.steps,
                    "lr": args.lr,
                    "line_weight": args.line_weight,
                    "image_weight": args.image_weight,
                    "auto_lambda": bool(args.auto_lambda),
                    "alpha_frozen_by_gate": bool(result.alpha_frozen_by_gate),
                    "hit_focal_limit": bool(result.hit_focal_limit),
                    "history": result.history,
                },
                handle,
                indent=1,
            )
        print(f"Wrote {args.json}", flush=True)


if __name__ == "__main__":
    main()
