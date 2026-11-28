#!/usr/bin/env python
"""Score one checkpoint on cameras whose true focal length and alpha we know.

The question this answers is the one the weekend goal is measured by: given images of a camera the
model has never seen and is not told about, how close does it get to that camera, and does it still
reconstruct well. The test cameras are made by resampling the 195 degree `lcam_fish` ground truth
frames of an environment into a camera of our choosing (`training/camera_augmentation.py`), so the
true field of view is exact by construction and the true alpha is either exact (pinhole, UCM) or the
UCM fit of the true mapping (equidistant fisheye). `--camera-set front` instead evaluates the real
`lcam_front` pinhole frames as they are.

Three properties make the numbers comparable across checkpoints, and all three are deliberate:

* the test camera list is fixed in code, with no randomness in it at all,
* the frames are selected from a seed that does not depend on the checkpoint or on the camera, so
  every model sees the same pixels,
* the scoring mask is a function of the TEST camera alone, never of the predicted one, so two models
  with different predicted cameras are scored on the very same pixel set (`--mask`).

Output in `--out-dir`:

* `cameras.csv`, one row per test camera, schema `CSV_FIELDS` below,
* `summary.txt`, the same numbers as a table plus the pass or fail of the five degree criterion,
* `panel_<family>.png`, ground truth next to the rendering for every camera of one model family.

Example:

    python scripts/camera_benchmark.py \
        --config config/erayzer_train_tartanair.yaml \
        --checkpoint /mnt/projects/p3dcv/anycam/checkpoints/runs/b90-ucm-long/last.pt \
        --data /mnt/projects/p3dcv/anycam/data/tartanair/HongKong/Data_hard/image_lcam_fish.zip \
        --out-dir ~/runs/bench/b90-ucm-long-HongKong
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from easydict import EasyDict as edict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import evaluate  # noqa: E402  (the repository root has to be on the path first)
from erayzer_core.model.utils_gaussian import UCM_MODELS, camera_theta  # noqa: E402
from erayzer_core.utils.camera_eval import evaluate_camera_pose_metrics  # noqa: E402
from training.camera_augmentation import (  # noqa: E402
    DEFAULT_SUPERSAMPLE,
    FISH_FOV_DEG,
    AugCamera,
    remap_views,
    true_alpha,
    ucm_theta_max,
)
from training.datasets import build_dataset as build_config_dataset  # noqa: E402

# --- the test cameras ---------------------------------------------------------------------------

PINHOLE_FOVS: Tuple[float, ...] = (60.0, 90.0, 120.0)
"""Pinhole fields of view across the image width. 120 is as wide as a pinhole stays usable."""

FISHEYE_FOVS: Tuple[float, ...] = (120.0, 150.0, 195.0)
"""Equidistant fisheye fields of view. 195 is the source camera itself, so that row is a resize."""

UCM_FOVS: Tuple[float, ...] = (130.0, 160.0, 190.0)
UCM_ALPHAS: Tuple[float, ...] = (0.4, 0.62, 0.8)
"""UCM cameras, every combination whose alpha can reach its field of view (see `test_cameras`).

0.62 is roughly the alpha an equidistant 195 degree camera fits to, so that column is the shape the
training data has; 0.4 and 0.8 sit on either side of it.
"""

FOV_LIMIT_DEG = 5.0
"""The success criterion: the mean field of view error of a camera may be about five degrees."""

WIDE_LIMIT_DEG = 120.0
"""Above this field of view the alpha of a camera is visible in the image, so its error counts."""

FRONT_FOV_DEG = 90.0
"""Field of view across the width of the real TartanAir `lcam_front` pinhole frames.

Not measured by us. It is the convention `training/tartanair_dataset.py` already builds its
intrinsics on (fx = fy = width / 2, which is exactly 90 degrees across the width), and TartanAir
documents its pinhole cameras as 90 degree ones. `docs/datasets/tartanair.md` still records the
exact intrinsics as an open point, so `--front-fov` overrides this number rather than hiding it.
"""


def camera_name(camera: AugCamera) -> str:
    """Stable row name of a test camera, safe as a file name and sortable within a family."""
    if camera.model == "ucm":
        return f"ucm_{int(round(camera.fov_deg)):03d}_a{int(round(float(camera.alpha) * 100)):03d}"
    return f"{camera.model}_{int(round(camera.fov_deg)):03d}"


def test_cameras() -> List[AugCamera]:
    """The fixed list of remap test cameras, in report order and without any randomness.

    A UCM camera is only listed when its alpha can image half its field of view at all: beyond
    `ucm_theta_max(alpha)` the projection is no longer injective, there is no focal length that puts
    that angle on the image edge, and `focal_for_fov` would refuse the camera.
    """
    cameras = [AugCamera("pinhole", fov) for fov in PINHOLE_FOVS]
    cameras += [AugCamera("fisheye", fov) for fov in FISHEYE_FOVS]
    for fov in UCM_FOVS:
        for alpha in UCM_ALPHAS:
            if math.radians(fov) / 2.0 < ucm_theta_max(alpha):
                cameras.append(AugCamera("ucm", fov, alpha))
    return cameras


# --- the camera the model predicts --------------------------------------------------------------


def fov_from_fx(
    fx_norm: torch.Tensor,
    camera_model: str,
    alpha: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Field of view across the image width in degrees, from fx normalised by the width.

    The image edge sits half a width from the principal point, so its normalised image radius is
    0.5 / fx_norm, and `camera_theta` turns that into the angle of the ray that lands there. Every
    camera model has its own law in there, which is the whole point of going through it rather than
    writing an arc tangent: pinhole atan(rho), equidistant fisheye rho itself, (E)UCM the closed form
    inverse of its projection. This is the expression `train.camera_stats` logs during training.
    """
    rho = 0.5 / fx_norm.clamp_min(1e-6)
    return torch.rad2deg(2.0 * camera_theta(rho, camera_model, alpha, beta))


def predicted_camera(fxfycxcy: torch.Tensor, camera_model: str) -> Tuple[float, Optional[float]]:
    """Field of view in degrees and alpha of one sample, from its predicted intrinsics [v, d].

    The views of a sample all look through the same lens, so the focal lengths are averaged over the
    views first and converted once, as `train.aug_stats` does. alpha is a single prediction per scene
    already, the mean over the views only drops the repetition.
    """
    values = fxfycxcy.detach().float()
    width = values.shape[-1]
    needed = {"ucm": 5, "eucm": 6}.get(camera_model, 4)
    if width < needed:
        raise ValueError(
            f"the checkpoint predicts {width} intrinsics, the {camera_model} camera model needs "
            f"{needed}. Pass the config of the run with --config so its camera model matches."
        )
    fx = values[..., 0].mean()
    alpha = values[..., 4].mean() if camera_model in UCM_MODELS else None
    beta = values[..., 5].mean() if camera_model == "eucm" else None
    fov = fov_from_fx(fx[None], camera_model, None if alpha is None else alpha[None],
                      None if beta is None else beta[None])
    return float(fov.item()), (None if alpha is None else float(alpha.item()))


# --- the scoring mask ---------------------------------------------------------------------------

MASK_MODES = ("circle", "valid", "full")


def scoring_mask(
    valid: Optional[torch.Tensor], fxfycxcy_px: torch.Tensor, h: int, w: int, mode: str
) -> Optional[torch.Tensor]:
    """The pixels every checkpoint is scored on for one test camera, [1, 1, h, w] or None.

    It depends on the TEST camera only, never on what a model predicted, which is what makes PSNR
    and SSIM of two models comparable at all. `valid` is the remap's own validity mask, 1 where the
    source fisheye really covered the pixel.

    * "circle" intersects it with the circle spanning the image width, the reference area
      `evaluate.image_circle_disc` defines for the cross model comparison. A wide camera fills
      roughly that circle anyway; a narrow pinhole loses its corners, which is the price of scoring
      a 60 degree and a 195 degree camera on the same shape.
    * "valid" is the remap mask alone, so a pinhole keeps its corners.
    * "full" scores every pixel, which only makes sense for data that fills its frame.
    """
    if mode not in MASK_MODES:
        raise ValueError(f"unknown mask mode {mode!r}, use one of {', '.join(MASK_MODES)}")
    if mode == "full":
        return None
    if mode == "valid":
        return valid
    disc = evaluate.image_circle_disc(fxfycxcy_px.reshape(1, -1), h, w)
    return disc if valid is None else valid * disc


# --- deterministic frame selection ----------------------------------------------------------------


@dataclass(frozen=True)
class Sample:
    """One evaluated scene: which trajectory and which frames of it, chosen once per run."""

    trajectory: int
    frames: Tuple[int, ...]


def select_samples(
    trajectory_lengths: Sequence[int], count: int, num_views: int, span: int, seed: int
) -> List[Sample]:
    """`count` samples drawn from a seed alone, so every checkpoint sees identical frames.

    Trajectories are visited in a shuffled order and then round robin, so a small `count` still
    spreads over the environment instead of taking the first trajectory only. Within a trajectory the
    frames are `num_views` evenly spaced indices over `span` frames, the eval rule of
    `TartanAirDataset`, placed at a start the generator picks.

    The generator is created here and never touched elsewhere, so the result depends on the seed, the
    trajectory lengths and nothing else. It is explicitly independent of the test camera, so a sample
    shows the same scene through all of them.
    """
    usable = [i for i, length in enumerate(trajectory_lengths) if length >= num_views]
    if not usable:
        raise ValueError(
            f"no trajectory has the {num_views} frames a sample needs "
            f"(longest is {max(trajectory_lengths) if trajectory_lengths else 0})"
        )
    rng = np.random.default_rng(int(seed))
    order = [usable[i] for i in rng.permutation(len(usable))]

    samples: List[Sample] = []
    for step in range(int(count)):
        trajectory = order[step % len(order)]
        length = int(trajectory_lengths[trajectory])
        reach = max(num_views - 1, min(int(span), length - 1))
        start = int(rng.integers(0, length - reach)) if length - reach > 0 else 0
        frames = tuple(int(start + offset) for offset in np.linspace(0, reach, num_views, dtype=int))
        samples.append(Sample(trajectory, frames))
    return samples


# --- the run ---------------------------------------------------------------------------------------

CSV_FIELDS = [
    "camera",
    "camera_model",
    "fov_true",
    "fov_pred_mean",
    "fov_pred_std",
    "fov_err",
    "fov_pass",
    "alpha_true",
    "alpha_pred_mean",
    "alpha_pred_std",
    "alpha_err",
    "psnr",
    "ssim",
    "mse",
    "mask_fraction",
    "RPA@5",
    "RPA@15",
    "RPA@30",
    "n_samples",
]
"""Columns of `cameras.csv`, in this order. Appending is allowed, reordering breaks every reader."""

FLOAT_FORMAT = {
    "fov_true": "{:.4f}",
    "fov_pred_mean": "{:.4f}",
    "fov_pred_std": "{:.4f}",
    "fov_err": "{:.4f}",
    "alpha_true": "{:.6f}",
    "alpha_pred_mean": "{:.6f}",
    "alpha_pred_std": "{:.6f}",
    "alpha_err": "{:.6f}",
    "psnr": "{:.4f}",
    "ssim": "{:.6f}",
    "mse": "{:.8f}",
    "mask_fraction": "{:.6f}",
    "RPA@5": "{:.6f}",
    "RPA@15": "{:.6f}",
    "RPA@30": "{:.6f}",
}


def _mean(values: Sequence[float]) -> float:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def _std(values: Sequence[float]) -> float:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    return float(np.std(finite)) if finite else float("nan")


def summarise_camera(camera: AugCamera, name: str, records: List[Dict[str, float]]) -> Dict[str, object]:
    """One CSV row out of the per sample records of a single test camera."""
    fov_true = float(camera.fov_deg)
    alpha_true_value = true_alpha(camera)
    fov_pred = [r["fov_pred"] for r in records]
    alpha_pred = [r["alpha_pred"] for r in records if r["alpha_pred"] is not None]
    fov_err = _mean([abs(value - fov_true) for value in fov_pred])
    alpha_err = _mean([abs(value - alpha_true_value) for value in alpha_pred])
    mse = _mean([r["mse"] for r in records])
    row: Dict[str, object] = {
        "camera": name,
        "camera_model": camera.model,
        "fov_true": fov_true,
        "fov_pred_mean": _mean(fov_pred),
        "fov_pred_std": _std(fov_pred),
        "fov_err": fov_err,
        "fov_pass": int(math.isfinite(fov_err) and fov_err <= FOV_LIMIT_DEG),
        "alpha_true": alpha_true_value,
        "alpha_pred_mean": _mean(alpha_pred),
        "alpha_pred_std": _std(alpha_pred),
        "alpha_err": alpha_err,
        "psnr": _mean([r["psnr"] for r in records]),
        "ssim": _mean([r["ssim"] for r in records]),
        "mse": mse,
        "mask_fraction": _mean([r["mask_fraction"] for r in records]),
        "RPA@5": _mean([r["RPA@5"] for r in records]),
        "RPA@15": _mean([r["RPA@15"] for r in records]),
        "RPA@30": _mean([r["RPA@30"] for r in records]),
        "n_samples": len(records),
    }
    return row


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: FLOAT_FORMAT.get(key, "{}").format(row[key]) for key in CSV_FIELDS}
            )


def _baseline_index(path: Optional[str]) -> Dict[str, Dict[str, str]]:
    if not path:
        return {}
    if not os.path.isfile(path):
        raise FileNotFoundError(f"--baseline-csv not found: {path}")
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return {row["camera"]: row for row in csv.DictReader(handle) if row.get("camera")}


def summary_text(
    rows: List[Dict[str, object]],
    header: Dict[str, object],
    baseline_csv: Optional[str],
    samples: Optional[Sequence[str]] = None,
) -> str:
    """The table a human reads, with the pass or fail of the five degree criterion in it."""
    baseline = _baseline_index(baseline_csv)
    lines = ["camera benchmark"]
    for key, value in header.items():
        lines.append(f"  {key} {value}")
    lines.append("")
    columns = (
        f"{'camera':<18}{'fov_true':>9}{'fov_pred':>10}{'fov_err':>9}{'pass':>6}"
        f"{'a_true':>8}{'a_pred':>8}{'a_err':>8}{'psnr':>8}{'ssim':>8}"
        f"{'RPA@5':>8}{'RPA@15':>8}{'RPA@30':>8}{'n':>4}"
    )
    lines.append(columns)
    lines.append("-" * len(columns))
    for row in rows:
        lines.append(
            f"{row['camera']:<18}{row['fov_true']:>9.2f}{row['fov_pred_mean']:>10.2f}"
            f"{row['fov_err']:>9.2f}{('ok' if row['fov_pass'] else 'FAIL'):>6}"
            f"{row['alpha_true']:>8.3f}{row['alpha_pred_mean']:>8.3f}{row['alpha_err']:>8.3f}"
            f"{row['psnr']:>8.2f}{row['ssim']:>8.4f}"
            f"{row['RPA@5']:>8.3f}{row['RPA@15']:>8.3f}{row['RPA@30']:>8.3f}{row['n_samples']:>4}"
        )

    lines.append("")
    fov_errors = [float(row["fov_err"]) for row in rows]
    mean_err = _mean(fov_errors)
    failed = [str(row["camera"]) for row in rows if not row["fov_pass"]]
    lines.append(
        f"mean field of view error over {len(rows)} cameras {mean_err:.2f} deg, limit "
        f"{FOV_LIMIT_DEG:.1f} deg, {'PASS' if math.isfinite(mean_err) and mean_err <= FOV_LIMIT_DEG else 'FAIL'}"
    )
    lines.append(
        f"cameras above the limit {len(failed)} of {len(rows)}"
        + (f" ({', '.join(failed)})" if failed else "")
    )
    wide = [row for row in rows if float(row["fov_true"]) > WIDE_LIMIT_DEG]
    lines.append(
        f"mean alpha error above {WIDE_LIMIT_DEG:.0f} deg "
        f"{_mean([float(row['alpha_err']) for row in wide]):.4f} over {len(wide)} cameras"
    )
    lines.append(f"mean psnr {_mean([float(row['psnr']) for row in rows]):.3f}")
    lines.append(f"mean ssim {_mean([float(row['ssim']) for row in rows]):.4f}")

    if baseline:
        lines.append("")
        lines.append("against the baseline, positive is better than the baseline")
        lines.append(f"{'camera':<18}{'d_psnr':>9}{'d_ssim':>9}{'d_RPA@5':>9}{'d_fov_err':>11}")
        for row in rows:
            other = baseline.get(str(row["camera"]))
            if other is None:
                continue
            lines.append(
                f"{row['camera']:<18}"
                f"{float(row['psnr']) - float(other['psnr']):>9.3f}"
                f"{float(row['ssim']) - float(other['ssim']):>9.4f}"
                f"{float(row['RPA@5']) - float(other['RPA@5']):>9.3f}"
                f"{float(other['fov_err']) - float(row['fov_err']):>11.3f}"
            )
    if samples:
        # Written out so a second checkpoint can be shown to have been given the same input.
        lines.append("")
        lines.append("frames, identical for every camera and every checkpoint at this seed")
        lines.extend(f"  {entry}" for entry in samples)
    return "\n".join(lines) + "\n"


def family_panel(pairs: List[Tuple[str, torch.Tensor, torch.Tensor]], path: str, gap: int = 4) -> None:
    """Ground truth next to the rendering for every camera of one family, one PNG, rows in order."""
    if not pairs:
        return
    rows = []
    for _, target, pred in pairs:
        left = target.detach().float().cpu().clamp(0, 1)
        right = pred.detach().float().cpu().clamp(0, 1)
        bar = torch.ones(3, left.shape[1], gap)
        rows.append(torch.cat([left, bar, right], dim=2))
    width = max(row.shape[2] for row in rows)
    padded = [
        row if row.shape[2] == width else torch.cat([row, torch.ones(3, row.shape[1], width - row.shape[2])], dim=2)
        for row in rows
    ]
    stacked = []
    for index, row in enumerate(padded):
        if index:
            stacked.append(torch.ones(3, gap, width))
        stacked.append(row)
    evaluate._save_tensor_image(torch.cat(stacked, dim=1), path)


# --- model and data ---------------------------------------------------------------------------------


def load_model(config, checkpoint: str, device: torch.device):
    """The checkpoint as an eval mode model, with the fly through render switched off.

    `evaluate.build_model` reads the path out of `config.evaluation`, and `ERayZer.forward` renders a
    thirty frame fly through for every batch as soon as that section exists. The benchmark calls the
    model a few hundred times and never looks at a video, so the section is removed again right after
    the weights are loaded.
    """
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"no checkpoint at {checkpoint}")
    config.evaluation = edict({"checkpoint_path": checkpoint, "out_dir": ""})
    model = evaluate.build_model(config, device, rank=0)
    config.pop("evaluation", None)
    config.inference = False
    model.eval()
    return model


def build_frame_source(config, data_path: str, camera: str, span: int):
    """The eval dataset of one environment, used for its frames and poses only.

    Built through `training.datasets.build_dataset` so the config keys map exactly as they do in
    training and evaluation. The benchmark then reads frames out of it itself, because it needs the
    native 1000 x 1000 fisheye frames for the remap, which the normal item path resizes away.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"no TartanAir data at {data_path}")
    if not isinstance(config.get("dataset", None), dict):
        config.dataset = edict()
    config.dataset.name = "tartanair"
    config.training.dataset = "tartanair"
    config.dataset.eval_path = data_path
    config.dataset.camera_name = camera
    config.dataset.camera_augmentation = False
    config.dataset.detect_lines = False
    config.data_repeat = 1
    # The benchmark picks its own frames (`select_samples`), so the selector only matters if someone
    # iterates the dataset the normal way; pinned here so that path would agree with this one.
    config.training.view_selector = dict(config.training.get("view_selector", {}))
    config.training.view_selector["use_curriculum"] = False
    config.training.view_selector["max_frame_dist"] = int(span)
    return build_config_dataset(config, split="eval")


def sample_batch(
    dataset, sample: Sample, device: torch.device
) -> Tuple[List[np.ndarray], torch.Tensor, torch.Tensor]:
    """The native frames, the poses and the index field of one sample.

    `_load_native_image`, `_load_poses` and `trajectories` are internals of `TartanAirDataset`. They
    are called rather than copied: the benchmark needs the unresized frames, which no public entry
    point hands out, and reimplementing the zip reading here would be a second copy to keep in step.
    """
    record = dataset.trajectories[sample.trajectory]
    frames = [dataset._load_native_image(record, index) for index in sample.frames]
    poses = dataset._load_poses(sample.trajectory)
    c2w = torch.from_numpy(np.stack([poses[index] for index in sample.frames])).float()
    image_index = torch.tensor(sample.frames, dtype=torch.long).unsqueeze(-1)
    index = torch.cat([image_index, torch.full_like(image_index, sample.trajectory)], dim=-1)
    return frames, c2w.to(device), index.to(device)


def front_batch(
    dataset, sample: Sample, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Images, pixel intrinsics, poses and the index field of one real `lcam_front` sample."""
    record = dataset.trajectories[sample.trajectory]
    images, intrinsics = [], []
    for index in sample.frames:
        image, intr, _ = dataset._load_image_and_intrinsics(record, index)
        images.append(image)
        intrinsics.append(intr)
    stacked = torch.stack(images)
    size = int(stacked.shape[-1])
    # _load_image_and_intrinsics normalises by the image size when the config asks it to; the mask
    # wants pixels, so undo that here rather than guessing the convention later.
    intr_px = torch.stack(intrinsics)[0] * (size if dataset.normalize_intrinsics else 1.0)
    poses = dataset._load_poses(sample.trajectory)
    c2w = torch.from_numpy(np.stack([poses[index] for index in sample.frames])).float()
    image_index = torch.tensor(sample.frames, dtype=torch.long).unsqueeze(-1)
    idx = torch.cat([image_index, torch.full_like(image_index, sample.trajectory)], dim=-1)
    return stacked.to(device), intr_px, c2w.to(device), idx.to(device)


def run_model(model, images, c2w, index, data_mask, device, amp_dtype, amp_enabled):
    """One forward pass on a single sample, batch of one.

    The batch deliberately carries no `fxfycxcy`: the whole question is what the model reads off the
    images, so handing it the true intrinsics of the test camera would answer it for free. `c2w` is
    the ground truth the pose metric is scored against and the model does not read it either.
    `data_mask` is the validity region of the test camera, which the camera augmentation also passes
    during training and which is the same for every checkpoint, so it changes no comparison.
    """
    batch = {
        "image": images[None],
        "c2w": c2w[None],
        "index": index[None],
    }
    if data_mask is not None:
        batch["data_mask"] = data_mask.expand(images.shape[0], -1, -1, -1)[None].contiguous().to(device)
    ctx = (
        torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled)
        if device.type == "cuda"
        else nullcontext()
    )
    with torch.no_grad(), ctx:
        return model(batch)


def score_sample(result, mask, camera_model, gt_c2w) -> Dict[str, float]:
    """PSNR, SSIM, RPA and the predicted camera of one forward pass."""
    pred = result.render.float().clamp(0, 1)
    target = result.get("target_image", None)
    if target is None:
        raise RuntimeError("the model returned no target_image, so there is nothing to score against")
    target = target.float()
    if pred.shape != target.shape:
        raise RuntimeError(f"render {tuple(pred.shape)} and target {tuple(target.shape)} differ in shape")
    b, v, c, h, w = pred.shape
    if mask is not None and tuple(mask.shape[-2:]) != (h, w):
        raise RuntimeError(
            f"the scoring mask is {tuple(mask.shape[-2:])} but the render is {(h, w)}; the mask is "
            "built at the size of the input images, so the two have to agree"
        )
    masks = None if mask is None else mask.to(pred.device, pred.dtype).expand(b * v, -1, h, w)
    metrics = evaluate.image_metrics(pred, target, masks)
    mse = float(metrics["mse"].mean().item())
    fov_pred, alpha_pred = predicted_camera(result.fxfycxcy, camera_model)
    views = gt_c2w.shape[0]
    rpa = evaluate_camera_pose_metrics(gt_c2w.detach(), result.c2w[0, :views].detach().float())
    return {
        "mse": mse,
        "psnr": -10.0 * math.log10(max(mse, 1e-8)),
        "ssim": float(metrics["ssim"].mean().item()),
        "mask_fraction": float(metrics["mask_fraction"].mean().item()),
        "fov_pred": fov_pred,
        "alpha_pred": alpha_pred,
        **{key: float(value) for key, value in rpa.items()},
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", "-c", required=True, help="the config of the run the checkpoint comes from")
    parser.add_argument("--checkpoint", required=True, help="the checkpoint to score")
    parser.add_argument("--data", required=True, help="an image_<camera>.zip, a directory or a manifest txt")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--camera-set", choices=("remap", "front"), default="remap")
    parser.add_argument("--samples", type=int, default=10, help="scenes per test camera")
    parser.add_argument("--seed", type=int, default=20261128, help="frame selection seed, fixed on purpose")
    parser.add_argument("--span", type=int, default=45, help="frames between the first and the last view")
    parser.add_argument("--supersample", type=int, default=DEFAULT_SUPERSAMPLE)
    parser.add_argument("--mask", choices=MASK_MODES, default="circle")
    parser.add_argument("--front-fov", type=float, default=FRONT_FOV_DEG)
    parser.add_argument("--front-camera", default="lcam_front")
    parser.add_argument("--source-camera", default="lcam_fish", help="the camera the remap reads from")
    parser.add_argument("--cameras", default="", help="comma separated row names, empty means all")
    parser.add_argument("--no-data-mask", action="store_true", help="hide the test camera validity from the model")
    parser.add_argument("--baseline-csv", default="", help="a cameras.csv to report the difference against")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--set", "-s", action="append", nargs=2, default=[], metavar=("KEY", "VALUE"))
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"no config at {args.config}")
    config = evaluate.load_config(args.config, args.set)
    config.inference = False

    camera_model = str(config.model.get("camera_model", "pinhole"))
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )

    remap = args.camera_set == "remap"
    source_camera = args.source_camera if remap else args.front_camera
    if remap and "fish" not in source_camera:
        raise ValueError(
            f"--camera-set remap resamples the 195 degree lcam_fish frames, so --source-camera has "
            f"to be a fisheye camera, got {source_camera!r}"
        )
    cameras = test_cameras() if remap else [AugCamera("pinhole", float(args.front_fov), None)]
    names = [camera_name(c) for c in cameras] if remap else ["front_" + args.front_camera]
    if args.cameras:
        wanted = {part.strip() for part in args.cameras.split(",") if part.strip()}
        unknown = wanted - set(names)
        if unknown:
            raise ValueError(f"unknown camera names {sorted(unknown)}, available {names}")
        keep = [i for i, name in enumerate(names) if name in wanted]
        cameras = [cameras[i] for i in keep]
        names = [names[i] for i in keep]

    dataset = build_frame_source(config, args.data, source_camera, args.span)
    num_views = int(config.training.num_views)
    lengths = [len(record.frame_entries) for record in dataset.trajectories]
    samples = select_samples(lengths, args.samples, num_views, args.span, args.seed)
    model = load_model(config, args.checkpoint, device)

    amp_dtype = torch.bfloat16 if str(config.training.get("amp_dtype", "bf16")).lower() == "bf16" else torch.float16
    amp_enabled = bool(config.training.get("use_amp", True)) and device.type == "cuda"
    size = int(config.model.image_tokenizer.image_size)

    out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    records: Dict[str, List[Dict[str, float]]] = {name: [] for name in names}
    panels: Dict[str, List[Tuple[str, torch.Tensor, torch.Tensor]]] = {}

    for sample_number, sample in enumerate(samples):
        if remap:
            frames, gt_c2w, index = sample_batch(dataset, sample, device)
        else:
            images_front, intr_px_front, gt_c2w, index = front_batch(dataset, sample, device)
        for camera, name in zip(cameras, names):
            if remap:
                images, valid, intr_px, _ = remap_views(frames, camera, size, args.supersample)
                images = images.to(device)
                valid = valid[:1]
            else:
                images, valid, intr_px = images_front, None, intr_px_front
            # From the images, not from the config: ERayZer renders at the resolution it is fed, and
            # the resize of the real pinhole frames is what finally decides that resolution.
            height = width = int(images.shape[-1])
            mask = scoring_mask(valid, intr_px, height, width, args.mask)
            result = run_model(
                model,
                images,
                gt_c2w,
                index,
                None if (args.no_data_mask or valid is None) else valid,
                device,
                amp_dtype,
                amp_enabled,
            )
            records[name].append(score_sample(result, mask, camera_model, gt_c2w))
            if sample_number == 0:
                panels.setdefault(camera.model, []).append(
                    (name, result.target_image[0, 0].float(), result.render[0, 0].float())
                )
            print(
                f"[{sample_number + 1}/{len(samples)}] {name:<18} "
                f"fov {records[name][-1]['fov_pred']:7.2f} true {camera.fov_deg:7.2f} "
                f"psnr {records[name][-1]['psnr']:6.2f}",
                flush=True,
            )

    rows = [summarise_camera(camera, name, records[name]) for camera, name in zip(cameras, names)]
    write_csv(os.path.join(out_dir, "cameras.csv"), rows)
    header = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "data": args.data,
        "camera_model": camera_model,
        "camera_set": args.camera_set,
        "samples": len(samples),
        "seed": args.seed,
        "mask": args.mask,
        "image_size": size,
        "source": f"{source_camera} at {FISH_FOV_DEG} deg" if remap else source_camera,
    }
    text = summary_text(
        rows,
        header,
        args.baseline_csv or None,
        [
            f"{dataset.trajectories[sample.trajectory].name} frames {list(sample.frames)}"
            for sample in samples
        ],
    )
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as handle:
        handle.write(text)
    for family, pairs in panels.items():
        family_panel(pairs, os.path.join(out_dir, f"panel_{family}.png"))
    print(text)
    print(f"written to {out_dir}")


if __name__ == "__main__":
    main()
