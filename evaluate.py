from __future__ import annotations

import argparse
import csv
import importlib
import os
from contextlib import nullcontext
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from easydict import EasyDict as edict
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import erayzer_core  # noqa: F401
from erayzer_core.model.utils_gaussian import loss_render_mask, render_mask
from erayzer_core.utils import camera_utils
from erayzer_core.utils.camera_eval import evaluate_camera_pose_metrics
from erayzer_core.utils.image_metrics import ssim_map
from training.datasets import build_dataset as build_config_dataset

# Weights the released erayzer_multi.pt carries that the published model code does not define: the
# VGG 19 of the perceptual loss, 16 convolutions with a weight and a bias each. Loading the file
# through the training path therefore reported "Unexpected keys: 32"; they are dropped here instead,
# so a real mismatch between checkpoint and model stays visible in that number.
TRAINING_ONLY_PREFIXES = ("loss_computer.",)
# Metric columns the evaluation has always written, in their original order and format.
BASE_FIELDS = ["basename", "mse", "psnr", "RPA@5", "RPA@15", "RPA@30"]
EXTRA_FIELDS = ["mse_unmasked", "psnr_unmasked", "ssim", "ssim_unmasked", "mask_fraction"]
FIELD_FORMATS = {
    "mse": "{:.8f}",
    "psnr": "{:.4f}",
    "RPA@5": "{:.6f}",
    "RPA@15": "{:.6f}",
    "RPA@30": "{:.6f}",
    "mse_unmasked": "{:.8f}",
    "psnr_unmasked": "{:.4f}",
    "ssim": "{:.6f}",
    "ssim_unmasked": "{:.6f}",
    "mask_fraction": "{:.6f}",
}


def _to_edict(value: Any) -> Any:
    if isinstance(value, dict):
        return edict({k: _to_edict(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_edict(v) for v in value]
    return value


def _parse_value(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _set_nested(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = config
    parts = dotted_key.split(".")
    for key in parts[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[parts[-1]] = value


def load_config(path: str, overrides: Iterable[Tuple[str, str]]) -> edict:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    for key, value in overrides:
        _set_nested(data, key, _parse_value(value))
    return _to_edict(data)


def setup_distributed() -> Tuple[torch.device, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    return device, rank, local_rank, world_size


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def extract_state_dict(checkpoint: Any) -> Dict[str, Any]:
    """Model weights of a training checkpoint or of the released erayzer_multi.pt.

    train.py stores them under "model" and prefixes every key with "module." when it trained under
    DDP; a bare state dict is taken as is. The released checkpoint has the same layout but also
    carries the perceptual loss network of the original training code under loss_computer., which
    this repository does not define, so those tensors are dropped rather than reported as unexpected.
    """
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                state_dict = value
                break
    cleaned = {}
    for key, value in state_dict.items():
        key = key.removeprefix("module.")
        if key.startswith(TRAINING_ONLY_PREFIXES):
            continue
        cleaned[key] = value
    return cleaned


def build_model(config: edict, device: torch.device, rank: int) -> torch.nn.Module:
    module_name, class_name = config.model.class_name.rsplit(".", 1)
    model_cls = importlib.import_module(module_name).__dict__[class_name]
    model = model_cls(config).to(device)
    ckpt_path = config.evaluation.checkpoint_path
    # On the CPU: a training checkpoint also holds the optimiser state, which is twice the size of
    # the weights and is never needed here, and load_state_dict copies onto the device anyway.
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = extract_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if rank == 0:
        print(f"Loaded checkpoint: {ckpt_path}")
    if missing and rank == 0:
        print(f"Missing keys: {len(missing)} {sorted(missing)[:8]}")
    if unexpected and rank == 0:
        print(f"Unexpected keys: {len(unexpected)} {sorted(unexpected)[:8]}")
    return model


def build_dataloader(config: edict, rank: int, world_size: int) -> DataLoader:
    training = config.training
    dataset = build_config_dataset(config, split="eval")
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    return DataLoader(
        dataset,
        batch_size=int(training.get("batch_size_per_gpu", 1)),
        shuffle=False,
        sampler=sampler,
        num_workers=int(training.get("num_workers", 2)),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _save_tensor_image(tensor: torch.Tensor, path: str) -> None:
    array = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    array = (array * 255.0).round().astype(np.uint8)
    Image.fromarray(array).save(path)


def save_panel(target: torch.Tensor, pred: torch.Tensor, path: str, gap: int = 4) -> None:
    """One image of the ground truth next to the rendering, both [3, h, w], separated by a white bar."""
    left = target.detach().cpu().clamp(0, 1)
    right = pred.detach().cpu().clamp(0, 1)
    bar = torch.ones(3, left.shape[1], gap)
    _save_tensor_image(torch.cat([left, bar, right], dim=2), path)


def evaluation_section(config: edict) -> edict:
    section = config.get("evaluation", None)
    return section if isinstance(section, dict) else edict()


def evaluation_mask(config: edict, fxfycxcy: torch.Tensor, h: int, w: int):
    """Pixels the evaluation scores, [n, 1, h, w], or None for the whole image.

    Without evaluation.mask_override this is what it has always been: nothing for a pinhole model and
    otherwise loss_render_mask, the very mask the training loss used, so a model is scored on the
    pixels it was trained on. With evaluation.mask_override "circle" every model, pinhole included,
    is scored on the circle spanning the image width without the 90 degree cap, which is what makes
    numbers from a pinhole model and from a 195 degree fisheye model comparable at all.
    """
    override = evaluation_section(config).get("mask_override", None)
    override = "" if override is None else str(override).strip().lower()
    camera_model = config.model.get("camera_model", "pinhole")
    if override in {"", "none", "null", "unset", "false"}:
        if camera_model == "pinhole":
            return None
        return loss_render_mask(config, fxfycxcy, h, w)
    if override == "circle":
        return render_mask(fxfycxcy, h, w, camera_model, image_circle=True, exact=True)
    raise ValueError(f"Unknown evaluation.mask_override: {override!r}, use unset or circle")


def image_metrics(pred: torch.Tensor, target: torch.Tensor, mask) -> Dict[str, torch.Tensor]:
    """Per image mse and SSIM with and without the mask, each [b, v].

    pred and target are [b, v, c, h, w] in 0..1, mask is [b * v, 1, h, w] or None.
    """
    b, v, c, h, w = pred.shape
    mse_unmasked = (pred - target).pow(2).flatten(2).mean(dim=2)
    smap = ssim_map(pred.reshape(b * v, c, h, w), target.reshape(b * v, c, h, w))
    ssim_unmasked = smap.flatten(1).mean(1).reshape(b, v)
    if mask is None:
        ones = torch.ones_like(mse_unmasked)
        return {
            "mse": mse_unmasked,
            "mse_unmasked": mse_unmasked,
            "ssim": ssim_unmasked,
            "ssim_unmasked": ssim_unmasked,
            "mask_fraction": ones,
        }
    mask_bv = mask.reshape(b, v, 1, h, w)
    # Unchanged from the version without a mask override, so the default numbers do not move.
    mse = ((pred - target).pow(2) * mask_bv).flatten(2).sum(2) / (mask_bv.flatten(2).sum(2) * c).clamp_min(1.0)
    weights = mask.float().expand_as(smap)
    ssim_masked = ((smap * weights).flatten(1).sum(1) / weights.flatten(1).sum(1).clamp_min(1.0)).reshape(b, v)
    return {
        "mse": mse,
        "mse_unmasked": mse_unmasked,
        "ssim": ssim_masked,
        "ssim_unmasked": ssim_unmasked,
        "mask_fraction": mask_bv.flatten(2).mean(2),
    }


def parse_view_indices(value: Any) -> list[int]:
    """evaluation.sample_views, written as "0,2,4" or as a yaml list, as a list of target view indices."""
    if value is None or value == "" or value is False:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(part) for part in str(value).replace(" ", "").split(",") if part != ""]


def novel_pose_c2ws(c2w_input: torch.Tensor, count: int = 3, lateral_offset: float = 0.3) -> torch.Tensor:
    """`count` camera to world poses that no input view had, [count, 4, 4].

    Each one sits halfway between two neighbouring input poses, rotation by quaternion slerp and
    centre on the straight line, and is then moved sideways along its own right axis by
    lateral_offset times the scene radius, the largest distance of an input camera centre from their
    mean. Halfway and not on a grid of fractions on purpose: a fraction of zero or one would hand
    back an input pose, and the point of these renders is that no input view ever saw them. They
    have no ground truth, they only show what the model does away from its inputs.
    """
    poses = c2w_input.detach().to(torch.float64).cpu().numpy()
    centres = poses[:, :3, 3]
    radius = float(np.linalg.norm(centres - centres.mean(axis=0, keepdims=True), axis=1).max())
    if not np.isfinite(radius) or radius <= 0.0:
        radius = 1.0
    num_input = poses.shape[0]
    novel = []
    for step in range(count):
        if num_input < 2:
            index, fraction = 0, 0.0
        else:  # the segments are spread over the whole chain of input poses
            index = min(int(step * (num_input - 1) / max(count, 1)), num_input - 2)
            fraction = 0.5
        quat_a = camera_utils.quaternion_from_matrix(poses[index][:3, :3])
        quat_b = camera_utils.quaternion_from_matrix(poses[min(index + 1, num_input - 1)][:3, :3])
        rotation = camera_utils.quaternion_matrix(camera_utils.quaternion_slerp(quat_a, quat_b, fraction))[:3, :3]
        centre = (1.0 - fraction) * centres[index] + fraction * centres[min(index + 1, num_input - 1)]
        sign = 1.0 if step % 2 == 0 else -1.0
        pose = np.eye(4)
        pose[:3, :3] = rotation
        pose[:3, 3] = centre + sign * lateral_offset * radius * rotation[:, 0]
        novel.append(pose)
    return torch.from_numpy(np.stack(novel)).to(c2w_input.dtype)


def render_novel_views(
    renderer: torch.nn.Module,
    gaussians: Any,
    c2ws: torch.Tensor,
    fxfycxcy: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Render the Gaussians of one scene from poses [n, 4, 4], returns [n, 3, height, width].

    gaussians is one entry of result.gaussians, which holds the raw values the renderer expects.
    """
    features = gaussians._features_dc
    if gaussians._features_rest is not None:
        features = torch.cat([features, gaussians._features_rest], dim=1)
    device = gaussians._xyz.device
    poses = c2ws.to(device=device, dtype=torch.float32)[None]
    intrinsics = fxfycxcy.to(device=device, dtype=torch.float32).reshape(1, 1, -1).expand(1, poses.shape[1], -1)
    out = renderer(
        gaussians._xyz[None],
        features[None],
        gaussians._scaling[None],
        gaussians._rotation[None],
        gaussians._opacity[None],
        height,
        width,
        C2W=poses,
        fxfycxcy=intrinsics.contiguous(),
    )
    return out.render[0]


def write_summary(out_dir: str, rows: list, totals: Dict[str, Any]) -> None:
    """metrics.txt and summary.csv. The first six lines and the first six columns are unchanged."""
    summary_path = os.path.join(out_dir, "metrics.txt")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(f"mse: {totals['mse']:.8f}\n")
        handle.write(f"psnr: {totals['psnr']:.4f}\n")
        handle.write(f"RPA@5: {totals['RPA@5']:.6f}\n")
        handle.write(f"RPA@15: {totals['RPA@15']:.6f}\n")
        handle.write(f"RPA@30: {totals['RPA@30']:.6f}\n")
        handle.write(f"num_images: {totals['num_images']}\n")
        for field in EXTRA_FIELDS:
            handle.write(f"{field}: {FIELD_FORMATS[field].format(totals[field])}\n")
        handle.write(f"mask: {totals['mask']}\n")

    def formatted(row: Dict[str, Any], basename: str) -> Dict[str, Any]:
        out = {"basename": basename}
        for field in BASE_FIELDS[1:] + EXTRA_FIELDS:
            out[field] = FIELD_FORMATS[field].format(row[field])
        return out

    csv_path = os.path.join(out_dir, "summary.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BASE_FIELDS + EXTRA_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(formatted(row, row["basename"]))
        handle.write("\n")
        writer.writerow(formatted(totals, "average"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--set", "-s", action="append", nargs=2, default=[])
    args = parser.parse_args()

    config = load_config(args.config, args.set)
    config.inference = False
    device, rank, local_rank, world_size = setup_distributed()

    dataloader = build_dataloader(config, rank, world_size)
    model = build_model(config, device, rank)
    if world_size > 1:
        device_ids = [local_rank] if device.type == "cuda" else None
        model = DDP(model, device_ids=device_ids)
    model.eval()

    out_dir = os.path.abspath(config.evaluation.out_dir)
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    amp_dtype = torch.bfloat16 if str(config.training.get("amp_dtype", "bf16")).lower() == "bf16" else torch.float16
    amp_enabled = bool(config.training.get("use_amp", True)) and device.type == "cuda"
    max_batches = int(config.evaluation.get("max_batches", 0))
    save_examples = int(config.evaluation.get("save_examples", 8))
    # Target view indices rendered next to their ground truth, the same ones for every model.
    sample_views = parse_view_indices(config.evaluation.get("sample_views", None))
    # Poses no input view had, rendered without a score because there is no ground truth for them.
    novel_views = int(config.evaluation.get("novel_views", 0))
    novel_offset = float(config.evaluation.get("novel_offset", 0.3))
    base_model = model.module if isinstance(model, DDP) else model

    mse_sum = torch.tensor(0.0, device=device)
    image_count = torch.tensor(0.0, device=device)
    sample_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            ctx = (
                torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled)
                if device.type == "cuda"
                else nullcontext()
            )
            with ctx:
                result = model(batch)

            pred = result.render.clamp(0, 1)
            target = result.get("target_image", batch["image"])
            if pred.shape[-2:] != target.shape[-2:]:
                b, v = target.shape[:2]
                target = F.interpolate(
                    target.reshape(b * v, *target.shape[2:]),
                    size=pred.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).reshape(b, v, target.shape[2], pred.shape[-2], pred.shape[-1])

            b, v, c, h, w = pred.shape
            fxfycxcy = result.fxfycxcy_target.float().reshape(b * v, -1)
            # One mask for training and evaluation, as far out as the renderer in the config reaches,
            # unless evaluation.mask_override asks for the common image circle instead.
            mask = evaluation_mask(config, fxfycxcy, h, w)
            metrics = image_metrics(pred, target, mask)
            mse_per_image = metrics["mse"]
            mse_sum += mse_per_image.sum()
            image_count += mse_per_image.numel()
            psnr_per_image = -10.0 * torch.log10(mse_per_image.clamp_min(1e-8))
            psnr_unmasked_per_image = -10.0 * torch.log10(metrics["mse_unmasked"].clamp_min(1e-8))

            for sample_idx in range(pred.shape[0]):
                scene_id = int(batch["index"][sample_idx, 0, 1].detach().cpu().item())
                sample_mse = float(mse_per_image[sample_idx].mean().detach().cpu().item())
                sample_psnr = float(psnr_per_image[sample_idx].mean().detach().cpu().item())
                rpa = evaluate_camera_pose_metrics(
                    batch["c2w"][sample_idx].detach(),
                    result.c2w[sample_idx].detach(),
                )
                sample_rows.append(
                    {
                        "basename": f"{scene_id:06d}",
                        "mse": sample_mse,
                        "psnr": sample_psnr,
                        **rpa,
                        "mse_unmasked": float(metrics["mse_unmasked"][sample_idx].mean().detach().cpu().item()),
                        "psnr_unmasked": float(psnr_unmasked_per_image[sample_idx].mean().detach().cpu().item()),
                        "ssim": float(metrics["ssim"][sample_idx].mean().detach().cpu().item()),
                        "ssim_unmasked": float(metrics["ssim_unmasked"][sample_idx].mean().detach().cpu().item()),
                        "mask_fraction": float(metrics["mask_fraction"][sample_idx].mean().detach().cpu().item()),
                    }
                )

            if rank == 0 and batch_idx < save_examples:
                sample_dir = os.path.join(out_dir, f"sample_{batch_idx:04d}")
                os.makedirs(sample_dir, exist_ok=True)
                for view_idx in range(pred.shape[1]):
                    _save_tensor_image(pred[0, view_idx], os.path.join(sample_dir, f"pred_{view_idx:02d}.png"))
                    _save_tensor_image(target[0, view_idx], os.path.join(sample_dir, f"target_{view_idx:02d}.png"))
                for view_idx in sample_views:
                    if 0 <= view_idx < pred.shape[1]:
                        save_panel(
                            target[0, view_idx],
                            pred[0, view_idx],
                            os.path.join(sample_dir, f"panel_{view_idx:02d}.png"),
                        )
                if novel_views > 0:
                    poses = novel_pose_c2ws(result.c2w_input[0], novel_views, novel_offset)
                    novel = render_novel_views(
                        base_model.renderer, result.gaussians[0], poses, fxfycxcy[0], h, w
                    )
                    for view_idx in range(novel.shape[0]):
                        _save_tensor_image(novel[view_idx], os.path.join(sample_dir, f"novel_{view_idx:02d}.png"))

            if max_batches > 0 and batch_idx + 1 >= max_batches:
                break

    if world_size > 1:
        dist.all_reduce(mse_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(image_count, op=dist.ReduceOp.SUM)
        gathered_rows: list[list[dict[str, Any]]] = [None for _ in range(world_size)]  # type: ignore[list-item]
        dist.all_gather_object(gathered_rows, sample_rows)
        sample_rows = [row for rows in gathered_rows for row in rows]

    if rank == 0:
        unique_rows = {}
        for row in sample_rows:
            unique_rows.setdefault(row["basename"], row)
        sample_rows = sorted(unique_rows.values(), key=lambda row: row["basename"])
        num_views = int(config.training.get("num_views", 0))
        dedup_image_count = len(sample_rows) * num_views
        raw_mse = (mse_sum / image_count.clamp_min(1)).item()
        raw_psnr = -10.0 * np.log10(max(raw_mse, 1e-8))
        avg_sample_mse = float(np.mean([row["mse"] for row in sample_rows])) if sample_rows else raw_mse
        avg_sample_psnr = float(np.mean([row["psnr"] for row in sample_rows])) if sample_rows else raw_psnr
        mse = avg_sample_mse
        psnr = avg_sample_psnr
        num_images = dedup_image_count if sample_rows else int(image_count.item())
        avg_rpa = {
            key: float(np.mean([row[key] for row in sample_rows])) if sample_rows else 0.0
            for key in ["RPA@5", "RPA@15", "RPA@30"]
        }
        override = str(evaluation_section(config).get("mask_override", "") or "").strip().lower()
        totals = {
            "mse": mse,
            "psnr": psnr,
            **avg_rpa,
            "num_images": num_images,
            **{
                key: float(np.mean([row[key] for row in sample_rows])) if sample_rows else 0.0
                for key in EXTRA_FIELDS
            },
            "mask": "image_circle" if override == "circle" else "model",
        }
        write_summary(out_dir, sample_rows, totals)
        print(f"Evaluation written to {out_dir}")
        print(f"mse={mse:.8f}, psnr={psnr:.4f}, num_images={num_images}")
        print(
            f"psnr_unmasked={totals['psnr_unmasked']:.4f}, ssim={totals['ssim']:.6f}, "
            f"ssim_unmasked={totals['ssim_unmasked']:.6f}, mask={totals['mask']}"
        )

    cleanup_distributed()


if __name__ == "__main__":
    main()
