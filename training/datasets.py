from __future__ import annotations

from typing import Any

from easydict import EasyDict as edict
from torch.utils.data import Dataset

from training.camera_augmentation import DEFAULT_SUPERSAMPLE
from training.dl3dv_dataset import DL3DVDataset
from training.line_detection import DEFAULT_MAX_COUNT, DEFAULT_MIN_LENGTH, DEFAULT_NUM_SAMPLES
from training.tartanair_dataset import TartanAirDataset


def _get_section(config: edict, name: str) -> edict:
    value = config.get(name, edict())
    if isinstance(value, dict):
        return edict(value)
    return value


def _get_dataset_name(config: edict) -> str:
    dataset = config.get("dataset", None)
    if isinstance(dataset, str):
        return dataset
    if isinstance(dataset, dict):
        name = dataset.get("name")
        if name:
            return str(name)
    training = _get_section(config, "training")
    if training.get("dataset"):
        return str(training.dataset)
    return ""


def _get_manifest_path(config: edict, split: str) -> str:
    dataset = _get_section(config, "dataset")
    split_key = f"{split}_manifest_path"
    manifest_path = dataset.get(split_key) or dataset.get("manifest_path")
    if manifest_path:
        return str(manifest_path)

    raise ValueError(
        f"config.dataset.{split_key} or config.dataset.manifest_path is required"
    )


def _get_tartanair_path(config: edict, split: str) -> str:
    dataset = _get_section(config, "dataset")
    for key in (f"{split}_path", f"{split}_manifest_path", "data_path", "manifest_path"):
        value = dataset.get(key)
        if value:
            return str(value)
    raise ValueError(
        f"config.dataset.{split}_path or config.dataset.data_path is required for tartanair"
    )


def build_dataset(config: edict, *, split: str) -> Dataset[Any]:
    dataset_name = _get_dataset_name(config)
    dataset = _get_section(config, "dataset")
    training = _get_section(config, "training")

    if dataset_name == "dl3dv":
        return DL3DVDataset(
            _get_manifest_path(config, split),
            image_size=int(config.model.image_tokenizer.image_size),
            patch_size=int(config.model.image_tokenizer.patch_size),
            num_views=int(training.num_views),
            num_input_views=int(training.num_input_views),
            view_selector=dict(training.view_selector),
            covisibility_root=dataset.get("covisibility_root"),
            feature_similarity_root=dataset.get("feature_similarity_root"),
            center_crop=bool(training.get("center_crop", True)),
            random_shuffle=bool(training.get("random_shuffle", False)),
            max_resample_tries=int(training.get("max_resample_tries", 100)),
            data_repeat=int(config.get("data_repeat", 1)),
            is_eval=split == "eval",
        )

    if dataset_name == "tartanair":
        return TartanAirDataset(
            _get_tartanair_path(config, split),
            image_size=int(config.model.image_tokenizer.image_size),
            patch_size=int(config.model.image_tokenizer.patch_size),
            num_views=int(training.num_views),
            num_input_views=int(training.num_input_views),
            view_selector=dict(training.view_selector),
            camera_name=dataset.get("camera_name"),
            center_crop=bool(training.get("center_crop", True)),
            random_shuffle=bool(training.get("random_shuffle", False)),
            max_resample_tries=int(training.get("max_resample_tries", 100)),
            data_repeat=int(config.get("data_repeat", 1)),
            is_eval=split == "eval",
            camera_augmentation=bool(dataset.get("camera_augmentation", False)),
            camera_aug_mix=dataset.get("camera_aug_mix"),
            camera_aug_supersample=int(dataset.get("camera_aug_supersample", DEFAULT_SUPERSAMPLE)),
            detect_lines=bool(dataset.get("detect_lines", False)),
            line_detector=str(dataset.get("line_detector", "auto")),
            line_min_length=float(dataset.get("line_min_length", DEFAULT_MIN_LENGTH)),
            line_max_count=int(dataset.get("line_max_count", DEFAULT_MAX_COUNT)),
            line_samples=int(dataset.get("line_samples", DEFAULT_NUM_SAMPLES)),
            line_chain=bool(dataset.get("line_chain", True)),
        )

    raise ValueError(f"Unsupported config.dataset.name: {dataset_name!r}")
