"""Checkpoints, reconstructions and Gaussian scenes for the viewer.

A viewer session holds at most one model and one scene. The model comes from a checkpoint under
`checkpoints/pretrained` or `checkpoints/runs/<run>/`, and a run brings its own `config.yaml`, which
is where the camera settings of that training live: a fisheye or UCM run must not be loaded as a
pinhole. The scene is the Gaussian cloud of one forward pass, kept on the GPU so that every frame is
only a render call.

For development without the model there is `make_synthetic_scene`, and `save_scene` / `load_scene`
carry a real reconstruction from the cluster to a smaller GPU.
"""

from __future__ import annotations

import importlib
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import yaml
from easydict import EasyDict as edict

from erayzer_core.model.gaussians_renderer import RGB2SH, GaussianModel

CAMERA_KEYS = ("camera_model", "renderer", "geer_tile_mode", "near_plane", "alpha_init", "f_scale")
FALLBACK_CONFIG = "config/erayzer_train_tartanair.yaml"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Weights that only the original E-RayZer training carried. Its perceptual loss lived inside the
# model as `loss_computer.perceptual_loss_module`, a VGG 19 with sixteen convolutions, so
# `erayzer_multi.pt` holds 32 tensors that no model in this repository builds. Those are the
# "32 unexpected keys" the pretrained load used to report. Dropping them up front leaves a load
# that says nothing when it went well and names real architecture mismatches when it did not.
TRAINING_ONLY_PREFIXES = ("loss_computer.",)


@dataclass
class CheckpointEntry:
    id: str
    label: str
    ckpt_path: str
    config_path: Optional[str]
    camera_settings: Dict[str, Any] = field(default_factory=dict)
    warning: str = ""


@dataclass
class EnvEntry:
    name: str
    path: str
    split: str  # "test" or "train"


@dataclass
class ModelBundle:
    model: Any
    config: edict
    ckpt_id: str
    amp_dtype: Optional[torch.dtype]


@dataclass
class Scene:
    pc: GaussianModel
    sh_degree: int
    c2w_input: torch.Tensor  # [v, 4, 4]
    fxfycxcy_input: torch.Tensor  # [v, 4], UCM [v, 5], normalized by the image size
    camera_model: str
    near_plane: float
    tile_mode: str
    center: torch.Tensor  # [3]
    radius: float
    name: str
    source: str
    num_gaussians: int = 0
    alpha_init: float = 0.5
    input_hw: tuple = (256, 256)  # the image size the input intrinsics are normalized by


# ---------------------------------------------------------------------------------------------
# configs
# ---------------------------------------------------------------------------------------------


def to_edict(value: Any) -> Any:
    if isinstance(value, dict):
        return edict({k: to_edict(v) for k, v in value.items()})
    if isinstance(value, list):
        return [to_edict(v) for v in value]
    return value


def read_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def camera_settings_of(config: Dict[str, Any]) -> Dict[str, Any]:
    """The camera relevant part of a training config, with the defaults the training itself uses."""
    model = dict(config.get("model") or {})
    defaults = {
        "camera_model": "pinhole",
        "renderer": "gsplat",
        "geer_tile_mode": "pbf",
        "near_plane": 0.2,
        "alpha_init": 0.5,
        "f_scale": 1.0,
    }
    return {key: model.get(key, defaults[key]) for key in CAMERA_KEYS}


# ---------------------------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------------------------


def _pick_checkpoint_file(run_dir: str) -> Optional[str]:
    """`last.pt` if the run wrote one, otherwise the highest numbered `ckpt_*.pt`.

    `train.py` names them `ckpt_{step:016d}.pt`, so the zero padding makes the lexicographic order
    the step order.
    """
    last = os.path.join(run_dir, "last.pt")
    if os.path.isfile(last):
        return last
    numbered = sorted(n for n in os.listdir(run_dir) if n.startswith("ckpt_") and n.endswith(".pt"))
    return os.path.join(run_dir, numbered[-1]) if numbered else None


def list_checkpoints(pretrained_path: Optional[str], runs_dir: Optional[str]) -> List[CheckpointEntry]:
    """Every checkpoint the viewer offers, the pretrained weights first and then the runs by name."""
    entries: List[CheckpointEntry] = []
    if pretrained_path and os.path.isfile(pretrained_path):
        entries.append(
            CheckpointEntry(
                id="pretrained",
                label=f"{os.path.basename(pretrained_path)} (pinhole)",
                ckpt_path=pretrained_path,
                config_path=None,
                camera_settings=camera_settings_of({}),
            )
        )

    if runs_dir and os.path.isdir(runs_dir):
        for name in sorted(os.listdir(runs_dir)):
            run_dir = os.path.join(runs_dir, name)
            if not os.path.isdir(run_dir):
                continue
            ckpt = _pick_checkpoint_file(run_dir)
            if ckpt is None:
                continue
            config_path = os.path.join(run_dir, "config.yaml")
            warning = ""
            if os.path.isfile(config_path):
                try:
                    settings = camera_settings_of(read_config(config_path))
                except Exception as error:  # a half written config must not hide the whole run
                    settings, config_path, warning = camera_settings_of({}), None, f"unreadable config.yaml ({error})"
            else:
                settings, config_path = camera_settings_of({}), None
                warning = "no config.yaml, camera settings guessed"
            label = (
                f"runs/{name} ({settings['camera_model']}, {settings['renderer']}, {settings['geer_tile_mode']})"
                f"{' [' + warning + ']' if warning else ''}"
            )
            entries.append(
                CheckpointEntry(
                    id=f"runs/{name}",
                    label=label,
                    ckpt_path=ckpt,
                    config_path=config_path,
                    camera_settings=settings,
                    warning=warning,
                )
            )
    return entries


def list_envs(data_root: Optional[str], camera_name: str = "lcam_fish") -> List[EnvEntry]:
    """TartanAir environments under `data_root`, the test split first.

    An environment counts as training data when `train_envs.txt` next to it names it, which is the
    same list the training uses.
    """
    if not data_root or not os.path.isdir(data_root):
        return []
    train_names = set()
    train_list = os.path.join(data_root, "train_envs.txt")
    if os.path.isfile(train_list):
        with open(train_list, "r", encoding="utf-8") as handle:
            train_names = {line.strip() for line in handle if line.strip()}

    entries: List[EnvEntry] = []
    for name in sorted(os.listdir(data_root)):
        archive = os.path.join(data_root, name, "Data_hard", f"image_{camera_name}.zip")
        if os.path.isfile(archive):
            entries.append(EnvEntry(name=name, path=archive, split="train" if name in train_names else "test"))
    entries.sort(key=lambda entry: (entry.split == "train", entry.name))
    return entries


# ---------------------------------------------------------------------------------------------
# model and reconstruction
# ---------------------------------------------------------------------------------------------


def _prepare_config(config: edict) -> edict:
    """Switch a training config into the single sample inference the viewer needs."""
    config.inference = False
    config.pop("evaluation", None)  # would send the model into its video rendering path
    config.create_visual = False
    config.training.random_split = False
    config.training.num_workers = 0
    config.training.batch_size_per_gpu = 1
    config.data_repeat = 1
    return config


def model_state_dict(checkpoint: Any) -> tuple:
    """The model weights of a checkpoint, ready for `load_state_dict`, plus the names dropped.

    Three checkpoint shapes reach the viewer: a bare state dict, one wrapped in `{"model": ...}` by
    `train.py`, and the pretrained `erayzer_multi.pt`, which is the second kind plus the weights of
    the perceptual loss of its training. Distributed runs prefix every name with `module.`.
    """
    raw = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    state, dropped = {}, []
    for key, value in raw.items():
        name = key.removeprefix("module.")
        if name.startswith(TRAINING_ONLY_PREFIXES):
            dropped.append(name)
        else:
            state[name] = value
    return state, dropped


def load_model(entry: CheckpointEntry, device: torch.device, repo_root: Optional[str] = None) -> ModelBundle:
    """Load a checkpoint with the config of its own run, ready for one forward pass."""
    checkpoint = torch.load(entry.ckpt_path, map_location="cpu")
    if entry.config_path and os.path.isfile(entry.config_path):
        data = read_config(entry.config_path)
    elif isinstance(checkpoint.get("config"), dict):
        data = dict(checkpoint["config"])
    else:
        fallback = os.path.join(repo_root or os.getcwd(), FALLBACK_CONFIG)
        data = read_config(fallback)
    config = _prepare_config(to_edict(dict(data)))

    amp_name = str(config.training.get("amp_dtype", "bf16")).lower()
    amp_dtype = torch.bfloat16 if amp_name == "bf16" else torch.float16
    # xformers runs its attention in bfloat16 only from compute capability 8.0, the RTX 30xx
    # generation. `torch.cuda.is_bf16_supported` is more generous than that and says yes on a
    # Turing card, where the forward pass then dies inside xformers, so ask the card itself.
    if device.type == "cuda" and amp_dtype is torch.bfloat16 and torch.cuda.get_device_capability(device)[0] < 8:
        name = torch.cuda.get_device_name(device)
        major, minor = torch.cuda.get_device_capability(device)
        raise RuntimeError(
            f"{name} has compute capability {major}.{minor}, and E-RayZer needs 8.0 for its bfloat16 attention, "
            "so the forward pass cannot run here. Reconstruct on the cluster and start the viewer with "
            "--scene <file>, or use --scene synthetic."
        )
    if not (config.training.get("use_amp", True) and device.type == "cuda"):
        amp_dtype = None

    module_name, class_name = config.model.class_name.rsplit(".", 1)
    model = importlib.import_module(module_name).__dict__[class_name](config).to(device)
    state_dict, dropped = model_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    report = f"Loaded {entry.ckpt_path}, {len(state_dict)} tensors"
    if dropped:
        report += f", {len(dropped)} training only weights skipped"
    if missing:
        report += f", MISSING {len(missing)} ({', '.join(missing[:4])}...)"
    if unexpected:
        report += f", UNEXPECTED {len(unexpected)} ({', '.join(unexpected[:4])}...)"
    print(report, flush=True)
    del checkpoint, state_dict
    return ModelBundle(model=model.eval(), config=config, ckpt_id=entry.id, amp_dtype=amp_dtype)


def _scene_extent(xyz: torch.Tensor):
    """Centre and radius of a Gaussian cloud, robust against the few points that fly off."""
    center = xyz.median(dim=0).values
    distance = (xyz - center).norm(dim=-1)
    radius = float(torch.quantile(distance.float(), 0.9))
    return center, max(radius, 1e-3)


def _scene_from_result(result, config: edict, name: str, source: str) -> Scene:
    camera_model = config.model.get("camera_model", "pinhole")
    pc = result.gaussians[0]  # a detached float GaussianModel, already on the device
    center, radius = _scene_extent(pc.get_xyz)
    return Scene(
        pc=pc,
        sh_degree=int(config.model.gaussians.sh_degree),
        c2w_input=result.c2w_input[0].detach().float(),
        fxfycxcy_input=result.fxfycxcy_input[0].detach().float(),
        camera_model=camera_model,
        near_plane=float(config.model.get("near_plane", 0.2)),
        tile_mode=config.model.get("geer_tile_mode", "pbf"),
        center=center,
        radius=radius,
        name=name,
        source=source,
        num_gaussians=int(pc.get_xyz.shape[0]),
        alpha_init=float(config.model.get("alpha_init", 0.5)),
        input_hw=(int(config.model.target_image.height), int(config.model.target_image.width)),
    )


def _forward(bundle: ModelBundle, batch: Dict[str, Any], device: torch.device):
    from contextlib import nullcontext

    context = (
        torch.autocast(device_type=device.type, dtype=bundle.amp_dtype)
        if device.type == "cuda" and bundle.amp_dtype is not None
        else nullcontext()
    )
    with torch.no_grad(), context:
        return bundle.model(batch)


def reconstruct_upload(bundle: ModelBundle, image_paths: List[str], device: torch.device) -> Scene:
    """One reconstruction from uploaded images, the batch built exactly as the Gradio demo builds it."""
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF
    from PIL import Image

    if not image_paths:
        raise ValueError("no images")
    num_views = int(bundle.config.training.num_views)
    paths = sorted(image_paths, key=os.path.basename)
    if len(paths) > num_views:
        print(f"Warning: {len(paths)} images given, the model takes {num_views}, using the first ones", flush=True)
        paths = paths[:num_views]

    size = int(bundle.config.model.image_tokenizer.image_size)
    transform = T.Compose(
        [
            T.Lambda(lambda img: TF.center_crop(img, min(img.size))),
            T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.ToTensor(),
        ]
    )
    images = torch.stack([transform(Image.open(path).convert("RGB")) for path in paths], dim=0)[None]
    batch = {
        "image": images.to(device),
        "fxfycxcy": torch.tensor([[[1.0, 1.0, 0.5, 0.5]] * len(paths)], dtype=torch.float32, device=device),
        "num_real_views": len(paths),
    }
    result = _forward(bundle, batch, device)
    scene = _scene_from_result(result, bundle.config, f"upload ({len(paths)} images)", "upload")
    del result, batch, images
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return scene


def reconstruct_tartanair(bundle: ModelBundle, env: EnvEntry, index: int, device: torch.device) -> Scene:
    """One reconstruction from a TartanAir sample, loaded through the dataset of the training config."""
    from torch.utils.data import DataLoader

    from training.datasets import build_dataset

    bundle.config.dataset.eval_path = env.path
    dataset = build_dataset(bundle.config, split="eval")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    batch = None
    for position, candidate in enumerate(loader):
        if position == index:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in candidate.items()}
            break
    if batch is None:
        raise IndexError(f"sample {index} is past the end of {env.name} with {len(dataset)} samples")

    result = _forward(bundle, batch, device)
    scene = _scene_from_result(result, bundle.config, f"{env.name} #{index}", "tartanair")
    del result, batch, loader, dataset
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return scene


# ---------------------------------------------------------------------------------------------
# synthetic scene
# ---------------------------------------------------------------------------------------------


def _hsv_to_rgb(hue: torch.Tensor) -> torch.Tensor:
    """Fully saturated colours for hue in [0, 1), [n] to [n, 3]."""
    h = (hue % 1.0) * 6.0
    i = torch.floor(h)
    f = h - i
    zero, one = torch.zeros_like(h), torch.ones_like(h)
    table = torch.stack(
        [
            torch.stack([one, f, zero], -1),
            torch.stack([1 - f, one, zero], -1),
            torch.stack([zero, one, f], -1),
            torch.stack([zero, 1 - f, one], -1),
            torch.stack([f, zero, one], -1),
            torch.stack([one, zero, 1 - f], -1),
        ],
        dim=0,
    )  # [6, n, 3]
    return table.gather(0, i.long().clamp(0, 5)[None, :, None].expand(1, -1, 3))[0]


def make_synthetic_scene(device: torch.device, seed: int = 0) -> Scene:
    """A deterministic scene for development without a model, about twenty thousand Gaussians.

    It holds a chequered floor, three coloured blobs and a ring of sixty four coloured columns around
    the origin. The ring is what makes rendering beyond ninety degrees visible: from the centre a
    pinhole shows a slice of it, a 195 degree fisheye shows more than half of it at once.
    """
    generator = torch.Generator().manual_seed(seed)
    points, colors, scales = [], [], []

    # Chequered floor at y = 1.2, which is below the origin in the OpenCV frame the viewer uses.
    side = 96
    axis = torch.linspace(-4.0, 4.0, side)
    gz, gx = torch.meshgrid(axis, axis, indexing="ij")
    floor = torch.stack([gx.reshape(-1), torch.full((side * side,), 1.2), gz.reshape(-1)], dim=-1)
    tile = ((torch.arange(side)[:, None] // 6 + torch.arange(side)[None, :] // 6) % 2).reshape(-1).float()
    points.append(floor)
    colors.append(torch.lerp(torch.full((side * side, 3), 0.18), torch.full((side * side, 3), 0.80), tile[:, None]))
    scales.append(torch.full((side * side,), 0.05))

    # Three blobs near the centre, so that there is something to orbit around.
    blob_centers = [(-1.5, 0.55, 1.6), (1.6, 0.25, 1.9), (0.0, -0.45, -1.7)]
    blob_colors = [(0.90, 0.25, 0.20), (0.20, 0.80, 0.35), (0.25, 0.45, 0.95)]
    for (cx, cy, cz), rgb in zip(blob_centers, blob_colors):
        offsets = torch.randn(2500, 3, generator=generator) * 0.22
        points.append(offsets + torch.tensor([cx, cy, cz]))
        colors.append(torch.tensor(rgb)[None].expand(2500, -1).clone())
        scales.append(torch.full((2500,), 0.045))

    # The azimuth ring, one column every 5.6 degrees, coloured by its azimuth.
    columns, per_column = 64, 48
    azimuth = torch.arange(columns, dtype=torch.float32) / columns * 2.0 * math.pi
    height = torch.linspace(1.15, -0.65, per_column)
    ring_x = (4.0 * torch.cos(azimuth))[:, None].expand(-1, per_column).reshape(-1)
    ring_z = (4.0 * torch.sin(azimuth))[:, None].expand(-1, per_column).reshape(-1)
    ring_y = height[None, :].expand(columns, -1).reshape(-1)
    points.append(torch.stack([ring_x, ring_y, ring_z], dim=-1))
    colors.append(_hsv_to_rgb(azimuth / (2.0 * math.pi))[:, None, :].expand(-1, per_column, -1).reshape(-1, 3))
    scales.append(torch.full((columns * per_column,), 0.07))

    xyz = torch.cat(points).to(device)
    rgb = torch.cat(colors).clamp(0.0, 1.0).to(device)
    scale = torch.cat(scales).to(device)
    n = xyz.shape[0]

    # `set_data` takes raw values: scaling before exp, opacity before sigmoid, colour as SH.
    pc = GaussianModel(0).set_data(
        xyz,
        RGB2SH(rgb)[:, None, :].contiguous(),
        torch.log(scale)[:, None].expand(-1, 3).contiguous(),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(n, -1).contiguous(),
        torch.full((n, 1), math.log(0.92 / 0.08), device=device),
    )

    # Five stand in input cameras on a circle around the centre, so that frusta and the jump buttons
    # work without a model.
    c2w = []
    for k in range(5):
        angle = k / 5.0 * 2.0 * math.pi
        position = torch.tensor([2.5 * math.cos(angle), 0.0, 2.5 * math.sin(angle)])
        forward = -position / position.norm()
        # OpenCV: x right, y down, z forward, and in this scene the world y axis points down as well.
        right = torch.linalg.cross(torch.tensor([0.0, 1.0, 0.0]), forward)
        right = right / right.norm()
        down = torch.linalg.cross(forward, right)
        matrix = torch.eye(4)
        matrix[:3, :3] = torch.stack([right, down, forward], dim=1)
        matrix[:3, 3] = position
        c2w.append(matrix)

    center, radius = _scene_extent(xyz)
    return Scene(
        pc=pc,
        sh_degree=0,
        c2w_input=torch.stack(c2w).to(device),
        fxfycxcy_input=torch.tensor([[1.0, 1.0, 0.5, 0.5]], device=device).expand(5, -1).contiguous(),
        camera_model="pinhole",
        near_plane=0.05,
        tile_mode="projection",
        center=center,
        radius=radius,
        name="synthetic",
        source="synthetic",
        num_gaussians=n,
        alpha_init=0.5,
    )


# ---------------------------------------------------------------------------------------------
# saving and loading
# ---------------------------------------------------------------------------------------------


def save_scene(scene: Scene, path: str) -> str:
    """Write a scene as the raw Gaussian tensors plus its camera defaults, readable anywhere."""
    pc = scene.pc
    features = pc.get_features.detach().cpu()
    payload = {
        "xyz": pc._xyz.detach().cpu(),
        "features": features,
        "scaling": pc._scaling.detach().cpu(),
        "rotation": pc._rotation.detach().cpu(),
        "opacity": pc._opacity.detach().cpu(),
        "sh_degree": scene.sh_degree,
        "c2w_input": scene.c2w_input.detach().cpu(),
        "fxfycxcy_input": scene.fxfycxcy_input.detach().cpu(),
        "camera_model": scene.camera_model,
        "near_plane": scene.near_plane,
        "tile_mode": scene.tile_mode,
        "alpha_init": scene.alpha_init,
        "input_hw": list(scene.input_hw),
        "name": scene.name,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    torch.save(payload, path)
    return path


def load_scene(path: str, device: torch.device) -> Scene:
    """Read a scene written by `save_scene`."""
    payload = torch.load(path, map_location=device)
    pc = GaussianModel(int(payload["sh_degree"])).set_data(
        payload["xyz"].float().to(device),
        payload["features"].float().to(device),
        payload["scaling"].float().to(device),
        payload["rotation"].float().to(device),
        payload["opacity"].float().to(device),
    )
    center, radius = _scene_extent(pc.get_xyz)
    return Scene(
        pc=pc,
        sh_degree=int(payload["sh_degree"]),
        c2w_input=payload["c2w_input"].float().to(device),
        fxfycxcy_input=payload["fxfycxcy_input"].float().to(device),
        camera_model=payload.get("camera_model", "pinhole"),
        near_plane=float(payload.get("near_plane", 0.2)),
        tile_mode=payload.get("tile_mode", "projection"),
        center=center,
        radius=radius,
        name=payload.get("name", os.path.basename(path)),
        source="file",
        num_gaussians=int(pc.get_xyz.shape[0]),
        alpha_init=float(payload.get("alpha_init", 0.5)),
        input_hw=tuple(payload.get("input_hw", (256, 256))),
    )
