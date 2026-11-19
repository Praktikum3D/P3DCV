from __future__ import annotations

import io
import os
import random
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# TartanAir NED body axes (x forward, y right, z down) -> OpenCV camera axes (x right, y down, z forward).
_R_CV_FROM_NED = np.array(
    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
    dtype=np.float64,
)


@dataclass(frozen=True)
class _TrajectoryRecord:
    name: str
    archive_path: Optional[str]
    frame_entries: Tuple[str, ...]
    pose_entry: Optional[str]


def _quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = q / max(float(np.linalg.norm(q)), 1e-12)
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _ned_poses_to_c2w(poses: np.ndarray) -> np.ndarray:
    c2w = np.broadcast_to(np.eye(4, dtype=np.float32), (len(poses), 4, 4)).copy()
    for i, row in enumerate(poses):
        rot = _quat_xyzw_to_matrix(row[3:7]) @ _R_CV_FROM_NED.T
        c2w[i, :3, :3] = rot.astype(np.float32)
        c2w[i, :3, 3] = row[:3].astype(np.float32)
    return c2w


def _read_manifest_lines(path: Path) -> List[Path]:
    entries: List[Path] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            item = Path(line)
            if not item.is_absolute():
                item = (path.parent / item).resolve()
            entries.append(item)
    return entries


def _index_zip_archive(archive_path: Path) -> List[_TrajectoryRecord]:
    frames: Dict[str, List[str]] = defaultdict(list)
    poses: Dict[str, str] = {}
    with zipfile.ZipFile(archive_path) as zf:
        for name in zf.namelist():
            parts = Path(name).parts
            traj = next((p for p in parts if p.startswith("P") and p[1:].isdigit()), None)
            if traj is None:
                continue
            if name.endswith(".png"):
                frames[traj].append(name)
            elif Path(name).name.startswith("pose_") and name.endswith(".txt"):
                poses[traj] = name

    records: List[_TrajectoryRecord] = []
    prefix = f"{archive_path.parent.parent.name}/{archive_path.parent.name}/{archive_path.stem}"
    for traj in sorted(frames):
        records.append(
            _TrajectoryRecord(
                name=f"{prefix}/{traj}",
                archive_path=str(archive_path),
                frame_entries=tuple(sorted(frames[traj])),
                pose_entry=poses.get(traj),
            )
        )
    return records


def _index_unpacked_dir(root: Path, camera_name: Optional[str]) -> List[_TrajectoryRecord]:
    records: List[_TrajectoryRecord] = []
    img_dir_pattern = f"image_{camera_name}" if camera_name else "image_*"
    for img_dir in sorted(root.rglob(img_dir_pattern)):
        if not img_dir.is_dir():
            continue
        pngs = sorted(str(p) for p in img_dir.glob("*.png"))
        if not pngs:
            continue
        traj_dir = img_dir.parent
        pose_candidates = sorted(traj_dir.glob("pose_*.txt"))
        pose_path = str(pose_candidates[0]) if pose_candidates else None
        records.append(
            _TrajectoryRecord(
                name=str(traj_dir.relative_to(root)) if traj_dir != root else traj_dir.name,
                archive_path=None,
                frame_entries=tuple(pngs),
                pose_entry=pose_path,
            )
        )
    return records


def _discover_trajectories(
    data_path: str,
    *,
    camera_name: Optional[str] = None,
) -> List[_TrajectoryRecord]:
    root = Path(data_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"TartanAir data path not found: {root}")

    targets: List[Path]
    if root.is_file() and root.suffix == ".txt":
        targets = _read_manifest_lines(root)
    else:
        targets = [root]

    records: List[_TrajectoryRecord] = []
    expected_zip = f"image_{camera_name}.zip" if camera_name else None

    for target in targets:
        if target.is_file() and target.suffix == ".zip":
            if expected_zip and target.name != expected_zip:
                continue
            records.extend(_index_zip_archive(target))
        elif target.is_dir():
            zip_pattern = expected_zip or "image_*.zip"
            zips = sorted(target.rglob(zip_pattern))
            if zips:
                for zp in zips:
                    records.extend(_index_zip_archive(zp))
            else:
                records.extend(_index_unpacked_dir(target, camera_name))
        else:
            raise ValueError(f"Unsupported TartanAir entry: {target}")

    if not records:
        raise ValueError(f"No TartanAir trajectories found under {root} (camera_name={camera_name!r})")
    return records


class TartanAirDataset(Dataset):
    """PyTorch dataset for TartanAir V2 trajectories (pinhole or fisheye).

    Reads directly from zipped `image_<camera>.zip` archives so archives can stay
    packed on the cluster. Also accepts a directory of archives, an unpacked
    trajectory tree, or a text manifest listing paths.
    """

    def __init__(
        self,
        dataset_path: str,
        *,
        image_size: int = 256,
        patch_size: int = 16,
        num_views: int = 10,
        num_input_views: int = 5,
        view_selector: Optional[Dict[str, Any]] = None,
        camera_name: Optional[str] = None,
        center_crop: bool = True,
        random_shuffle: bool = False,
        max_resample_tries: int = 100,
        normalize_intrinsics: bool = True,
        data_repeat: int = 1,
        is_eval: bool = False,
    ) -> None:
        self.dataset_path = os.path.abspath(os.path.expanduser(dataset_path))
        self.camera_name = camera_name
        self.trajectories = _discover_trajectories(self.dataset_path, camera_name=camera_name)

        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.num_views = int(num_views)
        self.num_input_views = int(num_input_views)
        self.view_selector = dict(view_selector or {"type": "two_frame", "min_frame_dist": 16, "max_frame_dist": 48})
        self.center_crop = bool(center_crop)
        self.random_shuffle = bool(random_shuffle)
        self.max_resample_tries = max(1, int(max_resample_tries))
        self.normalize_intrinsics = bool(normalize_intrinsics)
        self.data_repeat = max(1, int(data_repeat))
        self.is_eval = bool(is_eval)
        self.current_iteration = 0

        # Open ZipFile handles lazily per worker process (ZipFile is not fork/pickle safe).
        self._zip_cache: Dict[Tuple[int, str], zipfile.ZipFile] = {}
        self._pose_cache: Dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.trajectories) * self.data_repeat

    def update_iteration(self, iteration: int) -> None:
        self.current_iteration = int(iteration)

    def _get_zip(self, archive_path: str) -> zipfile.ZipFile:
        key = (os.getpid(), archive_path)
        zf = self._zip_cache.get(key)
        if zf is None:
            zf = zipfile.ZipFile(archive_path, "r")
            self._zip_cache[key] = zf
        return zf

    def _load_poses(self, traj_idx: int) -> np.ndarray:
        cached = self._pose_cache.get(traj_idx)
        if cached is not None:
            return cached

        record = self.trajectories[traj_idx]
        if record.pose_entry is None:
            c2w = np.broadcast_to(
                np.eye(4, dtype=np.float32), (len(record.frame_entries), 4, 4)
            ).copy()
        elif record.archive_path is not None:
            raw = self._get_zip(record.archive_path).read(record.pose_entry)
            poses = np.loadtxt(io.BytesIO(raw)).reshape(-1, 7)
            c2w = _ned_poses_to_c2w(poses)
        else:
            poses = np.loadtxt(record.pose_entry).reshape(-1, 7)
            c2w = _ned_poses_to_c2w(poses)

        self._pose_cache[traj_idx] = c2w
        return c2w

    def _load_image_and_intrinsics(
        self, record: _TrajectoryRecord, frame_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        entry = record.frame_entries[frame_idx]
        if record.archive_path is not None:
            raw = self._get_zip(record.archive_path).read(entry)
            image = Image.open(io.BytesIO(raw)).convert("RGB")
        else:
            image = Image.open(entry).convert("RGB")

        width, height = image.size
        # TartanAir-V2 pinhole (90 deg FoV): fx = fy = W/2, cx = W/2, cy = H/2.
        # Kept as nominal values for fisheye until a non-pinhole camera head is introduced.
        fx = width * 0.5
        fy = height * 0.5
        cx = width * 0.5
        cy = height * 0.5

        if height < width:
            resize_y = self.image_size
            resize_x = int(width * (self.image_size / height))
        else:
            resize_x = self.image_size
            resize_y = int(height * (self.image_size / width))
        resize_x = max(self.patch_size, (resize_x // self.patch_size) * self.patch_size)
        resize_y = max(self.patch_size, (resize_y // self.patch_size) * self.patch_size)

        scale_x = resize_x / width
        scale_y = resize_y / height
        image = image.resize((resize_x, resize_y), Image.Resampling.LANCZOS)
        fx *= scale_x
        fy *= scale_y
        cx *= scale_x
        cy *= scale_y
        width, height = image.size

        if self.center_crop and width != height:
            crop_size = min(width, height)
            left = (width - crop_size) // 2
            top = (height - crop_size) // 2
            image = image.crop((left, top, left + crop_size, top + crop_size))
            cx -= left
            cy -= top

        if self.normalize_intrinsics:
            intr = torch.tensor(
                [fx / self.image_size, fy / self.image_size, cx / self.image_size, cy / self.image_size],
                dtype=torch.float32,
            )
        else:
            intr = torch.tensor([fx, fy, cx, cy], dtype=torch.float32)

        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return tensor, intr

    def _select_views(self, num_frames: int) -> List[int]:
        if num_frames < self.num_views:
            return []

        selector_type = str(self.view_selector.get("type", "two_frame"))
        if selector_type == "uniform":
            return np.linspace(0, num_frames - 1, self.num_views, dtype=int).tolist()

        if selector_type == "random":
            indices = sorted(random.sample(range(num_frames), self.num_views))
            if self.random_shuffle:
                random.shuffle(indices)
            return indices

        if selector_type == "two_frame":
            return self._select_two_frame(num_frames)

        raise ValueError(f"Unsupported view_selector.type for TartanAir: {selector_type!r}")

    def _select_two_frame(self, num_frames: int) -> List[int]:
        use_curriculum = bool(self.view_selector.get("use_curriculum", False))
        if use_curriculum:
            max_iter = max(1, int(self.view_selector.get("curriculum_iter", 30000)))
            progress = min(self.current_iteration / max_iter, 1.0)
            final_min = float(self.view_selector.get("min_frame_dist", 10))
            final_max = float(self.view_selector.get("max_frame_dist", final_min))
            start_min = float(self.view_selector.get("curriculum_start_min_frame_dist", final_min))
            start_max = float(self.view_selector.get("curriculum_start_max_frame_dist", final_max))
            min_dist = int(start_min + (final_min - start_min) * progress)
            max_dist = int(start_max + (final_max - start_max) * progress)
        else:
            min_dist = int(self.view_selector.get("min_frame_dist", 10))
            max_dist = int(self.view_selector.get("max_frame_dist", min_dist))

        min_dist = max(self.num_views - 1, min_dist)
        max_dist = max(min_dist, max_dist)

        if self.is_eval:
            span = min(max_dist, num_frames - 1)
            return np.linspace(0, span, self.num_views, dtype=int).tolist()

        clamped_max = min(max_dist, num_frames - 1)
        clamped_min = min(min_dist, clamped_max)
        frame_dist = random.randint(clamped_min, clamped_max)
        if frame_dist < self.num_views - 1:
            return []

        start = random.randint(0, num_frames - frame_dist - 1)
        end = start + frame_dist
        middle_count = self.num_views - 2
        middle = sorted(random.sample(range(start + 1, end), middle_count))
        indices = [start, *middle, end]
        if self.random_shuffle:
            random.shuffle(indices)
        return indices

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        for attempt in range(self.max_resample_tries):
            traj_idx = (
                index % len(self.trajectories)
                if attempt == 0
                else random.randrange(len(self.trajectories))
            )
            record = self.trajectories[traj_idx]
            indices = self._select_views(len(record.frame_entries))
            if len(indices) < self.num_views:
                continue

            poses = self._load_poses(traj_idx)
            images: List[torch.Tensor] = []
            intrinsics: List[torch.Tensor] = []
            c2ws: List[torch.Tensor] = []

            for frame_idx in indices:
                img, intr = self._load_image_and_intrinsics(record, frame_idx)
                images.append(img)
                intrinsics.append(intr)
                c2ws.append(torch.from_numpy(poses[frame_idx]).float())

            image_indices = torch.tensor(indices, dtype=torch.long).unsqueeze(-1)
            scene_indices = torch.full_like(image_indices, traj_idx)
            return {
                "image": torch.stack(images, dim=0),
                "fxfycxcy": torch.stack(intrinsics, dim=0),
                "c2w": torch.stack(c2ws, dim=0),
                "index": torch.cat([image_indices, scene_indices], dim=-1),
            }

        raise RuntimeError(
            f"Could not sample {self.num_views} views after {self.max_resample_tries} tries"
        )

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_zip_cache"] = {}
        return state
