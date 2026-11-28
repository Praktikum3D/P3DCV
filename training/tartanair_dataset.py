from __future__ import annotations

import io
import os
import random
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from training.camera_augmentation import (
    DEFAULT_SUPERSAMPLE,
    FISH_SIZE,
    IDENTITY_FISHEYE,
    AugCamera,
    map_fisheye_pixels,
    normalize_mix,
    remap_views,
    sample_camera,
)
from training.line_cache import (
    RAW_MAX_COUNT_FACTOR,
    RAW_MIN_LENGTH,
    CacheHeader,
    LineCache,
    cache_file_name,
    detect_source_lines,
    frame_key,
    group_of,
)
from training.line_detection import (
    DEFAULT_MAX_COUNT,
    DEFAULT_MIN_LENGTH,
    DEFAULT_NUM_SAMPLES,
    PAD_VALUE,
    detect_line_segments,
    detect_lines_in_views,
    image_circle_mask,
    resolve_min_length,
)

_WARNED: set = set()
"""Messages already printed in this process, so a per sample condition warns once and not per step.

Workers are separate processes, so the warning appears once per worker, which is what makes it
visible at all: a worker that silently fell back to in step detection would just look slow.
"""


def _warn_once(message: str, key: Optional[str] = None) -> None:
    """Print `message` the first time this process sees `key` (the message itself by default).

    The separate key is for warnings whose text names the thing that went wrong: one line per
    MISSING FRAME would put thousands of lines in a training log and bury everything else, while
    one line per cache file says the same thing and names an example.
    """
    marker = key if key is not None else message
    if marker not in _WARNED:
        _WARNED.add(marker)
        print(f"WARNING: {message}", flush=True)

DEFAULT_LINE_NATIVE_SIZE = 512
"""Side the source frame is resized to before lines are detected on it, 0 for the full 1000.

Lines are detected on the source frame and mapped into the training view (see `_detect_lines`), so
this is the resolution that decides how accurate the detections are, independently of the 256 the
model trains at. It matters more than anything else in this module. Measured over eight
`data-sample` frames, mapped down to a 256 pixel view, at the operating point the current model
actually sits at (field of view 188 degrees instead of 195, so f at 1.0372 times the true one):

    detect at             256      384      512      768     1000
    gradient at 1.0372   AWAY  towards  towards  towards  towards
    residual x1.4 / x1.0  2.3      4.7      6.8     16.9     32.3
    ms per frame           81      154      248      526     1120

(end to end per frame, so decode and resize included; a busy frame costs several times a sparse
one. `tests/test_line_infra.py::test_the_native_path_is_sharper_on_real_frames` pins the sign of
the gradient at the operating point through the loader itself.)

Detecting on the 256 pixel view, which is what the model sees, is worse than useless: the
detector's endpoint noise is a fixed fraction of a pixel, so in angle it grows as the image
shrinks, and at 256 it buries the signal so thoroughly that the residual at the current operating
point is LOWER than at the true camera and its gradient pushes the field of view further away from
195 degrees. A line loss built on 256 pixel detections would work against the thing it is meant to
fix. 512 is the smallest size with a clear margin on the right side of that, at about five times
the cost of 256; 0 is sharper again at five times the cost of 512 and is the right setting for the
initialiser and the per scene refinement, which run on a handful of frames and do not care.
"""

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
        camera_augmentation: bool = False,
        camera_aug_mix: Optional[Dict[str, float]] = None,
        camera_aug_supersample: int = DEFAULT_SUPERSAMPLE,
        detect_lines: bool = False,
        line_detector: str = "auto",
        line_min_length: float = DEFAULT_MIN_LENGTH,
        line_max_count: int = DEFAULT_MAX_COUNT,
        line_samples: int = DEFAULT_NUM_SAMPLES,
        line_chain: bool = True,
        line_detect_native: bool = True,
        line_native_size: int = DEFAULT_LINE_NATIVE_SIZE,
        line_cache_dir: Optional[str] = None,
    ) -> None:
        self.dataset_path = os.path.abspath(os.path.expanduser(dataset_path))
        self.camera_name = camera_name

        # Camera augmentation: draw one camera per sample and resample every view of that sample to
        # it. Only lcam_fish can feed it, because only its 195 degree equidistant projection covers
        # every camera we want to draw. Checked before the trajectories are discovered, so a wrong
        # camera_name fails with this message rather than with an empty dataset.
        self.camera_augmentation = bool(camera_augmentation)
        self.camera_aug_supersample = max(1, int(camera_aug_supersample))
        self.camera_aug_mix = normalize_mix(camera_aug_mix) if self.camera_augmentation else None
        if self.camera_augmentation and "fish" not in str(self.camera_name or ""):
            raise ValueError(
                "dataset.camera_augmentation needs a fisheye source, set dataset.camera_name to "
                f"lcam_fish (got {self.camera_name!r})"
            )

        # Line detection: find straight segments on the image the model will actually see, so the
        # plumb line residual can be evaluated on them later. Default off, and off means not one
        # line of this runs; see `training/line_detection.py` for what a straight segment means on
        # a wide angle image.
        self.detect_lines = bool(detect_lines)
        self.line_detector = str(line_detector)
        self.line_min_length = float(line_min_length)
        self.line_max_count = max(1, int(line_max_count))
        self.line_samples = max(1, int(line_samples))
        self.line_chain = bool(line_chain)
        self.line_detect_native = bool(line_detect_native)
        self.line_native_size = max(0, int(line_native_size))
        # Where `scripts/precompute_lines.py` put its cache files. Unset means every sample detects
        # its own lines, which is correct but costs seconds per sample and starves the GPU; see
        # `training/line_cache.py`. With it set, `line_native_size` applies only to the frames that
        # miss the cache, because a cache is built at the native resolution already.
        self.line_cache_dir = str(line_cache_dir) if line_cache_dir else None
        self._line_caches: Dict[str, Optional[LineCache]] = {}

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

    @property
    def _is_fisheye_source(self) -> bool:
        """Whether the frames are full frame fisheye, so their corners carry no data.

        The same guess `loss_render_mask` makes from the dataset name, and no better: it is right
        as long as a `fish` dataset really delivers full `lcam_fish` frames.
        """
        return "fish" in str(self.camera_name or "")

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

    def _open_image(self, record: _TrajectoryRecord, frame_idx: int) -> Image.Image:
        entry = record.frame_entries[frame_idx]
        if record.archive_path is not None:
            raw = self._get_zip(record.archive_path).read(entry)
            return Image.open(io.BytesIO(raw)).convert("RGB")
        return Image.open(entry).convert("RGB")

    def _load_native_image(self, record: _TrajectoryRecord, frame_idx: int) -> np.ndarray:
        """One frame at its stored resolution, H x W x 3 uint8.

        The camera augmentation resamples from the full 1000 x 1000 fisheye, so it must not go
        through the resize and crop of `_load_image_and_intrinsics`.
        """
        array = np.asarray(self._open_image(record, frame_idx), dtype=np.uint8)
        if array.shape[:2] != (FISH_SIZE, FISH_SIZE):
            raise ValueError(
                f"camera augmentation expects {FISH_SIZE} x {FISH_SIZE} lcam_fish frames, "
                f"got {array.shape[:2]} from {record.name}"
            )
        return array

    def _load_image_and_intrinsics(
        self, record: _TrajectoryRecord, frame_idx: int, keep_native: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[np.ndarray]]:
        """The resized view and its intrinsics, plus the frame at its stored size on request.

        `keep_native` hands back the decoded frame before the resize, for the line detection, and
        only when it is a square `lcam_fish` frame, because only then is the resize a pure scale and
        `map_fisheye_pixels` able to place its pixels in the view. It costs one array copy of an
        image that is decoded anyway, never a second decode.
        """
        image = self._open_image(record, frame_idx)
        native = None
        if keep_native and image.size[0] == image.size[1] == FISH_SIZE:
            native = np.asarray(image, dtype=np.uint8)

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
        return tensor, intr, native

    def _augmented_views(
        self, record: _TrajectoryRecord, indices: Sequence[int]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], AugCamera, List[np.ndarray]]:
        """Draw one camera and resample all views of this sample to it.

        One camera for the whole sample, never one per view: the views of a sample are meant to show
        the same scene through the same lens, and the model predicts a single alpha per scene.

        The draw hangs off the `random` module, which the data loader seeds per worker and per epoch,
        so a fixed global seed reproduces the cameras exactly.

        Returns the images, the per view intrinsics, the extra batch fields `data_mask`,
        `aug_fov_true`, `aug_alpha_true` and `aug_model`, and, for the line detection, the drawn
        camera and the source frames it was drawn from.
        """
        rng = np.random.default_rng(random.getrandbits(63))
        camera = sample_camera(rng, self.camera_aug_mix)
        frames = [self._load_native_image(record, frame_idx) for frame_idx in indices]
        images, data_mask, intr_px, alpha_true = remap_views(
            frames, camera, self.image_size, self.camera_aug_supersample
        )

        intr = intr_px / self.image_size if self.normalize_intrinsics else intr_px
        intrinsics = intr[None].expand(len(indices), -1).contiguous()
        extra = {
            "data_mask": data_mask,
            "aug_fov_true": torch.tensor(float(camera.fov_deg), dtype=torch.float32),
            "aug_alpha_true": torch.tensor(float(alpha_true), dtype=torch.float32),
            "aug_model": torch.tensor(camera.code, dtype=torch.long),
        }
        return images, intrinsics, extra, camera, frames

    def _detect_lines(
        self,
        images: torch.Tensor,
        data_mask: Optional[torch.Tensor],
        native_frames: Optional[List[np.ndarray]],
        camera: AugCamera,
        record: Optional[_TrajectoryRecord] = None,
        indices: Optional[Sequence[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Lines of every view of this sample, in the pixels of the view, padded into batch tensors.

        Two ways to get there, and which one is used matters more than anything else here.

        With `native_frames` (the default where they can be loaded) the detector runs on the sharp
        `lcam_fish` source frame and the pixels it finds are mapped into the view analytically by
        `map_fisheye_pixels`: unproject with the known equidistant source law, reproject with the
        drawn camera. The lines then sit in exactly the pixels of the image the model sees, which is
        what the residual needs, but they carry the subpixel accuracy of the source rather than of a
        256 pixel thumbnail. Without them, the detector runs on the view itself, which is honest but
        so noisy at 256 pixels that the residual turns adversarial; see DEFAULT_LINE_NATIVE_SIZE for
        the numbers and `line_detect_native` to force it.

        Either way the border of the data is kept out. On the source that is the fisheye image
        circle, whose rim is a long smooth arc belonging to no 3D line; on an augmented view it is
        the sample's own `data_mask`; on an unaugmented view it is the image circle again, which is
        what `loss_render_mask` assumes of a `fish` dataset too.

        Every view is detected, not only the input views. The split into input and target views is
        drawn in the model (`ERayZer._split_indices`), not here, so the loader cannot know which
        views will be inputs; the consumer picks its views out of the per view tensors with the same
        index it uses for the images.

        Returns `line_pixels` [v, line_max_count, line_samples, 2] float32, padded with -1, and
        `line_count` [v] int64, the number of real lines of each view. The padded width is the fixed
        knob and not the largest count found, so that the default collate can stack samples.
        """
        if native_frames is not None:
            return self._detect_lines_on_source(images, data_mask, native_frames, camera, record, indices)

        views = [images[i] for i in range(images.shape[0])]
        if data_mask is not None:
            masks: Optional[List] = [data_mask[i] for i in range(data_mask.shape[0])]
        elif self._is_fisheye_source:
            circle = torch.from_numpy(image_circle_mask(images.shape[-1]))
            masks = [circle] * len(views)
        else:
            masks = None
        pixels, counts = detect_lines_in_views(
            views,
            masks=masks,
            min_length=self.line_min_length,
            max_count=self.line_max_count,
            num_samples=self.line_samples,
            detector=self.line_detector,
            chain=self.line_chain,
        )
        return {"line_pixels": pixels, "line_count": counts}

    def _wanted_cache_header(self, source_size: int) -> CacheHeader:
        """The header this dataset would have written, for comparing against a cache file's."""
        return CacheHeader(
            source_size=int(source_size),
            num_samples=self.line_samples,
            max_count=RAW_MAX_COUNT_FACTOR * self.line_max_count,
            min_length=RAW_MIN_LENGTH,
            detector=self.line_detector,
            chain=self.line_chain,
            camera_name=str(self.camera_name or ""),
        )

    def _line_cache_for(self, record: _TrajectoryRecord) -> Optional[LineCache]:
        """The cache file covering this record, loaded once per worker, or None.

        Every outcome that is not a plain hit warns exactly once per process: a missing file, a
        file that cannot be read, and knobs that disagree with this dataset's. A silent fallback
        would show up only as a slow job, which is the failure this whole module exists to avoid.
        """
        if not self.line_cache_dir:
            return None
        group = group_of(record.archive_path, record.name)
        if group in self._line_caches:
            return self._line_caches[group]

        path = os.path.join(self.line_cache_dir, cache_file_name(record.archive_path, record.name))
        cache: Optional[LineCache] = None
        if not os.path.isfile(path):
            _warn_once(f"no line cache at {path}, detecting in the training step instead")
        else:
            try:
                cache = LineCache.load(path)
            except Exception as error:  # a broken cache must not take the run down with it
                _warn_once(f"could not read the line cache {path} ({error}), detecting in the step")
            else:
                wanted = self._wanted_cache_header(cache.header.source_size)
                for difference in cache.header.differences(wanted):
                    _warn_once(f"line cache {os.path.basename(path)} was built with a different {difference}")
                if cache.header.num_samples != self.line_samples:
                    # The stored rows have the wrong K and cannot be reshaped into what the batch
                    # tensor needs, so this one really is unusable.
                    _warn_once(f"line cache {os.path.basename(path)} is unusable, detecting in the step")
                    cache = None
        self._line_caches[group] = cache
        return cache

    def _source_lines(
        self,
        record: Optional[_TrajectoryRecord],
        frame_idx: Optional[int],
        frame: np.ndarray,
        cache: Optional[LineCache],
    ) -> Tuple[np.ndarray, int]:
        """Chains of one source frame as [n, K, 2] plus the resolution they are expressed in.

        From the cache when it holds this frame, which costs a slice of a memory mapped tensor.
        Otherwise by detecting, which costs a second and a half per frame and warns once.
        """
        if cache is not None and record is not None and frame_idx is not None:
            key = frame_key(record.archive_path, record.name, record.frame_entries[frame_idx])
            rows = cache.get(key, max_count=RAW_MAX_COUNT_FACTOR * self.line_max_count)
            if rows is not None:
                return rows, int(cache.header.source_size)
            # Keyed on the cache file, not on the frame: a cache covering half an archive would
            # otherwise warn once per missing frame per worker and drown the log.
            _warn_once(
                f"line cache {os.path.basename(cache.path)} has no entry for {key} "
                "(and possibly more frames), detecting those in the training step",
                key=f"miss:{cache.path}",
            )

        source_size = int(frame.shape[0])
        if self.line_native_size and self.line_native_size < source_size:
            source_size = self.line_native_size
        return detect_source_lines(
            frame,
            source_size=source_size,
            num_samples=self.line_samples,
            max_count=RAW_MAX_COUNT_FACTOR * self.line_max_count,
            min_length=RAW_MIN_LENGTH,
            detector=self.line_detector,
            chain=self.line_chain,
        ), source_size

    def _detect_lines_on_source(
        self,
        images: torch.Tensor,
        data_mask: Optional[torch.Tensor],
        native_frames: List[np.ndarray],
        camera: AugCamera,
        record: Optional[_TrajectoryRecord] = None,
        indices: Optional[Sequence[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Map the source frames' chains into the view, detecting them only if no cache has them.

        The detection is the camera independent half and belongs in `scripts/precompute_lines.py`;
        everything below the mapping is the per sample half and is what this is supposed to cost.
        `record` and `indices` identify the frames in the cache, and without them (or without
        `dataset.line_cache_dir`) every frame is detected here as it always was.
        """
        size = int(images.shape[-1])
        cache = self._line_cache_for(record) if record is not None else None
        min_px = resolve_min_length(self.line_min_length, (size, size))

        pixels = torch.full(
            (len(native_frames), self.line_max_count, self.line_samples, 2), PAD_VALUE, dtype=torch.float32
        )
        counts = torch.zeros(len(native_frames), dtype=torch.long)
        for view, frame in enumerate(native_frames):
            frame_idx = int(indices[view]) if indices is not None else None
            # The length cut waits until the pixels are in the view: a line that is short on the
            # 195 degree source can fill a narrow drawn camera, and one that is long there can fall
            # outside it entirely. Generous limits at the source, the real ones after the map.
            source_pixels, source_size = self._source_lines(record, frame_idx, frame, cache)
            if len(source_pixels) == 0:
                continue

            mapped, valid = map_fisheye_pixels(source_pixels, camera, size, source_size)
            keep = valid.all(axis=-1)  # every sampled pixel of the line has to be imaged
            if data_mask is not None and keep.any():
                mask = data_mask[view, 0] > 0.5
                columns = np.clip(np.round(mapped[..., 0] - 0.5).astype(np.int64), 0, size - 1)
                rows = np.clip(np.round(mapped[..., 1] - 0.5).astype(np.int64), 0, size - 1)
                # The drawn cameras all have a convex valid region, so the sampled pixels decide.
                keep &= mask.numpy()[rows, columns].all(axis=-1)
            mapped = mapped[keep]
            if len(mapped) == 0:
                continue

            length = np.linalg.norm(np.diff(mapped, axis=1), axis=-1).sum(axis=-1)
            order = np.argsort(-length)[: self.line_max_count]
            order = order[length[order] >= min_px]
            if len(order) == 0:
                continue
            counts[view] = len(order)
            pixels[view, : len(order)] = torch.from_numpy(np.ascontiguousarray(mapped[order]))
        return {"line_pixels": pixels, "line_count": counts}

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
            c2ws = [torch.from_numpy(poses[frame_idx]).float() for frame_idx in indices]
            image_indices = torch.tensor(indices, dtype=torch.long).unsqueeze(-1)
            scene_indices = torch.full_like(image_indices, traj_idx)

            extra: Dict[str, torch.Tensor] = {}
            camera = IDENTITY_FISHEYE
            native_frames: Optional[List[np.ndarray]] = None
            # The source frames are only worth keeping when the line detection will read them; see
            # `_detect_lines`. They are the full 1000 x 1000 images, so holding them otherwise would
            # cost every worker three megabytes a view for nothing.
            keep_native = self.detect_lines and self.line_detect_native and self._is_fisheye_source
            if self.camera_augmentation:
                stacked_images, stacked_intr, extra, camera, frames = self._augmented_views(record, indices)
                native_frames = frames if keep_native else None
            else:
                images: List[torch.Tensor] = []
                intrinsics: List[torch.Tensor] = []
                natives: List[np.ndarray] = []
                for frame_idx in indices:
                    img, intr, native = self._load_image_and_intrinsics(record, frame_idx, keep_native)
                    images.append(img)
                    intrinsics.append(intr)
                    if native is not None:
                        natives.append(native)
                stacked_images = torch.stack(images, dim=0)
                stacked_intr = torch.stack(intrinsics, dim=0)
                native_frames = natives if len(natives) == len(indices) else None

            if self.detect_lines:
                extra.update(self._detect_lines(stacked_images, extra.get("data_mask"),
                                                native_frames, camera, record, indices))

            return {
                "image": stacked_images,
                "fxfycxcy": stacked_intr,
                "c2w": torch.stack(c2ws, dim=0),
                "index": torch.cat([image_indices, scene_indices], dim=-1),
                **extra,
            }

        raise RuntimeError(
            f"Could not sample {self.num_views} views after {self.max_resample_tries} tries"
        )

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_zip_cache"] = {}
        # Loaded lazily per worker, and memory mapped, so neither handle survives pickling.
        state["_line_caches"] = {}
        return state
