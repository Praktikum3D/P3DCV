"""Detect the lines of a source frame once, store them, and let training do only the cheap half.

Why this exists
---------------
The line detection is a CPU job and an expensive one: at the full 1000 pixel `lcam_fish`
resolution it costs about 1.5 seconds per frame, so a ten view training sample costs about fifteen
seconds. Run inside the training step that is not a slow loader, it is a stalled GPU: the sieve run
that tried it sat at 24 percent GPU and 98 percent CPU and was cancelled.

But the expensive half does not depend on anything the training step knows. Splitting the pathway
along that seam is what this module is for:

    once per source frame, here          every sample, in the loader
    ----------------------------         ---------------------------
    decode the 1000 px frame             draw a camera
    mask the fisheye image circle        map the cached pixels into it (map_fisheye_pixels)
    run LSD and the chaining             drop the lines that fall outside the view or the data mask
    sample K points per chain            keep the longest line_max_count above min_length
    -> pixels in SOURCE coordinates      -> pixels in VIEW coordinates

The left column is camera independent, which is exactly why it can be cached: a detected chain is a
property of the image, and the drawn camera only decides where those pixels land and which of them
survive. The right column is a handful of vectorized torch and numpy operations on at most a few
hundred points, microseconds against seconds.

What is in a cache file
-----------------------
One file per archive (or per unpacked trajectory), holding every frame of it:

    header   the detector knobs the cache was built with, see `CacheHeader`
    keys     one key per frame, root independent, see `frame_key`
    offsets  [num_frames + 1] int64, where each frame's rows start in `pixels`
    pixels   [total_lines, K, 2] float32, SOURCE pixel coordinates, longest line first

One flat tensor and an offset table rather than a dict of small tensors, so the file can be read
with `torch.load(..., mmap=True)` and a worker pays only for the frames it actually touches. The
rows of a frame are sorted longest first, which is what lets a reader with a smaller `max_count`
truncate instead of re-detecting and still get bit for bit what the in step path would have found.

Validity
--------
The header carries the knobs, and a reader compares them against its own. A different `num_samples`
makes the stored rows unusable, so that one falls back to detecting in the step; everything else is
a warning once per process and the cache is used. The one knob that deliberately does NOT have to
match is the dataset's `line_native_size`: a cache is built at the native resolution, which is
better than any downscale, so a cached run ignores that setting and says so.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from training.line_detection import (
    DEFAULT_MAX_COUNT,
    DEFAULT_NUM_SAMPLES,
    detect_line_segments,
    image_circle_mask,
)

CACHE_VERSION = 1
"""Bumped when the file layout changes. A reader refuses a version it does not know."""

RAW_MIN_LENGTH = 0.05
"""Shortest chain the SOURCE detection keeps, as a fraction of the source side.

Deliberately generous, and not the dataset's `line_min_length`. The real length cut belongs in the
view: a line that is short on a 195 degree source frame can fill a narrow drawn camera, and one
that is long there can fall outside it entirely. The dataset applies its own cut after the mapping.
"""

RAW_MAX_COUNT_FACTOR = 8
"""How many more chains the source detection keeps than the dataset will finally use.

The mapping and the data mask throw a lot of them away, so the source has to offer a surplus. The
factor is what the in step path has always used; the cache stores the same surplus so that the two
paths agree row for row.
"""


@dataclass
class CacheHeader:
    """The knobs a cache was built with, compared against the reader's on every load."""

    source_size: int
    num_samples: int = DEFAULT_NUM_SAMPLES
    max_count: int = DEFAULT_MAX_COUNT * RAW_MAX_COUNT_FACTOR
    min_length: float = RAW_MIN_LENGTH
    detector: str = "auto"
    chain: bool = True
    camera_name: str = "lcam_fish"
    version: int = CACHE_VERSION
    group: str = ""
    created: str = ""

    def differences(self, other: "CacheHeader") -> List[str]:
        """Knob names where `other` (the reader's wish) differs from this (the file's).

        `source_size`, `group` and `created` are left out on purpose: the first is the whole point
        of a native cache, the other two are provenance.
        """
        names = ("num_samples", "max_count", "min_length", "detector", "chain", "camera_name")
        out = []
        for name in names:
            mine, theirs = getattr(self, name), getattr(other, name)
            if name == "max_count":
                # More in the file than the reader wants is not a difference: the rows are sorted
                # longest first, so the reader truncates and gets what it would have detected.
                if int(mine) >= int(theirs):
                    continue
            elif name == "detector" and "auto" in (str(mine), str(theirs)):
                continue  # "auto" resolves to whatever the build offers, the same on both sides
            if mine != theirs:
                out.append(f"{name} {theirs!r} wanted, {mine!r} in the cache")
        return out


def frame_key(archive_path: Optional[str], record_name: str, entry: str) -> str:
    """A key for one frame that does not depend on where the data root is mounted.

    Inside an archive the zip member name already carries the trajectory and the file, so it is the
    key. An unpacked tree stores absolute paths, which would break the moment the root moved, so
    there the trajectory name and the file name are joined instead.
    """
    if archive_path is not None:
        return str(entry)
    return f"{record_name}/{os.path.basename(entry)}"


def group_of(archive_path: Optional[str], record_name: str) -> str:
    """Which cache file a record belongs to: its archive, or its own trajectory when unpacked."""
    return str(archive_path) if archive_path is not None else str(record_name)


def cache_file_name(archive_path: Optional[str], record_name: str) -> str:
    """`<env>_<difficulty>_<camera>.pt` for an archive, a flattened trajectory name otherwise."""
    if archive_path is not None:
        path = Path(archive_path)
        return f"{path.parent.parent.name}_{path.parent.name}_{path.stem}.pt"
    return str(record_name).strip("/").replace("/", "_").replace(os.sep, "_") + ".pt"


def detect_source_lines(
    frame: np.ndarray,
    *,
    source_size: int,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    max_count: int = DEFAULT_MAX_COUNT * RAW_MAX_COUNT_FACTOR,
    min_length: float = RAW_MIN_LENGTH,
    detector: str = "auto",
    chain: bool = True,
) -> np.ndarray:
    """The camera independent half: chains of one source frame as [n, K, 2] source pixels.

    This is the exact call `TartanAirDataset._detect_lines_on_source` makes in the step, kept in one
    place so that the cached and the uncached pathway cannot drift apart. `frame` is the decoded
    H x W x 3 uint8 source image; it is resized to `source_size` first if it is larger, which is
    what the dataset's `line_native_size` used to do and what a cache normally leaves alone.
    """
    import cv2  # local, so importing this module does not require OpenCV for the reader half

    if frame.shape[0] != source_size:
        frame = cv2.resize(frame, (source_size, source_size), interpolation=cv2.INTER_AREA)
    found = detect_line_segments(
        frame,
        mask=image_circle_mask(source_size),
        min_length=min_length,
        max_count=max_count,
        num_samples=num_samples,
        detector=detector,
        chain=chain,
    )
    if len(found) == 0:
        return np.zeros((0, int(num_samples), 2), dtype=np.float32)
    return np.ascontiguousarray(found.pixels, dtype=np.float32)


def write_cache(path: str, header: CacheHeader, rows: Sequence[Tuple[str, np.ndarray]]) -> Dict[str, Any]:
    """Write one cache file from `(key, pixels)` pairs. Returns a small summary dict."""
    keys = [str(key) for key, _ in rows]
    arrays = [np.asarray(p, dtype=np.float32).reshape(-1, header.num_samples, 2) for _, p in rows]
    counts = [len(a) for a in arrays]
    offsets = torch.zeros(len(arrays) + 1, dtype=torch.int64)
    offsets[1:] = torch.tensor(np.cumsum(counts), dtype=torch.int64)
    pixels = (
        torch.from_numpy(np.concatenate(arrays, axis=0))
        if arrays
        else torch.zeros(0, header.num_samples, 2, dtype=torch.float32)
    )
    header.created = header.created or time.strftime("%Y-%m-%dT%H:%M:%S")
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    torch.save({"header": asdict(header), "keys": keys, "offsets": offsets, "pixels": pixels}, path)
    return {
        "path": path,
        "frames": len(keys),
        "lines": int(pixels.shape[0]),
        "bytes": os.path.getsize(path),
    }


class LineCache:
    """Read side of one cache file: a key to rows lookup, memory mapped where torch allows it."""

    def __init__(self, path: str, payload: Dict[str, Any]):
        self.path = path
        raw = dict(payload.get("header", {}))
        raw.pop("version", None)
        self.header = CacheHeader(version=int(payload.get("header", {}).get("version", 0)), **raw)
        self.keys: List[str] = list(payload["keys"])
        self.offsets: torch.Tensor = payload["offsets"]
        self.pixels: torch.Tensor = payload["pixels"]
        self._index = {key: i for i, key in enumerate(self.keys)}

    @classmethod
    def load(cls, path: str) -> "LineCache":
        try:  # mmap keeps a worker's resident memory at the frames it actually reads
            payload = torch.load(path, map_location="cpu", mmap=True)
        except (TypeError, RuntimeError, ValueError):
            payload = torch.load(path, map_location="cpu")
        cache = cls(path, payload)
        if cache.header.version != CACHE_VERSION:
            raise ValueError(
                f"{path} was written by line cache version {cache.header.version}, "
                f"this code reads version {CACHE_VERSION}"
            )
        return cache

    def __len__(self) -> int:
        return len(self.keys)

    def get(self, key: str, max_count: Optional[int] = None) -> Optional[np.ndarray]:
        """The source pixels of one frame as [n, K, 2] float32, or None when it is not in here.

        `max_count` truncates to the longest n rows, which is exactly what a detection with that
        max_count would have returned, because the rows are stored longest first.
        """
        position = self._index.get(key)
        if position is None:
            return None
        start, stop = int(self.offsets[position]), int(self.offsets[position + 1])
        if max_count is not None and stop - start > int(max_count):
            stop = start + int(max_count)
        return np.asarray(self.pixels[start:stop].to(torch.float32).numpy(), dtype=np.float32)


def build_cache_for_frames(
    frames: Iterable[Tuple[str, np.ndarray]],
    header: CacheHeader,
    *,
    progress_every: int = 0,
    log=print,
) -> List[Tuple[str, np.ndarray]]:
    """Run `detect_source_lines` over `(key, frame)` pairs, optionally printing progress."""
    rows: List[Tuple[str, np.ndarray]] = []
    start = time.perf_counter()
    for i, (key, frame) in enumerate(frames):
        rows.append((key, detect_source_lines(
            frame,
            source_size=header.source_size,
            num_samples=header.num_samples,
            max_count=header.max_count,
            min_length=header.min_length,
            detector=header.detector,
            chain=header.chain,
        )))
        if progress_every and (i + 1) % progress_every == 0:
            elapsed = time.perf_counter() - start
            log(f"    {i + 1} frames, {elapsed / (i + 1):.2f} s per frame, {elapsed / 60.0:.1f} min so far")
    return rows
