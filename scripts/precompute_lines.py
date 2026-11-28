"""Detect the lines of every source frame once and cache them, so training does not have to.

The line detection is pure CPU and costs about 1.5 seconds per 1000 pixel frame, which is fine once
and ruinous in a training step: ten views per sample means fifteen seconds of CPU against roughly
one second of GPU, and a job that does that sits near 25 percent GPU utilisation and gets cancelled.
The detection does not depend on the camera the augmentation will draw, so it belongs here.
`training/line_cache.py` explains the split and the file format.

Run it as a cluster job on the `data` partition, with no GPU at all:

    sbatch --export=ALL,REPO_DIR=$PWD,DATA_PATH=/mnt/home/$USER/data/tartanair \\
        slurm/precompute_lines.sh

or directly, for one environment:

    python scripts/precompute_lines.py --data-path /mnt/.../tartanair --camera lcam_fish \\
        --cache-dir /mnt/home/$USER/data/line-cache --envs HongKong

One file per archive, named `<env>_<difficulty>_<camera>.pt`, holding every frame of it. Existing
files are skipped unless `--overwrite`, so an interrupted job can simply be resubmitted.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
import zipfile
from collections import OrderedDict
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.line_cache import (  # noqa: E402
    RAW_MAX_COUNT_FACTOR,
    RAW_MIN_LENGTH,
    CacheHeader,
    build_cache_for_frames,
    cache_file_name,
    detect_source_lines,
    frame_key,
    group_of,
    write_cache,
)
from training.line_detection import DEFAULT_MAX_COUNT, DEFAULT_NUM_SAMPLES  # noqa: E402
from training.tartanair_dataset import FISH_SIZE, _discover_trajectories  # noqa: E402


def open_frame(archive: Optional[zipfile.ZipFile], entry: str) -> np.ndarray:
    """One decoded source frame, H x W x 3 uint8, exactly as the dataset decodes it.

    The same `Image.open(...).convert("RGB")` and `np.asarray(..., uint8)` as
    `TartanAirDataset._load_native_image`, so that a cached detection and an in step detection see
    byte identical input and therefore produce byte identical output.
    """
    if archive is not None:
        return np.asarray(Image.open(io.BytesIO(archive.read(entry))).convert("RGB"), dtype=np.uint8)
    return np.asarray(Image.open(entry).convert("RGB"), dtype=np.uint8)


def grouped_records(records) -> "OrderedDict[str, list]":
    """Trajectories bucketed into the cache file they belong to, order preserved."""
    groups: "OrderedDict[str, list]" = OrderedDict()
    for record in records:
        groups.setdefault(group_of(record.archive_path, record.name), []).append(record)
    return groups


def frames_of(records, *, expect_size: Optional[int]) -> Iterator[Tuple[str, np.ndarray]]:
    """`(key, frame)` for every frame of these trajectories, opening each archive once."""
    archive: Optional[zipfile.ZipFile] = None
    try:
        for record in records:
            if record.archive_path is not None and archive is None:
                archive = zipfile.ZipFile(record.archive_path, "r")
            for entry in record.frame_entries:
                frame = open_frame(archive, entry)
                if expect_size and frame.shape[0] != expect_size:
                    raise ValueError(
                        f"{entry} is {frame.shape[:2]}, expected {expect_size} x {expect_size}; "
                        "the line cache is only meaningful for square lcam_fish frames"
                    )
                yield frame_key(record.archive_path, record.name, entry), frame
    finally:
        if archive is not None:
            archive.close()


def jobs_of(records) -> List[Tuple[Optional[str], str, str]]:
    """`(archive_path, record_name, entry)` for every frame, the unit a worker process gets.

    The frame itself is NOT passed: a decoded 1000 pixel frame is three megabytes, and shipping it
    to a worker costs more than the worker saves. Each process opens the archive itself instead.
    """
    return [
        (record.archive_path, record.name, entry)
        for record in records
        for entry in record.frame_entries
    ]


_WORKER: Dict[str, object] = {}
"""Per process state of the pool: the open archive and the header to detect with."""


def _worker_init(header: CacheHeader, expect_size: Optional[int]) -> None:
    _WORKER["header"] = header
    _WORKER["expect_size"] = expect_size
    _WORKER["archives"] = {}


def _worker_detect(job: Tuple[Optional[str], str, str]) -> Tuple[str, np.ndarray]:
    archive_path, record_name, entry = job
    archive = None
    if archive_path is not None:
        archives: Dict[str, zipfile.ZipFile] = _WORKER["archives"]  # type: ignore[assignment]
        archive = archives.get(archive_path)
        if archive is None:
            archive = archives[archive_path] = zipfile.ZipFile(archive_path, "r")
    frame = open_frame(archive, entry)
    expect_size = _WORKER["expect_size"]
    if expect_size and frame.shape[0] != expect_size:
        raise ValueError(
            f"{entry} is {frame.shape[:2]}, expected {expect_size} x {expect_size}; "
            "the line cache is only meaningful for square lcam_fish frames"
        )
    header: CacheHeader = _WORKER["header"]  # type: ignore[assignment]
    pixels = detect_source_lines(
        frame,
        source_size=header.source_size,
        num_samples=header.num_samples,
        max_count=header.max_count,
        min_length=header.min_length,
        detector=header.detector,
        chain=header.chain,
    )
    return frame_key(archive_path, record_name, entry), pixels


def detect_in_parallel(
    jobs: List[Tuple[Optional[str], str, str]],
    header: CacheHeader,
    *,
    workers: int,
    expect_size: Optional[int],
    progress_every: int,
) -> List[Tuple[str, np.ndarray]]:
    """`build_cache_for_frames` spread over a process pool, in the same order.

    Detection of one frame does not depend on any other, so this is exactly the serial result;
    `imap` with a chunk size keeps the order and the per task overhead low. At 1.2 seconds a frame
    a whole TartanAir environment is hours of single threaded work and minutes of parallel work,
    which is the difference between one job and a weekend.
    """
    import multiprocessing as mp

    rows: List[Tuple[str, np.ndarray]] = []
    started = time.perf_counter()
    context = mp.get_context("spawn")
    # Big enough to amortize the per task handoff, small enough that every worker gets a share of a
    # short archive; a fixed chunk of 8 would leave three of four processes idle on 16 frames.
    chunksize = max(1, min(8, len(jobs) // max(workers * 4, 1)))
    with context.Pool(workers, initializer=_worker_init, initargs=(header, expect_size)) as pool:
        for i, row in enumerate(pool.imap(_worker_detect, jobs, chunksize=chunksize)):
            rows.append(row)
            if progress_every and (i + 1) % progress_every == 0:
                elapsed = time.perf_counter() - started
                print(f"    {i + 1}/{len(jobs)} frames, {elapsed / (i + 1):.3f} s per frame "
                      f"wall clock, {elapsed / 60.0:.1f} min so far", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", required=True,
                        help="TartanAir root, a single image_*.zip, or a manifest .txt")
    parser.add_argument("--cache-dir", default=None,
                        help="where the cache files go, default /mnt/home/$USER/data/line-cache")
    parser.add_argument("--camera", default="lcam_fish", help="camera_name to index, as the dataset uses it")
    parser.add_argument("--envs", nargs="*", default=None,
                        help="only archives whose path contains one of these, for example HongKong")
    parser.add_argument("--source-size", type=int, default=0,
                        help="detect at this resolution, 0 (the default) for the native frame size")
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES,
                        help="K, the points sampled along each line; must match dataset.line_samples")
    parser.add_argument("--max-count", type=int, default=DEFAULT_MAX_COUNT,
                        help="dataset.line_max_count; the cache keeps this times "
                             f"{RAW_MAX_COUNT_FACTOR} chains per frame so the mapping has a surplus")
    parser.add_argument("--detector", default="auto", help="auto, lsd or hough")
    parser.add_argument("--no-chain", action="store_true", help="store raw detector segments, not chains")
    parser.add_argument("--overwrite", action="store_true", help="rebuild cache files that already exist")
    parser.add_argument("--workers", type=int, default=0,
                        help="detector processes, 0 for half the cores, 1 to stay single process")
    parser.add_argument("--limit-frames", type=int, default=0, help="stop each group after n frames, for a smoke test")
    parser.add_argument("--progress-every", type=int, default=200, help="print a line every n frames")
    args = parser.parse_args()

    cache_dir = args.cache_dir or os.path.expanduser(
        os.path.join("/mnt/home", os.environ.get("USER", "nobody"), "data", "line-cache")
    )
    os.makedirs(cache_dir, exist_ok=True)

    workers = int(args.workers) if args.workers > 0 else max(1, (os.cpu_count() or 2) // 2)
    records = _discover_trajectories(args.data_path, camera_name=args.camera)
    if args.envs:
        wanted = tuple(args.envs)
        records = [
            r for r in records
            if any(env in str(r.archive_path or "") or env in r.name for env in wanted)
        ]
        if not records:
            raise SystemExit(f"no trajectories matched --envs {' '.join(wanted)}")
    groups = grouped_records(records)
    print(f"{len(records)} trajectories in {len(groups)} cache groups, {workers} detector "
          f"processes, writing to {cache_dir}", flush=True)

    source_size = int(args.source_size) if args.source_size > 0 else FISH_SIZE
    written, skipped, total_frames, total_lines, total_bytes = 0, 0, 0, 0, 0
    started = time.perf_counter()
    for position, (group, group_records) in enumerate(groups.items(), start=1):
        first = group_records[0]
        path = os.path.join(cache_dir, cache_file_name(first.archive_path, first.name))
        frames = sum(len(r.frame_entries) for r in group_records)
        if os.path.isfile(path) and not args.overwrite:
            print(f"[{position}/{len(groups)}] {os.path.basename(path)} exists, skipping", flush=True)
            skipped += 1
            continue

        print(f"[{position}/{len(groups)}] {os.path.basename(path)}: {frames} frames from "
              f"{len(group_records)} trajectories", flush=True)
        header = CacheHeader(
            source_size=source_size,
            num_samples=int(args.num_samples),
            max_count=RAW_MAX_COUNT_FACTOR * int(args.max_count),
            min_length=RAW_MIN_LENGTH,
            detector=str(args.detector),
            chain=not args.no_chain,
            camera_name=str(args.camera),
            group=group,
        )
        expect_size = None if args.source_size else FISH_SIZE
        group_started = time.perf_counter()
        if workers > 1:
            jobs = jobs_of(group_records)
            if args.limit_frames:
                jobs = jobs[: args.limit_frames]
            rows = detect_in_parallel(jobs, header, workers=workers, expect_size=expect_size,
                                      progress_every=args.progress_every)
        else:
            stream = frames_of(group_records, expect_size=expect_size)
            if args.limit_frames:
                stream = (item for i, item in enumerate(stream) if i < args.limit_frames)
            rows = build_cache_for_frames(stream, header, progress_every=args.progress_every)
        summary = write_cache(path, header, rows)
        elapsed = time.perf_counter() - group_started
        print(f"    wrote {summary['frames']} frames, {summary['lines']} lines, "
              f"{summary['bytes'] / 1e6:.1f} MB, {elapsed / 60.0:.1f} min "
              f"({elapsed / max(summary['frames'], 1):.2f} s per frame)", flush=True)
        written += 1
        total_frames += summary["frames"]
        total_lines += summary["lines"]
        total_bytes += summary["bytes"]

    elapsed = time.perf_counter() - started
    print(
        f"Done: {written} cache files written, {skipped} skipped, {total_frames} frames, "
        f"{total_lines} lines, {total_bytes / 1e6:.1f} MB, {elapsed / 60.0:.1f} min total"
        + (f", {elapsed / total_frames:.2f} s per frame" if total_frames else ""),
        flush=True,
    )


if __name__ == "__main__":
    main()
