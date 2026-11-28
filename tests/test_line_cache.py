"""The precomputed line cache: the same lines as detecting in the step, at a thousandth of the cost.

The line detection is camera independent and expensive, the mapping into the drawn camera is cheap,
and `training/line_cache.py` splits the pathway along that seam. A cache is only worth having if it
is EXACTLY what the training step would have computed, so that is what most of this file checks,
bit for bit and through the real dataset rather than through the library alone.

The rest is what happens when the cache does not fit: a file that is not there, a frame that is not
in it, knobs that disagree. None of those may crash a run and none of them may be silent, because a
worker that quietly falls back to detecting in the step looks exactly like a worker that is simply
slow, which is the failure this whole thing exists to prevent.

Run from the repository root:
    pytest tests/test_line_cache.py
"""

import os
import random
import sys
import tempfile

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

cv2 = pytest.importorskip("cv2")

import training.tartanair_dataset as tad  # noqa: E402
from training.line_cache import (  # noqa: E402
    CACHE_VERSION,
    RAW_MAX_COUNT_FACTOR,
    RAW_MIN_LENGTH,
    CacheHeader,
    LineCache,
    build_cache_for_frames,
    cache_file_name,
    detect_source_lines,
    frame_key,
    group_of,
    write_cache,
)
from training.tartanair_dataset import TartanAirDataset, _discover_trajectories  # noqa: E402

SAMPLE_ROOT = "/praktikum/data-sample"
CAMERA = "lcam_fish"

needs_frames = pytest.mark.skipif(
    not os.path.isdir(SAMPLE_ROOT), reason=f"no lcam_fish sample frames at {SAMPLE_ROOT}"
)


@pytest.fixture(autouse=True)
def fresh_warnings():
    """Every test starts with an empty "already warned" set, or the second one would see nothing."""
    tad._WARNED.clear()
    yield
    tad._WARNED.clear()


def open_frame(record, index):
    from PIL import Image

    entry = record.frame_entries[index]
    if record.archive_path is not None:
        import io
        import zipfile

        with zipfile.ZipFile(record.archive_path) as zf:
            return np.asarray(Image.open(io.BytesIO(zf.read(entry))).convert("RGB"), dtype=np.uint8)
    return np.asarray(Image.open(entry).convert("RGB"), dtype=np.uint8)


def build_cache(cache_dir, *, source_size=256, num_samples=8, max_count=32, **header_kwargs):
    """Write a cache for every frame of the sample trajectory, as the precompute script would."""
    records = _discover_trajectories(SAMPLE_ROOT, camera_name=CAMERA)
    written = []
    for record in records:
        header = CacheHeader(
            source_size=source_size,
            num_samples=num_samples,
            max_count=RAW_MAX_COUNT_FACTOR * max_count,
            min_length=RAW_MIN_LENGTH,
            camera_name=CAMERA,
            group=group_of(record.archive_path, record.name),
            **header_kwargs,
        )
        frames = (
            (frame_key(record.archive_path, record.name, entry), open_frame(record, i))
            for i, entry in enumerate(record.frame_entries)
        )
        rows = build_cache_for_frames(frames, header, log=lambda *a: None)
        path = os.path.join(cache_dir, cache_file_name(record.archive_path, record.name))
        written.append(write_cache(path, header, rows))
    return written


def dataset(**kwargs):
    options = dict(
        image_size=256, patch_size=16, num_views=2, num_input_views=1,
        view_selector={"type": "uniform"}, camera_name=CAMERA,
        normalize_intrinsics=False, camera_augmentation=True,
        detect_lines=True, line_max_count=32, line_samples=8, data_repeat=16,
    )
    options.update(kwargs)
    return TartanAirDataset(SAMPLE_ROOT, **options)


def sample_at(data, index, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    return data[index]


@pytest.fixture(scope="module")
def cache_dir_256():
    """One cache at 256, built once: at the native size the detector costs a second per frame."""
    if not os.path.isdir(SAMPLE_ROOT):
        pytest.skip("no sample frames")
    with tempfile.TemporaryDirectory() as tmp:
        build_cache(tmp, source_size=256)
        yield tmp


# --------------------------------------------------------------------------------------------
# the one claim everything rests on
# --------------------------------------------------------------------------------------------

@needs_frames
def test_cached_lines_are_identical_to_detecting_in_the_step(cache_dir_256):
    """Same frames, same drawn camera, same tensors, bit for bit.

    The drawn camera comes out of the `random` module, so seeding before each access gives the two
    datasets the same camera and makes the line tensors directly comparable.
    """
    live = dataset(line_native_size=256)
    cached = dataset(line_cache_dir=cache_dir_256)
    for index in range(3):
        a = sample_at(live, index, 400 + index)
        b = sample_at(cached, index, 400 + index)
        assert torch.equal(a["image"], b["image"]), "the two runs did not draw the same camera"
        assert torch.equal(a["line_count"], b["line_count"]), index
        assert torch.equal(a["line_pixels"], b["line_pixels"]), index
        assert int(a["line_count"].sum()) > 0, "a sample with no lines proves nothing"


@needs_frames
def test_cached_lines_are_identical_at_the_native_resolution():
    """The same claim at the size a real cache is built with, on one sample because it is slow."""
    with tempfile.TemporaryDirectory() as tmp:
        build_cache(tmp, source_size=tad.FISH_SIZE)
        live = dataset(line_native_size=0)
        cached = dataset(line_cache_dir=tmp)
        a = sample_at(live, 0, 7)
        b = sample_at(cached, 0, 7)
        assert torch.equal(a["line_pixels"], b["line_pixels"])
        assert int(a["line_count"].sum()) > 0


@needs_frames
def test_the_unaugmented_pathway_matches_too(cache_dir_256):
    """Without augmentation the mapping is the identity scale, and must still agree."""
    live = dataset(camera_augmentation=False, line_native_size=256)
    cached = dataset(camera_augmentation=False, line_cache_dir=cache_dir_256)
    a = sample_at(live, 0, 11)
    b = sample_at(cached, 0, 11)
    assert torch.equal(a["line_pixels"], b["line_pixels"])


@needs_frames
def test_the_cache_is_much_faster(cache_dir_256, capsys):
    """Reported, not asserted as a threshold, because a shared box makes timings noisy."""
    import time

    live = dataset(line_native_size=0)
    cached = dataset(line_cache_dir=cache_dir_256)
    start = time.perf_counter(); sample_at(live, 0, 3); slow = time.perf_counter() - start
    start = time.perf_counter(); sample_at(cached, 0, 3); fast = time.perf_counter() - start
    with capsys.disabled():
        print(f"\n  two view sample: {slow:.2f} s detecting in the step, {fast:.3f} s from the cache")
    assert fast < slow


# --------------------------------------------------------------------------------------------
# when the cache does not fit
# --------------------------------------------------------------------------------------------

@needs_frames
def test_a_missing_cache_file_warns_once_and_falls_back(capsys):
    with tempfile.TemporaryDirectory() as empty:
        data = dataset(line_cache_dir=empty)
        first = sample_at(data, 0, 5)
        warnings = [line for line in capsys.readouterr().out.splitlines() if "WARNING" in line]
        assert len(warnings) == 1 and "no line cache" in warnings[0]
        assert int(first["line_count"].sum()) > 0, "the fallback must still produce lines"

        sample_at(data, 1, 6)
        sample_at(data, 2, 7)
        assert not [line for line in capsys.readouterr().out.splitlines() if "WARNING" in line], (
            "the warning repeated, which would flood a training log"
        )


@needs_frames
def test_a_frame_missing_from_the_cache_falls_back_and_warns_once(capsys):
    """A cache built over half the frames still has to deliver the other half, by detecting them."""
    with tempfile.TemporaryDirectory() as tmp:
        records = _discover_trajectories(SAMPLE_ROOT, camera_name=CAMERA)
        record = records[0]
        header = CacheHeader(source_size=256, num_samples=8,
                             max_count=RAW_MAX_COUNT_FACTOR * 32, camera_name=CAMERA)
        kept = list(record.frame_entries)[:1]  # only the very first frame
        rows = [
            (frame_key(record.archive_path, record.name, entry),
             detect_source_lines(open_frame(record, i), source_size=256))
            for i, entry in enumerate(record.frame_entries) if entry in kept
        ]
        write_cache(os.path.join(tmp, cache_file_name(record.archive_path, record.name)), header, rows)

        data = dataset(line_cache_dir=tmp, line_native_size=256)
        got = sample_at(data, 0, 9)
        text = capsys.readouterr().out
        misses = [line for line in text.splitlines() if "has no entry for" in line]
        assert len(misses) == 1, text
        assert int(got["line_count"].sum()) > 0

        # and the fallback produced exactly what a cacheless run would have
        live = dataset(line_native_size=256)
        assert torch.equal(got["line_pixels"], sample_at(live, 0, 9)["line_pixels"])


@needs_frames
def test_a_header_mismatch_warns_once_and_still_uses_the_cache(capsys):
    with tempfile.TemporaryDirectory() as tmp:
        build_cache(tmp, source_size=256, chain=False)
        data = dataset(line_cache_dir=tmp)  # the dataset wants chain=True
        sample_at(data, 0, 13)
        warnings = [line for line in capsys.readouterr().out.splitlines() if "WARNING" in line]
        assert len(warnings) == 1 and "different chain" in warnings[0], warnings
        sample_at(data, 1, 14)
        assert not [line for line in capsys.readouterr().out.splitlines() if "WARNING" in line]


@needs_frames
def test_a_different_sample_count_makes_the_cache_unusable(capsys):
    """K decides the shape of every row, so a mismatch cannot be worked around, only reported."""
    with tempfile.TemporaryDirectory() as tmp:
        build_cache(tmp, source_size=256, num_samples=6)
        data = dataset(line_cache_dir=tmp, line_native_size=256)  # wants line_samples=8
        got = sample_at(data, 0, 17)
        text = capsys.readouterr().out
        assert "unusable" in text, text
        assert got["line_pixels"].shape[-2] == 8
        live = dataset(line_native_size=256)
        assert torch.equal(got["line_pixels"], sample_at(live, 0, 17)["line_pixels"])


@needs_frames
def test_a_bigger_cache_is_truncated_to_what_the_reader_asked_for():
    """Rows are stored longest first, so truncation equals a detection with the smaller max_count."""
    records = _discover_trajectories(SAMPLE_ROOT, camera_name=CAMERA)
    frame = open_frame(records[0], 0)
    many = detect_source_lines(frame, source_size=256, max_count=64)
    few = detect_source_lines(frame, source_size=256, max_count=16)
    assert len(many) > len(few) >= 1
    assert np.array_equal(many[: len(few)], few)

    with tempfile.TemporaryDirectory() as tmp:
        header = CacheHeader(source_size=256, num_samples=8, max_count=64, camera_name=CAMERA)
        path = os.path.join(tmp, "x.pt")
        write_cache(path, header, [("k", many)])
        assert np.array_equal(LineCache.load(path).get("k", max_count=16), few)


def test_a_cache_from_another_version_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        header = CacheHeader(source_size=256, num_samples=8)
        path = os.path.join(tmp, "x.pt")
        write_cache(path, header, [("k", np.zeros((2, 8, 2), dtype=np.float32))])
        payload = torch.load(path, map_location="cpu")
        payload["header"]["version"] = CACHE_VERSION + 1
        torch.save(payload, path)
        with pytest.raises(ValueError, match="version"):
            LineCache.load(path)


# --------------------------------------------------------------------------------------------
# the file format and the keys
# --------------------------------------------------------------------------------------------

@needs_frames
def test_the_cache_round_trips_every_frame():
    with tempfile.TemporaryDirectory() as tmp:
        written = build_cache(tmp, source_size=256)
        assert len(written) == 1 and written[0]["frames"] > 0
        cache = LineCache.load(written[0]["path"])
        records = _discover_trajectories(SAMPLE_ROOT, camera_name=CAMERA)
        record = records[0]
        assert len(cache) == len(record.frame_entries)
        for i, entry in enumerate(record.frame_entries):
            key = frame_key(record.archive_path, record.name, entry)
            rows = cache.get(key)
            assert rows is not None and rows.dtype == np.float32
            assert rows.ndim == 3 and rows.shape[1:] == (8, 2)
            direct = detect_source_lines(open_frame(record, i), source_size=256)
            assert np.array_equal(rows, direct), entry
        assert cache.get("not a frame") is None


@needs_frames
def test_the_parallel_precompute_equals_the_serial_one():
    """A pool is only a speedup if it changes nothing, and the cache has to be reproducible.

    Detection of one frame is independent of every other, so `imap` in order must give exactly the
    serial list. Worth pinning because the parallel path is what will actually build the caches:
    at 1.2 seconds a frame, 27 archives single threaded is more than a day.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    precompute_lines = pytest.importorskip("precompute_lines")

    records = _discover_trajectories(SAMPLE_ROOT, camera_name=CAMERA)
    header = CacheHeader(source_size=256, num_samples=8,
                         max_count=RAW_MAX_COUNT_FACTOR * 32, camera_name=CAMERA)
    jobs = precompute_lines.jobs_of(records)
    assert len(jobs) > 1

    serial = build_cache_for_frames(
        ((frame_key(r.archive_path, r.name, e), open_frame(r, i))
         for r in records for i, e in enumerate(r.frame_entries)),
        header, log=lambda *a: None,
    )
    parallel = precompute_lines.detect_in_parallel(
        jobs, header, workers=2, expect_size=None, progress_every=0
    )
    assert [key for key, _ in parallel] == [key for key, _ in serial]
    for (_, a), (_, b) in zip(serial, parallel):
        assert np.array_equal(a, b)


def test_keys_and_file_names_do_not_depend_on_the_data_root():
    """A cache written on one mount must still be readable when the root moves."""
    assert frame_key("/a/b/HongKong/Data_hard/image_lcam_fish.zip", "x", "P000/img/0.png") == "P000/img/0.png"
    assert frame_key(None, "HongKong/Data_hard/P000", "/mnt/elsewhere/0.png") == "HongKong/Data_hard/P000/0.png"
    assert cache_file_name("/a/HongKong/Data_hard/image_lcam_fish.zip", "x") == "HongKong_Data_hard_image_lcam_fish.pt"
    assert cache_file_name(None, "HongKong/Data_hard/P000") == "HongKong_Data_hard_P000.pt"
    assert group_of("/a/b.zip", "n") == "/a/b.zip" and group_of(None, "n") == "n"


def test_the_header_compares_only_the_knobs_that_matter():
    built = CacheHeader(source_size=1000, num_samples=8, max_count=256, detector="lsd", chain=True)
    # a native cache read by a dataset that would have downscaled is not a mismatch
    assert built.differences(CacheHeader(source_size=512, num_samples=8, max_count=256,
                                         detector="lsd", chain=True)) == []
    # more rows in the file than the reader wants is fine, fewer is not
    assert built.differences(CacheHeader(source_size=1000, num_samples=8, max_count=128,
                                         detector="lsd", chain=True)) == []
    assert built.differences(CacheHeader(source_size=1000, num_samples=8, max_count=512,
                                         detector="lsd", chain=True)) != []
    # "auto" resolves to whatever the OpenCV build offers, so it never counts as different
    assert built.differences(CacheHeader(source_size=1000, num_samples=8, max_count=256,
                                         detector="auto", chain=True)) == []
    assert "num_samples" in built.differences(CacheHeader(source_size=1000, num_samples=4))[0]


def test_an_empty_frame_round_trips_as_zero_rows():
    """A frame the detector found nothing on is a legitimate entry, not a missing one."""
    with tempfile.TemporaryDirectory() as tmp:
        header = CacheHeader(source_size=256, num_samples=8)
        path = os.path.join(tmp, "x.pt")
        write_cache(path, header, [("a", np.zeros((0, 8, 2), np.float32)),
                                   ("b", np.ones((3, 8, 2), np.float32))])
        cache = LineCache.load(path)
        assert cache.get("a") is not None and len(cache.get("a")) == 0
        assert len(cache.get("b")) == 3


@needs_frames
def test_the_dataset_ignores_the_cache_when_the_flag_is_unset(cache_dir_256, capsys):
    """Unset means today's behaviour exactly, cache files on disk or not."""
    plain = dataset()
    assert plain.line_cache_dir is None
    sample_at(plain, 0, 21)
    assert "line cache" not in capsys.readouterr().out
