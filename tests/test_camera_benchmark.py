"""The success criteria harness: one checkpoint against cameras whose true parameters we know.

The weekend goal is measured by three numbers, and every one of them is only worth reading if the
harness that produced it is trustworthy:

* the field of view error per test camera, which needs the true field of view to be exact (it comes
  out of the remap, so it is exact by construction) and the predicted one to be read with the right
  formula (there is a different one per camera model, and getting it wrong moves the number by tens
  of degrees),
* the alpha error, which needs the true alpha of an equidistant camera to be its UCM fit,
* PSNR and SSIM, which are only comparable between two checkpoints if both are scored on exactly the
  same pixels, so the mask must be a function of the TEST camera and of nothing else.

Plus the thing that makes a comparison a comparison at all: two runs of the harness have to feed the
model the same frames, which is why the frame selection hangs off a seed and nothing else.

The model forward cannot run on a GTX 1660 (xformers wants compute capability 8.0), so these tests
replace the model with a stub that predicts a fixed camera. That is enough to pin everything the
harness itself computes; what the real network predicts is the subject of the cluster run.

Run from the repository root:
    pytest tests/test_camera_benchmark.py
"""

import csv
import math
import os
import sys

import numpy as np
import pytest
import torch
from easydict import EasyDict as edict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

cv2 = pytest.importorskip("cv2")

from PIL import Image  # noqa: E402

import evaluate  # noqa: E402
import scripts.camera_benchmark as bench  # noqa: E402
from erayzer_core.model.utils_gaussian import camera_theta  # noqa: E402
from training.camera_augmentation import (  # noqa: E402
    FISH_SIZE,
    AugCamera,
    fit_ucm_alpha,
    focal_for_fov,
    remap_views,
    ucm_rho_norm,
    ucm_theta_max,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(REPO_ROOT, "config", "erayzer_train_tartanair.yaml")


# --- the fixed test camera list ---------------------------------------------------------------


def test_the_camera_list_is_the_documented_one():
    """Fifteen cameras, three families, no randomness and no duplicate row names."""
    cameras = bench.test_cameras()
    names = [bench.camera_name(camera) for camera in cameras]
    assert names == [
        "pinhole_060", "pinhole_090", "pinhole_120",
        "fisheye_120", "fisheye_150", "fisheye_195",
        "ucm_130_a040", "ucm_130_a062", "ucm_130_a080",
        "ucm_160_a040", "ucm_160_a062", "ucm_160_a080",
        "ucm_190_a040", "ucm_190_a062", "ucm_190_a080",
    ]
    assert len(set(names)) == len(names)
    assert cameras == bench.test_cameras(), "the list must not depend on call order or on a seed"


def test_every_ucm_test_camera_can_reach_its_field_of_view():
    """theta_max(alpha) > fov / 2, otherwise the camera has no focal length at all."""
    for camera in bench.test_cameras():
        if camera.model != "ucm":
            continue
        half = math.radians(camera.fov_deg) / 2.0
        assert half < ucm_theta_max(float(camera.alpha)), camera
        # And the constructive proof: a focal length exists and puts the half angle on the edge.
        f = focal_for_fov("ucm", camera.fov_deg, 256, camera.alpha)
        assert ucm_rho_norm(half, float(camera.alpha)) * f == pytest.approx(128.0)


def test_the_fields_of_view_cover_the_required_range():
    """The criterion is stated for 60 to 195 degrees, so the list has to span it."""
    cameras = bench.test_cameras()
    fovs = [camera.fov_deg for camera in cameras]
    assert min(fovs) == 60.0 and max(fovs) == 195.0
    assert all(60.0 <= fov <= 195.0 for fov in fovs)
    assert all(fov < 180.0 for camera, fov in zip(cameras, fovs) if camera.model == "pinhole")
    assert sorted({camera.model for camera in cameras}) == ["fisheye", "pinhole", "ucm"]
    # The source camera is 195 degrees, so nothing wider can be resampled out of it.
    assert max(fovs) <= bench.FISH_FOV_DEG


def test_the_true_alpha_of_every_camera_is_defined():
    """Exact for pinhole and UCM, the fit of the true mapping for the equidistant fisheye."""
    for camera in bench.test_cameras():
        alpha = bench.true_alpha(camera)
        assert math.isfinite(alpha)
        if camera.model == "pinhole":
            assert alpha == 0.0
        elif camera.model == "ucm":
            assert alpha == pytest.approx(float(camera.alpha))
        else:
            assert alpha == pytest.approx(fit_ucm_alpha(camera.fov_deg))
            assert 0.0 < alpha < 1.0


# --- the remap gives back the camera that was asked for ----------------------------------------


@pytest.mark.parametrize("camera", bench.test_cameras(), ids=bench.camera_name)
def test_the_remap_intrinsics_round_trip_to_the_requested_field_of_view(camera):
    """The focal length `remap_views` reports, read back with the model's own law, is the input fov.

    This is the link between the true value in the report and the camera the pixels actually come
    from: if the remap and `fov_from_fx` disagreed, every error in the table would be that
    disagreement rather than the model's.
    """
    size = 64
    exact = focal_for_fov(camera.model, camera.fov_deg, size, camera.alpha)
    alpha = None if camera.alpha is None else torch.tensor([float(camera.alpha)], dtype=torch.float64)
    recovered = bench.fov_from_fx(torch.tensor([exact / size], dtype=torch.float64), camera.model, alpha)
    assert float(recovered.item()) == pytest.approx(camera.fov_deg, abs=1e-6)

    # And the remap really hands back that focal length, up to the float32 it stores it in.
    frame = np.zeros((FISH_SIZE, FISH_SIZE, 3), dtype=np.uint8)
    _, _, intrinsics, _ = remap_views([frame], camera, size, supersample=1)
    assert float(intrinsics[0]) == pytest.approx(exact, rel=1e-6)
    assert float(intrinsics[1]) == pytest.approx(exact, rel=1e-6)
    assert float(intrinsics[2]) == size / 2.0 and float(intrinsics[3]) == size / 2.0


# --- reading the predicted camera --------------------------------------------------------------


def test_fov_from_fx_follows_the_law_of_every_camera_model():
    """One formula per model, written out independently of `camera_theta` here."""
    fx = torch.tensor([0.5, 0.29383, 0.75], dtype=torch.float64)
    rho = 0.5 / fx

    pinhole = bench.fov_from_fx(fx, "pinhole")
    assert torch.allclose(pinhole, torch.rad2deg(2.0 * torch.atan(rho)))
    assert float(bench.fov_from_fx(torch.tensor([0.5], dtype=torch.float64), "pinhole")) == pytest.approx(90.0)

    fisheye = bench.fov_from_fx(fx, "fisheye")
    assert torch.allclose(fisheye, torch.rad2deg(2.0 * rho))

    alpha = torch.full_like(fx, 0.62)
    ucm = bench.fov_from_fx(fx, "ucm", alpha)
    # Forward direction: the recovered half angle projects back onto the image edge radius.
    half = torch.deg2rad(ucm) / 2.0
    assert torch.allclose(torch.sin(half) / (alpha + (1 - alpha) * torch.cos(half)), rho, atol=1e-12)

    beta = torch.ones_like(fx)
    eucm = bench.fov_from_fx(fx, "eucm", alpha, beta)
    assert torch.equal(eucm, ucm), "an EUCM camera with beta one is a UCM camera"
    assert not torch.allclose(ucm, pinhole), "the models must not collapse onto one formula"
    assert not torch.allclose(ucm, fisheye)


def test_fov_from_fx_matches_what_training_logs():
    """The harness and `train.camera_stats` have to report the same number for the same prediction."""
    import train

    fx = torch.tensor([0.31, 0.33], dtype=torch.float32)
    alpha = torch.tensor([0.55, 0.57], dtype=torch.float32)
    result = edict(fxfycxcy=torch.stack([fx, fx, fx, fx, alpha], dim=-1))
    logged = train.camera_stats(result, "ucm")["camera/fov_deg"]
    # camera_stats averages the angle over the views and converts once, which is the same number.
    expected = float(bench.fov_from_fx(fx, "ucm", alpha).mean())
    assert logged == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize("camera_model,width", [("pinhole", 4), ("fisheye", 4), ("ucm", 5), ("eucm", 6)])
def test_predicted_camera_reads_the_right_columns(camera_model, width):
    values = torch.tensor([[0.3, 0.3, 0.5, 0.5, 0.62, 1.0][:width]] * 4)
    fov, alpha = bench.predicted_camera(values, camera_model)
    assert fov == pytest.approx(float(bench.fov_from_fx(torch.tensor([0.3]), camera_model,
                                                        torch.tensor([0.62]) if width >= 5 else None,
                                                        torch.tensor([1.0]) if width >= 6 else None)))
    assert alpha == (pytest.approx(0.62) if camera_model in ("ucm", "eucm") else None)


def test_predicted_camera_refuses_a_checkpoint_without_the_extra_intrinsics():
    """A pinhole checkpoint read as a UCM one would silently report a wrong field of view."""
    with pytest.raises(ValueError, match="needs 5"):
        bench.predicted_camera(torch.rand(4, 4), "ucm")


# --- the same pixels for every model ------------------------------------------------------------


def camera_intrinsics(camera, size=64):
    f = focal_for_fov(camera.model, camera.fov_deg, size, camera.alpha)
    return torch.tensor([f, f, size / 2.0, size / 2.0], dtype=torch.float32)


@pytest.mark.parametrize("camera", bench.test_cameras()[::4], ids=bench.camera_name)
def test_the_scoring_mask_depends_on_the_test_camera_only(camera):
    """Two checkpoints with different camera models score the identical pixel set."""
    size = 64
    frame = np.zeros((FISH_SIZE, FISH_SIZE, 3), dtype=np.uint8)
    _, valid, intrinsics, _ = remap_views([frame], camera, size, supersample=1)
    masks = [bench.scoring_mask(valid[:1], intrinsics, size, size, "circle") for _ in range(2)]
    assert torch.equal(masks[0], masks[1])
    # Nothing in the signature can carry a predicted camera: the only inputs are the remap mask and
    # the true intrinsics of the test camera, both of which are the same for every checkpoint.
    assert masks[0].shape == (1, 1, size, size)
    assert masks[0].sum() > 0, "a test camera that scores no pixel at all would be meaningless"


def test_the_circle_mask_is_the_remap_mask_inside_the_image_circle():
    size = 64
    camera = AugCamera("fisheye", 150.0)
    frame = np.zeros((FISH_SIZE, FISH_SIZE, 3), dtype=np.uint8)
    _, valid, intrinsics, _ = remap_views([frame], camera, size, supersample=1)
    disc = evaluate.image_circle_disc(intrinsics.reshape(1, -1), size, size)
    assert torch.equal(bench.scoring_mask(valid[:1], intrinsics, size, size, "circle"), valid[:1] * disc)
    assert torch.equal(bench.scoring_mask(valid[:1], intrinsics, size, size, "valid"), valid[:1])
    assert bench.scoring_mask(valid[:1], intrinsics, size, size, "full") is None
    with pytest.raises(ValueError, match="unknown mask mode"):
        bench.scoring_mask(valid[:1], intrinsics, size, size, "square")


def test_a_pinhole_test_camera_keeps_its_corners_only_without_the_circle():
    size = 64
    camera = AugCamera("pinhole", 60.0)
    frame = np.zeros((FISH_SIZE, FISH_SIZE, 3), dtype=np.uint8)
    _, valid, intrinsics, _ = remap_views([frame], camera, size, supersample=1)
    assert float(valid[:1].mean()) == pytest.approx(1.0), "a 60 degree pinhole is covered everywhere"
    circle = bench.scoring_mask(valid[:1], intrinsics, size, size, "circle")
    assert float(circle.mean()) == pytest.approx(math.pi / 4.0, abs=0.02)


# --- deterministic frame selection ---------------------------------------------------------------


def test_the_same_seed_picks_the_same_frames():
    lengths = [80, 65, 120, 88]
    first = bench.select_samples(lengths, 10, 4, 45, seed=7)
    second = bench.select_samples(lengths, 10, 4, 45, seed=7)
    assert first == second
    assert first != bench.select_samples(lengths, 10, 4, 45, seed=8)


def test_the_frames_of_a_sample_are_inside_its_trajectory():
    lengths = [80, 65, 12, 120]
    samples = bench.select_samples(lengths, 12, 4, 45, seed=3)
    assert len(samples) == 12
    for sample in samples:
        length = lengths[sample.trajectory]
        assert len(sample.frames) == 4
        assert list(sample.frames) == sorted(sample.frames)
        assert 0 <= min(sample.frames) and max(sample.frames) < length
    # All four trajectories are long enough for four views, so all four are used.
    assert {sample.trajectory for sample in samples} == {0, 1, 2, 3}


def test_a_trajectory_that_is_too_short_is_dropped():
    samples = bench.select_samples([3, 40], 4, 10, 20, seed=1)
    assert {sample.trajectory for sample in samples} == {1}
    with pytest.raises(ValueError, match="frames a sample needs"):
        bench.select_samples([3, 4], 2, 10, 20, seed=1)


# --- the CSV ---------------------------------------------------------------------------------------


def test_the_csv_schema_is_the_documented_one(tmp_path):
    assert bench.CSV_FIELDS == [
        "camera", "camera_model", "fov_true", "fov_pred_mean", "fov_pred_std", "fov_err", "fov_pass",
        "alpha_true", "alpha_pred_mean", "alpha_pred_std", "alpha_err",
        "psnr", "ssim", "mse", "mask_fraction", "RPA@5", "RPA@15", "RPA@30", "n_samples",
    ]
    camera = AugCamera("ucm", 160.0, 0.62)
    records = [
        {"mse": 0.01, "psnr": 20.0, "ssim": 0.8, "mask_fraction": 0.7, "fov_pred": 158.0,
         "alpha_pred": 0.6, "RPA@5": 0.5, "RPA@15": 0.75, "RPA@30": 1.0},
        {"mse": 0.02, "psnr": 17.0, "ssim": 0.7, "mask_fraction": 0.7, "fov_pred": 164.0,
         "alpha_pred": 0.64, "RPA@5": 0.5, "RPA@15": 0.75, "RPA@30": 1.0},
    ]
    row = bench.summarise_camera(camera, "ucm_160_a062", records)
    assert row["fov_true"] == 160.0 and row["alpha_true"] == pytest.approx(0.62)
    assert row["fov_pred_mean"] == pytest.approx(161.0)
    assert row["fov_err"] == pytest.approx(3.0), "the mean of the per sample errors, not of the values"
    assert row["fov_pass"] == 1
    assert row["alpha_err"] == pytest.approx(0.02)
    assert row["n_samples"] == 2

    path = str(tmp_path / "cameras.csv")
    bench.write_csv(path, [row])
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == bench.CSV_FIELDS
        written = list(reader)
    assert len(written) == 1 and written[0]["camera"] == "ucm_160_a062"
    assert float(written[0]["fov_err"]) == pytest.approx(3.0)


def test_a_camera_above_the_limit_fails():
    camera = AugCamera("fisheye", 195.0)
    records = [{"mse": 0.01, "psnr": 20.0, "ssim": 0.8, "mask_fraction": 0.7, "fov_pred": 182.0,
                "alpha_pred": None, "RPA@5": 0.0, "RPA@15": 0.0, "RPA@30": 0.0}]
    row = bench.summarise_camera(camera, "fisheye_195", records)
    assert row["fov_err"] == pytest.approx(13.0)
    assert row["fov_pass"] == 0
    text = bench.summary_text([row], {"checkpoint": "none"}, None)
    assert "FAIL" in text and "fisheye_195" in text


# --- end to end with a stubbed model ---------------------------------------------------------------


FAKE_FX = 0.32
"""Normalised focal length the stub predicts, whatever it is shown. Fixed so the table is known."""


class FakeModel:
    """Stands in for ERayZer: predicts one fixed camera and renders the input back, dimmed.

    xformers needs compute capability 8.0, so the real forward cannot run on this machine. What the
    stub has to reproduce is only the shape of the result the harness reads: the predicted
    intrinsics, the predicted poses and a render next to its target.
    """

    def __init__(self, camera_model, views):
        self.camera_model = camera_model
        self.views = views
        self.seen = []

    def eval(self):
        return self

    def __call__(self, batch):
        images = batch["image"]
        b, v = images.shape[:2]
        self.seen.append(images.detach().clone())
        width = {"ucm": 5, "eucm": 6}.get(self.camera_model, 4)
        row = [FAKE_FX, FAKE_FX, 0.5, 0.5, 0.62, 1.0][:width]
        return edict(
            render=(images * 0.9).clamp(0, 1),
            target_image=images,
            fxfycxcy=torch.tensor([row] * (b * v), dtype=torch.float32, device=images.device),
            c2w=batch["c2w"].clone(),
        )


def write_tartanair_tree(root, camera, frames, size):
    """A tiny unpacked TartanAir trajectory the dataset discovers like a real one."""
    traj = root / "Env" / "Data_hard" / "P000"
    image_dir = traj / f"image_{camera}"
    image_dir.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(0, 255, size, dtype=np.uint8)
    for index in range(frames):
        pattern = np.stack(
            [
                np.broadcast_to(grid[None, :], (size, size)),
                np.broadcast_to(grid[:, None], (size, size)),
                np.full((size, size), (index * 17) % 256, dtype=np.uint8),
            ],
            axis=-1,
        )
        Image.fromarray(pattern).save(image_dir / f"{index:06d}_{camera}_image.png")
    poses = np.zeros((frames, 7), dtype=np.float64)
    poses[:, 0] = np.arange(frames) * 0.1
    poses[:, 6] = 1.0
    np.savetxt(traj / "pose_lcam_front.txt", poses)
    return str(root / "Env" / "Data_hard")


def run_benchmark(tmp_path, monkeypatch, camera_model="ucm", camera_set="remap", extra=()):
    views = 4
    size = 32
    source = "lcam_fish" if camera_set == "remap" else "lcam_front"
    data = write_tartanair_tree(tmp_path / f"data_{source}", source, 6, FISH_SIZE if camera_set == "remap" else 64)
    out_dir = tmp_path / f"out_{camera_model}_{camera_set}_{len(list(tmp_path.iterdir()))}"
    holder = {}

    def fake_build_model(config, device, rank):
        holder["model"] = FakeModel(str(config.model.get("camera_model", "pinhole")), views)
        return holder["model"]

    monkeypatch.setattr(evaluate, "build_model", fake_build_model)
    bench.main(
        [
            "--config", CONFIG,
            "--checkpoint", CONFIG,  # only has to exist, the stub never reads it
            "--data", data,
            "--out-dir", str(out_dir),
            "--camera-set", camera_set,
            "--samples", "1",
            "--device", "cpu",
            "--supersample", "1",
            "-s", "model.camera_model", camera_model,
            "-s", "model.image_tokenizer.image_size", str(size),
            "-s", "model.image_tokenizer.patch_size", "16",
            "-s", "training.num_views", str(views),
            "-s", "training.use_amp", "false",
            *extra,
        ]
    )
    return out_dir, holder["model"]


def read_rows(out_dir):
    with open(os.path.join(str(out_dir), "cameras.csv"), encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_a_whole_run_writes_the_table_the_summary_and_the_panels(tmp_path, monkeypatch):
    out_dir, model = run_benchmark(tmp_path, monkeypatch)
    rows = read_rows(out_dir)
    assert [row["camera"] for row in rows] == [bench.camera_name(c) for c in bench.test_cameras()]
    assert len(model.seen) == len(rows), "one forward pass per test camera"

    expected_fov = float(bench.fov_from_fx(torch.tensor([FAKE_FX]), "ucm", torch.tensor([0.62])))
    for row in rows:
        assert float(row["fov_pred_mean"]) == pytest.approx(expected_fov, abs=1e-3)
        assert float(row["fov_err"]) == pytest.approx(abs(expected_fov - float(row["fov_true"])), abs=1e-3)
        assert float(row["alpha_pred_mean"]) == pytest.approx(0.62, abs=1e-6)
        assert int(row["n_samples"]) == 1
        assert 0.0 < float(row["mask_fraction"]) <= 1.0
        assert float(row["psnr"]) > 0.0

    for family in ["pinhole", "fisheye", "ucm"]:
        assert os.path.isfile(os.path.join(str(out_dir), f"panel_{family}.png"))
    summary = open(os.path.join(str(out_dir), "summary.txt"), encoding="utf-8").read()
    assert "mean field of view error over 15 cameras" in summary
    assert "PASS" in summary or "FAIL" in summary
    assert "frames, identical for every camera and every checkpoint" in summary
    assert "frames [0, 1, 3, 5]" in summary, "the frames have to be named, not just counted"


def test_two_runs_feed_the_model_identical_frames(tmp_path, monkeypatch):
    """Determinism, the property that makes two checkpoints comparable at all."""
    first_dir, first = run_benchmark(tmp_path, monkeypatch)
    second_dir, second = run_benchmark(tmp_path, monkeypatch)
    assert len(first.seen) == len(second.seen)
    for a, b in zip(first.seen, second.seen):
        assert torch.equal(a, b)
    assert (
        open(os.path.join(str(first_dir), "cameras.csv"), encoding="utf-8").read()
        == open(os.path.join(str(second_dir), "cameras.csv"), encoding="utf-8").read()
    )


def test_two_camera_models_are_scored_on_the_same_pixels(tmp_path, monkeypatch):
    """The same mask for every model, measured through the whole harness rather than on the function."""
    ucm_dir, _ = run_benchmark(tmp_path, monkeypatch, camera_model="ucm")
    pinhole_dir, _ = run_benchmark(tmp_path, monkeypatch, camera_model="pinhole")
    ucm_rows = {row["camera"]: row for row in read_rows(ucm_dir)}
    pinhole_rows = {row["camera"]: row for row in read_rows(pinhole_dir)}
    assert set(ucm_rows) == set(pinhole_rows)
    for name, row in ucm_rows.items():
        assert row["mask_fraction"] == pinhole_rows[name]["mask_fraction"], name
        # The stub renders the input back, so the same pixel set also gives the same image scores.
        assert row["psnr"] == pinhole_rows[name]["psnr"], name
        assert row["ssim"] == pinhole_rows[name]["ssim"], name
    # The predicted camera does differ, which is what the mask must not depend on.
    assert ucm_rows["fisheye_195"]["fov_pred_mean"] != pinhole_rows["fisheye_195"]["fov_pred_mean"]


def test_the_front_set_scores_the_real_pinhole_frames(tmp_path, monkeypatch):
    out_dir, model = run_benchmark(tmp_path, monkeypatch, camera_model="pinhole", camera_set="front")
    rows = read_rows(out_dir)
    assert len(rows) == 1
    assert rows[0]["camera"] == "front_lcam_front"
    assert float(rows[0]["fov_true"]) == pytest.approx(bench.FRONT_FOV_DEG)
    assert float(rows[0]["alpha_true"]) == 0.0
    assert len(model.seen) == 1
    # No remap: the frames are the dataset's own resized views, so they fill the frame.
    assert float(model.seen[0].min()) >= 0.0 and float(model.seen[0].max()) <= 1.0


def test_the_front_fov_can_be_overridden(tmp_path, monkeypatch):
    out_dir, _ = run_benchmark(
        tmp_path, monkeypatch, camera_model="pinhole", camera_set="front", extra=("--front-fov", "85.5")
    )
    assert float(read_rows(out_dir)[0]["fov_true"]) == pytest.approx(85.5)


def test_a_subset_of_cameras_can_be_selected(tmp_path, monkeypatch):
    out_dir, _ = run_benchmark(tmp_path, monkeypatch, extra=("--cameras", "pinhole_090,ucm_190_a062"))
    assert [row["camera"] for row in read_rows(out_dir)] == ["pinhole_090", "ucm_190_a062"]


# --- the command line refuses to do something meaningless --------------------------------------------


def test_a_missing_checkpoint_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="no checkpoint at"):
        bench.load_model(edict({"model": {}}), str(tmp_path / "nothing.pt"), torch.device("cpu"))


def test_missing_data_is_reported():
    with pytest.raises(FileNotFoundError, match="no TartanAir data at"):
        bench.build_frame_source(edict({"dataset": {}, "training": {}}), "/no/such/archive.zip", "lcam_fish", 45)


def test_a_missing_config_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="no config at"):
        bench.main(["--config", str(tmp_path / "nope.yaml"), "--checkpoint", "x", "--data", "y",
                    "--out-dir", str(tmp_path)])


def test_the_remap_set_refuses_a_pinhole_source(tmp_path):
    with pytest.raises(ValueError, match="has to be a fisheye camera"):
        bench.main(["--config", CONFIG, "--checkpoint", CONFIG, "--data", str(tmp_path),
                    "--out-dir", str(tmp_path), "--source-camera", "lcam_front"])


def test_an_unknown_camera_name_is_reported(tmp_path):
    with pytest.raises(ValueError, match="unknown camera names"):
        bench.main(["--config", CONFIG, "--checkpoint", CONFIG, "--data", str(tmp_path),
                    "--out-dir", str(tmp_path), "--cameras", "ucm_200_a062"])


def test_an_unknown_baseline_csv_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="baseline-csv"):
        bench.summary_text([], {}, str(tmp_path / "missing.csv"))
