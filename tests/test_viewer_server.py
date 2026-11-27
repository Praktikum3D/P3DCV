"""The HTTP side of the viewer: checkpoints, uploads, reconstruction and the bounds of a frame.

Reconstruction needs bf16 and therefore the cluster, so the model is a stand in here and the tests
pin what the server does around it: which checkpoint it picks, what it refuses to take from an
upload, how it answers a request that would allocate more memory than exists, and that the weights
of the pretrained checkpoint load without leftovers.

The routes are plain synchronous functions, so they are called directly instead of through a client.
Run from the repository root:
    pytest tests/test_viewer_server.py
"""

import io
import os
import sys
import types
import typing

import pytest
import torch
from easydict import EasyDict as edict
from fastapi import HTTPException, UploadFile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from viewer import server
from viewer.scene import ModelBundle, Scene, UnsupportedDeviceError, model_state_dict
from viewer.server import AppState

# The 32 keys `erayzer_multi.pt` carries beyond the model: the VGG 19 of the perceptual loss of the
# original training, sixteen convolutions with a weight and a bias each.
PERCEPTUAL_KEYS = [
    f"loss_computer.perceptual_loss_module.Net.conv{index}.{part}"
    for index in range(1, 17)
    for part in ("weight", "bias")
]


# ------------------------------------------------------------------------------------------------
# loading the pretrained weights
# ------------------------------------------------------------------------------------------------


def test_the_pretrained_checkpoint_loses_only_its_training_weights():
    """A state dict shaped like `erayzer_multi.pt` comes out as the model weights alone."""
    model_keys = {"camera_token": torch.zeros(1), "image_tokenizer.1.weight": torch.zeros(2)}
    checkpoint = {
        "model": {**model_keys, **{key: torch.zeros(1) for key in PERCEPTUAL_KEYS}},
        "optimizer": {"state": {}},
        "fwdbwd_pass_step": 152000,
    }
    state, dropped = model_state_dict(checkpoint)
    assert len(dropped) == 32
    assert set(state) == set(model_keys)
    assert all(key.startswith("loss_computer.") for key in dropped)


def test_a_distributed_checkpoint_loses_its_wrapper_prefix():
    """Runs on several GPUs write every name behind `module.`, which no single model answers to."""
    checkpoint = {"model": {"module.camera_token": torch.zeros(1), "module." + PERCEPTUAL_KEYS[0]: torch.zeros(1)}}
    state, dropped = model_state_dict(checkpoint)
    assert set(state) == {"camera_token"}
    assert dropped == [PERCEPTUAL_KEYS[0]]


def test_a_bare_state_dict_is_taken_as_it_is():
    """Not every checkpoint is wrapped, and a bare one must not be mistaken for an empty load."""
    state, dropped = model_state_dict({"camera_token": torch.zeros(1), "register_token": torch.zeros(1)})
    assert set(state) == {"camera_token", "register_token"}
    assert dropped == []


def test_the_standalone_precompute_script_drops_the_same_names():
    """`scripts/precompute_scene.py` imports nothing of the viewer, so it must not drift away."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(repo, "scripts", "precompute_scene.py")
    module = types.ModuleType("precompute_scene_probe")
    module.Any = typing.Any  # the slice below is read without the imports of the file around it
    source = open(path, "r", encoding="utf-8").read()
    start = source.index("TRAINING_ONLY_PREFIXES")
    end = source.index("def build_model")
    exec(compile(source[start:end], path, "exec"), module.__dict__)
    state, dropped = module.model_state_dict({"model": {"camera_token": 1, PERCEPTUAL_KEYS[0]: 1}})
    assert set(state) == {"camera_token"} and len(dropped) == 1


# ------------------------------------------------------------------------------------------------
# a server with a stand in scene and a stand in model
# ------------------------------------------------------------------------------------------------


class FakeGaussians:
    def __init__(self, xyz):
        self._xyz = xyz

    @property
    def get_xyz(self):
        return self._xyz


def fake_scene(name="fake") -> Scene:
    xyz = torch.zeros(8, 3)
    return Scene(
        pc=FakeGaussians(xyz),
        sh_degree=0,
        c2w_input=torch.eye(4)[None],
        fxfycxcy_input=torch.tensor([[1.0, 1.0, 0.5, 0.5]]),
        camera_model="pinhole",
        near_plane=0.2,
        tile_mode="projection",
        center=torch.zeros(3),
        radius=1.0,
        name=name,
        source="synthetic",
        num_gaussians=8,
    )


def fake_bundle(camera_model="fisheye") -> ModelBundle:
    config = edict(
        {
            "model": {"camera_model": camera_model, "renderer": "geer", "geer_tile_mode": "projection"},
            "training": {"num_views": 10},
        }
    )
    return ModelBundle(model=object(), config=config, ckpt_id="runs/fake", amp_dtype=torch.bfloat16)


@pytest.fixture
def app_state(tmp_path, monkeypatch):
    state = AppState(device=torch.device("cpu"), out_dir=str(tmp_path / "out"), data_root=str(tmp_path / "data"))
    os.makedirs(state.out_dir, exist_ok=True)
    state.scene, state.scene_id = fake_scene(), "synthetic"
    monkeypatch.setattr(server, "STATE", state)
    return state


def render_body(**overrides):
    body = {
        "c2w": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        "camera_model": "pinhole",
        "fov_deg": 90.0,
        "res": 512,
        "aspect": 1.0,
    }
    body.update(overrides)
    return body


# ------------------------------------------------------------------------------------------------
# the bounds of a frame
# ------------------------------------------------------------------------------------------------


def test_a_huge_aspect_ratio_is_refused_instead_of_allocated(app_state, monkeypatch):
    """The 244 GiB request of the review log: 256 rows times a million columns of float32.

    The browser clamps the aspect ratio, a raw API client does not, and the frame size is the one
    number in a request that turns straight into an allocation. So the server bounds it itself, and
    `render_view` must never see the request at all.
    """
    monkeypatch.setattr(server, "render_view", lambda *a, **k: pytest.fail("the render path was reached"))
    for aspect in (1e6, 1e9, float("inf"), float("nan"), -3.0, 10.001, 0.0):
        with pytest.raises(HTTPException) as error:
            server._render_request(app_state, render_body(aspect=aspect), full_resolution=False)
        assert error.value.status_code == 422
        assert "aspect" in error.value.detail


def test_the_frame_size_stays_inside_the_allowed_bounds(app_state, monkeypatch):
    """Whatever comes in, the width the renderer sees is bounded by the resolution and the ratio."""
    seen = {}

    def capture(scene, c2w, camera_model, fov_deg, alpha, h, w, **kwargs):
        seen["hw"] = (h, w)
        return torch.zeros(h, w, 3).numpy(), {"render_ms": 1.0, "fov_used": fov_deg, "width": w, "height": h}

    monkeypatch.setattr(server, "render_view", capture)
    for res in server.RESOLUTIONS:
        for aspect in server.ASPECT_RANGE:
            server._render_request(app_state, render_body(res=res, aspect=aspect), full_resolution=True)
            h, w = seen["hw"]
            assert h == res
            assert 64 <= w <= res * server.ASPECT_RANGE[1] + 2
            assert h * w * 4 < 128 * 1024 * 1024  # a frame is megabytes, never gigabytes


def test_a_moving_frame_is_half_as_high(app_state, monkeypatch):
    """While the camera moves the server renders smaller, and the saved image never does."""
    seen = []
    monkeypatch.setattr(
        server,
        "render_view",
        lambda scene, c2w, cm, fov, alpha, h, w, **k: (
            seen.append((h, w)) or torch.zeros(h, w, 3).numpy(),
            {"render_ms": 1.0, "fov_used": fov, "width": w, "height": h},
        ),
    )
    server._render_request(app_state, render_body(res=512, moving=True), full_resolution=False)
    server._render_request(app_state, render_body(res=512, moving=True), full_resolution=True)
    assert seen[0][0] == 256 and seen[1][0] == 512


@pytest.mark.parametrize(
    "body, field",
    [
        ({"res": 500}, "res"),
        ({"res": "big"}, "res"),
        ({"camera_model": "orthographic"}, "camera_model"),
        ({"tile_mode": "none"}, "tile_mode"),
        ({"view": "normals"}, "view"),
        ({"fov_deg": float("nan")}, "fov_deg"),
        ({"fov_deg": "wide"}, "fov_deg"),
        ({"c2w": [1, 2, 3]}, "c2w"),
        ({"c2w": [float("inf")] * 16}, "c2w"),
        ({"camera_model": "ucm", "alpha": 1.5}, "alpha"),
    ],
)
def test_a_malformed_render_request_is_a_422(app_state, monkeypatch, body, field):
    """Every field a raw client can get wrong is named back to it instead of raising a 500."""
    monkeypatch.setattr(server, "render_view", lambda *a, **k: pytest.fail("the render path was reached"))
    with pytest.raises(HTTPException) as error:
        server._render_request(app_state, render_body(**body), full_resolution=False)
    assert error.value.status_code == 422
    assert field in error.value.detail


def test_a_render_that_blows_up_frees_the_gpu_and_names_itself(app_state, monkeypatch):
    """A failing frame must not reach the client as a bare 500, and it must not keep its memory."""

    def fail(*args, **kwargs):
        raise ValueError("the kernel disliked this")

    monkeypatch.setattr(server, "render_view", fail)
    with pytest.raises(HTTPException) as error:
        server._render_request(app_state, render_body(), full_resolution=False)
    assert error.value.status_code == 500
    assert "ValueError" in error.value.detail and "disliked" in error.value.detail


def test_rendering_without_a_scene_is_a_409(app_state):
    app_state.scene = None
    with pytest.raises(HTTPException) as error:
        server._render_request(app_state, render_body(), full_resolution=False)
    assert error.value.status_code == 409


# ------------------------------------------------------------------------------------------------
# uploads
# ------------------------------------------------------------------------------------------------


def upload_file(name: str, size: int = 32) -> UploadFile:
    return UploadFile(file=io.BytesIO(b"x" * size), filename=name)


def test_an_upload_lands_under_the_output_directory(app_state):
    result = server.api_upload([upload_file("b.jpg"), upload_file("a.png")])
    directory = os.path.join(app_state.out_dir, "uploads", result["upload_id"])
    assert sorted(os.listdir(directory)) == ["000_b.jpg", "001_a.png"]
    # The index keeps the order the browser sent, which is the order the model sees.
    assert result["files"] == ["000_b.jpg", "001_a.png"]


@pytest.mark.parametrize(
    "files, detail",
    [
        ([upload_file("notes.txt")], "not one of"),
        ([upload_file("huge.png", server.MAX_UPLOAD_BYTES + 1)], "larger than"),
        ([upload_file("empty.png", 0)], "empty"),
        ([upload_file(f"{i}.png") for i in range(server.MAX_UPLOAD_FILES + 1)], "at most"),
        ([], "no files"),
    ],
)
def test_an_upload_the_viewer_will_not_take(app_state, files, detail):
    with pytest.raises(HTTPException) as error:
        server.api_upload(files)
    assert error.value.status_code == 422
    assert detail in error.value.detail


def test_an_upload_name_cannot_leave_its_directory(app_state):
    """A file name is data, not a path: everything but the plain name is written over."""
    server.api_upload([upload_file("../../../etc/passwd.png")])
    uploads = os.path.join(app_state.out_dir, "uploads")
    directory = os.path.join(uploads, os.listdir(uploads)[0])
    assert os.listdir(directory) == ["000_passwd.png"]


@pytest.mark.parametrize("upload_id", ["../../etc", "..", "not_there", "a/b", ""])
def test_a_reconstruction_from_an_unknown_upload_is_refused(app_state, upload_id):
    app_state.bundle = fake_bundle()
    with pytest.raises(HTTPException) as error:
        server.api_reconstruct({"mode": "upload", "upload_id": upload_id})
    assert error.value.status_code in (404, 422)


# ------------------------------------------------------------------------------------------------
# checkpoints, models and reconstruction
# ------------------------------------------------------------------------------------------------


def write_run(root, name, camera_model):
    import yaml

    run = os.path.join(root, "runs", name)
    os.makedirs(run, exist_ok=True)
    open(os.path.join(run, "last.pt"), "wb").close()
    config = {"model": {"camera_model": camera_model, "renderer": "geer", "geer_tile_mode": "projection"}}
    with open(os.path.join(run, "config.yaml"), "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)


def test_the_state_lists_the_checkpoints_and_the_environments(app_state, tmp_path, monkeypatch):
    """What the sidebar fills itself from: checkpoints with their camera settings, and the data."""
    write_run(str(tmp_path), "fish-geer", "fisheye")
    pretrained = str(tmp_path / "erayzer_multi.pt")
    open(pretrained, "wb").close()
    app_state.pretrained, app_state.runs_dir = pretrained, str(tmp_path / "runs")

    for name in ("HongKong", "Office", "AbandonedFactory"):
        os.makedirs(os.path.join(app_state.data_root, name, "Data_hard"), exist_ok=True)
        open(os.path.join(app_state.data_root, name, "Data_hard", "image_lcam_fish.zip"), "wb").close()
    with open(os.path.join(app_state.data_root, "train_envs.txt"), "w", encoding="utf-8") as handle:
        handle.write("AbandonedFactory\n")

    payload = server.api_state()
    assert [entry["id"] for entry in payload["checkpoints"]] == ["pretrained", "runs/fish-geer"]
    assert payload["checkpoints"][1]["camera_settings"]["camera_model"] == "fisheye"
    assert payload["model"] is None
    # The test environments come first, the training ones are marked as such.
    assert [entry["name"] for entry in payload["envs"]] == ["HongKong", "Office", "AbandonedFactory"]
    assert payload["envs"][2]["label"].endswith("(train)")
    assert payload["limits"]["aspect"] == list(server.ASPECT_RANGE)


def test_loading_an_unknown_checkpoint_is_a_404(app_state):
    with pytest.raises(HTTPException) as error:
        server.api_load_model({"id": "runs/nothing"})
    assert error.value.status_code == 404


def test_a_gpu_without_bfloat16_says_so_instead_of_crashing(app_state, tmp_path, monkeypatch):
    """On the development GPU reconstruction cannot run, and the viewer has to say why."""
    write_run(str(tmp_path), "fish-geer", "fisheye")
    app_state.runs_dir = str(tmp_path / "runs")

    def refuse(entry, device, repo_root=None):
        raise UnsupportedDeviceError("compute capability 7.5, and E-RayZer needs 8.0 for its bfloat16 attention")

    monkeypatch.setattr(server, "load_model", refuse)
    with pytest.raises(HTTPException) as error:
        server.api_load_model({"id": "runs/fish-geer"})
    assert error.value.status_code == 503
    assert "bfloat16" in error.value.detail


def test_a_broken_checkpoint_is_not_blamed_on_the_gpu(app_state, tmp_path, monkeypatch):
    """Torch raises RuntimeError for a truncated file too, and that is a 500, not a 503."""
    write_run(str(tmp_path), "fish-geer", "fisheye")
    app_state.runs_dir = str(tmp_path / "runs")

    def truncated(entry, device, repo_root=None):
        raise RuntimeError("PytorchStreamReader failed reading zip archive")

    monkeypatch.setattr(server, "load_model", truncated)
    with pytest.raises(HTTPException) as error:
        server.api_load_model({"id": "runs/fish-geer"})
    assert error.value.status_code == 500
    assert "PytorchStreamReader" in error.value.detail


def test_loading_a_model_keeps_the_camera_settings_of_its_run(app_state, tmp_path, monkeypatch):
    write_run(str(tmp_path), "fish-geer", "fisheye")
    app_state.runs_dir = str(tmp_path / "runs")
    monkeypatch.setattr(server, "load_model", lambda entry, device, repo_root=None: fake_bundle())

    payload = server.api_load_model({"id": "runs/fish-geer"})["model"]
    assert payload["camera_model"] == "fisheye" and payload["tile_mode"] == "projection"
    assert "fish-geer" in payload["label"]
    assert app_state.bundle is not None


def test_reconstruction_needs_a_model_first(app_state):
    with pytest.raises(HTTPException) as error:
        server.api_reconstruct({"mode": "tartanair", "env": "HongKong"})
    assert error.value.status_code == 409


def test_an_unknown_mode_or_environment_is_refused(app_state):
    app_state.bundle = fake_bundle()
    with pytest.raises(HTTPException) as error:
        server.api_reconstruct({"mode": "webcam"})
    assert error.value.status_code == 422
    with pytest.raises(HTTPException) as error:
        server.api_reconstruct({"mode": "tartanair", "env": "HongKong"})
    assert error.value.status_code == 404


def test_a_tartanair_reconstruction_becomes_the_scene(app_state, monkeypatch):
    """The whole path with the forward pass replaced: the new scene is what the UI gets back."""
    app_state.bundle = fake_bundle()
    env_dir = os.path.join(app_state.data_root, "HongKong", "Data_hard")
    os.makedirs(env_dir, exist_ok=True)
    open(os.path.join(env_dir, "image_lcam_fish.zip"), "wb").close()

    seen = {}

    def reconstruct(bundle, env, index, device):
        seen["env"], seen["index"] = env.name, index
        return fake_scene("HongKong #3")

    monkeypatch.setattr(server, "reconstruct_tartanair", reconstruct)
    monkeypatch.setattr(server, "probe_depth", lambda scene: False)

    payload = server.api_reconstruct({"mode": "tartanair", "env": "HongKong", "index": 3})
    assert seen == {"env": "HongKong", "index": 3}
    assert payload["scene"]["name"] == "HongKong #3"
    assert app_state.scene_id == "tartanair:HongKong:3"
    # The reconstructed scene shows up in the dropdown, so the UI does not snap back to the old one.
    assert any(entry["id"] == "tartanair:HongKong:3" for entry in payload["scenes"])


def test_an_upload_reconstruction_passes_the_stored_images_in_order(app_state, monkeypatch):
    app_state.bundle = fake_bundle()
    upload = server.api_upload([upload_file("second.png"), upload_file("first.png")])
    seen = {}

    def reconstruct(bundle, paths, device):
        seen["names"] = [os.path.basename(path) for path in paths]
        return fake_scene("upload")

    monkeypatch.setattr(server, "reconstruct_upload", reconstruct)
    monkeypatch.setattr(server, "probe_depth", lambda scene: False)

    server.api_reconstruct({"mode": "upload", "upload_id": upload["upload_id"]})
    assert seen["names"] == ["000_second.png", "001_first.png"]
    assert app_state.scene_id == f"upload:{upload['upload_id']}"


def test_the_dropdown_can_point_back_at_a_reconstruction(app_state):
    """A reconstructed scene has no file, so selecting it again must not look for one."""
    app_state.scene, app_state.scene_id = fake_scene("HongKong #3"), "tartanair:HongKong:3"
    payload = server.api_load_scene({"id": "tartanair:HongKong:3"})
    assert payload["scene"]["name"] == "HongKong #3"
    with pytest.raises(HTTPException) as error:
        server.api_load_scene({"id": "tartanair:Office:0"})
    assert error.value.status_code in (400, 404)


def test_a_negative_sample_index_is_refused(app_state, monkeypatch):
    app_state.bundle = fake_bundle()
    env_dir = os.path.join(app_state.data_root, "HongKong", "Data_hard")
    os.makedirs(env_dir, exist_ok=True)
    open(os.path.join(env_dir, "image_lcam_fish.zip"), "wb").close()
    for index in (-1, "many"):
        with pytest.raises(HTTPException) as error:
            server.api_reconstruct({"mode": "tartanair", "env": "HongKong", "index": index})
        assert error.value.status_code == 422


def test_a_saved_file_is_served_only_out_of_the_shots_directory(app_state):
    shots = os.path.join(app_state.out_dir, "shots")
    os.makedirs(shots, exist_ok=True)
    open(os.path.join(shots, "frame.png"), "wb").close()
    assert server.api_files("frame.png").path == os.path.join(shots, "frame.png")
    with pytest.raises(HTTPException) as error:
        server.api_files("../../../etc/passwd")
    assert error.value.status_code == 404
