"""The viewer server: a small FastAPI app that renders one frame per request with 3DGEER.

There is no websocket, because the container image carries uvicorn without its websocket extras.
The browser instead runs a sequential fetch loop: it posts the current camera, waits for the JPEG and
only then asks for the next frame. That throttles itself to one request in flight, so the newest
camera always wins and the GPU never queues up stale frames.

Every route is a synchronous `def`, which FastAPI runs in its threadpool, and one lock serialises all
GPU work, so a second browser tab cannot render into a scene that is being replaced, and a frame
cannot arrive in the middle of a reconstruction.

A session picks a checkpoint (`/api/load_model`), then a scene, which is either a precomputed file
(`/api/load_scene`) or a fresh reconstruction of uploaded images or of a TartanAir sample
(`/api/upload`, `/api/reconstruct`), and then renders it (`/api/render`, `/api/save`). Reconstructing
needs bf16 and therefore a GPU of compute capability 8.0; rendering does not.

    python -m viewer.server --host 127.0.0.1 --port 7870 --scenes ~/scenes

The host is deliberately not 0.0.0.0 by default. Bind to the address of the interface you want to
reach the viewer on, and keep it inside the VPN.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response

from .camera import CAMERA_MODELS, estimated_camera_info
from .render import VIEWS, RenderStats, encode_jpeg, encode_png, probe_depth, render_view
from .scene import (
    IMAGE_SUFFIXES,
    CheckpointEntry,
    ModelBundle,
    Scene,
    list_checkpoints,
    list_envs,
    load_model,
    load_scene,
    make_synthetic_scene,
    reconstruct_tartanair,
    reconstruct_upload,
)

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESOLUTIONS = (256, 384, 512, 768)
TILE_MODES = ("projection", "pbf")
LOG_EVERY = 100

# A raw API client is not the browser and does not clamp anything, so the server does it. Without
# this an aspect ratio of a million turned a 256 pixel high frame into a 256 million pixel wide one
# and the render path asked CUDA for 244 GiB.
ASPECT_RANGE = (0.1, 10.0)
MAX_UPLOAD_FILES = 20
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
UPLOAD_NAME = re.compile(r"[^A-Za-z0-9._-]")


@dataclass
class AppState:
    device: torch.device
    out_dir: str
    scenes_dir: Optional[str] = None
    pretrained: Optional[str] = None
    runs_dir: Optional[str] = None
    data_root: Optional[str] = None
    camera_name: str = "lcam_fish"
    gpu_lock: threading.Lock = field(default_factory=threading.Lock)
    scene: Optional[Scene] = None
    scene_id: str = ""
    depth_available: bool = False
    bundle: Optional[ModelBundle] = None
    bundle_label: str = ""
    stats: RenderStats = field(default_factory=RenderStats)


STATE: Optional[AppState] = None
app = FastAPI(title="3DGEER viewer")


def state() -> AppState:
    if STATE is None:
        raise HTTPException(503, "the server has no state yet")
    return STATE


def available_scenes(app_state: AppState) -> List[Dict[str, str]]:
    """Every scene the viewer can load: the precomputed dumps plus the built in synthetic one.

    A scene that came out of a reconstruction has no file behind it, so it is listed as it is, which
    keeps the dropdown showing what is on screen instead of jumping back to the previous entry.
    """
    entries = [{"id": "synthetic", "label": "synthetic (development scene)"}]
    if app_state.scenes_dir and os.path.isdir(app_state.scenes_dir):
        for name in sorted(os.listdir(app_state.scenes_dir)):
            if name.endswith(".pt"):
                path = os.path.join(app_state.scenes_dir, name)
                size = os.path.getsize(path) / (1024.0 * 1024.0)
                entries.append({"id": name, "label": f"{name} ({size:.0f} MB)"})
    known = {entry["id"] for entry in entries}
    if app_state.scene is not None and app_state.scene_id not in known:
        entries.append({"id": app_state.scene_id, "label": f"{app_state.scene.name} (reconstructed)"})
    return entries


def checkpoint_entries(app_state: AppState) -> List[CheckpointEntry]:
    return list_checkpoints(app_state.pretrained, app_state.runs_dir)


def model_payload(app_state: AppState) -> Optional[Dict[str, Any]]:
    bundle = app_state.bundle
    if bundle is None:
        return None
    model_config = bundle.config.model
    return {
        "id": bundle.ckpt_id,
        "label": app_state.bundle_label,
        "camera_model": model_config.get("camera_model", "pinhole"),
        "renderer": model_config.get("renderer", "gsplat"),
        "tile_mode": model_config.get("geer_tile_mode", "pbf"),
        "alpha_init": float(model_config.get("alpha_init", 0.5)),
        "num_views": int(bundle.config.training.get("num_views", 10)),
        "amp_dtype": str(bundle.amp_dtype).replace("torch.", "") if bundle.amp_dtype else "fp32",
    }


def load_scene_by_id(app_state: AppState, scene_id: str) -> Scene:
    if scene_id == "synthetic":
        return make_synthetic_scene(app_state.device)
    if app_state.scene is not None and scene_id == app_state.scene_id:
        return app_state.scene  # a reconstruction has no file to read it back from
    if not app_state.scenes_dir:
        raise HTTPException(400, "no scene directory configured")
    path = os.path.join(app_state.scenes_dir, os.path.basename(scene_id))
    if not os.path.isfile(path):
        raise HTTPException(404, f"no scene {scene_id!r}")
    return load_scene(path, app_state.device)


def scene_payload(app_state: AppState) -> Optional[Dict[str, Any]]:
    scene = app_state.scene
    if scene is None:
        return None
    info = estimated_camera_info(scene.fxfycxcy_input, scene.camera_model)
    return {
        "id": app_state.scene_id,
        "name": scene.name,
        "source": scene.source,
        "num_gaussians": scene.num_gaussians,
        "camera_model": scene.camera_model,
        "tile_mode": scene.tile_mode,
        "near_plane": scene.near_plane,
        "alpha_init": scene.alpha_init,
        "center": [float(x) for x in scene.center],
        "radius": scene.radius,
        "views": [
            {"index": i, "c2w": [float(x) for x in scene.c2w_input[i].reshape(-1)], **info[i]}
            for i in range(scene.c2w_input.shape[0])
        ],
    }


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/app.js")
def app_js():
    return FileResponse(os.path.join(STATIC_DIR, "app.js"), media_type="application/javascript")


@app.get("/style.css")
def style_css():
    return FileResponse(os.path.join(STATIC_DIR, "style.css"), media_type="text/css")


@app.get("/api/state")
def api_state():
    app_state = state()
    return {
        "device": str(app_state.device),
        "scenes": available_scenes(app_state),
        "scene": scene_payload(app_state),
        "checkpoints": [
            {"id": entry.id, "label": entry.label, "camera_settings": entry.camera_settings}
            for entry in checkpoint_entries(app_state)
        ],
        "model": model_payload(app_state),
        "envs": [
            {"name": entry.name, "label": entry.name if entry.split == "test" else f"{entry.name} (train)"}
            for entry in list_envs(app_state.data_root, app_state.camera_name)
        ],
        "depth_available": app_state.depth_available,
        "resolutions": list(RESOLUTIONS),
        "limits": {
            "aspect": list(ASPECT_RANGE),
            "max_upload_files": MAX_UPLOAD_FILES,
            "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
            "image_suffixes": list(IMAGE_SUFFIXES),
        },
        "stats": app_state.stats.summary(),
    }


@app.post("/api/load_model")
def api_load_model(body: dict = Body(...)):
    """Build the model of one checkpoint with the config of its own run. Replaces any loaded model."""
    app_state = state()
    wanted = str(body.get("id", ""))
    entries = {entry.id: entry for entry in checkpoint_entries(app_state)}
    if wanted not in entries:
        raise HTTPException(404, f"no checkpoint {wanted!r}")
    entry = entries[wanted]
    with app_state.gpu_lock:
        app_state.bundle, app_state.bundle_label = None, ""  # free the old model before building
        if app_state.device.type == "cuda":
            torch.cuda.empty_cache()
        started = time.perf_counter()
        try:
            bundle = load_model(entry, app_state.device, repo_root=REPO_ROOT)
        except RuntimeError as error:  # no bfloat16 on this GPU is the expected one
            raise HTTPException(503, str(error))
        except Exception as error:  # a truncated file or a config that does not fit its weights
            raise HTTPException(500, f"loading {entry.ckpt_path} failed, {type(error).__name__}: {error}")
        app_state.bundle, app_state.bundle_label = bundle, entry.label
    return {"model": model_payload(app_state), "seconds": time.perf_counter() - started}


def _upload_dir(app_state: AppState, upload_id: str) -> str:
    """The folder of one upload, rejecting any id that tries to leave the upload root."""
    if not re.fullmatch(r"[0-9A-Za-z_-]{1,64}", upload_id):
        raise HTTPException(422, "upload_id must be the id a previous upload returned")
    path = os.path.join(app_state.out_dir, "uploads", upload_id)
    if not os.path.isdir(path):
        raise HTTPException(404, f"no upload {upload_id!r}")
    return path


@app.post("/api/upload")
def api_upload(files: List[UploadFile] = File(...)):
    """Take a handful of images, store them under the output directory and name the batch."""
    app_state = state()
    if not files:
        raise HTTPException(422, "no files")
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(422, f"at most {MAX_UPLOAD_FILES} images per upload, got {len(files)}")

    upload_id = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
    target = os.path.join(app_state.out_dir, "uploads", upload_id)
    os.makedirs(target, exist_ok=True)
    stored: List[str] = []
    for index, upload in enumerate(files):
        name = UPLOAD_NAME.sub("_", os.path.basename(upload.filename or ""))
        if os.path.splitext(name)[1].lower() not in IMAGE_SUFFIXES:
            raise HTTPException(422, f"{upload.filename!r} is not one of {', '.join(IMAGE_SUFFIXES)}")
        payload = upload.file.read(MAX_UPLOAD_BYTES + 1)
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(422, f"{name} is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
        if not payload:
            raise HTTPException(422, f"{name} is empty")
        path = os.path.join(target, f"{index:03d}_{name}")
        with open(path, "wb") as handle:
            handle.write(payload)
        stored.append(os.path.basename(path))
    return {"upload_id": upload_id, "files": stored}


@app.post("/api/reconstruct")
def api_reconstruct(body: dict = Body(...)):
    """One forward pass of the loaded model, from uploaded images or from a TartanAir sample."""
    app_state = state()
    if app_state.bundle is None:
        raise HTTPException(409, "load a checkpoint first")
    mode = str(body.get("mode", "upload"))
    if mode not in ("upload", "tartanair"):
        raise HTTPException(422, "mode must be 'upload' or 'tartanair'")

    if mode == "upload":
        directory = _upload_dir(app_state, str(body.get("upload_id", "")))
        paths = [
            os.path.join(directory, name)
            for name in sorted(os.listdir(directory))
            if os.path.splitext(name)[1].lower() in IMAGE_SUFFIXES
        ]
        if not paths:
            raise HTTPException(422, "that upload holds no images")
        scene_id = f"upload:{os.path.basename(directory)}"
    else:
        envs = {entry.name: entry for entry in list_envs(app_state.data_root, app_state.camera_name)}
        env_name = str(body.get("env", ""))
        if env_name not in envs:
            raise HTTPException(404, f"no environment {env_name!r} under the data root")
        try:
            index = int(body.get("index", 0))
        except (TypeError, ValueError):
            raise HTTPException(422, "index must be a whole number")
        if index < 0:
            raise HTTPException(422, "index must not be negative")
        scene_id = f"tartanair:{env_name}:{index}"

    started = time.perf_counter()
    with app_state.gpu_lock:
        app_state.scene, app_state.scene_id = None, ""  # the old scene goes before the new one comes
        if app_state.device.type == "cuda":
            torch.cuda.empty_cache()
        try:
            if mode == "upload":
                scene = reconstruct_upload(app_state.bundle, paths, app_state.device)
            else:
                scene = reconstruct_tartanair(app_state.bundle, envs[env_name], index, app_state.device)
        except (IndexError, ValueError, FileNotFoundError) as error:
            raise HTTPException(422, str(error))
        except torch.cuda.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            raise HTTPException(507, f"the GPU ran out of memory during the reconstruction, {error}")
        except Exception as error:  # a GPU the model cannot run on is the one that happens
            raise HTTPException(500, f"the reconstruction failed, {type(error).__name__}: {error}")
        app_state.scene, app_state.scene_id = scene, scene_id
        app_state.depth_available = probe_depth(scene)
    return {
        "scene": scene_payload(app_state),
        "scenes": available_scenes(app_state),
        "depth_available": app_state.depth_available,
        "seconds": time.perf_counter() - started,
    }


@app.post("/api/load_scene")
def api_load_scene(body: dict = Body(...)):
    app_state = state()
    scene_id = str(body.get("id", "synthetic"))
    with app_state.gpu_lock:
        if scene_id != app_state.scene_id or app_state.scene is None:
            app_state.scene = None  # free the old one before the new one is built
            if app_state.device.type == "cuda":
                torch.cuda.empty_cache()
        scene = load_scene_by_id(app_state, scene_id)
        app_state.scene, app_state.scene_id = scene, scene_id
        app_state.depth_available = probe_depth(scene)
    return {"scene": scene_payload(app_state), "depth_available": app_state.depth_available}


def _finite(body: dict, key: str, default: float) -> float:
    """One number out of a request body, with every shape a raw client might send turned into a 422."""
    try:
        value = float(body.get(key, default))
    except (TypeError, ValueError):
        raise HTTPException(422, f"{key} must be a number")
    if not math.isfinite(value):
        raise HTTPException(422, f"{key} must be finite")
    return value


def _render_request(app_state: AppState, body: dict, full_resolution: bool):
    scene = app_state.scene
    if scene is None:
        raise HTTPException(409, "no scene loaded")
    try:
        c2w = torch.tensor(body["c2w"], dtype=torch.float32).reshape(4, 4)
    except Exception as error:
        raise HTTPException(422, f"c2w must be 16 numbers, row major ({error})")
    if not bool(torch.isfinite(c2w).all()):
        raise HTTPException(422, "c2w must be finite")

    camera_model = str(body.get("camera_model", "pinhole"))
    if camera_model not in CAMERA_MODELS:
        raise HTTPException(422, f"camera_model must be one of {CAMERA_MODELS}")
    tile_mode = str(body.get("tile_mode", "projection"))
    if tile_mode not in TILE_MODES:
        raise HTTPException(422, f"tile_mode must be one of {TILE_MODES}")
    try:
        height = int(body.get("res", 512))
    except (TypeError, ValueError):
        raise HTTPException(422, f"res must be one of {RESOLUTIONS}")
    if height not in RESOLUTIONS:
        raise HTTPException(422, f"res must be one of {RESOLUTIONS}")
    if not full_resolution and bool(body.get("moving", False)):
        height = max(height // 2, 128)  # half resolution while the camera moves
    # The frame size is the one thing a request can make arbitrarily expensive, so it is bounded
    # here rather than in the client: an aspect ratio of a million once asked CUDA for 244 GiB.
    aspect = _finite(body, "aspect", 1.0)
    if not ASPECT_RANGE[0] <= aspect <= ASPECT_RANGE[1]:
        raise HTTPException(422, f"aspect must be between {ASPECT_RANGE[0]} and {ASPECT_RANGE[1]}")
    width = max(2 * int(round(height * aspect / 2.0)), 64)
    view = str(body.get("view", "rgb"))
    if view not in VIEWS:
        raise HTTPException(422, f"view must be one of {VIEWS}")
    if view == "depth" and not app_state.depth_available:
        view = "rgb"
    fov_deg = _finite(body, "fov_deg", 90.0)
    alpha = _finite(body, "alpha", scene.alpha_init)
    if camera_model == "ucm" and not 0.0 < alpha < 1.0:
        raise HTTPException(422, "alpha must lie strictly between 0 and 1")

    with app_state.gpu_lock:
        image, info = render_view(
            scene,
            c2w,
            camera_model,
            fov_deg,
            alpha,
            height,
            width,
            tile_mode=tile_mode,
            view=view,
            frusta=bool(body.get("frusta", True)),
        )
    app_state.stats.add(height, info["render_ms"])
    if app_state.stats.total % LOG_EVERY == 0:
        print(app_state.stats.line(), flush=True)
    return image, info


@app.post("/api/render")
def api_render(body: dict = Body(...)):
    app_state = state()
    image, info = _render_request(app_state, body, full_resolution=False)
    return Response(
        content=encode_jpeg(image),
        media_type="image/jpeg",
        headers={
            "X-Render-Ms": f"{info['render_ms']:.2f}",
            "X-Fov-Clamped": f"{info['fov_used']:.3f}",
            "X-Width": str(info["width"]),
            "X-Height": str(info["height"]),
            "Cache-Control": "no-store",
        },
    )


@app.post("/api/save")
def api_save(body: dict = Body(...)):
    app_state = state()
    image, info = _render_request(app_state, body, full_resolution=True)
    shots = os.path.join(app_state.out_dir, "shots")
    os.makedirs(shots, exist_ok=True)
    name = f"{time.strftime('%Y%m%d-%H%M%S')}_{body.get('camera_model', 'pinhole')}.png"
    with open(os.path.join(shots, name), "wb") as handle:
        handle.write(encode_png(image))
    return {"path": os.path.join(shots, name), "url": f"/api/files/{name}", "render_ms": info["render_ms"]}


@app.get("/api/files/{name}")
def api_files(name: str):
    app_state = state()
    path = os.path.join(app_state.out_dir, "shots", os.path.basename(name))
    if not os.path.isfile(path):
        raise HTTPException(404, "no such file")
    return FileResponse(path, media_type="image/png")


def build_state(args: argparse.Namespace) -> AppState:
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    app_state = AppState(
        device=device,
        out_dir=args.out,
        scenes_dir=args.scenes,
        pretrained=args.pretrained,
        runs_dir=args.runs_dir,
        data_root=args.data_root,
        camera_name=args.camera_name,
    )
    start = args.scene
    if start is None:
        entries = available_scenes(app_state)
        # Prefer a precomputed scene over the synthetic one when there is one.
        start = entries[1]["id"] if len(entries) > 1 else "synthetic"
    app_state.scene = load_scene_by_id(app_state, start)
    app_state.scene_id = start
    app_state.depth_available = probe_depth(app_state.scene)
    print(
        f"Scene {app_state.scene.name}, {app_state.scene.num_gaussians} Gaussians, "
        f"camera_model {app_state.scene.camera_model}, depth view {app_state.depth_available}",
        flush=True,
    )
    return app_state


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive 3DGEER viewer")
    parser.add_argument("--host", default="127.0.0.1", help="address to bind, keep it inside the VPN")
    parser.add_argument("--port", type=int, default=7870)
    parser.add_argument("--scenes", default=None, help="directory of precomputed .pt scenes")
    parser.add_argument("--scene", default=None, help="scene to start with, a file name or 'synthetic'")
    parser.add_argument("--out", default=os.path.expanduser("~/runs/viewer"), help="where saved images go")
    parser.add_argument("--pretrained", default=None, help="pretrained checkpoint, listed in the UI")
    parser.add_argument("--runs-dir", default=None, help="checkpoints/runs, listed in the UI")
    parser.add_argument("--data-root", default=None, help="TartanAir root, listed in the UI")
    parser.add_argument("--camera-name", default="lcam_fish", help="TartanAir camera of the listed scenes")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.host == "0.0.0.0":
        print("Warning: binding every interface. Only do this behind the VPN, never on a public host.", flush=True)

    global STATE
    STATE = build_state(args)
    print(f"URL: http://{args.host}:{args.port}", flush=True)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
