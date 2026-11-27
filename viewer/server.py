"""The viewer server: a small FastAPI app that renders one frame per request with 3DGEER.

There is no websocket, because the container image carries uvicorn without its websocket extras.
The browser instead runs a sequential fetch loop: it posts the current camera, waits for the JPEG and
only then asks for the next frame. That throttles itself to one request in flight, so the newest
camera always wins and the GPU never queues up stale frames.

Every route is a synchronous `def`, which FastAPI runs in its threadpool, and one lock serialises all
GPU work, so a second browser tab cannot render into a scene that is being replaced.

    python -m viewer.server --host 127.0.0.1 --port 7870 --scenes ~/scenes

The host is deliberately not 0.0.0.0 by default. Bind to the address of the interface you want to
reach the viewer on, and keep it inside the VPN.
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, Response

from .camera import CAMERA_MODELS, estimated_camera_info
from .render import VIEWS, RenderStats, encode_jpeg, encode_png, probe_depth, render_view
from .scene import Scene, list_checkpoints, load_scene, make_synthetic_scene

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
RESOLUTIONS = (256, 384, 512, 768)
TILE_MODES = ("projection", "pbf")
LOG_EVERY = 100


@dataclass
class AppState:
    device: torch.device
    out_dir: str
    scenes_dir: Optional[str] = None
    pretrained: Optional[str] = None
    runs_dir: Optional[str] = None
    gpu_lock: threading.Lock = field(default_factory=threading.Lock)
    scene: Optional[Scene] = None
    scene_id: str = ""
    depth_available: bool = False
    stats: RenderStats = field(default_factory=RenderStats)


STATE: Optional[AppState] = None
app = FastAPI(title="3DGEER viewer")


def state() -> AppState:
    if STATE is None:
        raise HTTPException(503, "the server has no state yet")
    return STATE


def available_scenes(app_state: AppState) -> List[Dict[str, str]]:
    """Every scene the viewer can load: the precomputed dumps plus the built in synthetic one."""
    entries = [{"id": "synthetic", "label": "synthetic (development scene)"}]
    if app_state.scenes_dir and os.path.isdir(app_state.scenes_dir):
        for name in sorted(os.listdir(app_state.scenes_dir)):
            if name.endswith(".pt"):
                path = os.path.join(app_state.scenes_dir, name)
                size = os.path.getsize(path) / (1024.0 * 1024.0)
                entries.append({"id": name, "label": f"{name} ({size:.0f} MB)"})
    return entries


def load_scene_by_id(app_state: AppState, scene_id: str) -> Scene:
    if scene_id == "synthetic":
        return make_synthetic_scene(app_state.device)
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
            for entry in list_checkpoints(app_state.pretrained, app_state.runs_dir)
        ],
        "depth_available": app_state.depth_available,
        "resolutions": list(RESOLUTIONS),
        "stats": app_state.stats.summary(),
    }


@app.post("/api/load_scene")
def api_load_scene(body: dict = Body(...)):
    app_state = state()
    scene_id = str(body.get("id", "synthetic"))
    with app_state.gpu_lock:
        app_state.scene = None  # free the old one before the new one is built
        if app_state.device.type == "cuda":
            torch.cuda.empty_cache()
        scene = load_scene_by_id(app_state, scene_id)
        app_state.scene, app_state.scene_id = scene, scene_id
        app_state.depth_available = probe_depth(scene)
    return {"scene": scene_payload(app_state), "depth_available": app_state.depth_available}


def _render_request(app_state: AppState, body: dict, full_resolution: bool):
    scene = app_state.scene
    if scene is None:
        raise HTTPException(409, "no scene loaded")
    try:
        c2w = torch.tensor(body["c2w"], dtype=torch.float32).reshape(4, 4)
    except Exception as error:
        raise HTTPException(422, f"c2w must be 16 numbers, row major ({error})")

    camera_model = str(body.get("camera_model", "pinhole"))
    if camera_model not in CAMERA_MODELS:
        raise HTTPException(422, f"camera_model must be one of {CAMERA_MODELS}")
    tile_mode = str(body.get("tile_mode", "projection"))
    if tile_mode not in TILE_MODES:
        raise HTTPException(422, f"tile_mode must be one of {TILE_MODES}")
    height = int(body.get("res", 512))
    if height not in RESOLUTIONS:
        raise HTTPException(422, f"res must be one of {RESOLUTIONS}")
    if not full_resolution and bool(body.get("moving", False)):
        height = max(height // 2, 128)  # half resolution while the camera moves
    aspect = float(body.get("aspect", 1.0))
    width = max(2 * int(round(height * aspect / 2.0)), 64)
    view = str(body.get("view", "rgb"))
    if view not in VIEWS:
        raise HTTPException(422, f"view must be one of {VIEWS}")
    if view == "depth" and not app_state.depth_available:
        view = "rgb"

    with app_state.gpu_lock:
        image, info = render_view(
            scene,
            c2w,
            camera_model,
            float(body.get("fov_deg", 90.0)),
            float(body.get("alpha", scene.alpha_init)),
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
