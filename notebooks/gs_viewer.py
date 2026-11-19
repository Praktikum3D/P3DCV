"""Interactive 3D Gaussian splat viewer for Jupyter notebooks.

A small WebGL2 splat renderer that runs inside the notebook output cell (VS Code
and JupyterLab), so it needs no extra Python package and no running server.
The same page can be written to a standalone .html file and opened in any
browser.

What it renders
---------------
Every Gaussian is drawn as a screen-space ellipse with the EWA projection used
by 3DGS / gsplat (including the 0.3 px low-pass), sorted back to front and
alpha blended. Colours are evaluated from the full degree-3 spherical
harmonics for the current eye position, as gsplat does. The model puts a lot
of its colour into the higher SH bands, so DC-only colours look washed out
(toggle "view-dependent colour" to see the difference). The SH coefficients
are sent as int8 with one scale per splat (about 1/255 colour error).

Coordinates follow the model: OpenCV cameras (x right, y down, z forward),
camera-to-world matrices, intrinsics normalised by the image size.

Controls: left drag orbits, right or shift drag pans, the wheel zooms,
W/A/S/D/Q/E fly, keys 1-9 and the camera buttons jump to a predicted camera.
"""

from __future__ import annotations

import base64
import html
import json
import uuid
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

SH_C0 = 0.28209479177387814

# One splat = 36 bytes: position f32x3, rotation quaternion (w, x, y, z) f16x4,
# scale f16x3 + padding, DC colour + opacity u8x4, source-view colour u8x4.
SPLAT_DTYPE = np.dtype(
    [
        ("pos", "<f4", (3,)),
        ("quat", "<f2", (4,)),
        ("scale", "<f2", (4,)),
        ("rgba", "u1", (4,)),
        ("vrgba", "u1", (4,)),
    ]
)
assert SPLAT_DTYPE.itemsize == 36


def view_palette(num_views: int) -> np.ndarray:
    """One distinct colour per view, [V, 3] in 0..1 (same hsv map as app_core.engine)."""
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("hsv")
    return np.array([cmap(i / max(num_views, 1))[:3] for i in range(num_views)], dtype=np.float32)


def pack_splats(
    xyz: np.ndarray,
    sh: np.ndarray,
    opacity: np.ndarray,
    scale: np.ndarray,
    quat_wxyz: np.ndarray,
    view_ids: Optional[np.ndarray] = None,
    num_views: Optional[int] = None,
    min_opacity: float = 0.02,
    max_splats: int = 150_000,
) -> dict:
    """Filter the Gaussians and pack them into the binary layout the viewer reads.

    xyz [N,3]; sh [N,16,3] raw SH coefficients (degree 3, DC first) or [N,3] DC only;
    opacity in 0..1; scale in world units; quaternions (w, x, y, z).
    Keeps the `max_splats` most opaque Gaussians with opacity >= min_opacity.
    """
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    n = xyz.shape[0]
    sh = np.asarray(sh, dtype=np.float32).reshape(n, -1, 3)
    opacity = np.asarray(opacity, dtype=np.float32).reshape(n)
    keep = np.flatnonzero((opacity >= min_opacity) & np.isfinite(xyz).all(axis=1))
    if keep.size > max_splats:
        keep = keep[np.argsort(-opacity[keep])[:max_splats]]

    out = np.zeros(keep.size, dtype=SPLAT_DTYPE)
    out["pos"] = xyz[keep]
    quat = np.asarray(quat_wxyz, dtype=np.float32).reshape(n, 4)[keep]
    quat /= np.linalg.norm(quat, axis=1, keepdims=True) + 1e-12
    out["quat"] = quat.astype(np.float16)
    out["scale"][:, :3] = np.clip(np.asarray(scale, dtype=np.float32).reshape(n, 3)[keep], 1e-6, 1e3)
    sh = sh[keep]
    dc_rgb = np.clip(0.5 + SH_C0 * sh[:, 0], 0.0, 1.0)
    rgba = np.concatenate([dc_rgb, opacity[keep, None]], axis=1)
    out["rgba"] = np.round(rgba * 255.0).astype(np.uint8)
    if view_ids is not None:
        view_ids = np.asarray(view_ids).reshape(n)[keep]
        palette = view_palette(num_views or int(view_ids.max()) + 1)
        out["vrgba"][:, :3] = np.round(palette[view_ids] * 255.0).astype(np.uint8)
    else:
        out["vrgba"][:, :3] = out["rgba"][:, :3]
    out["vrgba"][:, 3] = out["rgba"][:, 3]

    packed = {"splats": out, "sh": None, "sh_scale": None}
    if sh.shape[1] == 16:
        # int8 coefficients with one scale per splat
        flat = sh.reshape(-1, 48)
        sh_scale = np.maximum(np.abs(flat).max(axis=1), 1e-8).astype(np.float32)
        packed["sh"] = np.round(flat / sh_scale[:, None] * 127.0).astype(np.int8)
        packed["sh_scale"] = sh_scale
    return packed


def _camera_meta(c2ws: np.ndarray, fxfycxcy: np.ndarray, depths: Sequence[float], names: Sequence[str]) -> list:
    palette = view_palette(len(c2ws))
    cams = []
    for i, (c2w, intr) in enumerate(zip(c2ws, fxfycxcy)):
        color = "#%02x%02x%02x" % tuple(int(round(c * 255)) for c in palette[i])
        cams.append(
            {
                "c2w": np.asarray(c2w, dtype=float)[:3, :4].tolist(),
                "intr": [float(v) for v in intr],
                "depth": float(depths[i]),
                "color": color,
                "name": names[i],
            }
        )
    return cams


def _frustum_lines(c2ws: np.ndarray, fxfycxcy: np.ndarray, size: float) -> np.ndarray:
    """Line segments [M, 6] (xyz, rgb) that draw every camera as a small frustum."""
    palette = view_palette(len(c2ws))
    lines = []
    for i, (c2w, (fx, fy, cx, cy)) in enumerate(zip(c2ws, fxfycxcy)):
        R, t = c2w[:3, :3], c2w[:3, 3]
        corners = [((u - cx) / fx, (v - cy) / fy, 1.0) for u, v in ((0, 0), (1, 0), (1, 1), (0, 1))]
        corners = [t + R @ (np.array(c) * size) for c in corners]
        segments = [(t, c) for c in corners] + [(corners[k], corners[(k + 1) % 4]) for k in range(4)]
        # a tick on the top edge marks "up" (-y in OpenCV)
        top_mid = 0.5 * (corners[0] + corners[1])
        segments.append((top_mid, top_mid - R[:, 1] * size * 0.3))
        for a, b in segments:
            lines.append(np.concatenate([a, palette[i]]))
            lines.append(np.concatenate([b, palette[i]]))
    return np.asarray(lines, dtype=np.float32)


def build_viewer_html(
    packed: dict,
    c2ws: np.ndarray,
    fxfycxcy: np.ndarray,
    cam_depths: Optional[Sequence[float]] = None,
    cam_names: Optional[Sequence[str]] = None,
    up: Optional[np.ndarray] = None,
    size: int = 560,
    background: str = "white",
    title: str = "",
    standalone: bool = False,
) -> str:
    """Return the viewer as an HTML snippet (or a full page when standalone=True).

    packed    : output of pack_splats
    c2ws      : [V, 4, 4] camera-to-world (OpenCV) of the views to show
    fxfycxcy  : [V, 4] normalised intrinsics of those views
    cam_depths: per view a typical scene depth, used as orbit distance
    up        : world "up" for orbiting; defaults to -y of the first camera
    """
    splats = packed["splats"]
    c2ws = np.asarray(c2ws, dtype=np.float64)
    fxfycxcy = np.asarray(fxfycxcy, dtype=np.float64)
    if cam_depths is None:
        cam_depths = [2.0] * len(c2ws)
    if cam_names is None:
        cam_names = [f"view {i}" for i in range(len(c2ws))]
    if up is None:
        up = -c2ws[0, :3, 1]
    up = np.asarray(up, dtype=float)
    up = up / np.linalg.norm(up)

    centers = c2ws[:, :3, 3]
    spread = np.linalg.norm(centers - centers.mean(0), axis=1).max() if len(centers) > 1 else 0.0
    frustum_size = float(max(0.05 * float(np.median(cam_depths)), 0.4 * spread / max(len(c2ws), 1), 1e-3))

    has_sh = packed.get("sh") is not None
    meta = {
        "n": int(splats.shape[0]),
        "stride": SPLAT_DTYPE.itemsize // 4,
        "cams": _camera_meta(c2ws, fxfycxcy, cam_depths, cam_names),
        "up": up.tolist(),
        "size": int(size),
        "bg": background,
        "title": title,
        "hasSH": has_sh,
    }

    def b64(arr) -> str:
        return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")

    uid = "gsv_" + uuid.uuid4().hex[:10]
    body = (
        _TEMPLATE.replace("__UID__", uid)
        .replace("__META__", html.escape(json.dumps(meta), quote=False))
        .replace("__DATA__", b64(splats))
        .replace("__SH__", b64(packed["sh"]) if has_sh else "")
        .replace("__SHS__", b64(packed["sh_scale"]) if has_sh else "")
        .replace("__LINES__", b64(_frustum_lines(c2ws, fxfycxcy, frustum_size)))
        .replace("__JS__", _JS.replace("__UID__", uid))
    )
    if not standalone:
        return body
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title or 'Gaussian splats')}</title>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<style>body{margin:16px;font-family:system-ui,sans-serif;background:#f4f4f4;color:#222}</style>"
        f"</head><body>{body}</body></html>"
    )


def show(*args, save_to: Optional[str | Path] = None, **kwargs) -> None:
    """Display the viewer in the notebook; optionally also save a standalone .html."""
    from IPython.display import HTML, display

    if save_to is not None:
        Path(save_to).write_text(build_viewer_html(*args, standalone=True, **kwargs), encoding="utf-8")
    display(HTML(build_viewer_html(*args, **kwargs)))


def sh_colors_reference(packed: dict, eye: np.ndarray) -> np.ndarray:
    """Numpy version of the viewer's colour evaluation (for testing): [N, 3] in 0..1."""
    pos = packed["splats"]["pos"].astype(np.float64)
    sh = packed["sh"].astype(np.float64).reshape(-1, 16, 3) / 127.0 * packed["sh_scale"][:, None, None]
    d = pos - eye
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-20
    return np.clip(np.einsum("nk,nkc->nc", _sh_basis(d), sh) + 0.5, 0.0, 1.0)


def _sh_basis(d: np.ndarray) -> np.ndarray:
    """Real SH basis up to degree 3 in the 3DGS / gsplat convention, [N, 16]."""
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
    return np.stack(
        [
            np.full_like(x, SH_C0),
            -0.4886025119029199 * y, 0.4886025119029199 * z, -0.4886025119029199 * x,
            1.0925484305920792 * xy, -1.0925484305920792 * yz, 0.31539156525252005 * (2 * zz - xx - yy),
            -1.0925484305920792 * xz, 0.5462742152960396 * (xx - yy),
            -0.5900435899266435 * y * (3 * xx - yy), 2.890611442640554 * xy * z,
            -0.4570457994644658 * y * (4 * zz - xx - yy), 0.3731763325901154 * z * (2 * zz - 3 * xx - 3 * yy),
            -0.4570457994644658 * x * (4 * zz - xx - yy), 1.445305721320277 * z * (xx - yy),
            -0.5900435899266435 * x * (xx - 3 * yy),
        ],
        axis=1,
    )


_TEMPLATE = r"""
<div id="__UID__" class="gsv" style="font:12px system-ui,sans-serif;color:#222;max-width:100%">
  <style>
    #__UID__ .bar{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:6px 0}
    #__UID__ button{font:inherit;padding:2px 8px;border:1px solid #bbb;border-radius:4px;background:#fafafa;color:#222;cursor:pointer}
    #__UID__ button .sw{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:4px;vertical-align:-1px}
    #__UID__ label{display:inline-flex;gap:4px;align-items:center;white-space:nowrap}
    #__UID__ canvas{display:block;max-width:100%;border:1px solid #ccc;border-radius:4px;outline:none;touch-action:none}
    #__UID__ .info{color:#777;margin-top:4px}
  </style>
  <div class="title" style="font-weight:600"></div>
  <div class="bar cams"></div>
  <canvas tabindex="0"></canvas>
  <div class="bar">
    <label><input type="checkbox" class="shbox" checked> view-dependent colour (SH)</label>
    <label><input type="checkbox" class="byview"> colour by source view</label>
    <label><input type="checkbox" class="showcams" checked> cameras</label>
  </div>
  <div class="bar">
    <label>splat scale <input type="range" class="scale" min="0.05" max="2" step="0.05" value="1"></label>
    <label>min opacity <input type="range" class="minop" min="0" max="0.95" step="0.01" value="0"></label>
    <label>background <select class="bgsel"><option>white</option><option>black</option><option>grey</option></select></label>
  </div>
  <div class="info status"></div>
  <div class="info">left drag: orbit · right/shift drag: pan · wheel: zoom · W/A/S/D/Q/E: fly (click the canvas first) · 1-9: jump to camera · R: back to the first camera</div>
  <script type="application/json" class="meta">__META__</script>
  <script type="application/octet-stream" class="data">__DATA__</script>
  <script type="application/octet-stream" class="sh">__SH__</script>
  <script type="application/octet-stream" class="shs">__SHS__</script>
  <script type="application/octet-stream" class="lines">__LINES__</script>
  <script>__JS__</script>
</div>
"""

_JS = r"""
(function () {
  const root = document.getElementById("__UID__");
  if (!root || root.dataset.started) return;
  root.dataset.started = "1";
  const $ = (s) => root.querySelector(s);
  const status = $(".status");
  const meta = JSON.parse($(".meta").textContent);
  $(".title").textContent = meta.title || "";

  function b64ToBuffer(b64) {
    const s = atob(b64.trim());
    const out = new Uint8Array(s.length);
    for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
    return out.buffer;
  }
  const raw = b64ToBuffer($(".data").textContent);
  const N = meta.n, STRIDE = meta.stride;           // stride in 4-byte words
  const srcF32 = new Float32Array(raw), srcU32 = new Uint32Array(raw);
  const sortedU32 = new Uint32Array(N * STRIDE), sortedU8 = new Uint8Array(sortedU32.buffer);
  const lineData = new Float32Array(b64ToBuffer($(".lines").textContent));
  const shQ = meta.hasSH ? new Int8Array(b64ToBuffer($(".sh").textContent)) : null;
  const shS = meta.hasSH ? new Float32Array(b64ToBuffer($(".shs").textContent)).map((s) => s / 127) : null;
  if (!shQ) { $(".shbox").checked = false; $(".shbox").disabled = true; }

  const canvas = $("canvas");
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.style.width = meta.size + "px"; canvas.style.height = meta.size + "px";
  canvas.width = Math.round(meta.size * dpr); canvas.height = Math.round(meta.size * dpr);
  const gl = canvas.getContext("webgl2", { antialias: false, alpha: false, premultipliedAlpha: true });
  if (!gl) { status.textContent = "WebGL2 is not available in this view."; return; }

  // ---------- shaders ----------
  const splatVS = `#version 300 es
  precision highp float;
  uniform mat4 view; uniform vec4 intr; uniform vec2 viewport;
  uniform float scaleMul; uniform float minOpacity; uniform int colorMode;
  layout(location=0) in vec2 corner;
  layout(location=1) in vec3 center;
  layout(location=2) in vec4 quat;
  layout(location=3) in vec4 scale;
  layout(location=4) in vec4 rgba;
  layout(location=5) in vec4 vrgba;
  out vec4 vColor; out vec2 vPos;
  void main() {
    vec4 cam = view * vec4(center, 1.0);
    vec4 col = colorMode == 1 ? vrgba : rgba;
    if (cam.z < 0.01 || col.a < minOpacity) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }
    vec4 q = normalize(quat);
    float w = q.x, x = q.y, y = q.z, z = q.w;
    mat3 R = transpose(mat3(
      1.0 - 2.0*(y*y + z*z), 2.0*(x*y - w*z),       2.0*(x*z + w*y),
      2.0*(x*y + w*z),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z - w*x),
      2.0*(x*z - w*y),       2.0*(y*z + w*x),       1.0 - 2.0*(x*x + y*y)));
    vec3 s = scale.xyz * scaleMul;
    mat3 M = R * mat3(s.x, 0.0, 0.0, 0.0, s.y, 0.0, 0.0, 0.0, s.z);
    mat3 Sigma = M * transpose(M);
    float zc = cam.z;
    mat3 J = transpose(mat3(
      intr.x / zc, 0.0, -intr.x * cam.x / (zc * zc),
      0.0, intr.y / zc, -intr.y * cam.y / (zc * zc),
      0.0, 0.0, 0.0));
    mat3 T = J * mat3(view);
    mat3 cov = T * Sigma * transpose(T);
    float a = cov[0][0] + 0.3, b = cov[0][1], c = cov[1][1] + 0.3;
    float mid = 0.5 * (a + c);
    float rad = length(vec2(0.5 * (a - c), b));
    float l1 = mid + rad, l2 = max(mid - rad, 0.1);
    vec2 dv = vec2(b, l1 - a);
    dv = length(dv) < 1e-9 ? (a >= c ? vec2(1.0, 0.0) : vec2(0.0, 1.0)) : normalize(dv);
    vec2 major = min(sqrt(2.0 * l1), 1024.0) * dv;
    vec2 minor = min(sqrt(2.0 * l2), 1024.0) * vec2(dv.y, -dv.x);
    vec2 pix = vec2(intr.x * cam.x / zc + intr.z, intr.y * cam.y / zc + intr.w);
    vec2 p = pix + corner.x * major + corner.y * minor;
    gl_Position = vec4(2.0 * p.x / viewport.x - 1.0, 1.0 - 2.0 * p.y / viewport.y, 0.0, 1.0);
    vColor = col;
    vPos = corner;
  }`;
  const splatFS = `#version 300 es
  precision highp float;
  in vec4 vColor; in vec2 vPos; out vec4 fragColor;
  void main() {
    float A = -dot(vPos, vPos);
    if (A < -4.0) discard;
    float B = min(exp(A) * vColor.a, 0.99);
    fragColor = vec4(B * vColor.rgb, B);
  }`;
  const lineVS = `#version 300 es
  precision highp float;
  uniform mat4 view; uniform vec4 intr; uniform vec2 viewport;
  layout(location=0) in vec3 pos; layout(location=1) in vec3 color;
  out vec3 vc;
  void main() {
    vec4 c = view * vec4(pos, 1.0);
    gl_Position = vec4(2.0 * (intr.x * c.x + intr.z * c.z) / viewport.x - c.z,
                       c.z - 2.0 * (intr.y * c.y + intr.w * c.z) / viewport.y,
                       c.z - 0.02, c.z);
    vc = color;
  }`;
  const lineFS = `#version 300 es
  precision highp float;
  in vec3 vc; out vec4 fragColor;
  void main() { fragColor = vec4(vc, 1.0); }`;

  function program(vs, fs) {
    const p = gl.createProgram();
    for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
      const s = gl.createShader(type);
      gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      gl.attachShader(p, s);
    }
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    return p;
  }
  const splatProg = program(splatVS, splatFS);
  const lineProg = program(lineVS, lineFS);
  const U = (p, n) => gl.getUniformLocation(p, n);

  // ---------- buffers ----------
  const splatVAO = gl.createVertexArray();
  gl.bindVertexArray(splatVAO);
  const quadBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-2, -2, 2, -2, 2, 2, -2, 2]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  const splatBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, splatBuf);
  gl.bufferData(gl.ARRAY_BUFFER, sortedU32.byteLength, gl.DYNAMIC_DRAW);
  const B = STRIDE * 4;
  const attrs = [[1, 3, gl.FLOAT, false, 0], [2, 4, gl.HALF_FLOAT, false, 12], [3, 4, gl.HALF_FLOAT, false, 20],
                 [4, 4, gl.UNSIGNED_BYTE, true, 28], [5, 4, gl.UNSIGNED_BYTE, true, 32]];
  for (const [loc, size, type, norm, off] of attrs) {
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, size, type, norm, B, off);
    gl.vertexAttribDivisor(loc, 1);
  }
  const lineVAO = gl.createVertexArray();
  gl.bindVertexArray(lineVAO);
  const lineBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, lineBuf);
  gl.bufferData(gl.ARRAY_BUFFER, lineData, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 24, 0);
  gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 24, 12);
  gl.bindVertexArray(null);

  // ---------- camera (OpenCV: x right, y down, z forward) ----------
  const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const len = (a) => Math.hypot(a[0], a[1], a[2]);
  const norm = (a) => mul(a, 1 / len(a));
  function rotate(v, axis, ang) {     // Rodrigues
    const c = Math.cos(ang), s = Math.sin(ang);
    return add(add(mul(v, c), mul(cross(axis, v), s)), mul(axis, dot(axis, v) * (1 - c)));
  }
  const up = norm(meta.up);
  const cam = { x: [1, 0, 0], y: [0, 1, 0], z: [0, 0, 1], p: [0, 0, 0], target: [0, 0, 1], intr: [1, 1, 0.5, 0.5] };
  let dirty = true, needSort = true, drag = null;

  function setCamera(i) {
    const c = meta.cams[i]; if (!c) return;
    const m = c.c2w;
    cam.x = norm([m[0][0], m[1][0], m[2][0]]);
    cam.y = norm([m[0][1], m[1][1], m[2][1]]);
    cam.z = norm([m[0][2], m[1][2], m[2][2]]);
    cam.p = [m[0][3], m[1][3], m[2][3]];
    cam.target = add(cam.p, mul(cam.z, c.depth));
    cam.intr = c.intr.slice();
    dirty = true; needSort = true;
  }
  function viewMatrix() {   // world -> camera, column-major
    const { x, y, z, p } = cam;
    return new Float32Array([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0,
                             -dot(x, p), -dot(y, p), -dot(z, p), 1]);
  }
  function orbit(yaw, pitch) {
    let o = sub(cam.p, cam.target);
    for (const [axis, ang] of [[up, yaw], [cam.x, pitch]]) {
      o = rotate(o, axis, ang);
      cam.x = norm(rotate(cam.x, axis, ang)); cam.y = norm(rotate(cam.y, axis, ang)); cam.z = norm(rotate(cam.z, axis, ang));
    }
    cam.p = add(cam.target, o);
  }
  function translate(d) { cam.p = add(cam.p, d); cam.target = add(cam.target, d); }

  // ---------- depth sort (16-bit counting sort, back to front) + SH colours ----------
  const depths = new Float32Array(N), keys = new Uint32Array(N), order = new Uint32Array(N);
  const counts = new Uint32Array(65536), starts = new Uint32Array(65536);
  const C1 = 0.4886025119029199, C0 = 0.28209479177387814;
  const bas = new Float32Array(16);
  function shBasis(x, y, z) {
    const xx = x * x, yy = y * y, zz = z * z, xy = x * y, yz = y * z, xz = x * z;
    bas[0] = C0; bas[1] = -C1 * y; bas[2] = C1 * z; bas[3] = -C1 * x;
    bas[4] = 1.0925484305920792 * xy; bas[5] = -1.0925484305920792 * yz;
    bas[6] = 0.31539156525252005 * (2 * zz - xx - yy); bas[7] = -1.0925484305920792 * xz;
    bas[8] = 0.5462742152960396 * (xx - yy);
    bas[9] = -0.5900435899266435 * y * (3 * xx - yy); bas[10] = 2.890611442640554 * xy * z;
    bas[11] = -0.4570457994644658 * y * (4 * zz - xx - yy); bas[12] = 0.3731763325901154 * z * (2 * zz - 3 * xx - 3 * yy);
    bas[13] = -0.4570457994644658 * x * (4 * zz - xx - yy); bas[14] = 1.445305721320277 * z * (xx - yy);
    bas[15] = -0.5900435899266435 * x * (xx - 3 * yy);
  }
  function sortSplats() {
    const z = cam.z;
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < N; i++) {
      const k = i * STRIDE;
      const d = z[0] * srcF32[k] + z[1] * srcF32[k + 1] + z[2] * srcF32[k + 2];
      depths[i] = d; if (d < lo) lo = d; if (d > hi) hi = d;
    }
    const sc = 65535 / Math.max(hi - lo, 1e-9);
    counts.fill(0);
    for (let i = 0; i < N; i++) { const k = ((depths[i] - lo) * sc) | 0; keys[i] = k; counts[k]++; }
    let acc = 0;
    for (let k = 65535; k >= 0; k--) { starts[k] = acc; acc += counts[k]; }
    for (let i = 0; i < N; i++) order[starts[keys[i]]++] = i;
    const useSH = shQ && ui.sh;
    const e = cam.p;
    for (let j = 0; j < N; j++) {
      const i = order[j], s = i * STRIDE, d = j * STRIDE;
      for (let w = 0; w < STRIDE; w++) sortedU32[d + w] = srcU32[s + w];
      if (useSH) {
        let dx = srcF32[s] - e[0], dy = srcF32[s + 1] - e[1], dz = srcF32[s + 2] - e[2];
        const inv = 1 / Math.sqrt(dx * dx + dy * dy + dz * dz + 1e-20);
        shBasis(dx * inv, dy * inv, dz * inv);
        const o = i * 48, b = d * 4 + 28, f = shS[i];
        for (let ch = 0; ch < 3; ch++) {
          let v = 0;
          for (let k = 0; k < 16; k++) v += bas[k] * shQ[o + k * 3 + ch];
          v = 0.5 + v * f;
          sortedU8[b + ch] = v <= 0 ? 0 : v >= 1 ? 255 : (v * 255 + 0.5) | 0;
        }
      }
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, splatBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, sortedU32);
  }

  // ---------- UI state ----------
  const ui = { sh: !!shQ, byView: false, showCams: true, scale: 1, minOp: 0, bg: meta.bg };
  const bgColors = { white: [1, 1, 1], black: [0, 0, 0], grey: [0.5, 0.5, 0.5] };
  $(".bgsel").value = meta.bg in bgColors ? meta.bg : "white";
  $(".shbox").onchange = (e) => { ui.sh = e.target.checked; dirty = true; needSort = true; };
  $(".byview").onchange = (e) => { ui.byView = e.target.checked; dirty = true; };
  $(".showcams").onchange = (e) => { ui.showCams = e.target.checked; dirty = true; };
  $(".scale").oninput = (e) => { ui.scale = parseFloat(e.target.value); dirty = true; };
  $(".minop").oninput = (e) => { ui.minOp = parseFloat(e.target.value); dirty = true; };
  $(".bgsel").onchange = (e) => { ui.bg = e.target.value; dirty = true; };
  const camBar = $(".cams");
  meta.cams.forEach((c, i) => {
    const b = document.createElement("button");
    b.innerHTML = `<span class="sw" style="background:${c.color}"></span>${c.name}`;
    b.onclick = () => { setCamera(i); canvas.focus(); };
    camBar.appendChild(b);
  });

  // ---------- input ----------
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  canvas.addEventListener("pointerdown", (e) => {
    canvas.setPointerCapture(e.pointerId); canvas.focus();
    drag = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey };
  });
  canvas.addEventListener("pointerup", () => { drag = null; });
  canvas.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    drag.x = e.clientX; drag.y = e.clientY;
    if (drag.pan) {
      const s = len(sub(cam.p, cam.target)) / (cam.intr[0] * meta.size);
      translate(add(mul(cam.x, -dx * s), mul(cam.y, -dy * s)));
    } else {
      orbit(-dx * 0.006, -dy * 0.006);
    }
    dirty = true; needSort = true;
  });
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const o = sub(cam.p, cam.target);
    const f = Math.exp(Math.max(-0.5, Math.min(0.5, e.deltaY * 0.0015)));
    cam.p = add(cam.target, mul(o, Math.max(f, 1e-3 / len(o))));
    dirty = true; needSort = true;
  }, { passive: false });
  canvas.addEventListener("keydown", (e) => {
    const step = 0.03 * len(sub(cam.p, cam.target));
    const moves = { w: cam.z, s: mul(cam.z, -1), d: cam.x, a: mul(cam.x, -1), e: mul(cam.y, -1), q: cam.y };
    const k = e.key.toLowerCase();
    if (moves[k]) { translate(mul(moves[k], step)); }
    else if (k === "r") { setCamera(0); }
    else if (/^[1-9]$/.test(k)) { setCamera(parseInt(k) - 1); }
    else return;
    e.preventDefault(); dirty = true; needSort = true;
  });

  // ---------- draw (only when something changed) ----------
  let sortMs = 0;
  function draw() {
    requestAnimationFrame(draw);
    if (!dirty) return;
    dirty = false;
    if (needSort) { const t = performance.now(); sortSplats(); sortMs = performance.now() - t; needSort = false; }
    const W = canvas.width, H = canvas.height;
    const intr = [cam.intr[0] * W, cam.intr[1] * H, cam.intr[2] * W, cam.intr[3] * H];
    const view = viewMatrix();
    gl.viewport(0, 0, W, H);
    const bg = bgColors[ui.bg] || [1, 1, 1];
    gl.clearColor(bg[0], bg[1], bg[2], 1); gl.clear(gl.COLOR_BUFFER_BIT);
    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.useProgram(splatProg);
    gl.uniformMatrix4fv(U(splatProg, "view"), false, view);
    gl.uniform4fv(U(splatProg, "intr"), intr);
    gl.uniform2f(U(splatProg, "viewport"), W, H);
    gl.uniform1f(U(splatProg, "scaleMul"), ui.scale);
    gl.uniform1f(U(splatProg, "minOpacity"), ui.minOp);
    gl.uniform1i(U(splatProg, "colorMode"), ui.byView ? 1 : 0);
    gl.bindVertexArray(splatVAO);
    gl.drawArraysInstanced(gl.TRIANGLE_FAN, 0, 4, N);
    if (ui.showCams && lineData.length) {
      gl.disable(gl.BLEND);
      gl.useProgram(lineProg);
      gl.uniformMatrix4fv(U(lineProg, "view"), false, view);
      gl.uniform4fv(U(lineProg, "intr"), intr);
      gl.uniform2f(U(lineProg, "viewport"), W, H);
      gl.bindVertexArray(lineVAO);
      gl.drawArrays(gl.LINES, 0, lineData.length / 6);
    }
    gl.bindVertexArray(null);
    const d = len(sub(cam.p, cam.target));
    status.textContent = `${N.toLocaleString()} splats · sort + colour ${sortMs.toFixed(0)} ms · ` +
      `eye (${cam.p.map((v) => v.toFixed(2)).join(", ")}) · orbit radius ${d.toFixed(2)}`;
  }
  setCamera(0);
  requestAnimationFrame(draw);
})();
"""
