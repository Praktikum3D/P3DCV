// The viewer client. It keeps the camera, draws whatever JPEG the server sends back and runs one
// request at a time, so the newest camera always wins and the GPU never works on a stale frame.

"use strict";

const canvas = document.getElementById("canvas");
const context = canvas.getContext("2d");
const element = (id) => document.getElementById(id);

const DEFAULT_FOV = { pinhole: 90, fisheye: 195, ucm: 160 };
const FOV_RANGE = { pinhole: [20, 170], fisheye: [20, 270], ucm: [20, 300] };

const state = {
  scene: null,
  depthAvailable: false,
  // The camera lives as an orbit around a target, in the OpenCV frame the renderer uses:
  // x right, y down, z forward, so the world up direction is negative y.
  target: [0, 0, 0],
  yaw: 0,
  pitch: 0.2,
  distance: 4,
  radius: 4,
  inFlight: false,
  dirty: true,
  moving: false,
  movingUntil: 0,
  keys: new Set(),
  shift: false,
  lastRoundTrip: 0,
};

// ----------------------------------------------------------------------------- camera maths

function direction() {
  const cp = Math.cos(state.pitch);
  return [cp * Math.sin(state.yaw), Math.sin(state.pitch), cp * Math.cos(state.yaw)];
}

function normalize(v) {
  const n = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / n, v[1] / n, v[2] / n];
}

function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function basis() {
  const forward = direction();
  // The world y axis points down, which is what the camera y axis has to line up with.
  let right = cross([0, 1, 0], forward);
  if (Math.hypot(right[0], right[1], right[2]) < 1e-6) right = [1, 0, 0];
  right = normalize(right);
  const down = cross(forward, right);
  return { forward, right, down };
}

function cameraMatrix() {
  const { forward, right, down } = basis();
  const eye = [
    state.target[0] - forward[0] * state.distance,
    state.target[1] - forward[1] * state.distance,
    state.target[2] - forward[2] * state.distance,
  ];
  // Row major 4x4, columns are right, down, forward, position.
  return [
    right[0], down[0], forward[0], eye[0],
    right[1], down[1], forward[1], eye[1],
    right[2], down[2], forward[2], eye[2],
    0, 0, 0, 1,
  ];
}

function adoptMatrix(c2w) {
  // Put the orbit target in front of a given pose, so the jump buttons keep the orbit usable.
  const forward = [c2w[2], c2w[6], c2w[10]];
  const eye = [c2w[3], c2w[7], c2w[11]];
  state.distance = Math.max(0.25 * state.radius, 1e-3);
  state.target = [
    eye[0] + forward[0] * state.distance,
    eye[1] + forward[1] * state.distance,
    eye[2] + forward[2] * state.distance,
  ];
  state.pitch = Math.asin(Math.max(-1, Math.min(1, forward[1])));
  state.yaw = Math.atan2(forward[0], forward[2]);
  invalidate();
}

function resetView() {
  if (!state.scene) return;
  state.target = state.scene.center.slice();
  state.radius = state.scene.radius;
  state.distance = 2 * state.scene.radius;
  state.yaw = 0;
  state.pitch = 0.15;
  invalidate();
}

// ----------------------------------------------------------------------------- render loop

function invalidate() {
  state.dirty = true;
}

function markMoving() {
  state.moving = true;
  state.movingUntil = performance.now() + 250;
  invalidate();
}

function currentRequest() {
  const stage = element("stage").getBoundingClientRect();
  return {
    c2w: cameraMatrix(),
    camera_model: cameraModel(),
    fov_deg: Number(element("fov").value),
    alpha: Number(element("alpha").value),
    res: Number(element("res").value),
    // The same bounds the server enforces, so an odd window shape never produces a rejected frame.
    aspect: Math.min(Math.max(stage.width / Math.max(stage.height, 1), 0.2), 10),
    moving: state.moving,
    tile_mode: element("tile-mode").value,
    frusta: element("frusta").checked,
    view: document.querySelector("input[name=view]:checked").value,
  };
}

async function renderFrame() {
  const started = performance.now();
  const body = currentRequest();
  state.inFlight = true;
  state.dirty = false;
  try {
    const response = await fetch("/api/render", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      element("stats").textContent = `render failed: ${response.status} ${await response.text()}`;
      return;
    }
    const blob = await response.blob();
    const bitmap = await createImageBitmap(blob);
    const stage = element("stage").getBoundingClientRect();
    canvas.width = Math.max(Math.round(stage.width), 1);
    canvas.height = Math.max(Math.round(stage.height), 1);
    context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    bitmap.close();

    const renderMs = Number(response.headers.get("X-Render-Ms") || 0);
    const clamped = Number(response.headers.get("X-Fov-Clamped") || body.fov_deg);
    state.lastRoundTrip = performance.now() - started;
    if (Math.abs(clamped - body.fov_deg) > 0.5) element("fov").value = Math.round(clamped);
    showFov(clamped, Number(response.headers.get("X-Width")), Number(response.headers.get("X-Height")));
    element("stats").textContent =
      `render ${renderMs.toFixed(1)} ms, round trip ${state.lastRoundTrip.toFixed(0)} ms, ` +
      `${(1000 / Math.max(state.lastRoundTrip, 1)).toFixed(1)} fps, ` +
      `${response.headers.get("X-Width")}x${response.headers.get("X-Height")}`;
  } catch (error) {
    element("stats").textContent = `render failed: ${error}`;
  } finally {
    state.inFlight = false;
  }
}

function loop() {
  flyStep();
  if (state.moving && performance.now() > state.movingUntil) {
    state.moving = false;
    invalidate(); // one more frame at the full resolution once the camera comes to rest
  }
  if (state.dirty && !state.inFlight && state.scene) renderFrame();
  requestAnimationFrame(loop);
}

// ----------------------------------------------------------------------------- navigation

let dragging = null;

canvas.addEventListener("pointerdown", (event) => {
  canvas.setPointerCapture(event.pointerId);
  dragging = { x: event.clientX, y: event.clientY, pan: event.shiftKey || event.button === 2 };
});

canvas.addEventListener("pointermove", (event) => {
  if (!dragging) return;
  const dx = event.clientX - dragging.x;
  const dy = event.clientY - dragging.y;
  dragging.x = event.clientX;
  dragging.y = event.clientY;
  if (dragging.pan) {
    const { right, down } = basis();
    const step = (state.distance * 0.0022);
    for (let i = 0; i < 3; i++) {
      state.target[i] -= right[i] * dx * step + down[i] * dy * step;
    }
  } else {
    state.yaw -= dx * 0.005;
    state.pitch = Math.max(-1.5, Math.min(1.5, state.pitch + dy * 0.005));
  }
  markMoving();
});

canvas.addEventListener("pointerup", () => { dragging = null; invalidate(); });
canvas.addEventListener("pointercancel", () => { dragging = null; });
canvas.addEventListener("contextmenu", (event) => event.preventDefault());

canvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  state.distance = Math.max(1e-3, Math.min(state.distance * Math.exp(event.deltaY * 0.0012), 500 * state.radius));
  markMoving();
}, { passive: false });

window.addEventListener("keydown", (event) => {
  state.shift = event.shiftKey;
  if (event.target.tagName === "INPUT" || event.target.tagName === "SELECT") return;
  if ("wasdqe".includes(event.key.toLowerCase())) {
    state.keys.add(event.key.toLowerCase());
    event.preventDefault();
  }
});

window.addEventListener("keyup", (event) => {
  state.shift = event.shiftKey;
  state.keys.delete(event.key.toLowerCase());
});
window.addEventListener("blur", () => { state.keys.clear(); state.shift = false; });

function flyStep() {
  if (state.keys.size === 0) return;
  const { forward, right, down } = basis();
  const step = 0.02 * state.radius * (state.shift ? 5 : 1);
  const move = [0, 0, 0];
  const add = (axis, sign) => { for (let i = 0; i < 3; i++) move[i] += axis[i] * sign * step; };
  if (state.keys.has("w")) add(forward, 1);
  if (state.keys.has("s")) add(forward, -1);
  if (state.keys.has("d")) add(right, 1);
  if (state.keys.has("a")) add(right, -1);
  if (state.keys.has("e")) add(down, 1);
  if (state.keys.has("q")) add(down, -1);
  for (let i = 0; i < 3; i++) state.target[i] += move[i];
  markMoving();
}

window.addEventListener("resize", invalidate);

// ----------------------------------------------------------------------------- controls

function cameraModel() {
  return document.querySelector("input[name=cam]:checked").value;
}

function ucmThetaMax(alpha) {
  const a = Math.min(Math.max(alpha, 1e-6), 1 - 1e-6);
  return Math.acos(-Math.min(a, 1 - a) / Math.max(a, 1 - a));
}

function syncCameraControls(keepFov) {
  const model = cameraModel();
  const slider = element("fov");
  let [low, high] = FOV_RANGE[model];
  if (model === "ucm") {
    high = Math.min(high, Math.floor(2 * ucmThetaMax(Number(element("alpha").value)) * 180 / Math.PI) - 1);
  }
  slider.min = low;
  slider.max = high;
  if (!keepFov) slider.value = Math.min(DEFAULT_FOV[model], high);
  slider.value = Math.min(Math.max(Number(slider.value), low), high);
  element("alpha-row").hidden = model !== "ucm";
  element("alpha-value").textContent = Number(element("alpha").value).toFixed(2);
  showFov(Number(slider.value), 0, 0);
  invalidate();
}

function showFov(fovDeg, width, height) {
  // Only the label follows the server, never the slider, so a reply cannot fight an ongoing drag.
  element("fov-value").textContent = `${fovDeg.toFixed(0)}°`;
  if (width > 0) {
    const half = (fovDeg * Math.PI / 180) / 2;
    const model = cameraModel();
    const alpha = Number(element("alpha").value);
    let rho = Math.tan(half);
    if (model === "fisheye") rho = half;
    if (model === "ucm") rho = Math.sin(half) / (alpha + (1 - alpha) * Math.cos(half));
    element("focal-status").textContent = `f = ${((width / 2) / rho).toFixed(1)} px at ${width}x${height}`;
  }
}

function fillViewButtons() {
  const holder = element("view-buttons");
  holder.innerHTML = "";
  if (!state.scene) return;
  state.scene.views.forEach((view) => {
    const button = document.createElement("button");
    button.textContent = `View ${view.index + 1}`;
    const alpha = view.alpha === undefined ? "" : `, α ${view.alpha.toFixed(2)}`;
    button.title = `fov ${view.fov_deg.toFixed(1)}°, fx ${view.fx_norm.toFixed(3)}${alpha}`;
    button.addEventListener("click", () => adoptMatrix(view.c2w));
    holder.appendChild(button);
  });
}

function fillSelect(id, entries, valueOf, labelOf) {
  const select = element(id);
  const previous = select.value;
  select.innerHTML = "";
  entries.forEach((entry) => {
    const option = document.createElement("option");
    option.value = valueOf(entry);
    option.textContent = labelOf(entry);
    select.appendChild(option);
  });
  if (previous) select.value = previous;
}

function applyState(data) {
  if (data.scenes) fillSelect("scene-select", data.scenes, (e) => e.id, (e) => e.label);
  if (data.checkpoints) fillSelect("ckpt-select", data.checkpoints, (e) => e.id, (e) => e.label);
  if (data.envs) fillSelect("env-select", data.envs, (e) => e.name, (e) => e.label);
  if (data.model) showModel(data.model);
  if (data.resolutions && element("res").options.length === 0) {
    data.resolutions.forEach((value) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = `${value} px`;
      option.selected = value === 512;
      element("res").appendChild(option);
    });
  }
  state.depthAvailable = Boolean(data.depth_available);
  element("depth-row").hidden = !state.depthAvailable;
  if (data.scene) adoptScene(data.scene);
}

function adoptScene(scene) {
  state.scene = scene;
  element("scene-select").value = scene.id;
  element("scene-status").textContent =
    `${scene.name}, ${scene.num_gaussians.toLocaleString()} Gaussians, ` +
    `${scene.camera_model}, tile mode ${scene.tile_mode}`;
  element("tile-mode").value = scene.tile_mode === "pbf" ? "pbf" : "projection";
  element("alpha").value = scene.alpha_init;
  fillViewButtons();
  resetView();
  syncCameraControls(true);
}

async function loadScene() {
  const id = element("scene-select").value;
  element("scene-status").textContent = `loading ${id} ...`;
  const response = await fetch("/api/load_scene", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id }),
  });
  if (!response.ok) {
    element("scene-status").textContent = `load failed: ${await response.text()}`;
    return;
  }
  const data = await response.json();
  state.depthAvailable = Boolean(data.depth_available);
  element("depth-row").hidden = !state.depthAvailable;
  adoptScene(data.scene);
}

function showModel(model) {
  element("ckpt-status").textContent = model
    ? `${model.label}, ${model.camera_model}, ${model.renderer}, ${model.tile_mode}, ${model.num_views} views`
    : "no model loaded";
  if (model) element("ckpt-select").value = model.id;
}

async function postJson(url, body, status) {
  element(status).textContent = "working ...";
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    element(status).textContent = `failed: ${response.status} ${await response.text()}`;
    return null;
  }
  return response.json();
}

async function loadModel() {
  const data = await postJson("/api/load_model", { id: element("ckpt-select").value }, "ckpt-status");
  if (!data) return;
  showModel(data.model);
  element("ckpt-status").textContent += `, loaded in ${data.seconds.toFixed(1)} s`;
}

function adoptReconstruction(data) {
  if (!data) return;
  if (data.scenes) fillSelect("scene-select", data.scenes, (e) => e.id, (e) => e.label);
  state.depthAvailable = Boolean(data.depth_available);
  element("depth-row").hidden = !state.depthAvailable;
  adoptScene(data.scene);
  element("scene-status").textContent += `, reconstructed in ${data.seconds.toFixed(1)} s`;
}

async function uploadAndReconstruct() {
  const files = element("upload-files").files;
  if (!files || files.length === 0) {
    element("scene-status").textContent = "pick some images first";
    return;
  }
  const form = new FormData();
  for (const file of files) form.append("files", file);
  element("scene-status").textContent = `uploading ${files.length} images ...`;
  const upload = await fetch("/api/upload", { method: "POST", body: form });
  if (!upload.ok) {
    element("scene-status").textContent = `upload failed: ${upload.status} ${await upload.text()}`;
    return;
  }
  const { upload_id } = await upload.json();
  adoptReconstruction(await postJson("/api/reconstruct", { mode: "upload", upload_id }, "scene-status"));
}

async function reconstructTartanair() {
  const body = { mode: "tartanair", env: element("env-select").value, index: Number(element("env-index").value) };
  adoptReconstruction(await postJson("/api/reconstruct", body, "scene-status"));
}

function syncSource() {
  const source = document.querySelector("input[name=source]:checked").value;
  ["precomputed", "upload", "tartanair"].forEach((name) => {
    element(`source-${name}`).hidden = name !== source;
  });
}

async function saveImage() {
  element("save-status").textContent = "saving ...";
  const body = currentRequest();
  body.moving = false;
  const response = await fetch("/api/save", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    element("save-status").textContent = `save failed: ${await response.text()}`;
    return;
  }
  const data = await response.json();
  element("save-status").innerHTML = `saved <a href="${data.url}" target="_blank">${data.path}</a>`;
}

element("scene-load").addEventListener("click", loadScene);
element("ckpt-load").addEventListener("click", loadModel);
element("upload-run").addEventListener("click", uploadAndReconstruct);
element("tartanair-run").addEventListener("click", reconstructTartanair);
document.querySelectorAll("input[name=source]").forEach((input) =>
  input.addEventListener("change", syncSource));
element("reset-view").addEventListener("click", resetView);
element("save-image").addEventListener("click", saveImage);
element("fov").addEventListener("input", () => { showFov(Number(element("fov").value), 0, 0); markMoving(); });
element("alpha").addEventListener("input", () => { syncCameraControls(true); markMoving(); });
element("res").addEventListener("change", invalidate);
element("tile-mode").addEventListener("change", invalidate);
element("frusta").addEventListener("change", invalidate);
document.querySelectorAll("input[name=cam]").forEach((input) =>
  input.addEventListener("change", () => syncCameraControls(false)));
document.querySelectorAll("input[name=view]").forEach((input) =>
  input.addEventListener("change", invalidate));

fetch("/api/state")
  .then((response) => response.json())
  .then((data) => { applyState(data); syncSource(); syncCameraControls(false); loop(); })
  .catch((error) => { element("stats").textContent = `cannot reach the server: ${error}`; });
