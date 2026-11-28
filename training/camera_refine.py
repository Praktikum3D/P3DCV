"""Refine the camera of ONE reconstructed scene after inference, without touching the network.

What this is for
----------------
E-RayZer predicts a focal length and an (E)UCM alpha per scene, and those predictions carry the
error the weekend goal is about. Everything else the network produced, the Gaussians and the poses,
is a consistent explanation of the input images UNDER that camera, so the camera is the one quantity
that can still be corrected once the forward pass is over: freeze the scene, keep the pixels, and
move only f and alpha until the images and the straight lines in them agree with the camera. It
costs a few seconds per scene and no training at all, which is why it is worth measuring separately
from the training loss that shares the same residual.

Two terms, and why both are needed
----------------------------------
* The photometric term re-renders the INPUT views and compares them with the observed images. It is
  the only term that knows what the scene looks like, but on its own it is weak and ambiguous here:
  the Gaussians are pixel aligned to the input views, so shrinking the camera and pulling the
  Gaussians along would reproduce the same image. It cannot pull the Gaussians (they are frozen), so
  it does constrain f, but through the disagreement between views rather than through anything
  inside one view.
* The plumb line term (`training/line_geometry.py`) knows nothing about the scene and everything
  about the camera: the viewing rays of the pixels of one straight 3D line are coplanar with the
  projection centre exactly when the intrinsics are right. It is blind to the overall brightness, to
  the depth and to the pose, and that is what makes it a good partner for the photometric term.

Gradients reach f and alpha because this project renders through rays built in PyTorch
(`render_opencv_cam_geer` hands `rays_o`/`rays_d` to 3DGEER and gets gradients back for them), so
differentiating the rendering with respect to the intrinsics needs nothing special here.

Parametrization
---------------
The free variables are offsets that start at exactly zero, so the refinement starts at the
prediction by construction and a zero step count returns it unchanged:

    f     = f0 * exp(s)                       one scalar for the scene, fx and fy together,
    alpha = sigmoid(logit(alpha0) + u)        stays inside (0, 1) whatever the optimizer does,
    beta  = beta0 * exp(t)                    stays positive, only when the camera model has one.

cx and cy are not refined. They are pinned at the image centre by the camera head and a scene with a
handful of views has no leverage on them worth the risk.

The pixel set is frozen
-----------------------
Both terms are evaluated on the pixels the INITIAL camera can render, computed once and kept fixed.
A mask that follows the current camera would hand the optimizer a way to improve the loss by
scoring fewer pixels, which is exactly the local minimum this refinement is supposed to avoid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from easydict import EasyDict as edict

from erayzer_core.model.utils_gaussian import (
    UCM_MODELS,
    Renderer,
    camera_theta,
    render_mask,
)
from training.line_detection import DEFAULT_MAX_COUNT, DEFAULT_NUM_SAMPLES, detect_line_segments
from training.line_geometry import DEFAULT_TRIM, plumbline_residual_mean, valid_segment_mask

DEFAULT_STEPS = 200
"""Adam steps. The focal length moves by about `lr` in log space per step, so 200 steps at the
default learning rate can travel a factor of e, far more than any plausible correction."""

DEFAULT_LR = 0.01
"""Adam learning rate on the offsets. One step changes f by about one percent."""

DEFAULT_LINE_WEIGHT = 1.0
"""Weight of the plumb line residual against the masked photometric MSE.

`refine_camera` reports both terms at step 0, and `auto_lambda` sets the weight so they start out
equal. The constant here is only the fallback for a caller that wants a fixed number, and the
measurement says the fallback is not what a wide camera wants.

Measured on `hongkong_b90_fisheye`, a real scene dumped from a checkpoint whose field of view had
collapsed to 97 degrees against a true 195: over the whole correction the photometric term grows by
0.34 while the plumb line term falls by 0.0026, a ratio of about 130. So any line weight below that
leaves the photometric term in charge, and it votes for the camera the scene was built with, which
is the wrong one. Observed exactly that way:

    line weight     1      300          0 (image weight 0)
    field of view  97.2    175.7        192.7 degrees

The reason is structural and not a property of this scene. The Gaussians are pixel aligned to the
input views under the PREDICTED camera, so re-rendering those views reproduces the input images
however wrong that camera is; the photometric term is therefore not a measurement of the camera but
an anchor to wherever the network left it. A refinement that is meant to move the camera should run
with `image_weight=0` and keep the photometric number as a diagnostic.
"""

DEFAULT_DETECT_SIZE = 1024
"""Side the input views are rendered at for the line detection, 0 for the working resolution.

The resolution of the DETECTION, not of the refinement. `training/line_detection.py` documents why
it is the single most important setting here: the detector's endpoint noise is a fixed fraction of
a pixel, so in angle it grows as the image shrinks, and at 256 pixels it buries the signal and turns
the residual adversarial. A refinement runs on a handful of frames and can afford the sharpest
setting, which is why the dataset's compromise of 512 does not apply.

Be aware of what that buys on a scene whose images are themselves only 256 pixels: rendering them
at 1024 does not create detail that is not there, so the accuracy is still a 256 pixel accuracy
spread over four times as many pixels. It costs little and cannot hurt, but it is not the same thing
as detecting on a sharp 1000 pixel source frame.
"""


@dataclass
class RefineScene:
    """One reconstructed scene, frozen, plus the images its input views are supposed to show.

    * `xyz` [n, 3], `features` [n, k, 3], `scaling` [n, 3], `rotation` [n, 4], `opacity` [n, 1] as
      the model's `to_gs` produces them, raw: `GaussianModel` applies exp and sigmoid itself.
    * `c2w` [v, 4, 4] and `fxfycxcy` [v, d] IN PIXELS, d = 4, 5 with alpha, 6 with beta.
    * `images` [v, 3, h, w] in [0, 1], the observed input views.
    * `camera_model`, `near_plane`, `tile_mode` as the run that produced the scene used them.
    """

    xyz: torch.Tensor
    features: torch.Tensor
    scaling: torch.Tensor
    rotation: torch.Tensor
    opacity: torch.Tensor
    sh_degree: int
    c2w: torch.Tensor
    fxfycxcy: torch.Tensor
    images: torch.Tensor
    camera_model: str = "fisheye"
    near_plane: float = 0.01
    tile_mode: str = "projection"
    name: str = ""

    @property
    def device(self) -> torch.device:
        return self.xyz.device

    @property
    def image_hw(self) -> Tuple[int, int]:
        return int(self.images.shape[-2]), int(self.images.shape[-1])

    def to(self, device) -> "RefineScene":
        moved = {}
        for key, value in self.__dict__.items():
            moved[key] = value.to(device) if torch.is_tensor(value) else value
        return RefineScene(**moved)


def renderer_for(scene: RefineScene) -> Renderer:
    """A `Renderer` configured exactly like the run that produced `scene`.

    The same class the training loop renders with, so the refinement optimizes through the very code
    path it is correcting, 3DGEER with rays built in PyTorch included.
    """
    config = edict(
        model=edict(
            camera_model=scene.camera_model,
            renderer="geer",
            geer_tile_mode=scene.tile_mode,
            near_plane=float(scene.near_plane),
            use_gsplat=True,
            use_deferred_rendering=False,
            gaussians=edict(sh_degree=int(scene.sh_degree)),
        )
    )
    return Renderer(config)


def render_views(
    scene: RefineScene,
    renderer: Renderer,
    fxfycxcy: torch.Tensor,
    height: int,
    width: int,
    views: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """Render the scene's views with `fxfycxcy` [v, d] in pixels, [v, 3, height, width].

    Differentiable with respect to `fxfycxcy`; the Gaussians and the poses go in as they are and are
    never asked for a gradient, because nothing in this module optimizes them.
    """
    index = torch.arange(scene.c2w.shape[0]) if views is None else torch.as_tensor(list(views))
    c2w = scene.c2w[index.to(scene.c2w.device)]
    out = renderer(
        scene.xyz[None],
        scene.features[None],
        scene.scaling[None],
        scene.rotation[None],
        scene.opacity[None],
        height,
        width,
        C2W=c2w[None],
        fxfycxcy=fxfycxcy[None],
    )
    return out.render[0]


def scale_intrinsics(fxfycxcy: torch.Tensor, scale: float) -> torch.Tensor:
    """fx, fy, cx, cy times `scale`, alpha and beta untouched: the same camera on a resized image."""
    factor = torch.ones_like(fxfycxcy)
    factor[..., :4] = float(scale)
    return fxfycxcy * factor


def fov_degrees(fxfycxcy: torch.Tensor, camera_model: str, width: int) -> float:
    """Field of view across the image width in degrees, for `fxfycxcy` [v, d] in pixels.

    The same quantity `train.camera_stats` logs as `camera/fov_deg`, so the numbers a refinement
    reports can be put next to a training curve without converting anything.
    """
    f = fxfycxcy[..., 0].float()
    alpha = fxfycxcy[..., 4].float() if camera_model in UCM_MODELS else None
    beta = fxfycxcy[..., 5].float() if camera_model == "eucm" else None
    theta = camera_theta((0.5 * width) / f.clamp_min(1e-6), camera_model, alpha, beta)
    return float(math.degrees(2.0 * theta.mean().item()))


def detect_scene_lines(
    scene: RefineScene,
    *,
    detect_size: int = DEFAULT_DETECT_SIZE,
    max_count: int = DEFAULT_MAX_COUNT,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    min_length: float = 0.2,
    detector: str = "auto",
    images: Optional[torch.Tensor] = None,
    renderer: Optional[Renderer] = None,
) -> torch.Tensor:
    """The lines of every input view, [v, max_count, num_samples, 2], padded as the batch tensors are.

    Detected ONCE, before the refinement starts, and then held fixed: the pixels are the measurement,
    they do not move when the camera does. Which is also why they must be found on images that do not
    depend on the camera being right, that is on the observed input views.

    `detect_size` renders the views again at that resolution for the detection only and scales the
    pixels back into the working resolution afterwards; see DEFAULT_DETECT_SIZE. The rendering is
    done with the scene's own (initial) camera, so the detection sees the image the model believes
    in. Pass `images` to detect on given images instead, which is what a caller with the real input
    frames at full size should do.

    Pixels outside what the initial camera can render are kept out through `render_mask`, so the
    white rim of the rendered image circle, which belongs to no 3D line, cannot enter.
    """
    height, width = scene.image_hw
    if images is None:
        size = int(detect_size) if detect_size and int(detect_size) > 0 else width
        scale = float(size) / float(width)
        intrinsics = scale_intrinsics(scene.fxfycxcy, scale)
        if abs(scale - 1.0) < 1e-9:
            images = scene.images
        else:
            renderer = renderer_for(scene) if renderer is None else renderer
            # One view at a time and straight to the CPU: the detection runs there anyway, and a
            # batch of high resolution renders is the largest allocation anything here makes.
            with torch.no_grad():
                images = torch.cat(
                    [render_views(scene, renderer, intrinsics[i: i + 1], size, size, views=[i]).cpu()
                     for i in range(scene.c2w.shape[0])]
                )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        size = int(images.shape[-1])
        scale = float(size) / float(width)
        intrinsics = scale_intrinsics(scene.fxfycxcy, scale)

    mask = render_mask(
        intrinsics.detach().float(), int(images.shape[-2]), size, scene.camera_model,
        exact=scene.tile_mode == "projection",
    )  # [v, 1, h, w]

    pixels = torch.full((images.shape[0], max_count, num_samples, 2), -1.0, dtype=torch.float32)
    for view in range(images.shape[0]):
        found = detect_line_segments(
            images[view].detach().float().clamp(0.0, 1.0).cpu(),
            mask=mask[view].cpu(),
            min_length=min_length,
            max_count=max_count,
            num_samples=num_samples,
            detector=detector,
        )
        if len(found):
            # Back into the working resolution, where the intrinsics the refinement optimizes live.
            pixels[view, : len(found)] = torch.from_numpy(found.pixels) / scale
    return pixels.to(scene.device)


class CameraOffsets(torch.nn.Module):
    """The handful of numbers the refinement is allowed to move, all starting at zero.

    `forward()` returns the full `fxfycxcy` [v, d] in pixels, built from the scene's initial values
    so that an unstepped module reproduces them exactly (exp(0) and a sigmoid of the original logit).
    """

    def __init__(self, fxfycxcy: torch.Tensor, camera_model: str, learn_beta: bool = False,
                 learn_alpha: bool = True):
        super().__init__()
        self.register_buffer("initial", fxfycxcy.detach().clone().float())
        self.camera_model = camera_model
        self.log_f = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.alpha_offset = None
        self.log_beta = None
        if camera_model in UCM_MODELS and learn_alpha and fxfycxcy.shape[-1] > 4:
            alpha0 = self.initial[..., 4].mean().clamp(1e-4, 1 - 1e-4)
            self.register_buffer("alpha_logit", torch.log(alpha0 / (1 - alpha0)))
            self.alpha_offset = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        if camera_model == "eucm" and learn_beta and fxfycxcy.shape[-1] > 5:
            self.log_beta = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        columns = [
            self.initial[..., 0] * torch.exp(self.log_f),
            self.initial[..., 1] * torch.exp(self.log_f),
            self.initial[..., 2],
            self.initial[..., 3],
        ]
        if self.alpha_offset is not None:
            alpha = torch.sigmoid(self.alpha_logit + self.alpha_offset)
            columns.append(alpha.expand_as(columns[0]))
        elif self.initial.shape[-1] > 4:
            columns.append(self.initial[..., 4])
        if self.initial.shape[-1] > 5:
            beta = self.initial[..., 5]
            if self.log_beta is not None:
                beta = beta * torch.exp(self.log_beta)
            columns.append(beta)
        return torch.stack(columns, dim=-1)


def masked_image_loss(
    render: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor]
) -> torch.Tensor:
    """Mean squared error over the masked pixels, the training loss's own normalisation.

    `compute_loss` in `train.py` replaces the masked out pixels by the target and divides by the
    fraction that is left, so that a smaller mask does not look like a smaller error. The same here,
    with a mask that never changes, so the division is a constant and only the comparison moves.
    """
    if mask is None:
        return torch.nn.functional.mse_loss(render, target)
    inside = max(float(mask.mean()), 1e-3)
    blended = render * mask + target * (1 - mask)
    return torch.nn.functional.mse_loss(blended, target) / inside


@dataclass
class RefineResult:
    """What a refinement did. `history` has one row per step, step 0 being the starting point."""

    fxfycxcy: torch.Tensor
    initial_fxfycxcy: torch.Tensor
    camera_model: str
    history: List[Dict[str, float]] = field(default_factory=list)
    num_lines: int = 0

    @property
    def first(self) -> Dict[str, float]:
        return self.history[0] if self.history else {}

    @property
    def last(self) -> Dict[str, float]:
        return self.history[-1] if self.history else {}

    def summary(self) -> str:
        if not self.history:
            return "no steps"
        a, b = self.first, self.last
        text = (
            f"fov {a['fov_deg']:.2f} -> {b['fov_deg']:.2f} deg, "
            f"fx {a['fx']:.2f} -> {b['fx']:.2f} px, "
            f"image {a['image_loss']:.6f} -> {b['image_loss']:.6f}, "
            f"line {a['residual']:.6f} -> {b['residual']:.6f}"
        )
        if "alpha" in a:
            text += f", alpha {a['alpha']:.4f} -> {b['alpha']:.4f}"
        return text


def refine_camera(
    scene: RefineScene,
    *,
    line_pixels: Optional[torch.Tensor] = None,
    steps: int = DEFAULT_STEPS,
    lr: float = DEFAULT_LR,
    line_weight: float = DEFAULT_LINE_WEIGHT,
    image_weight: float = 1.0,
    auto_lambda: bool = False,
    learn_alpha: bool = True,
    learn_beta: bool = False,
    trim: float = DEFAULT_TRIM,
    renderer: Optional[Renderer] = None,
    progress: Optional[int] = None,
) -> RefineResult:
    """Minimise `image_weight * photometric + line_weight * plumbline` over f, alpha and beta.

    `line_pixels` [v, n, K, 2] comes from `detect_scene_lines` (or from the dataset, in the same
    padded layout); without it the refinement runs on the photometric term alone, which is the
    comparison the sieve needs. With `auto_lambda` the weight is replaced by the ratio of the two
    terms measured at step 0, so both start out exactly equal and the number is a measurement rather
    than a guess.

    `learn_alpha` can be turned off to refine the focal length alone. That is worth having because
    f and alpha are not independent under a few lines: several (f, alpha) pairs bend the rays almost
    the same way over the pixels that were actually observed, so a scene with a handful of lines can
    be better served by moving one parameter than two. The local demonstration in
    `scripts/refine_camera.py` shows exactly that contrast.

    Returns a `RefineResult` whose `history` holds, per step, the two terms, fx in pixels, the field
    of view in degrees and alpha where there is one. Nothing in the scene is modified.
    """
    height, width = scene.image_hw
    renderer = renderer_for(scene) if renderer is None else renderer
    offsets = CameraOffsets(
        scene.fxfycxcy, scene.camera_model, learn_beta=learn_beta, learn_alpha=learn_alpha
    ).to(scene.device)
    target = scene.images.detach().float()

    # Frozen once, see the module docstring: the pixel set must not follow the camera.
    mask = render_mask(
        scene.fxfycxcy.detach().float(), height, width, scene.camera_model,
        exact=scene.tile_mode == "projection",
    ).to(device=scene.device, dtype=target.dtype)

    # A padded row is marked by its negative first coordinate, the convention `line_detection` sets
    # and `line_geometry.valid_segment_mask` reads; with no real line at all the term is left out
    # entirely rather than added as a zero that still costs a backward pass.
    num_lines = int(valid_segment_mask(line_pixels).sum()) if line_pixels is not None else 0
    has_lines = num_lines > 0

    photometric = float(image_weight) != 0.0

    def terms(fxfycxcy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # With a zero image weight the rendering is still done, but only to report the number; not
        # asking for its gradient is what makes a line only refinement the cheap one.
        with torch.enable_grad() if photometric else torch.no_grad():
            render = render_views(scene, renderer, fxfycxcy, height, width)
            image = masked_image_loss(render, target, mask)
        if not has_lines:
            return image, fxfycxcy.new_zeros(())
        residual = plumbline_residual_mean(
            fxfycxcy[:, None, :], scene.camera_model, line_pixels, (height, width), trim=trim
        )
        return image, residual

    optimizer = torch.optim.Adam(offsets.parameters(), lr=float(lr))
    history: List[Dict[str, float]] = []
    weight = float(line_weight)
    for step in range(int(steps) + 1):
        fxfycxcy = offsets()
        image, residual = terms(fxfycxcy)
        if step == 0 and auto_lambda and has_lines:
            # Both terms the same size at the start, which is what makes the two parts comparable
            # instead of one of them being decoration.
            weight = float(image.detach()) / max(float(residual.detach()), 1e-12)
        total = float(image_weight) * image + weight * residual
        row = {
            "step": float(step),
            "image_loss": float(image.detach()),
            "residual": float(residual.detach()),
            "total": float(total.detach()),
            "fx": float(fxfycxcy[..., 0].detach().mean()),
            "fov_deg": fov_degrees(fxfycxcy.detach(), scene.camera_model, width),
        }
        if fxfycxcy.shape[-1] > 4:
            row["alpha"] = float(fxfycxcy[..., 4].detach().mean())
        if fxfycxcy.shape[-1] > 5:
            row["beta"] = float(fxfycxcy[..., 5].detach().mean())
        history.append(row)
        if progress and (step % int(progress) == 0 or step == int(steps)):
            print(
                f"  step {step:4d}  image {row['image_loss']:.6f}  line {row['residual']:.6f}  "
                f"fx {row['fx']:.2f}  fov {row['fov_deg']:.2f}",
                flush=True,
            )
        if step == int(steps) or not total.requires_grad:
            # Nothing to optimize happens when both terms are switched off; the history still holds
            # the starting point, so a caller gets a report rather than an exception.
            break
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()

    with torch.no_grad():
        final = offsets().detach()
    return RefineResult(
        fxfycxcy=final,
        initial_fxfycxcy=scene.fxfycxcy.detach().clone(),
        camera_model=scene.camera_model,
        history=history,
        num_lines=num_lines,
    )
