
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict

from .gaussians_renderer import (
    GaussianModel,
    render_opencv_cam_geer,
    render_opencv_cam_gsplat,
    deferred_gaussian_render,
)


# The camera models built on the (Enhanced) Unified Camera Model. "ucm" carries one extra intrinsic,
# alpha in column 4; "eucm" carries beta in column 5 on top of it. UCM is exactly EUCM with beta = 1,
# which is why the two share all the math below and differ only in where beta comes from.
UCM_MODELS = ("ucm", "eucm")


def _ucm_rho_squared(rho, beta):
    """beta * rho^2, the only place beta enters the (E)UCM unprojection.

    beta is None for "ucm", which is EUCM with beta fixed at 1. Multiplying by a beta of exactly 1.0
    is exact in IEEE 754, so an eucm camera with beta = 1 produces the very same floats as a ucm one.
    """
    return rho**2 if beta is None else beta * rho**2


def camera_theta(rho, camera_model, alpha=None, beta=None):
    """Angle to the optical axis for the image radius rho divided by f.

    The Enhanced Unified Camera Model projects a ray (x, y, z) to the normalized image radius

        rho = m / (alpha * d + (1 - alpha) * z),   m = sqrt(x^2 + y^2),   d = sqrt(beta * m^2 + z^2),

    so for a unit ray, m = sin(theta) and z = cos(theta). With beta = 1 the auxiliary distance d is
    the length of the unit ray itself, d = 1, and the formula collapses to the UCM
    rho = sin(theta) / (alpha + (1 - alpha) cos(theta)). alpha = 0 is the pinhole in both cases.

    Unprojecting it is a quadratic in z and has the closed form theta = atan2(rho, mz) with

        mz = (1 - beta alpha^2 rho^2) / (alpha sqrt(1 - (2 alpha - 1) beta rho^2) + 1 - alpha),

    which is the known UCM expression with rho^2 replaced by beta rho^2. Derivation: write a unit ray
    as m = sin(theta), z = cos(theta), put g = sqrt(beta m^2 + z^2) and D = alpha g + (1 - alpha) z,
    so that rho = m / D. Then

        D^2 - beta alpha^2 m^2 = z (z (alpha^2 + (1 - alpha)^2) + 2 alpha (1 - alpha) g)   and
        D^2 - (2 alpha - 1) beta m^2 = ((1 - alpha) g + alpha z)^2,

    so the numerator of mz is that first line over D^2, the square root is ((1 - alpha) g + alpha z) / D,
    the denominator becomes (z (alpha^2 + (1 - alpha)^2) + 2 alpha (1 - alpha) g) / D, and the whole
    fraction reduces to mz = z / D. With rho = m / D that gives atan2(rho, mz) = atan2(m, z) = theta.

    The square root is clamped at zero, which is the border of the image circle: beyond it the camera
    images nothing and pixel_angles marks those pixels invalid.
    """
    if camera_model in UCM_MODELS:  # (enhanced) unified camera model, alpha = 0 is pinhole
        if camera_model == "eucm" and beta is None:
            raise ValueError("the eucm camera model needs a beta")
        rho2 = _ucm_rho_squared(rho, beta)
        mz = (1 - alpha**2 * rho2) / (alpha * torch.sqrt((1 - (2 * alpha - 1) * rho2).clamp_min(0)) + 1 - alpha)
        return torch.atan2(rho, mz)
    return rho if camera_model == "fisheye" else torch.atan(rho)


def camera_rho(theta, camera_model, alpha=None, beta=None):
    """Image radius divided by f for the angle theta, the forward direction of camera_theta.

    The (E)UCM projection of the unit ray (sin theta, 0, cos theta), see camera_theta for the model
    and for the inverse. The auxiliary distance d = sqrt(beta sin^2 + cos^2) is evaluated in the
    equivalent form sqrt(1 + (beta - 1) sin^2): it loses no accuracy near theta = 0 and, more
    importantly here, a beta of exactly 1 makes it exactly 1.0, where sqrt(sin^2 + cos^2) would only
    be 1 up to rounding. That is what keeps eucm with beta = 1 bit for bit equal to ucm.
    """
    if camera_model in UCM_MODELS:
        if camera_model == "eucm" and beta is None:
            raise ValueError("the eucm camera model needs a beta")
        s, c = torch.sin(theta), torch.cos(theta)
        one = torch.ones_like(s)
        d = one if beta is None else torch.sqrt((one + (beta - 1) * s**2).clamp_min(0))
        return s / (alpha * d + (1 - alpha) * c)
    return theta if camera_model == "fisheye" else torch.tan(theta)


def camera_rays(c2w, fxfycxcy, h, w, camera_model="pinhole"):
    """Ray origins and normalized directions in world space for every pixel, both [b, h, w, 3].

    c2w [b, 4, 4], fxfycxcy [b, 4] in pixels, for UCM [b, 5] with alpha last and for EUCM [b, 6] with
    beta after it. This is the ray math of cam_info_to_plucker, kept in one place so the plucker
    embeddings and the 3DGEER renderer agree.
    """
    b = c2w.shape[0]
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    y, x = y.to(c2w), x.to(c2w)
    x = x[None, :, :].expand(b, -1, -1).reshape(b, -1)
    y = y[None, :, :].expand(b, -1, -1).reshape(b, -1)
    x = (x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
    y = (y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
    z = torch.ones_like(x)
    if camera_model != "pinhole":  # fisheye or (E)UCM, direction from the angle to the optical axis
        rho = torch.sqrt(x * x + y * y + 1e-12)
        alpha = fxfycxcy[:, 4:5] if camera_model in UCM_MODELS else None
        beta = fxfycxcy[:, 5:6] if camera_model == "eucm" else None
        theta = camera_theta(rho, camera_model, alpha, beta)
        x, y, z = x * torch.sin(theta) / rho, y * torch.sin(theta) / rho, torch.cos(theta)
    ray_d = torch.stack([x, y, z], dim=2)  # [b, h*w, 3]
    ray_d = torch.bmm(ray_d, c2w[:, :3, :3].transpose(1, 2))  # [b, h*w, 3]
    ray_d = ray_d / torch.norm(ray_d, dim=2, keepdim=True)  # [b, h*w, 3]
    ray_o = c2w[:, :3, 3][:, None, :].expand_as(ray_d)  # [b, h*w, 3]
    return ray_o.reshape(b, h, w, 3), ray_d.reshape(b, h, w, 3)


def pixel_angles(fxfycxcy, h, w, camera_model, exact=False):
    """Offsets dx, dy from the principal point, their radius, the angle to the optical axis and the pixels
    the camera can render, all [n, h, w]. fxfycxcy [n, 4] in pixels, for UCM [n, 5] with alpha last and
    for EUCM [n, 6] with beta after it.

    gsplat and 3DGEER in its "pbf" tile mode drop everything behind the camera plane (more than 90 degrees off
    axis), 3DGEER culls the Gaussian centres there and only splat tails reach further. The (E)UCM is undefined
    outside its own image circle. With exact the angle is not capped at all, which the Kannala Brandt fit needs
    to know its own range and which 3DGEER in its "projection" tile mode really renders, out to the image border.
    """
    y, x = torch.meshgrid(torch.arange(h, device=fxfycxcy.device), torch.arange(w, device=fxfycxcy.device), indexing="ij")
    dx, dy = x + 0.5 - fxfycxcy[:, 2, None, None], y + 0.5 - fxfycxcy[:, 3, None, None]
    radius = torch.sqrt(dx**2 + dy**2)
    rho = radius / fxfycxcy[:, 0, None, None]
    alpha = fxfycxcy[:, 4, None, None] if camera_model in UCM_MODELS else None
    beta = fxfycxcy[:, 5, None, None] if camera_model == "eucm" else None
    theta = camera_theta(rho, camera_model, alpha, beta)
    valid = theta <= (math.pi if exact else math.pi / 2)
    if camera_model in UCM_MODELS:
        # The gsplat warp interpolates, so stay 1.5 px inside the 90 degree border where gsplat leaves background.
        # The second condition is the image circle, where the square root of camera_theta turns negative.
        margin = 0.0 if exact else 1.5 / fxfycxcy[:, 0, None, None]
        valid = (theta <= (math.pi if exact else math.pi / 2) - margin) & \
                (1 - (2 * alpha - 1) * _ucm_rho_squared(rho, beta) >= 0)
    return dx, dy, radius, theta, valid


def render_mask(fxfycxcy, h, w, camera_model, image_circle=False, exact=False):
    """Pixels the camera model can render, [n, 1, h, w]. With image_circle only the circle spanning the
    image width counts (fisheye data such as TartanAir lcam_fish).
    """
    _, _, radius, _, valid = pixel_angles(fxfycxcy, h, w, camera_model, exact)
    if image_circle:
        valid = valid & (radius <= w / 2)
    return valid[:, None].float()


def loss_render_mask(config, fxfycxcy, h, w, has_data_mask=False):
    """The pixels the training loss and the evaluation score, [n, 1, h, w]. fxfycxcy [n, 4], UCM [n, 5],
    EUCM [n, 6].

    Both of them call this, so they always mask the same pixels. The mask normally stops at 90 degrees off
    axis, because neither gsplat nor 3DGEER in its "pbf" tile mode draws anything further out and those
    pixels would be background without a gradient. With the geer renderer in its "projection" tile mode the
    renderer fills the whole image circle, so the mask is the camera model's own range instead.

    The image circle is a guess from the dataset name, right only as long as every frame really is a full
    frame lcam_fish image. With camera augmentation the caller passes the sample's own data mask instead,
    which states exactly which pixels carry data, so the guess is then dropped: keeping it would cut the
    corners off an augmented pinhole sample, whose data mask is all ones and whose corners carry the
    strongest focal length signal. Default False, so evaluation and every unaugmented run are unchanged.
    """
    camera_model = config.model.get("camera_model", "pinhole")
    image_circle = (not has_data_mask) and "fish" in str(config.dataset.get("camera_name", ""))
    beyond_90 = (config.model.get("renderer", "gsplat") == "geer"
                 and config.model.get("geer_tile_mode", "pbf") == "projection")
    return render_mask(fxfycxcy, h, w, camera_model, image_circle, exact=beyond_90)


def ucm_radial_coeffs(fxfycxcy, h, w, camera_model="ucm"):
    """Kannala Brandt coefficients [v, 4] that reproduce the (E)UCM (alpha in column 4, beta in column 5
    for eucm) up to the largest valid angle, by least squares on
    theta_d / theta - 1 = k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8.

    An eucm camera with beta = 1 is a ucm camera, and the fit it produces is the same one bit for bit:
    camera_rho evaluates the auxiliary distance d so that beta = 1 gives exactly 1.0.

    A view whose learned camera has collapsed gets zero coefficients, which is the equidistant fisheye
    3DGEER renders by default. That happens when no pixel is inside the UCM image circle any more, so
    theta_max is zero and the sample grid collapses onto the origin, or when an intrinsic is not finite.
    There is nothing left to fit in that case, and surviving matters more than accuracy here: the renderer
    keeps producing finite images and a defined gradient, so the camera can recover, while the old code
    handed a rank deficient matrix to the solver and killed a 20000 step run (job 31792) at step 3930.
    """
    assert camera_model in UCM_MODELS, f"{camera_model} is not a unified camera model"
    _, _, _, theta, valid = pixel_angles(fxfycxcy, h, w, camera_model, exact=True)
    theta_max = torch.where(valid, theta, torch.zeros_like(theta)).flatten(1).amax(1)  # [v]
    a = fxfycxcy[:, 4]  # [v]
    b = fxfycxcy[:, 5] if camera_model == "eucm" else None  # [v]
    # Below this angle the sample grid th is numerically zero and all four columns of A vanish together.
    degenerate = ~torch.isfinite(fxfycxcy).all(1) | ~torch.isfinite(theta_max) | (theta_max < 1e-2)  # [v]
    theta_max = torch.where(degenerate, torch.ones_like(theta_max), theta_max)  # harmless stand in
    a = torch.where(degenerate, torch.zeros_like(a), a)[:, None]  # [v, 1]
    if b is not None:  # a collapsed camera keeps the beta of plain UCM, the coefficients are dropped anyway
        b = torch.where(degenerate, torch.ones_like(b), b)[:, None]  # [v, 1]
    th = theta_max[:, None] * torch.linspace(0.02, 1.0, 64, device=theta.device)[None]  # [v, 64]
    target = camera_rho(th, camera_model, a, b) / th - 1
    A = torch.stack([th**2, th**4, th**6, th**8], dim=-1)  # [v, 64, 4]
    # Solved through the pseudo inverse instead of torch.linalg.lstsq: the only lstsq driver CUDA offers is
    # gels (QR), which assumes full rank and raises as soon as the fit degenerates, and the rank safe driver
    # gelsd runs on CPU only, so it would force a .cpu() round trip of the inputs and the result. The SVD
    # behind pinv stays on the device and blocks on the stream exactly as the old lstsq did, so it adds no
    # new synchronisation, it drops vanishing singular values and returns the minimum norm solution, and on
    # a healthy fit it agrees with lstsq far below the float32 the result is cast to. It costs about 6 ms
    # against 1 ms for lstsq on a GTX 1660, once per batch element per step, a few percent of a UCM step.
    coeffs = (torch.linalg.pinv(A.double()) @ target.double()[..., None])[..., 0]  # [v, 4]
    return torch.where(degenerate[:, None], torch.zeros_like(coeffs), coeffs).float()


class Renderer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.sh_degree = config.model.gaussians.sh_degree
        self.gaussians_model = GaussianModel(config.model.gaussians.sh_degree)

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(
        self,
        xyz,
        features,
        scaling,
        rotation,
        opacity,
        height,
        width,
        C2W,
        fxfycxcy,
        deferred=True,
    ):
        """
        xyz: [b, n_gaussians, 3]
        features: [b, n_gaussians, (sh_degree+1)^2, 3]
        scaling: [b, n_gaussians, 3]
        rotation: [b, n_gaussians, 4]
        opacity: [b, n_gaussians, 1]

        height: int
        width: int
        C2W: [b, v, 4, 4]
        fxfycxcy: [b, v, 4], UCM [b, v, 5] with alpha last, EUCM [b, v, 6] with beta after it

        output: [b, v, 3, height, width]
        """
        if self.config.model.get("camera_model", "pinhole") != "pinhole":
            assert self.config.model.get("use_gsplat", True) and not self.config.model.get("use_deferred_rendering", False), \
                "camera models other than pinhole need the gsplat renderer"

        # if deferred:
        if self.config.model.get("use_deferred_rendering", False):
            renderings = deferred_gaussian_render(
                xyz, features, scaling, rotation, opacity, height, width, C2W, fxfycxcy
            )
            b, v = C2W.size(0), C2W.size(1)
            depth = torch.zeros(
                b, v, 1, height, width, dtype=torch.float32, device=xyz.device
            )
            alpha = torch.zeros(
                b, v, 1, height, width, dtype=torch.float32, device=xyz.device
            )
        else:
            b, v = C2W.size(0), C2W.size(1)
            renderings = torch.zeros(
                b, v, 3, height, width, dtype=torch.float32, device=xyz.device
            )

            depth = torch.zeros(
                b, v, 1, height, width, dtype=torch.float32, device=xyz.device
            )
            alpha = torch.zeros(
                b, v, 1, height, width, dtype=torch.float32, device=xyz.device
            )

            for i in range(b):
                pc = self.gaussians_model.set_data(
                    xyz[i], features[i], scaling[i], rotation[i], opacity[i]
                )
                if self.config.model.get("use_gsplat", True): # False
                    near_plane = self.config.model.get("near_plane", 0.2)
                    camera_model = self.config.model.get("camera_model", "pinhole")
                    if self.config.model.get("renderer", "gsplat") == "geer":
                        radial = None
                        if camera_model in UCM_MODELS:  # 3DGEER has no (E)UCM, a fitted Kannala Brandt fisheye replaces it
                            radial = ucm_radial_coeffs(fxfycxcy[i].detach(), height, width, camera_model)
                        elif camera_model == "fisheye":
                            radial = torch.zeros(fxfycxcy.shape[1], 4, device=xyz.device)
                        buffers = render_opencv_cam_geer(
                            pc, height, width, C2W[i], fxfycxcy[i], self.sh_degree, near_plane=near_plane,
                            camera_model="pinhole" if camera_model == "pinhole" else "fisheye", radial_coeffs=radial,
                            ray_camera_model=camera_model,
                            geer_tile_mode=self.config.model.get("geer_tile_mode", "pbf"),
                        )
                    elif camera_model in UCM_MODELS:
                        # gsplat has no (E)UCM. Render an equidistant fisheye image whose focal length keeps every
                        # valid pixel inside it, then look up each (E)UCM pixel by its angle to the optical axis.
                        dx, dy, radius, theta, valid = pixel_angles(fxfycxcy[i], height, width, camera_model)
                        ratio = torch.where(valid, radius / theta.clamp_min(1e-6), torch.full_like(radius, float("inf")))
                        f_e = ratio.flatten(1).amin(1).detach()
                        K_e = torch.stack([f_e, f_e, fxfycxcy[i][:, 2], fxfycxcy[i][:, 3]], dim=-1)
                        buffers = render_opencv_cam_gsplat(
                            pc, height, width, C2W[i], K_e, self.sh_degree, near_plane=near_plane, camera_model="fisheye"
                        )
                        scale = f_e[:, None, None] * theta / radius.clamp_min(1e-6)
                        grid = torch.stack([(fxfycxcy[i][:, 2, None, None] + dx * scale) * 2 / width - 1,
                                            (fxfycxcy[i][:, 3, None, None] + dy * scale) * 2 / height - 1], dim=-1)
                        buffers = {"render": F.grid_sample(buffers["render"], grid, align_corners=False, padding_mode="border")}
                    else:
                        buffers = render_opencv_cam_gsplat(
                            pc, height, width, C2W[i], fxfycxcy[i], self.sh_degree,
                            near_plane=near_plane,
                            camera_model=camera_model,
                        )
                    renderings[i] = buffers["render"]
                    if "depth" in buffers and buffers["depth"] is not None:
                        depth[i] = buffers["depth"]
                    if "alpha" in buffers and buffers["alpha"] is not None:
                        alpha[i] = buffers["alpha"]
                else:
                    for j in range(v):
                        # renderings[i, j] = render_opencv_cam(
                        #     pc, height, width, C2W[i, j], fxfycxcy[i, j]
                        # )["render"]
                        buffers = render_opencv_cam(
                            pc, height, width, C2W[i, j], fxfycxcy[i, j]
                        )
                        renderings[i, j] = buffers["render"]
                        if "depth" in buffers and buffers["depth"] is not None: 
                            depth[i, j] = buffers["depth"]
                        if "alpha" in buffers and buffers["alpha"] is not None:
                            alpha[i, j] = buffers["alpha"]

        # return renderings
        return edict(render=renderings, depth=depth, alpha=alpha)


def get_point_range_func(gaussians_config):
    range_setting = gaussians_config.get(
        "range_setting", edict({"type": "object_centric_depth"})
    )

    if range_setting.type == "object_centric_depth":
        rangefunc = lambda t : (2.0 * torch.sigmoid(t) - 1.0) * 1.5 + 2.7
        return rangefunc
    elif range_setting.type == "linear_depth":
        near = range_setting.get("near", 0.0)
        far = range_setting.get("far", 500.0)
        rangefunc = lambda t : torch.sigmoid(t) * (far - near) + near
        return rangefunc
    elif range_setting.type == "log_depth":
        near = range_setting.get("near", -6.2)
        far = range_setting.get("far", 6.2)
        # rangefunc = lambda t : torch.exp(torch.sigmoid(t) * (far - near) + near)
        rangefunc = lambda t : torch.exp(torch.sigmoid(t) * (far - near) + near)
        return rangefunc
    elif range_setting.type == "disparity":
        near = range_setting.get("near", 0.1)
        far = range_setting.get("far", 500.0)
        rangefunc = lambda t : 1.0 / (torch.sigmoid(t) * (1.0 / near - 1.0 / far) + 1.0 / far) 
        return rangefunc
    else:
        raise NotImplementedError

    return lambda t : t
