
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


def camera_theta(rho, camera_model, alpha=None):
    """Angle to the optical axis for the image radius rho divided by f."""
    if camera_model == "ucm":  # unified camera model, alpha = 0 is pinhole
        mz = (1 - alpha**2 * rho**2) / (alpha * torch.sqrt((1 - (2 * alpha - 1) * rho**2).clamp_min(0)) + 1 - alpha)
        return torch.atan2(rho, mz)
    return rho if camera_model == "fisheye" else torch.atan(rho)


def camera_rays(c2w, fxfycxcy, h, w, camera_model="pinhole"):
    """Ray origins and normalized directions in world space for every pixel, both [b, h, w, 3].

    c2w [b, 4, 4], fxfycxcy [b, 4] in pixels, for UCM [b, 5] with alpha last. This is the ray math of
    cam_info_to_plucker, kept in one place so the plucker embeddings and the 3DGEER renderer agree.
    """
    b = c2w.shape[0]
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    y, x = y.to(c2w), x.to(c2w)
    x = x[None, :, :].expand(b, -1, -1).reshape(b, -1)
    y = y[None, :, :].expand(b, -1, -1).reshape(b, -1)
    x = (x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
    y = (y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
    z = torch.ones_like(x)
    if camera_model != "pinhole":  # fisheye or UCM, direction from the angle to the optical axis
        rho = torch.sqrt(x * x + y * y + 1e-12)
        theta = camera_theta(rho, camera_model, fxfycxcy[:, 4:5] if camera_model == "ucm" else None)
        x, y, z = x * torch.sin(theta) / rho, y * torch.sin(theta) / rho, torch.cos(theta)
    ray_d = torch.stack([x, y, z], dim=2)  # [b, h*w, 3]
    ray_d = torch.bmm(ray_d, c2w[:, :3, :3].transpose(1, 2))  # [b, h*w, 3]
    ray_d = ray_d / torch.norm(ray_d, dim=2, keepdim=True)  # [b, h*w, 3]
    ray_o = c2w[:, :3, 3][:, None, :].expand_as(ray_d)  # [b, h*w, 3]
    return ray_o.reshape(b, h, w, 3), ray_d.reshape(b, h, w, 3)


def pixel_angles(fxfycxcy, h, w, camera_model, exact=False):
    """Offsets dx, dy from the principal point, their radius, the angle to the optical axis and the pixels
    the camera can render, all [n, h, w]. fxfycxcy [n, 4] in pixels, for UCM [n, 5] with alpha last.

    gsplat and 3DGEER in its "pbf" tile mode drop everything behind the camera plane (more than 90 degrees off
    axis), 3DGEER culls the Gaussian centres there and only splat tails reach further. UCM is undefined outside
    its own image circle. With exact the angle is not capped at all, which the Kannala Brandt fit needs to know
    its own range and which 3DGEER in its "projection" tile mode really renders, out to the image border.
    """
    y, x = torch.meshgrid(torch.arange(h, device=fxfycxcy.device), torch.arange(w, device=fxfycxcy.device), indexing="ij")
    dx, dy = x + 0.5 - fxfycxcy[:, 2, None, None], y + 0.5 - fxfycxcy[:, 3, None, None]
    radius = torch.sqrt(dx**2 + dy**2)
    rho = radius / fxfycxcy[:, 0, None, None]
    alpha = fxfycxcy[:, 4, None, None] if camera_model == "ucm" else None
    theta = camera_theta(rho, camera_model, alpha)
    valid = theta <= (math.pi if exact else math.pi / 2)
    if camera_model == "ucm":
        # The gsplat warp interpolates, so stay 1.5 px inside the 90 degree border where gsplat leaves background.
        margin = 0.0 if exact else 1.5 / fxfycxcy[:, 0, None, None]
        valid = (theta <= (math.pi if exact else math.pi / 2) - margin) & (1 - (2 * alpha - 1) * rho**2 >= 0)
    return dx, dy, radius, theta, valid


def render_mask(fxfycxcy, h, w, camera_model, image_circle=False, exact=False):
    """Pixels the camera model can render, [n, 1, h, w]. With image_circle only the circle spanning the
    image width counts (fisheye data such as TartanAir lcam_fish).
    """
    _, _, radius, _, valid = pixel_angles(fxfycxcy, h, w, camera_model, exact)
    if image_circle:
        valid = valid & (radius <= w / 2)
    return valid[:, None].float()


def loss_render_mask(config, fxfycxcy, h, w):
    """The pixels the training loss and the evaluation score, [n, 1, h, w]. fxfycxcy [n, 4], UCM [n, 5].

    Both of them call this, so they always mask the same pixels. The mask normally stops at 90 degrees off
    axis, because neither gsplat nor 3DGEER in its "pbf" tile mode draws anything further out and those
    pixels would be background without a gradient. With the geer renderer in its "projection" tile mode the
    renderer fills the whole image circle, so the mask is the camera model's own range instead.
    """
    camera_model = config.model.get("camera_model", "pinhole")
    image_circle = "fish" in str(config.dataset.get("camera_name", ""))
    beyond_90 = (config.model.get("renderer", "gsplat") == "geer"
                 and config.model.get("geer_tile_mode", "pbf") == "projection")
    return render_mask(fxfycxcy, h, w, camera_model, image_circle, exact=beyond_90)


def ucm_radial_coeffs(fxfycxcy, h, w):
    """Kannala Brandt coefficients [v, 4] that reproduce UCM (alpha in column 4) up to the largest valid angle,
    by least squares on theta_d / theta - 1 = k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8."""
    _, _, _, theta, valid = pixel_angles(fxfycxcy, h, w, "ucm", exact=True)
    theta_max = torch.where(valid, theta, torch.zeros_like(theta)).flatten(1).amax(1)  # [v]
    th = theta_max[:, None] * torch.linspace(0.02, 1.0, 64, device=theta.device)[None]  # [v, 64]
    a = fxfycxcy[:, 4:5]
    target = torch.sin(th) / (a + (1 - a) * torch.cos(th)) / th - 1
    A = torch.stack([th**2, th**4, th**6, th**8], dim=-1)  # [v, 64, 4]
    return torch.linalg.lstsq(A.double(), target.double()[..., None]).solution[..., 0].float()


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
        fxfycxcy: [b, v, 4]

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
                        if camera_model == "ucm":  # 3DGEER has no UCM, a fitted Kannala Brandt fisheye replaces it
                            radial = ucm_radial_coeffs(fxfycxcy[i].detach(), height, width)
                        elif camera_model == "fisheye":
                            radial = torch.zeros(fxfycxcy.shape[1], 4, device=xyz.device)
                        buffers = render_opencv_cam_geer(
                            pc, height, width, C2W[i], fxfycxcy[i], self.sh_degree, near_plane=near_plane,
                            camera_model="pinhole" if camera_model == "pinhole" else "fisheye", radial_coeffs=radial,
                            ray_camera_model=camera_model,
                            geer_tile_mode=self.config.model.get("geer_tile_mode", "pbf"),
                        )
                    elif camera_model == "ucm":
                        # gsplat has no UCM. Render an equidistant fisheye image whose focal length keeps every
                        # valid pixel inside it, then look up each UCM pixel by its angle to the optical axis.
                        dx, dy, radius, theta, valid = pixel_angles(fxfycxcy[i], height, width, "ucm")
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
