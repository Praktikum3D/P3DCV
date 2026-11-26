
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict

from .gaussians_renderer import (
    GaussianModel,
    render_opencv_cam_gsplat,
    deferred_gaussian_render,
)


def camera_theta(rho, camera_model, alpha=None):
    """Angle to the optical axis for the image radius rho divided by f."""
    if camera_model == "ucm":  # unified camera model, alpha = 0 is pinhole
        mz = (1 - alpha**2 * rho**2) / (alpha * torch.sqrt((1 - (2 * alpha - 1) * rho**2).clamp_min(0)) + 1 - alpha)
        return torch.atan2(rho, mz)
    return rho if camera_model == "fisheye" else torch.atan(rho)


def pixel_angles(fxfycxcy, h, w, camera_model):
    """Offsets dx, dy from the principal point, their radius, the angle to the optical axis and the pixels
    the camera can render, all [n, h, w]. fxfycxcy [n, 4] in pixels, for UCM [n, 5] with alpha last.

    gsplat drops everything behind the camera plane (more than 90 degrees off axis), and UCM is undefined
    outside its own image circle.
    """
    y, x = torch.meshgrid(torch.arange(h, device=fxfycxcy.device), torch.arange(w, device=fxfycxcy.device), indexing="ij")
    dx, dy = x + 0.5 - fxfycxcy[:, 2, None, None], y + 0.5 - fxfycxcy[:, 3, None, None]
    radius = torch.sqrt(dx**2 + dy**2)
    rho = radius / fxfycxcy[:, 0, None, None]
    alpha = fxfycxcy[:, 4, None, None] if camera_model == "ucm" else None
    theta = camera_theta(rho, camera_model, alpha)
    valid = theta <= math.pi / 2
    if camera_model == "ucm":
        # The warp interpolates, so stay 1.5 px inside the 90 degree border where gsplat leaves background.
        valid = (theta <= math.pi / 2 - 1.5 / fxfycxcy[:, 0, None, None]) & (1 - (2 * alpha - 1) * rho**2 >= 0)
    return dx, dy, radius, theta, valid


def render_mask(fxfycxcy, h, w, camera_model, image_circle=False):
    """Pixels the camera model can render, [n, 1, h, w]. With image_circle only the circle spanning the
    image width counts (fisheye data such as TartanAir lcam_fish).
    """
    _, _, radius, _, valid = pixel_angles(fxfycxcy, h, w, camera_model)
    if image_circle:
        valid = valid & (radius <= w / 2)
    return valid[:, None].float()


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
                    if camera_model == "ucm":
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
