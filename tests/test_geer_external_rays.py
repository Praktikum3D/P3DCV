"""3DGEER rendering from rays handed in from outside, checked against the old detour.

Run from the repository root with the gsplat-geer build of gsplat installed:
    pytest tests/test_geer_external_rays.py
"""

import math
import os
import sys

import pytest
import torch
from easydict import EasyDict as edict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gsplat import rasterization

from erayzer_core.model.gaussians_renderer import GaussianModel, eval_sh, render_opencv_cam_geer
from erayzer_core.model.utils_gaussian import (
    Renderer,
    camera_rays,
    camera_theta,
    pixel_angles,
    render_mask,
    ucm_radial_coeffs,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="3DGEER needs a GPU")

H, W = 64, 64


def _rotmat_to_quat(R):
    """Rotation matrix [3, 3] to a unit quaternion (w, x, y, z), from the old detour code."""
    t = R[0, 0] + R[1, 1] + R[2, 2]
    cands = torch.stack([
        torch.stack([1 + t, R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]),
        torch.stack([R[2, 1] - R[1, 2], 1 + R[0, 0] - R[1, 1] - R[2, 2], R[0, 1] + R[1, 0], R[0, 2] + R[2, 0]]),
        torch.stack([R[0, 2] - R[2, 0], R[0, 1] + R[1, 0], 1 - R[0, 0] + R[1, 1] - R[2, 2], R[1, 2] + R[2, 1]]),
        torch.stack([R[1, 0] - R[0, 1], R[0, 2] + R[2, 0], R[1, 2] + R[2, 1], 1 - R[0, 0] - R[1, 1] + R[2, 2]]),
    ])
    q = cands[torch.argmax(torch.stack([t, R[0, 0], R[1, 1], R[2, 2]]))]
    return q / q.norm()


def _quat_multiply(a, b):
    """Hamilton product of a [4] and b [N, 4], both (w, x, y, z), from the old detour code."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], dim=-1)


def render_detour(pc, height, width, C2W, fxfycxcy, sh_degree, near_plane=0.01, camera_model="fisheye",
                  radial_coeffs=None, bg_color=(1.0, 1.0, 1.0)):
    """The old render_opencv_cam_geer: Gaussians rotated into the camera frame, rendered from the origin,
    rays built by the kernel from the camera model. Kept here so the new path can be compared to it."""
    means, quats, scales = pc.get_xyz, pc.get_rotation, pc.get_scaling
    opacity, shs = pc.get_opacity.squeeze(-1), pc.get_features.transpose(1, 2)
    W2C = C2W.inverse()
    eye = torch.eye(4, device=C2W.device)[None]
    bg = torch.tensor([bg_color], dtype=torch.float32, device=C2W.device)
    renders = []
    for j in range(C2W.shape[0]):
        R, t = W2C[j, :3, :3], W2C[j, :3, 3]
        dirs = torch.nn.functional.normalize(means - C2W[j, :3, 3], dim=-1)
        colors = (eval_sh(sh_degree, shs, dirs) + 0.5).clamp_min(0.0)
        K = torch.zeros(1, 3, 3, device=C2W.device)
        K[0, 0, 0], K[0, 1, 1], K[0, 0, 2], K[0, 1, 2], K[0, 2, 2] = fxfycxcy[j, 0], fxfycxcy[j, 1], fxfycxcy[j, 2], fxfycxcy[j, 3], 1.0
        out, _, _ = rasterization(
            means @ R.T + t, _quat_multiply(_rotmat_to_quat(R), quats), scales, opacity, colors[None], eye, K,
            width, height, near_plane=near_plane, packed=False, backgrounds=bg, camera_model=camera_model,
            radial_coeffs=None if radial_coeffs is None else radial_coeffs[j:j + 1], with_geer=True, with_eval3d=True,
        )
        renders.append(out[0])
    return {"render": torch.stack(renders).permute(0, 3, 1, 2), "depth": None}


def make_scene(device, sh_degree=1, n=1500, seed=0):
    """A shell of Gaussians around the origin, so that even a 180 degree view is covered. The Gaussians
    are big enough that 3DGEER keeps their gradients, and every tensor is a leaf."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    direction = torch.nn.functional.normalize(torch.randn(n, 3, generator=g), dim=-1)
    xyz = (direction * (2.0 + torch.rand(n, 1, generator=g) * 2.5)).to(device)
    features = ((torch.rand(n, (sh_degree + 1) ** 2, 3, generator=g) - 0.5) * 0.8).to(device)
    features[:, 0] += 0.6
    scaling = torch.log(torch.rand(n, 3, generator=g).to(device) * 0.07 + 0.06)
    rotation = torch.randn(n, 4, generator=g).to(device)
    opacity = torch.rand(n, 1, generator=g).to(device) * 2.0 + 1.0
    leaves = [t.requires_grad_() for t in (xyz, features, scaling, rotation, opacity)]
    return GaussianModel(sh_degree).set_data(*leaves), leaves


def make_cameras(device, camera_model, v=3):
    """v poses near the origin looking in different directions, and a matching fxfycxcy [v, 4 or 5]."""
    c2ws = []
    for i in range(v):
        azim, elev = 1.1 * i + 0.3, 0.3 * i - 0.3
        forward = torch.tensor([math.cos(elev) * math.cos(azim), math.sin(elev), math.cos(elev) * math.sin(azim)])
        right = torch.linalg.cross(forward, torch.tensor([0.0, 1.0, 0.0]))
        right = right / right.norm()
        down = torch.linalg.cross(forward, right)
        c2w = torch.eye(4)
        c2w[:3, 0], c2w[:3, 1], c2w[:3, 2] = right, down, forward
        c2w[:3, 3] = torch.tensor([0.1 * i, 0.05 * i, -0.1 * i])
        c2ws.append(c2w)
    C2W = torch.stack(c2ws).to(device)
    f = {"pinhole": 45.0, "fisheye": 18.5, "ucm": 24.0}[camera_model]
    fxfycxcy = torch.tensor([[f, f, W / 2, H / 2]], device=device).repeat(v, 1)
    if camera_model == "ucm":
        fxfycxcy = torch.cat([fxfycxcy, torch.full((v, 1), 0.6, device=device)], dim=1)
    return C2W, fxfycxcy


def geer_args(fxfycxcy, camera_model, device):
    """The tile assignment arguments render_opencv_cam_geer gets from the Renderer."""
    if camera_model == "ucm":
        return "fisheye", ucm_radial_coeffs(fxfycxcy.detach(), H, W)
    if camera_model == "fisheye":
        return "fisheye", torch.zeros(fxfycxcy.shape[0], 4, device=device)
    return "pinhole", None


@pytest.mark.parametrize("camera_model", ["pinhole", "fisheye", "ucm"])
def test_external_rays_match_the_detour(camera_model):
    """The external rays reproduce the image the kernel built its own rays for."""
    device = "cuda"
    sh_degree = 1
    pc, _ = make_scene(device, sh_degree)
    C2W, fxfycxcy = make_cameras(device, camera_model)
    tile_model, radial = geer_args(fxfycxcy, camera_model, device)

    old = render_detour(pc, H, W, C2W, fxfycxcy, sh_degree, camera_model=tile_model, radial_coeffs=radial)["render"]
    new = render_opencv_cam_geer(pc, H, W, C2W, fxfycxcy, sh_degree, camera_model=tile_model,
                                 radial_coeffs=radial, ray_camera_model=camera_model)["render"]

    assert torch.isfinite(new).all()
    assert new.shape == old.shape
    assert (old.amin(1) < 0.99).float().mean().item() > 0.5, "the views have to show something"
    diff = (new - old).abs()
    # UCM is the one case where the two paths really use different rays: the kernel only knows the
    # Kannala Brandt fit of the camera, the external rays are the exact UCM ones. The fit is good to a
    # small fraction of a pixel, so the images stay close but not as close as for the other models.
    bound = 5e-4 if camera_model == "ucm" else 5e-5
    assert diff.max().item() < bound, f"{camera_model}: max abs diff {diff.max().item()}"
    assert diff.mean().item() < bound / 25, f"{camera_model}: mean abs diff {diff.mean().item()}"


def test_the_rays_really_drive_the_image():
    """Guard against the rasterizer quietly ignoring the rays, which would make the test above vacuous.

    Same Gaussians, same poses, same tile assignment, only the rays come from the pinhole model instead
    of the fisheye one. If the rays were ignored the two images would be identical.
    """
    device = "cuda"
    pc, _ = make_scene(device, sh_degree=1)
    C2W, fxfycxcy = make_cameras(device, "fisheye", v=1)
    tile_model, radial = geer_args(fxfycxcy, "fisheye", device)
    kwargs = dict(camera_model=tile_model, radial_coeffs=radial)
    right = render_opencv_cam_geer(pc, H, W, C2W, fxfycxcy, 1, ray_camera_model="fisheye", **kwargs)["render"]
    wrong = render_opencv_cam_geer(pc, H, W, C2W, fxfycxcy, 1, ray_camera_model="pinhole", **kwargs)["render"]
    assert (right - wrong).abs().max().item() > 0.1
    assert (right - wrong).abs().mean().item() > 1e-3


@pytest.mark.parametrize("camera_model", ["pinhole", "fisheye", "ucm"])
def test_gradients_reach_pose_intrinsics_and_gaussians(camera_model):
    """Degree zero colors, so the pose can only get its gradient through the rays, not through the SH."""
    device = "cuda"
    pc, leaves = make_scene(device, sh_degree=0)
    C2W, fxfycxcy = make_cameras(device, camera_model)
    C2W, fxfycxcy = C2W.requires_grad_(), fxfycxcy.requires_grad_()
    tile_model, radial = geer_args(fxfycxcy, camera_model, device)

    out = render_opencv_cam_geer(pc, H, W, C2W, fxfycxcy, 0, camera_model=tile_model,
                                 radial_coeffs=radial, ray_camera_model=camera_model)["render"]
    weight = torch.linspace(-1.0, 1.0, H * W, device=device).reshape(1, 1, H, W)
    (out * weight).sum().backward()

    for name, grad in [("C2W", C2W.grad), ("fxfycxcy", fxfycxcy.grad)] + list(zip(
            ["xyz", "features", "scaling", "rotation", "opacity"], [t.grad for t in leaves])):
        assert grad is not None, f"{camera_model}: no gradient for {name}"
        assert torch.isfinite(grad).all(), f"{camera_model}: non finite gradient for {name}"
        assert grad.abs().max().item() > 0.0, f"{camera_model}: zero gradient for {name}"
    # rotation and translation of every view, not just one
    assert (C2W.grad[:, :3, :3].abs().amax(dim=(1, 2)) > 0).all()
    assert (C2W.grad[:, :3, 3].abs().amax(dim=1) > 0).all()
    assert (fxfycxcy.grad[:, 0].abs() > 0).all(), "no focal length gradient"
    if camera_model == "ucm":
        assert (fxfycxcy.grad[:, 4].abs() > 0).all(), "no UCM alpha gradient"


@pytest.mark.parametrize("camera_model", ["pinhole", "fisheye", "ucm"])
def test_focal_length_gradient_matches_central_differences(camera_model):
    """The focal length gradient is the one the pinhole experiment was missing, so check its value.

    3DGEER drops the gradient of Gaussians whose footprint is below a pixel or which are saturated, so
    the analytic value sits a few percent off the central difference. The tolerances allow for that.
    """
    device = "cuda"
    pc, _ = make_scene(device, sh_degree=0)
    C2W, fxfycxcy = make_cameras(device, camera_model, v=1)
    tile_model, radial = geer_args(fxfycxcy, camera_model, device)

    def loss_of(f):
        k = fxfycxcy.clone()
        k[:, 0], k[:, 1] = f, f
        return render_opencv_cam_geer(pc, H, W, C2W, k, 0, camera_model=tile_model,
                                      radial_coeffs=radial, ray_camera_model=camera_model)["render"].sum()

    f0 = fxfycxcy[0, 0].clone().requires_grad_()
    loss_of(f0).backward()
    eps = 0.05
    with torch.no_grad():
        numeric = (loss_of(fxfycxcy[0, 0] + eps) - loss_of(fxfycxcy[0, 0] - eps)) / (2 * eps)
    assert numeric.abs().item() > 1.0
    rel = (f0.grad - numeric).abs().item() / numeric.abs().item()
    assert rel < 0.1, f"{camera_model}: analytic {f0.grad.item()} vs numeric {numeric.item()}"


@pytest.mark.parametrize("camera_model", ["pinhole", "fisheye", "ucm"])
def test_camera_translation_gradient_matches_central_differences(camera_model):
    """The pose gradient now travels through the ray origins, with the view matrix detached."""
    device = "cuda"
    pc, _ = make_scene(device, sh_degree=0)
    C2W, fxfycxcy = make_cameras(device, camera_model, v=1)
    tile_model, radial = geer_args(fxfycxcy, camera_model, device)

    def loss_of(c2w):
        return render_opencv_cam_geer(pc, H, W, c2w, fxfycxcy, 0, camera_model=tile_model,
                                      radial_coeffs=radial, ray_camera_model=camera_model)["render"].sum()

    pose = C2W.clone().requires_grad_()
    loss_of(pose).backward()
    analytic = pose.grad[0, :3, 3]
    numeric = torch.zeros(3, device=device)
    eps = 2e-3
    with torch.no_grad():
        for k in range(3):
            plus, minus = C2W.clone(), C2W.clone()
            plus[0, k, 3] += eps
            minus[0, k, 3] -= eps
            numeric[k] = (loss_of(plus) - loss_of(minus)) / (2 * eps)
    cosine = torch.nn.functional.cosine_similarity(analytic, numeric, dim=0).item()
    rel = (analytic - numeric).norm().item() / numeric.norm().item()
    assert numeric.norm().item() > 1.0
    assert cosine > 0.99, f"{camera_model}: analytic {analytic.tolist()} vs numeric {numeric.tolist()}"
    assert rel < 0.2, f"{camera_model}: relative error {rel}"


def test_ucm_beyond_the_valid_region_stays_finite():
    """A focal length short enough that the image corners leave the UCM image circle.

    camera_theta clamps 1 - (2 alpha - 1) rho^2 at zero, so those pixels get a finite direction instead
    of a NaN, just one that points backwards. 3DGEER leaves them at the background colour (its kernels
    skip a pixel whose ray is not finite or has length zero, see world_ray_is_valid) and render_mask
    keeps them out of the loss, exactly as before. Only alpha exactly 1 together with rho exactly 1
    would divide zero by zero, and alpha comes from a sigmoid computed in float32, so it stays below 1.
    """
    device = "cuda"
    pc, _ = make_scene(device, sh_degree=0)
    C2W, fxfycxcy = make_cameras(device, "ucm", v=2)
    for alpha in [0.6, 0.9, 0.999]:
        with torch.no_grad():  # corners reach rho 2.83, for alpha 0.6 the circle already ends at 2.24
            fxfycxcy[:, 0], fxfycxcy[:, 1], fxfycxcy[:, 4] = 16.0, 16.0, alpha
        _, _, _, theta, valid = pixel_angles(fxfycxcy, H, W, "ucm", exact=True)
        assert torch.isfinite(theta).all()
        assert not valid.all(), f"alpha {alpha}: the test needs pixels outside the valid region"

        rays_o, rays_d = camera_rays(C2W, fxfycxcy, H, W, "ucm")
        assert torch.isfinite(rays_o).all() and torch.isfinite(rays_d).all()

        tile_model, radial = geer_args(fxfycxcy, "ucm", device)
        assert torch.isfinite(radial).all()
        out = render_opencv_cam_geer(pc, H, W, C2W, fxfycxcy, 0, camera_model=tile_model,
                                     radial_coeffs=radial, ray_camera_model="ucm")["render"]
        assert torch.isfinite(out).all()
        mask = render_mask(fxfycxcy, H, W, "ucm")
        assert mask.sum() > 0 and mask.min() == 0.0


@pytest.mark.parametrize("camera_model", ["pinhole", "fisheye", "ucm"])
def test_camera_rays_match_the_old_plucker_math(camera_model):
    """camera_rays was lifted out of cam_info_to_plucker, so it has to give the very same rays.

    cam_info_to_plucker itself is not imported here because erayzer.py pulls in xformers; the body below
    is the code that stood in it before the refactor.
    """
    device = "cuda"
    C2W, fxfycxcy = make_cameras(device, camera_model, v=3)
    b, h, w = C2W.shape[0], H, W

    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    y, x = y.to(C2W), x.to(C2W)
    x = x[None, :, :].expand(b, -1, -1).reshape(b, -1)
    y = y[None, :, :].expand(b, -1, -1).reshape(b, -1)
    x = (x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
    y = (y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
    z = torch.ones_like(x)
    if camera_model != "pinhole":
        rho = torch.sqrt(x * x + y * y + 1e-12)
        theta = camera_theta(rho, camera_model, fxfycxcy[:, 4:5] if camera_model == "ucm" else None)
        x, y, z = x * torch.sin(theta) / rho, y * torch.sin(theta) / rho, torch.cos(theta)
    ray_d = torch.stack([x, y, z], dim=2)
    ray_d = torch.bmm(ray_d, C2W[:, :3, :3].transpose(1, 2))
    ray_d = ray_d / torch.norm(ray_d, dim=2, keepdim=True)
    ray_o = C2W[:, :3, 3][:, None, :].expand_as(ray_d)

    new_o, new_d = camera_rays(C2W, fxfycxcy, h, w, camera_model)
    assert torch.equal(new_o, ray_o.reshape(b, h, w, 3))
    assert torch.equal(new_d, ray_d.reshape(b, h, w, 3))


@pytest.mark.parametrize("camera_model", ["pinhole", "fisheye", "ucm"])
def test_renderer_runs_under_autocast(camera_model):
    """The real call site, with the mixed precision the cluster trains in around it.

    Renderer.forward casts its inputs to float32 and switches autocast off, and the renderer switches it
    off again around the rasterization, so 3DGEER never sees a half precision tensor. The local GPU has
    no bf16, float16 autocast exercises the same path.
    """
    device = "cuda"
    pc, leaves = make_scene(device, sh_degree=1)
    C2W, fxfycxcy = make_cameras(device, camera_model, v=2)
    config = edict({"model": {"gaussians": {"sh_degree": 1}, "renderer": "geer", "camera_model": camera_model,
                              "near_plane": 0.01, "use_gsplat": True, "use_deferred_rendering": False}})
    renderer = Renderer(config)
    xyz, features, scaling, rotation, opacity = (t[None] for t in leaves)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        out = renderer(xyz, features, scaling, rotation, opacity, H, W, C2W[None], fxfycxcy[None])
    assert out.render.shape == (1, 2, 3, H, W)
    assert out.render.dtype == torch.float32
    assert torch.isfinite(out.render).all()
    out.render.sum().backward()
    assert torch.isfinite(leaves[0].grad).all() and leaves[0].grad.abs().max() > 0


def test_centre_pixel_ray_is_the_optical_axis():
    """cam_info_to_plucker regularizes rho with 1e-12 instead of branching, so check the principal point."""
    device = "cuda"
    for camera_model in ["pinhole", "fisheye", "ucm"]:
        C2W, fxfycxcy = make_cameras(device, camera_model, v=1)
        with torch.no_grad():  # put the principal point on the centre of a pixel, so dx = dy = 0 there
            fxfycxcy[:, 2], fxfycxcy[:, 3] = W // 2 + 0.5, H // 2 + 0.5
        _, rays_d = camera_rays(C2W, fxfycxcy, H, W, camera_model)
        assert torch.isfinite(rays_d).all()
        centre = rays_d[0, H // 2, W // 2]
        assert torch.allclose(centre, C2W[0, :3, 2], atol=1e-5), f"{camera_model}: {centre.tolist()}"
