"""Tests 7, 8 and 9: KB unprojection and the KB Plücker map (CPU, float64).

7. cam_info_to_plucker without radial_coeffs returns exactly what the original
   implementation returns; with k = 0 it is the equidistant model.
8. kb_unproject and the renderer's KB forward (_kb_proj, gsplat reference)
   invert each other.
9. gradcheck of kb_unproject and of the full cam_info_to_plucker; invalid pixels
   are finite and have zero gradient with respect to the intrinsics and k.
"""

import math

import pytest
import torch
from easydict import EasyDict as edict
from einops import rearrange

from conftest import (
    KB_MILD,
    KB_STRONG_K,
    TARTANAIR_FISH,
    import_camera_utils,
    import_erayzer,
    load_torch_impl,
)

ti = load_torch_impl()
cu = import_camera_utils()
er = import_erayzer()
DT = torch.float64


def original_cam_info_to_plucker(c2w, fxfycxcy, target_imgs_info, normalized=True, return_moment=True):
    """Verbatim copy of cam_info_to_plucker before the KB change (commit 77a29b8)."""
    if len(c2w.shape) == 3:
        b = c2w.shape[0]
    elif len(c2w.shape) == 4:
        c2w, fxfycxcy = rearrange(c2w.clone(), "b v n d -> (b v) n d"), \
            rearrange(fxfycxcy.clone(), "b v d -> (b v) d")
        b = c2w.shape[0]

    device = c2w.device
    h, w = target_imgs_info.height, target_imgs_info.width

    fxfycxcy = fxfycxcy.clone()
    if normalized:
        fxfycxcy[:, 0] *= w
        fxfycxcy[:, 1] *= h
        fxfycxcy[:, 2] *= w
        fxfycxcy[:, 3] *= h

    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    y, x = y.to(c2w), x.to(c2w)
    x = x[None, :, :].expand(b, -1, -1).reshape(b, -1)
    y = y[None, :, :].expand(b, -1, -1).reshape(b, -1)
    x = (x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
    y = (y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
    z = torch.ones_like(x)
    ray_d = torch.stack([x, y, z], dim=2)  # [b, h*w, 3]
    ray_d = torch.bmm(ray_d, c2w[:, :3, :3].transpose(1, 2))  # [b, h*w, 3]
    ray_d = ray_d / torch.norm(ray_d, dim=2, keepdim=True)  # [b, h*w, 3]
    ray_o = c2w[:, :3, 3][:, None, :].expand_as(ray_d)  # [b, h*w, 3]

    ray_o = ray_o.reshape(b, h, w, 3).permute(0, 3, 1, 2)   # [b,3,h,w]
    ray_d = ray_d.reshape(b, h, w, 3).permute(0, 3, 1, 2)

    if return_moment:
        plucker = torch.cat([torch.cross(ray_o, ray_d, dim=1), ray_d], dim=1)
    else:
        plucker = torch.cat([ray_o, ray_d], dim=1)
    return plucker     # [b,c=6,h,w]


def random_c2w(*batch, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(*batch, 4, generator=g, dtype=dtype)
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    R = torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*batch, 3, 3)
    c2w = torch.eye(4, dtype=dtype).repeat(*batch, 1, 1)
    c2w[..., :3, :3] = R
    c2w[..., :3, 3] = torch.randn(*batch, 3, generator=g, dtype=dtype)
    return c2w


# ---------------------------------------------------------------------------
# 7. Pinhole path unchanged, k = 0 is equidistant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("normalized", [True, False])
@pytest.mark.parametrize("return_moment", [True, False])
@pytest.mark.parametrize("batched_views", [False, True])
def test_pinhole_unchanged(normalized, return_moment, batched_views):
    info = edict(height=24, width=32)
    if batched_views:
        c2w = random_c2w(2, 3)
        fxfycxcy = torch.tensor([0.9, 1.2, 0.5, 0.48]).repeat(2, 3, 1)
    else:
        c2w = random_c2w(4)
        fxfycxcy = torch.tensor([0.9, 1.2, 0.5, 0.48]).repeat(4, 1)
    if not normalized:
        fxfycxcy = fxfycxcy * torch.tensor([32.0, 24.0, 32.0, 24.0])
    ref = original_cam_info_to_plucker(c2w, fxfycxcy, info, normalized, return_moment)
    out = er.cam_info_to_plucker(c2w, fxfycxcy, info, normalized=normalized, return_moment=return_moment)
    assert torch.equal(out, ref)
    out2, mask = er.cam_info_to_plucker(
        c2w, fxfycxcy, info, normalized=normalized, return_moment=return_moment, return_valid_mask=True
    )
    assert torch.equal(out2, ref) and bool(mask.all()) and mask.shape == (ref.shape[0], 1, 24, 32)


def test_k0_is_equidistant_not_pinhole():
    info = edict(height=40, width=50)
    c2w = random_c2w(2, dtype=DT)
    fxfycxcy = torch.tensor([[30.0, 32.0, 24.5, 19.0], [41.0, 40.0, 25.0, 20.5]], dtype=DT)
    k = torch.zeros(2, 4, dtype=DT)
    pl = er.cam_info_to_plucker(c2w, fxfycxcy, info, normalized=False, return_moment=False, radial_coeffs=k)
    d = pl[:, 3:]  # [b,3,h,w]

    y, x = torch.meshgrid(torch.arange(40, dtype=DT), torch.arange(50, dtype=DT), indexing="ij")
    mx = (x[None] + 0.5 - fxfycxcy[:, 2, None, None]) / fxfycxcy[:, 0, None, None]
    my = (y[None] + 0.5 - fxfycxcy[:, 3, None, None]) / fxfycxcy[:, 1, None, None]
    td = torch.sqrt(mx**2 + my**2)  # theta = theta_d for the equidistant model
    dc = torch.stack([torch.sin(td) * mx / td, torch.sin(td) * my / td, torch.cos(td)], dim=1)
    expected = torch.einsum("bij,bjhw->bihw", c2w[:, :3, :3], dc)
    torch.testing.assert_close(d, expected, rtol=0, atol=1e-12)
    torch.testing.assert_close(pl[:, :3], c2w[:, :3, 3, None, None].expand(-1, -1, 40, 50))

    pinhole = er.cam_info_to_plucker(c2w, fxfycxcy, info, normalized=False, return_moment=False)
    assert (pinhole[:, 3:] - d).abs().max() > 0.1


def test_kb_plucker_batched_views_and_moment():
    """[b, v, ...] inputs with [b, v, 4] or [b, 4] coefficients, moment = o x d."""
    info = edict(height=8, width=10)
    c2w = random_c2w(2, 3, dtype=DT)
    fxfycxcy = torch.tensor([0.6, 0.7, 0.5, 0.5], dtype=DT).repeat(2, 3, 1)
    k_bv = torch.tensor([0.03, -0.01, 0.002, 0.0], dtype=DT).repeat(2, 3, 1)
    a = er.cam_info_to_plucker(c2w, fxfycxcy, info, radial_coeffs=k_bv)
    b = er.cam_info_to_plucker(c2w, fxfycxcy, info, radial_coeffs=k_bv[:, 0])
    torch.testing.assert_close(a, b)
    od = er.cam_info_to_plucker(c2w, fxfycxcy, info, return_moment=False, radial_coeffs=k_bv)
    torch.testing.assert_close(a[:, :3], torch.cross(od[:, :3], od[:, 3:], dim=1))
    torch.testing.assert_close(a[:, 3:].norm(dim=1), torch.ones(6, 8, 10, dtype=DT))


# ---------------------------------------------------------------------------
# 8. kb_unproject inverts the renderer's KB forward
# ---------------------------------------------------------------------------

CAMERAS = {
    "tartanair_equidistant": TARTANAIR_FISH,
    "mild": KB_MILD,
    "strong": dict(width=640, height=640, f=180.0, c=319.5, k=KB_STRONG_K),
}


@pytest.mark.parametrize("cam", list(CAMERAS), ids=list(CAMERAS))
def test_unproject_then_project_returns_pixel(cam):
    cfg = CAMERAS[cam]
    W, H, f, c = cfg["width"], cfg["height"], cfg["f"], cfg["c"]
    k = torch.tensor(cfg["k"], dtype=DT)
    fx, fy, cx, cy = f, f * 1.01, c, c - 3.0
    y, x = torch.meshgrid(
        torch.arange(0, H, 7, dtype=DT), torch.arange(0, W, 7, dtype=DT), indexing="ij"
    )
    u, v = x.reshape(-1) + 0.5, y.reshape(-1) + 0.5
    pts = torch.stack([(u - cx) / fx, (v - cy) / fy], dim=-1)
    dirs, valid = cu.kb_unproject(pts, k)
    assert valid.sum() > 100
    if cam == "tartanair_equidistant":
        # the whole 195 degree image circle is valid, including theta > pi / 2
        r = torch.sqrt((u - c) ** 2 + (v - c + 3.0) ** 2)
        assert bool(valid[r < 495].all())
        assert bool((dirs[valid][:, 2] < 0).any())
    torch.testing.assert_close(dirs.norm(dim=-1), torch.ones_like(u), rtol=0, atol=1e-12)

    means = (dirs[valid] * 2.5)[None]  # [1, M, 3]
    covars = torch.eye(3, dtype=DT).expand(1, means.shape[1], 3, 3) * 1e-4
    Ks = torch.tensor([[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]], dtype=DT)
    m2d, _ = ti._kb_proj(means, covars, Ks, k[None], W, H)
    err = (m2d[0] - torch.stack([u, v], -1)[valid]).abs().max()
    assert err < 1e-6, float(err)


@pytest.mark.parametrize("cam", list(CAMERAS), ids=list(CAMERAS))
def test_project_then_unproject_returns_direction(cam):
    cfg = CAMERAS[cam]
    k = torch.tensor(cfg["k"], dtype=DT)
    lim = float(cu.kb_theta_max(k))
    g = torch.Generator().manual_seed(0)
    n = 2000
    theta = torch.rand(n, generator=g, dtype=DT) * lim * 0.995
    theta[:3] = torch.tensor([0.0, 1e-7, 1e-3], dtype=DT)
    phi = torch.rand(n, generator=g, dtype=DT) * 2 * math.pi
    dirs = torch.stack([torch.sin(theta) * torch.cos(phi), torch.sin(theta) * torch.sin(phi), torch.cos(theta)], -1)
    U, V, _ = ti._kb_proj_normalized(dirs[:, 0], dirs[:, 1], dirs[:, 2], k.expand(n, 4))
    back, valid = cu.kb_unproject(torch.stack([U, V], -1), k)
    assert bool(valid.all())
    torch.testing.assert_close(back, dirs, rtol=0, atol=1e-9)


# ---------------------------------------------------------------------------
# 9. Gradients of kb_unproject and cam_info_to_plucker
# ---------------------------------------------------------------------------


def test_gradcheck_kb_unproject():
    k = torch.tensor(KB_STRONG_K, dtype=DT)
    lim = float(cu.kb_theta_max(k))
    td_lim = float(cu.kb_psi(torch.tensor(lim, dtype=DT), k))
    pts = torch.tensor(
        [
            [0.0, 0.0],  # principal point
            [1e-6, -2e-6],  # Taylor branch
            [3e-4, 1e-4],
            [0.3, -0.2],
            [0.5, 0.45],
            [td_lim * 0.999 * 0.6, td_lim * 0.999 * 0.8],  # just inside theta_max
        ],
        dtype=DT,
        requires_grad=True,
    )
    kk = k.clone().requires_grad_(True)

    def fn(p, kk):
        d, valid = cu.kb_unproject(p, kk)
        assert bool(valid.all())
        return d

    torch.autograd.gradcheck(fn, (pts, kk), eps=1e-7, atol=1e-6)


def test_invalid_pixels_finite_and_zero_grad():
    k = torch.tensor(KB_STRONG_K, dtype=DT, requires_grad=True)
    pts = torch.tensor([[0.2, 0.1], [3.0, 2.0], [-5.0, 0.0]], dtype=DT, requires_grad=True)
    d, valid = cu.kb_unproject(pts, k)
    assert valid.tolist() == [True, False, False]
    assert torch.isfinite(d).all()
    torch.testing.assert_close(d[1:], torch.tensor([[0.0, 0.0, 1.0]] * 2, dtype=DT))
    (d * torch.randn_like(d)).sum().backward()
    assert torch.isfinite(pts.grad).all() and torch.isfinite(k.grad).all()
    assert bool((pts.grad[1:] == 0).all())


def test_gradcheck_cam_info_to_plucker():
    """Small image containing the principal point and pixels near and beyond theta_max."""
    info = edict(height=5, width=6)
    k = torch.tensor([KB_STRONG_K], dtype=DT)
    lim = float(cu.kb_theta_max(k[0]))
    td_lim = float(cu.kb_psi(torch.tensor(lim, dtype=DT), k[0]))
    # cx = 2.5, cy = 2.5: pixel (2, 2) is exactly at the principal point. Focal
    # length such that pixel (0, 2) is just inside theta_max; the corners are outside.
    f = 2.0 / (td_lim * 0.995)
    fxfycxcy = torch.tensor([[f, f, 2.5, 2.5]], dtype=DT, requires_grad=True)
    kk = k.clone().requires_grad_(True)
    c2w = random_c2w(1, dtype=DT).requires_grad_(True)

    pl, mask = er.cam_info_to_plucker(
        c2w, fxfycxcy, info, normalized=False, radial_coeffs=kk, return_valid_mask=True
    )
    assert bool(mask[0, 0, 2, 2]) and bool(mask[0, 0, 0, 2]) and not bool(mask[0, 0, 0, 0])
    assert torch.isfinite(pl).all()

    for return_moment in [True, False]:
        torch.autograd.gradcheck(
            lambda c, fi, kk: er.cam_info_to_plucker(
                c, fi, info, normalized=False, return_moment=return_moment, radial_coeffs=kk
            ),
            (c2w, fxfycxcy, kk),
            eps=1e-7,
            atol=1e-6,
        )

    # normalised intrinsics, batched views
    fi_n = (fxfycxcy.detach() / torch.tensor([6.0, 5.0, 6.0, 5.0], dtype=DT))[None].requires_grad_(True)
    torch.autograd.gradcheck(
        lambda c, fi, kk: er.cam_info_to_plucker(c[None], fi, info, normalized=True, radial_coeffs=kk[None]),
        (c2w, fi_n, kk),
        eps=1e-7,
        atol=1e-6,
    )


def test_float32_kb_plucker_tartanair_finite():
    """Full-size TartanAir fisheye in float32, as in training: finite values and gradients."""
    info = edict(height=1000, width=1000)
    fxfycxcy = torch.tensor([[293.8 / 1000, 293.8 / 1000, 0.4995, 0.4995]], requires_grad=True)
    k = torch.zeros(1, 4, requires_grad=True)
    pl, mask = er.cam_info_to_plucker(
        torch.eye(4)[None], fxfycxcy, info, radial_coeffs=k, return_valid_mask=True
    )
    assert torch.isfinite(pl).all()
    # image circle valid (195 degrees), far corners invalid (theta > 0.75 pi)
    assert bool(mask[0, 0, 500, 2]) and bool(mask[0, 0, 2, 500]) and not bool(mask[0, 0, 0, 0])
    pl.sum().backward()
    assert torch.isfinite(fxfycxcy.grad).all() and torch.isfinite(k.grad).all()
