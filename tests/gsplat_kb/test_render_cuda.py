"""Tests 3, 6 and 10: end-to-end rendering through rasterization() (CUDA).

3. dL/d(fx, fy, cx, cy, k1..k4) from rasterization() against central finite
   differences of the full render, L = sum(render * W).
6. Recovery: render with known KB parameters, perturb them, optimise only the
   camera parameters on the photometric loss and check that they come back.
10. The renderer and the Plücker map agree: tiny Gaussians placed along the
   KB Plücker rays of chosen pixels render onto those pixels.
"""

import math
from types import SimpleNamespace

import pytest
import torch
from easydict import EasyDict as edict

from conftest import KB_MILD, TARTANAIR_FISH, import_erayzer, report, requires_cuda

pytestmark = requires_cuda
DEV = "cuda"


def make_scene(cam, N, seed=0, theta_max=None, sigma_px=4.0, rho_range=(2.0, 6.0)):
    """Random Gaussians seen by a camera at the origin looking along +z."""
    g = torch.Generator().manual_seed(seed)
    if theta_max is None:
        theta_max = 0.95 * min(cam["width"], cam["height"]) / 2 / cam["f"]
    theta = torch.sqrt(torch.rand(N, generator=g)) * theta_max
    phi = torch.rand(N, generator=g) * 2 * math.pi
    rho = rho_range[0] + (rho_range[1] - rho_range[0]) * torch.rand(N, generator=g)
    means = torch.stack(
        [rho * torch.sin(theta) * torch.cos(phi), rho * torch.sin(theta) * torch.sin(phi), rho * torch.cos(theta)], -1
    )
    quats = torch.randn(N, 4, generator=g)
    scales = (sigma_px * rho / cam["f"])[:, None] * torch.exp(torch.randn(N, 3, generator=g) * 0.3)
    opacities = 0.5 + 0.4 * torch.rand(N, generator=g)
    colors = torch.rand(N, 3, generator=g)
    return {n: t.to(DEV) for n, t in dict(means=means, quats=quats, scales=scales, opacities=opacities,
                                           colors=colors).items()}


def render(scene, cam, intr, k, packed=False, rasterize_mode="classic"):
    """intr: [4] = (fx, fy, cx, cy), k: [4]."""
    from gsplat import rasterization

    O = torch.zeros((), device=DEV, dtype=intr.dtype)
    I = torch.ones((), device=DEV, dtype=intr.dtype)
    Ks = torch.stack([torch.stack([intr[0], O, intr[2]]), torch.stack([O, intr[1], intr[3]]),
                      torch.stack([O, O, I])])[None]
    img, _, _ = rasterization(
        scene["means"], scene["quats"], scene["scales"], scene["opacities"], scene["colors"],
        torch.eye(4, device=DEV)[None], Ks, cam["width"], cam["height"],
        camera_model="kb", radial_coeffs=k[None], packed=packed, rasterize_mode=rasterize_mode,
        near_plane=0.1,
    )
    return img[0]


FD_CASES = [
    ("kb_mild_dense_classic", KB_MILD, False, "classic"),
    ("kb_mild_packed_antialiased", KB_MILD, True, "antialiased"),
    ("tartanair_equidistant_packed_classic", dict(TARTANAIR_FISH, theta_max=1.68), True, "classic"),
]


@pytest.mark.parametrize("name,cam,packed,mode", FD_CASES, ids=[c[0] for c in FD_CASES])
def test_finite_differences(name, cam, packed, mode):
    scene = make_scene(cam, N=4000, theta_max=cam.get("theta_max"))
    g = torch.Generator().manual_seed(1)
    Wt = torch.randn(cam["height"], cam["width"], 3, generator=g, dtype=torch.float64).to(DEV)
    intr0 = torch.tensor([cam["f"], cam["f"] * 1.01, cam["c"], cam["c"] - 2.0], device=DEV)
    k0 = torch.tensor(cam["k"], device=DEV)

    def loss(intr, k):
        return (render(scene, cam, intr, k, packed, mode).double() * Wt).sum()

    intr = intr0.clone().requires_grad_(True)
    k = k0.clone().requires_grad_(True)
    loss(intr, k).backward()
    analytic = torch.cat([intr.grad, k.grad]).double().cpu()

    steps = [1e-3 * cam["f"], 1e-3 * cam["f"], 1e-2, 1e-2, 1e-4, 1e-4, 1e-4, 1e-4]
    fd = []
    with torch.no_grad():
        for i, h in enumerate(steps):
            e = torch.zeros(8, device=DEV)
            e[i] = h
            lp = loss(intr0 + e[:4], k0 + e[4:])
            lm = loss(intr0 - e[:4], k0 - e[4:])
            fd.append(float(lp - lm) / (2 * h))
    fd = torch.tensor(fd, dtype=torch.float64)
    names = ["fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"]
    rel = ((analytic - fd).abs() / fd.abs().clamp(min=1e-12)).tolist()
    report(f"finite_differences[{name}]", {
        **{f"d{n}": float(a) for n, a in zip(names, analytic)},
        **{f"fd_{n}": float(b) for n, b in zip(names, fd)},
        **{f"rel_{n}": r for n, r in zip(names, rel)},
    })
    for n, r, b in zip(names, rel, fd.tolist()):
        tol = 0.05 if n in ("fx", "fy", "cx", "cy") else 0.10
        assert r < tol, (n, r, b, rel)


def test_recovery():
    """Recover f, c, k1, k2 from a photometric loss (k3, k4 fixed at the truth).

    k1..k4 are poorly conditioned (theta^3 ... theta^9 are nearly collinear and
    f trades off against k1), so this test optimises f, c, k1 and k2 only.
    """
    cam = dict(width=256, height=256, f=80.0, c=127.5)
    k_true = torch.tensor([0.02, -0.005, 0.0, 0.0], device=DEV)
    intr_true = torch.tensor([80.0, 80.0, 127.5, 127.5], device=DEV)
    scene = make_scene(cam, N=8000, theta_max=1.55, sigma_px=1.5, rho_range=(3.0, 5.0))
    with torch.no_grad():
        target = render(scene, cam, intr_true, k_true)

    log_f = torch.full((2,), math.log(1.1), device=DEV, requires_grad=True)
    c = intr_true[2:].clone().requires_grad_(True)
    k12 = (k_true[:2] + torch.tensor([0.05, 0.0], device=DEV)).requires_grad_(True)
    steps = 2000
    opt = torch.optim.Adam([{"params": [log_f], "lr": 5e-3}, {"params": [c], "lr": 0.05},
                            {"params": [k12], "lr": 5e-3}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    history = []
    for it in range(steps):
        f = intr_true[:2] * torch.exp(log_f)
        k = torch.cat([k12, k_true[2:]])
        img = render(scene, cam, torch.cat([f, c]), k)
        loss = ((img - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if it % 250 == 0 or it == steps - 1:
            history.append(dict(it=it, loss=float(loss), fx=float(f[0]), fy=float(f[1]), cx=float(c[0]),
                                cy=float(c[1]), k1=float(k12[0]), k2=float(k12[1])))
    final = history[-1]
    report("recovery", dict(truth=dict(f=80.0, c=127.5, k1=0.02, k2=-0.005), history=history))
    assert abs(final["fx"] / 80.0 - 1) < 0.01 and abs(final["fy"] / 80.0 - 1) < 0.01, final
    assert abs(final["cx"] - 127.5) < 0.5 and abs(final["cy"] - 127.5) < 0.5, final
    assert abs(final["k1"] - 0.02) < 0.01, final


def test_renderer_and_plucker_agree():
    """Gaussians at o + t d along KB Plücker rays land on their pixels."""
    from erayzer_core.model.gaussians_renderer import render_opencv_cam_gsplat

    er = import_erayzer()
    H = W = 1000
    fx, fy, cx, cy = 250.0, 252.0, 499.5, 497.0
    k = torch.tensor([[0.02, -0.005, 0.001, 0.0]], device=DEV)
    a = 0.3
    c2w = torch.eye(4, device=DEV)
    c2w[:3, :3] = torch.tensor([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    c2w[:3, 3] = torch.tensor([0.5, -0.2, 1.0])
    fxfycxcy = torch.tensor([[fx, fy, cx, cy]], device=DEV)

    pl = er.cam_info_to_plucker(c2w[None], fxfycxcy, edict(height=H, width=W), normalized=False,
                                return_moment=False, radial_coeffs=k)
    # pixels (row, col): centre, mid-field, and two beyond 90 degrees (z < 0)
    pixels = [(497, 499), (300, 650), (820, 260), (497, 28), (40, 520)]
    t = 2.0
    xyz = torch.stack([pl[0, :3, i, j] + t * pl[0, 3:, i, j] for i, j in pixels])
    ray_z = torch.stack([pl[0, 3:, i, j] for i, j in pixels]) @ c2w[:3, :3]  # camera-space directions
    assert float(ray_z[3, 2]) < 0 and float(ray_z[4, 2]) < 0  # behind the image plane

    n = len(pixels)
    pc = SimpleNamespace(
        get_xyz=xyz,
        get_opacity=torch.ones(n, 1, device=DEV),
        get_scaling=torch.full((n, 3), 2e-3 * t, device=DEV),
        get_rotation=torch.tensor([[1.0, 0, 0, 0]], device=DEV).repeat(n, 1),
        get_features=torch.full((n, 1, 3), 0.5 / 0.28209479177387814, device=DEV),  # colour 1
    )
    out = render_opencv_cam_gsplat(pc, H, W, c2w[None], fxfycxcy, sh_degree=0, bg_color=(0.0, 0.0, 0.0),
                                   camera_model="kb", radial_coeffs=k)
    img = out["render"][0].mean(0)  # [H, W]
    errs = []
    for i, j in pixels:
        r = 6
        win = img[i - r:i + r + 1, j - r:j + r + 1] if i >= r else img[: i + r + 1, j - r:j + r + 1]
        i0 = max(i - r, 0)
        ys, xs = torch.meshgrid(torch.arange(win.shape[0], device=DEV) + i0,
                                torch.arange(win.shape[1], device=DEV) + j - r, indexing="ij")
        wsum = win.sum()
        assert float(wsum) > 0.5, (i, j, float(wsum))
        ci, cj = float((win * ys).sum() / wsum), float((win * xs).sum() / wsum)
        errs.append(math.hypot(ci - i, cj - j))
    report("renderer_vs_plucker", dict(pixels=pixels, centroid_err_px=errs))
    assert max(errs) < 0.5, errs

    # The default call (pinhole, no coefficients) still works and differs.
    out_pin = render_opencv_cam_gsplat(pc, H, W, c2w[None], fxfycxcy, sh_degree=0, bg_color=(0.0, 0.0, 0.0))
    assert not torch.equal(out_pin["render"], out["render"])
