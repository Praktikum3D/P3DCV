"""Reference outputs of gsplat for the pinhole and equidistant fisheye paths.

Run once with the UNMODIFIED gsplat fork (the one in the container image) to
save the reference, then test_regression_cuda.py recomputes the same outputs
with the modified fork and compares:

    python tests/gsplat_kb/regression_reference.py --save /path/ref.pt

Only uses rasterization() arguments that the unmodified fork accepts. Inputs
are generated on the CPU with fixed seeds, so both runs see identical tensors.
"""

import argparse
import math

import torch

CONFIGS = [
    # name, camera_model, packed, rasterize_mode, render_mode
    ("pinhole_packed_classic", "pinhole", True, "classic", "RGB"),
    ("pinhole_dense_classic", "pinhole", False, "classic", "RGB+ED"),
    ("pinhole_packed_antialiased", "pinhole", True, "antialiased", "RGB+D"),
    ("pinhole_dense_antialiased", "pinhole", False, "antialiased", "RGB"),
    ("fisheye_packed_classic", "fisheye", True, "classic", "RGB+ED"),
    ("fisheye_dense_antialiased", "fisheye", False, "antialiased", "RGB"),
]


def make_inputs(camera_model, seed=0, device="cuda"):
    g = torch.Generator().manual_seed(seed)
    N, C, W, H = 3000, 2, 320, 240
    means = torch.randn(N, 3, generator=g) * torch.tensor([1.5, 1.2, 0.8]) + torch.tensor([0.0, 0.0, 4.0])
    quats = torch.randn(N, 4, generator=g)
    scales = torch.exp(torch.rand(N, 3, generator=g) * 2.0 - 4.5)
    opacities = torch.rand(N, generator=g) * 0.8 + 0.1
    sh = torch.randn(N, 4, 3, generator=g) * 0.3  # sh_degree 1
    viewmats = torch.eye(4).repeat(C, 1, 1)
    a = 0.1
    viewmats[1, :3, :3] = torch.tensor(
        [[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]
    )
    viewmats[1, :3, 3] = torch.tensor([0.2, -0.1, 0.3])
    if camera_model == "fisheye":
        f = 120.0
    else:
        f = 260.0
    Ks = torch.tensor([[[f, 0, 159.5], [0, f * 1.02, 119.5], [0, 0, 1]]]).repeat(C, 1, 1)
    Ks[1, 0, 2] += 3.0
    weights = torch.randn(C, H, W, 4, generator=g)
    tensors = dict(means=means, quats=quats, scales=scales, opacities=opacities, sh=sh, viewmats=viewmats, Ks=Ks)
    tensors = {n: t.to(device).requires_grad_(True) for n, t in tensors.items()}
    return tensors, weights.to(device), W, H


def compute(config):
    from gsplat import rasterization

    name, camera_model, packed, rasterize_mode, render_mode = config
    t, weights, W, H = make_inputs(camera_model)
    render, alphas, _ = rasterization(
        t["means"], t["quats"], t["scales"], t["opacities"], t["sh"], t["viewmats"], t["Ks"], W, H,
        sh_degree=1, packed=packed, rasterize_mode=rasterize_mode, render_mode=render_mode,
        camera_model=camera_model, near_plane=0.2,
        backgrounds=torch.ones(2, 4 if render_mode != "RGB" else 3, device="cuda") * 0.5,
    )
    D = render.shape[-1]
    L = (render * weights[..., :D]).sum() + (alphas[..., 0] * weights[..., 0]).sum()
    L.backward()
    out = {"render": render.detach().cpu(), "alphas": alphas.detach().cpu()}
    for n, x in t.items():
        out["grad_" + n] = x.grad.detach().cpu()
    return out


def compute_all():
    return {c[0]: compute(c) for c in CONFIGS}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", required=True)
    args = parser.parse_args()
    import gsplat

    ref = compute_all()
    ref["_gsplat_file"] = gsplat.__file__
    torch.save(ref, args.save)
    print(f"saved {len(CONFIGS)} configs to {args.save} using {gsplat.__file__}")
