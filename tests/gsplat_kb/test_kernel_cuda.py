"""Tests 2 and 5: CUDA projection kernels against the float64 PyTorch reference.

2. camera_model="kb": forward (radii, means2d, depths, conics, compensations) and
   all backward outputs (v_means, v_quats, v_scales, v_viewmats, v_Ks,
   v_radial_coeffs) of fully_fused_projection, packed and not packed, with and
   without compensations (antialiased mode), two cameras with different k.
5. camera_model="kb" with k = 0 matches camera_model="fisheye"; the fisheye
   backward now returns dL/d(fx, fy, cx, cy) consistent with the reference.

gradcheck on float32 kernels fails on tolerance, so the kernels are compared
with the float64 reference instead, as in gsplat's own tests/test_basic.py.
"""

import math

import pytest
import torch

from conftest import KB_STRONG_K, load_torch_impl, report, requires_cuda

pytestmark = requires_cuda
ti = load_torch_impl()
DEV = "cuda"


def _rotation(seed):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(4, generator=g, dtype=torch.float64)
    return ti._quat_to_rotmat(q[None])[0]


def make_scene(N=600, seed=0, max_theta=1.75):
    """Gaussians around the origin (where both cameras sit), at incidence angles
    up to max_theta for camera 0 (so z < 0 is covered), plus a few on the axis."""
    g = torch.Generator().manual_seed(seed)
    D = torch.float64
    theta = torch.rand(N, generator=g, dtype=D) * max_theta
    theta[:4] = torch.tensor([0.0, 1e-4, 5e-4, 2e-3], dtype=D)  # series branch
    phi = torch.rand(N, generator=g, dtype=D) * 2 * math.pi
    rho = 1.0 + 4.0 * torch.rand(N, generator=g, dtype=D)
    means = torch.stack(
        [rho * torch.sin(theta) * torch.cos(phi), rho * torch.sin(theta) * torch.sin(phi), rho * torch.cos(theta)],
        -1,
    )
    quats = torch.randn(N, 4, generator=g, dtype=D)
    scales = torch.exp(torch.rand(N, 3, generator=g, dtype=D) * 1.5 - 4.5)  # 0.011 .. 0.05
    viewmats = torch.eye(4, dtype=D).repeat(2, 1, 1)
    viewmats[1, :3, :3] = _rotation(seed + 1)
    viewmats[1, :3, 3] = torch.tensor([0.05, -0.02, 0.1], dtype=D)
    Ks = torch.tensor(
        [[[293.8, 0, 499.5], [0, 290.0, 495.0], [0, 0, 1]], [[200.0, 0, 330.0], [0, 205.0, 310.0], [0, 0, 1]]],
        dtype=D,
    )
    k = torch.tensor([[0.03, -0.01, 0.002, -0.0001], list(KB_STRONG_K)], dtype=D)
    return dict(means=means, quats=quats, scales=scales, viewmats=viewmats, Ks=Ks, k=k, W=1000, H=1000)


def _leafs(scene, dtype, device):
    return {n: scene[n].to(device=device, dtype=dtype).clone().requires_grad_(True) for n in
            ["means", "quats", "scales", "viewmats", "Ks", "k"]}


def run_cuda(scene, packed, calc_comp, camera_model="kb", k_override=None):
    from gsplat.cuda._wrapper import fully_fused_projection

    p = _leafs(scene, torch.float32, DEV)
    k = p["k"] if k_override is None else k_override
    out = fully_fused_projection(
        p["means"], None, p["quats"], p["scales"], p["viewmats"], p["Ks"], scene["W"], scene["H"],
        packed=packed, calc_compensations=calc_comp, camera_model=camera_model,
        radial_coeffs=k if camera_model == "kb" else None,
    )
    C, N = scene["viewmats"].shape[0], scene["means"].shape[0]
    if packed:
        cam_ids, gs_ids, radii, means2d, depths, conics, comps = out

        def dense(x, fill=0.0):
            shape = (C, N) + tuple(x.shape[1:])
            d = torch.full(shape, fill, dtype=x.dtype, device=x.device)
            return d.index_put((cam_ids, gs_ids), x)

        res = dict(radii=dense(radii, 0), means2d=dense(means2d), depths=dense(depths), conics=dense(conics),
                   comps=dense(comps) if comps is not None else None, raw=(means2d, depths, conics, comps),
                   ids=(cam_ids, gs_ids))
    else:
        radii, means2d, depths, conics, comps = out
        res = dict(radii=radii, means2d=means2d, depths=depths, conics=conics, comps=comps,
                   raw=(means2d, depths, conics, comps), ids=None)
    res["leafs"] = p
    return res


def run_ref(scene, calc_comp, camera_model="kb"):
    p = _leafs(scene, torch.float64, "cpu")
    covars, _ = ti._quat_scale_to_covar_preci(p["quats"], p["scales"], True, False, triu=False)
    radii, means2d, depths, conics, comps = ti._fully_fused_projection(
        p["means"], covars, p["viewmats"], p["Ks"], scene["W"], scene["H"],
        calc_compensations=calc_comp, camera_model=camera_model,
        radial_coeffs=p["k"] if camera_model == "kb" else None,
    )
    return dict(radii=radii, means2d=means2d, depths=depths, conics=conics, comps=comps, leafs=p)


def rel_err(a, b):
    a, b = a.double().cpu(), b.double().cpu()
    return float((a - b).abs().max() / b.abs().max().clamp(min=1e-12))


def _masked(t, mask):
    """Select valid entries with where(): invalid dense outputs are uninitialised."""
    m = mask.to(t.device)
    return torch.where(m[..., None] if t.dim() == 3 else m, t, torch.zeros_like(t))


def grads_cuda(res, w, mask):
    """Backprop sum(w * outputs) through the CUDA projection."""
    outs = dict(zip("mdco", res["raw"]))
    L = 0.0
    for key, out in outs.items():
        if out is None:
            continue
        if res["ids"] is not None:  # packed: weights of the nnz entries
            cam_ids, gs_ids = res["ids"]
            wk = w[key].float().to(DEV)[cam_ids, gs_ids]
            L = L + (out * wk).sum()
        else:
            L = L + (_masked(out, mask) * w[key].float().to(DEV)).sum()
    L.backward()
    return {n: t.grad for n, t in res["leafs"].items()}


def grads_ref(ref, w, mask):
    outs = dict(m=ref["means2d"], d=ref["depths"], c=ref["conics"], o=ref["comps"])
    L = 0.0
    for key, out in outs.items():
        if out is not None:
            L = L + (_masked(out, mask) * w[key]).sum()
    L.backward()
    return {n: t.grad for n, t in ref["leafs"].items()}


@pytest.mark.parametrize("packed", [False, True], ids=["dense", "packed"])
@pytest.mark.parametrize("calc_comp", [False, True], ids=["classic", "antialiased"])
def test_kb_kernel_matches_reference(packed, calc_comp):
    scene = make_scene()
    cu_ = run_cuda(scene, packed, calc_comp)
    ref = run_ref(scene, calc_comp)

    valid_cu = (cu_["radii"] > 0).cpu()
    valid_ref = ref["radii"] > 0
    disagree = int((valid_cu != valid_ref).sum())
    # validity may only differ for Gaussians right at a culling boundary (the
    # packed kernel also clamps the radius computation at 0.1 instead of 0.01)
    assert disagree <= 0.01 * valid_ref.numel(), disagree
    mask = valid_cu & valid_ref
    assert int(mask.sum()) > 500
    # both cameras, behind the image plane, and the series branch are exercised
    assert bool(mask[0, :4].all())

    errs = {}
    for name in ["means2d", "depths", "conics"] + (["comps"] if calc_comp else []):
        a = cu_[name].cpu()[mask].double()
        b = ref[name].detach()[mask]
        errs[name] = rel_err(a, b)
        torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-3 * float(b.abs().max()))

    g = torch.Generator().manual_seed(1)
    C, N = mask.shape
    w = dict(m=torch.randn(C, N, 2, generator=g, dtype=torch.float64),
             d=torch.randn(C, N, generator=g, dtype=torch.float64),
             c=torch.randn(C, N, 3, generator=g, dtype=torch.float64) * 1e3,
             o=torch.randn(C, N, generator=g, dtype=torch.float64))
    w = {n: t * (mask[..., None] if t.dim() == 3 else mask) for n, t in w.items()}
    gc = grads_cuda(cu_, w, mask)
    gr = grads_ref(ref, w, mask)
    for name in ["means", "quats", "scales", "viewmats", "Ks", "k"]:
        a, b = gc[name], gr[name]
        if name == "Ks":
            idx = torch.tensor([0, 2, 4, 5])  # fx, cx, fy, cy
            a, b = a.reshape(-1, 9)[:, idx], b.reshape(-1, 9)[:, idx]
        errs["v_" + name] = rel_err(a, b)
        assert errs["v_" + name] < 1e-3, (name, errs)
    report(f"kernel_vs_reference[{'packed' if packed else 'dense'},{'aa' if calc_comp else 'classic'}]",
           dict(disagree=disagree, n_valid=int(mask.sum()), **errs))


def test_kb_default_coeffs_and_errors():
    from gsplat.cuda._wrapper import fully_fused_projection

    scene = make_scene(N=50)
    p = _leafs(scene, torch.float32, DEV)
    a = fully_fused_projection(p["means"], None, p["quats"], p["scales"], p["viewmats"], p["Ks"], 1000, 1000,
                               camera_model="kb")
    b = fully_fused_projection(p["means"], None, p["quats"], p["scales"], p["viewmats"], p["Ks"], 1000, 1000,
                               camera_model="kb", radial_coeffs=torch.zeros(2, 4, device=DEV))
    assert torch.equal(a[0], b[0])  # radii
    valid = a[0] > 0
    for x, y in zip(a[1:4], b[1:4]):  # means2d, depths, conics (uninitialised where radii == 0)
        assert torch.equal(x[valid], y[valid])
    with pytest.raises(AssertionError):
        fully_fused_projection(p["means"], None, p["quats"], p["scales"], p["viewmats"], p["Ks"], 1000, 1000,
                               camera_model="kb", radial_coeffs=torch.zeros(3, 4, device=DEV))


def test_kb_k0_matches_fisheye_and_fisheye_intrinsics_grad():
    """Test 5: KB with k = 0 equals the equidistant fisheye (in front of the
    camera, where gsplat's fisheye is defined), and the fisheye backward now has
    intrinsics gradients that match the reference."""
    scene = make_scene(max_theta=1.45)
    zeros = torch.zeros(2, 4, device=DEV)
    kb = run_cuda(scene, False, False, "kb", k_override=zeros)
    fe = run_cuda(scene, False, False, "fisheye")
    means_c = torch.einsum("cij,nj->cni", scene["viewmats"][:, :3, :3], scene["means"]) + scene["viewmats"][:, None, :3, 3]
    xy = means_c[..., :2].abs().min(dim=-1).values
    # away from x = 0 and y = 0, where gsplat's fisheye regulariser x*x + 1e-7 matters
    mask = (kb["radii"] > 0).cpu() & (fe["radii"] > 0).cpu() & (xy > 0.05)
    assert int(mask.sum()) > 300
    e_m = rel_err(kb["means2d"].cpu()[mask], fe["means2d"].cpu()[mask])
    max_px = float((kb["means2d"].cpu()[mask] - fe["means2d"].cpu()[mask]).abs().max())
    e_c = rel_err(kb["conics"].cpu()[mask], fe["conics"].cpu()[mask])
    assert max_px < 2e-3 and e_c < 1e-3, (max_px, e_c)
    # radii (3 sigma) agree as well
    assert int((kb["radii"].cpu()[mask] != fe["radii"].cpu()[mask]).sum()) <= 2

    # fisheye v_Ks against the reference
    ref = run_ref(scene, False, camera_model="fisheye")
    vmask = (fe["radii"] > 0).cpu() & (ref["radii"] > 0)
    g = torch.Generator().manual_seed(2)
    C, N = vmask.shape
    w = dict(m=torch.randn(C, N, 2, generator=g, dtype=torch.float64) * vmask[..., None],
             d=torch.zeros(C, N, dtype=torch.float64),
             c=torch.randn(C, N, 3, generator=g, dtype=torch.float64) * 1e3 * vmask[..., None],
             o=torch.zeros(C, N, dtype=torch.float64))
    gc = grads_cuda(fe, w, vmask)
    gr = grads_ref(ref, w, vmask)
    idx = torch.tensor([0, 2, 4, 5])
    a, b = gc["Ks"].reshape(-1, 9)[:, idx], gr["Ks"].reshape(-1, 9)[:, idx]
    e_k = rel_err(a, b)
    assert float(b.abs().max()) > 0 and float(a.abs().max()) > 0
    assert e_k < 1e-3, e_k
    report("kb_k0_vs_fisheye", dict(means2d_max_px=max_px, conics_rel=e_c, fisheye_vKs_rel=e_k))
