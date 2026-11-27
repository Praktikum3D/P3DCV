"""Test 1: the float64 PyTorch reference of the KB projection (CPU only).

* gradcheck of mean2d and cov2d with respect to the camera-space mean, fx, fy,
  cx, cy and k1..k4, at theta = 1e-5 (series branch), 0.5, 1.4, z < 0 and just
  below theta_max;
* k = 0 reproduces gsplat's equidistant reference _fisheye_proj;
* the projected mean agrees with the independent Aria fisheye624_project with
  its tangential and thin-prism terms set to zero;
* kb_theta_limit finds the first root of psi'.
"""

import math

import pytest
import torch

from conftest import KB_STRONG_K, import_camera_utils, load_torch_impl, sphere_point

ti = load_torch_impl()
DT = torch.float64


def _proj_fn(p, fx, fy, cx, cy, k, cov):
    Ks = torch.zeros(1, 3, 3, dtype=DT)
    Ks = Ks.index_put((torch.tensor([0]), torch.tensor([0]), torch.tensor([0])), fx)
    Ks = Ks.index_put((torch.tensor([0]), torch.tensor([1]), torch.tensor([1])), fy)
    Ks = Ks.index_put((torch.tensor([0]), torch.tensor([0]), torch.tensor([2])), cx)
    Ks = Ks.index_put((torch.tensor([0]), torch.tensor([1]), torch.tensor([2])), cy)
    means2d, cov2d = ti._kb_proj(p[None, None], cov[None, None], Ks, k[None], 1000, 1000)
    return means2d, cov2d


def _theta_cases():
    k_mild = (0.03, -0.01, 0.002, -0.0001)
    lim = float(ti.kb_theta_limit(torch.tensor([KB_STRONG_K], dtype=DT), 0.75 * math.pi))
    return [
        ("theta=1e-5", 1e-5, k_mild),
        ("theta=0.5", 0.5, k_mild),
        ("theta=1.4", 1.4, k_mild),
        ("z<0 theta=1.7", 1.7, k_mild),
        ("z<0 equidistant theta=1.69", 1.69, (0.0, 0.0, 0.0, 0.0)),
        ("theta=1.4 strong", 1.4, KB_STRONG_K),
        ("just below theta_max", lim - 2e-3, KB_STRONG_K),
    ]


@pytest.mark.parametrize("name,theta,k", _theta_cases(), ids=[c[0] for c in _theta_cases()])
def test_gradcheck_kb_reference(name, theta, k):
    torch.manual_seed(0)
    p = torch.tensor(sphere_point(theta), dtype=DT, requires_grad=True)
    fx = torch.tensor(293.8, dtype=DT, requires_grad=True)
    fy = torch.tensor(301.2, dtype=DT, requires_grad=True)
    cx = torch.tensor(499.5, dtype=DT, requires_grad=True)
    cy = torch.tensor(487.0, dtype=DT, requires_grad=True)
    kk = torch.tensor(k, dtype=DT, requires_grad=True)
    A = torch.randn(3, 3, dtype=DT) * 0.03
    cov = A @ A.T + 1e-4 * torch.eye(3, dtype=DT)
    assert torch.allclose(
        torch.atan2(p[:2].norm(), p[2]), torch.tensor(theta, dtype=DT)
    )
    torch.autograd.gradcheck(
        lambda *a: _proj_fn(*a, cov),
        (p, fx, fy, cx, cy, kk),
        eps=1e-6,
        atol=1e-6,
    )


def test_series_branch_matches_exact_formula():
    """Below the series threshold the series equals the closed form (float64)."""
    k = torch.tensor([0.03, -0.01, 0.002, -0.0001], dtype=DT)
    for theta in [2e-4, 9e-4]:
        x, y, z = (torch.tensor(v, dtype=DT) for v in sphere_point(theta))
        U, V, Jn = ti._kb_proj_normalized(x, y, z, k)
        r = torch.sqrt(x * x + y * y)
        rho2 = r * r + z * z
        th = torch.atan2(r, z)
        psi = ti._kb_psi(th, k)
        dpsi = ti._kb_dpsi(th, k)
        s = psi / r
        # with theta ~ 1e-3, the textbook g is still accurate enough in float64
        g = dpsi * z / (r * r * rho2) - psi / r**3
        h = dpsi / rho2
        ref = torch.stack([s + x * x * g, x * y * g, -x * h, x * y * g, s + y * y * g, -y * h])
        torch.testing.assert_close(U, s * x, rtol=1e-12, atol=0)
        torch.testing.assert_close(Jn.reshape(-1), ref, rtol=1e-6, atol=1e-9)


def test_k0_matches_equidistant_fisheye_reference():
    """k = 0 is gsplat's camera_model="fisheye" (up to its eps = 1e-7 regularisers)."""
    torch.manual_seed(0)
    C, N = 2, 400
    means = torch.randn(C, N, 3, dtype=DT)
    means[..., 2] = means[..., 2].abs() + 0.3
    # keep |x|, |y| away from 0, where gsplat's x*x + 1e-7 regulariser of
    # _fisheye_proj is not negligible
    means[..., :2] = means[..., :2].sign() * (means[..., :2].abs() + 0.1)
    A = torch.randn(C, N, 3, 3, dtype=DT) * 0.05
    covars = A @ A.transpose(-1, -2)
    Ks = torch.tensor(
        [[[293.8, 0, 499.5], [0, 293.8, 499.5], [0, 0, 1]], [[200.0, 0, 320], [0, 210, 240], [0, 0, 1]]],
        dtype=DT,
    )
    m_kb, c_kb = ti._kb_proj(means, covars, Ks, torch.zeros(C, 4, dtype=DT), 1000, 1000)
    m_fe, c_fe = ti._fisheye_proj(means, covars, Ks, 1000, 1000)
    torch.testing.assert_close(m_kb, m_fe, rtol=1e-6, atol=1e-4)
    torch.testing.assert_close(c_kb, c_fe, rtol=1e-4, atol=1e-6)


def test_mean_matches_fisheye624_radial_part():
    """Independent cross-check against the Aria Fisheye624 model in camera_utils."""
    cu = import_camera_utils()
    torch.manual_seed(1)
    N = 500
    dirs = torch.randn(N, 3, dtype=DT)
    dirs[:, 2] = dirs[:, 2].abs() + 0.05  # fisheye624_project uses atan(r / z): z > 0 only
    means = dirs * (0.5 + torch.rand(N, 1, dtype=DT) * 3)
    k = torch.tensor([0.03, -0.01, 0.002, -0.0001], dtype=DT)
    fx, fy, cx, cy = 293.8, 301.2, 499.5, 487.0
    Ks = torch.tensor([[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]], dtype=DT)
    covars = torch.eye(3, dtype=DT).expand(1, N, 3, 3) * 1e-4
    m_kb, _ = ti._kb_proj(means[None], covars, Ks, k[None], 1000, 1000)
    params = torch.zeros(1, 16, dtype=DT)
    params[0, :4] = torch.tensor([fx, fy, cx, cy], dtype=DT)
    params[0, 4:8] = k  # k0..k3 = k1..k4, k4 = k5 = 0, no tangential / thin prism
    m_aria = cu.fisheye624_project(means[None], params)
    torch.testing.assert_close(m_kb, m_aria, rtol=1e-9, atol=1e-7)


def test_theta_limit():
    for k1 in [-0.1, -0.3, -1.0]:
        k = torch.tensor([[k1, 0.0, 0.0, 0.0]], dtype=DT)
        lim = float(ti.kb_theta_limit(k, 3.0))
        expected = math.sqrt(-1.0 / (3 * k1))
        if expected < 3.0:
            assert abs(lim - expected) < 1e-7, (k1, lim, expected)
    # psi' > 0 everywhere -> max_theta
    k = torch.tensor([[0.02, 0.0, 0.0, 0.0]], dtype=DT)
    assert float(ti.kb_theta_limit(k, 2.0)) == 2.0
    # the Plücker side uses the same algorithm
    cu = import_camera_utils()
    ks = torch.tensor([KB_STRONG_K, (0.03, -0.01, 0.002, -0.0001), (-1.0, 0, 0, 0)], dtype=DT)
    torch.testing.assert_close(
        cu.kb_theta_max(ks, 0.75 * math.pi), ti.kb_theta_limit(ks, 0.75 * math.pi), rtol=0, atol=1e-12
    )


def test_reference_fully_fused_kb_culls_and_uses_distance():
    torch.manual_seed(0)
    N = 6
    pts = torch.tensor(
        [sphere_point(t) for t in [0.0, 0.5, 1.4, 1.7, 1.62, 2.5]], dtype=DT
    )  # 1.62 and 2.5 are beyond theta_max of KB_STRONG_K
    covars = torch.eye(3, dtype=DT).expand(N, 3, 3) * 1e-3
    viewmats = torch.eye(4, dtype=DT)[None]
    Ks = torch.tensor([[[200.0, 0, 320], [0, 200, 320], [0, 0, 1]]], dtype=DT)
    radii, means2d, depths, conics, _ = ti._fully_fused_projection(
        pts, covars, viewmats, Ks, 640, 640, camera_model="kb",
        radial_coeffs=torch.tensor([KB_STRONG_K], dtype=DT),
    )
    torch.testing.assert_close(depths[0], torch.full((N,), 2.0, dtype=DT))
    valid = (radii[0] > 0).tolist()
    assert valid[:3] == [True, True, True]
    assert valid[4] is False and valid[5] is False
