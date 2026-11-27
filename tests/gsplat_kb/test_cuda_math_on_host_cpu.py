"""The KB (and equidistant v_K) device math of csrc/utils.cuh, compiled for the CPU.

The camera projection functions in utils.cuh are plain templated C++ apart from
the __device__ qualifiers. This test cuts them out of the header, compiles them
with the host C++ compiler against the vendored glm, and compares kb_proj,
kb_proj_vjp and the new intrinsics gradient of fisheye_proj_vjp with the float64
PyTorch reference (autograd). In double precision the agreement must be ~1e-9,
which checks the formulas and the dual-number derivatives; in float the
agreement shows what precision the CUDA kernels can reach.

Skipped if no C++ compiler is available. The real kernels are tested on the GPU
in test_kernel_cuda.py.
"""

import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import KB_STRONG_K, REPO_ROOT, load_torch_impl, sphere_point

ti = load_torch_impl()
CSRC = REPO_ROOT / "third_party/gsplat/gsplat/cuda/csrc"
GLM = CSRC / "third_party/glm"
CXX = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")

HARNESS = r"""
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#define __device__
#define __host__
#define GSPLAT_PRAGMA_UNROLL
using std::atan2;
using std::sqrt;
namespace gsplat {
template <typename T> using vec2 = glm::vec<2, T>;
template <typename T> using vec3 = glm::vec<3, T>;
template <typename T> using mat2 = glm::mat<2, 2, T>;
template <typename T> using mat3 = glm::mat<3, 3, T>;
template <typename T> using mat3x2 = glm::mat<3, 2, T>;
@@SOURCE@@
} // namespace gsplat
using namespace gsplat;

// stdin: n, then per case 24 numbers:
//   model(0 = fisheye, 1 = kb) x y z fx fy cx cy k1..k4 cov(xx xy xz yy yz zz)
//   v_mean(2) v_cov(00 01 11) theta_limit
// stdout per case: valid mean2d(2) cov2d(00 01 11) v_mean3d(3) v_cov3d(9, row-major)
//   v_fx v_fy v_cx v_cy v_k(4)
template <typename T> void run() {
    int n;
    if (scanf("%d", &n) != 1) return;
    for (int c = 0; c < n; c++) {
        double a[24];
        for (int i = 0; i < 24; i++) scanf("%lf", &a[i]);
        const int model = (int)a[0];
        vec3<T> p((T)a[1], (T)a[2], (T)a[3]);
        T fx = a[4], fy = a[5], cx = a[6], cy = a[7];
        T k[4] = {(T)a[8], (T)a[9], (T)a[10], (T)a[11]};
        mat3<T> cov((T)a[12], (T)a[13], (T)a[14], (T)a[13], (T)a[15], (T)a[16],
                    (T)a[14], (T)a[16], (T)a[17]);
        vec2<T> v_mean((T)a[18], (T)a[19]);
        mat2<T> v_cov((T)a[20], (T)a[21], (T)a[21], (T)a[22]);
        T theta_limit = a[23];
        mat2<T> cov2d(0);
        vec2<T> mean2d(0);
        int valid = 1;
        vec3<T> v_mean3d(0);
        mat3<T> v_cov3d(0), v_K(0);
        T v_k[4] = {0, 0, 0, 0};
        if (model == 1) {
            valid = kb_proj<T>(p, cov, fx, fy, cx, cy, k, theta_limit, 1000, 1000, cov2d, mean2d);
            kb_proj_vjp<T>(p, cov, fx, fy, cx, cy, k, 1000, 1000, v_cov, v_mean, v_mean3d, v_cov3d, v_K, v_k);
        } else {
            fisheye_proj<T>(p, cov, fx, fy, cx, cy, 1000, 1000, cov2d, mean2d);
            fisheye_proj_vjp<T>(p, cov, fx, fy, cx, cy, 1000, 1000, v_cov, v_mean, v_mean3d, v_cov3d, v_K);
        }
        printf("%d %.17g %.17g %.17g %.17g %.17g", valid, (double)mean2d.x, (double)mean2d.y,
               (double)cov2d[0][0], (double)cov2d[0][1], (double)cov2d[1][1]);
        for (int i = 0; i < 3; i++) printf(" %.17g", (double)v_mean3d[i]);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) printf(" %.17g", (double)v_cov3d[j][i]);
        printf(" %.17g %.17g %.17g %.17g", (double)v_K[0][0], (double)v_K[1][1],
               (double)v_K[2][0], (double)v_K[2][1]);
        for (int i = 0; i < 4; i++) printf(" %.17g", (double)v_k[i]);
        printf("\n");
    }
}
int main(int argc, char **argv) {
    if (argc > 1 && argv[1][0] == 'f') run<float>(); else run<double>();
    return 0;
}
"""


def _extract_source():
    text = (CSRC / "utils.cuh").read_text()
    start = text.index("template <typename T>\ninline __device__ void fisheye_proj(")
    end = text.index("template <typename T>\ninline __device__ void pos_world_to_cam(")
    return text[start:end]


@pytest.fixture(scope="module")
def harness(tmp_path_factory):
    if CXX is None:
        pytest.skip("no C++ compiler")
    d = tmp_path_factory.mktemp("kb_host")
    src = d / "kb_host.cpp"
    src.write_text(HARNESS.replace("@@SOURCE@@", _extract_source()))
    exe = d / "kb_host"
    cmd = [CXX, "-std=c++17", "-O1", f"-I{GLM}", str(src), "-o", str(exe)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-4000:]
    return exe


def _cases():
    """(model, p, fx, fy, cx, cy, k, cov6, v_mean, v_cov3) for a spread of angles."""
    rng = np.random.default_rng(0)
    out = []
    k_mild = (0.03, -0.01, 0.002, -0.0001)
    lim_strong = float(ti.kb_theta_limit(torch.tensor([KB_STRONG_K], dtype=torch.float64), 0.75 * math.pi))
    angles = [
        (1e-5, k_mild), (2e-4, k_mild), (5e-3, k_mild), (0.1, k_mild), (0.5, k_mild),
        (1.0, k_mild), (1.4, k_mild), (1.7, k_mild), (2.2, k_mild),
        (0.3, (0, 0, 0, 0)), (1.69, (0, 0, 0, 0)), (1e-4, (0, 0, 0, 0)),
        (0.8, KB_STRONG_K), (lim_strong - 0.01, KB_STRONG_K),
    ]
    for theta, k in angles:
        for phi in [0.3, 2.0, -2.6]:
            A = rng.normal(size=(3, 3)) * 0.05
            cov = A @ A.T + 1e-3 * np.eye(3)
            G = rng.normal(size=(2, 2)) * 1e-2
            G = G + G.T
            out.append(
                dict(
                    model=1,
                    p=sphere_point(theta, phi, rho=0.5 + rng.random() * 3),
                    K=(293.8 * (1 + 0.1 * rng.random()), 293.8, 499.5, 487.0),
                    k=k,
                    cov=cov,
                    v_mean=rng.normal(size=2),
                    v_cov=G,
                )
            )
    return out


def _run(exe, cases, precision):
    lines = [str(len(cases))]
    for c in cases:
        cov = c["cov"]
        lim = float(ti.kb_theta_limit(torch.tensor([c["k"]], dtype=torch.float64), 0.75 * math.pi))
        vals = (
            [c["model"], *c["p"], *c["K"], *c["k"]]
            + [cov[0, 0], cov[0, 1], cov[0, 2], cov[1, 1], cov[1, 2], cov[2, 2]]
            + [*c["v_mean"], c["v_cov"][0, 0], c["v_cov"][0, 1], c["v_cov"][1, 1], lim]
        )
        lines.append(" ".join(repr(float(v)) for v in vals))
    res = subprocess.run(
        [str(exe), precision], input="\n".join(lines), capture_output=True, text=True, check=True
    )
    return np.array([[float(x) for x in l.split()] for l in res.stdout.strip().splitlines()])


def _reference(c):
    """float64 autograd of the reference projection; same output layout as the harness."""
    DT = torch.float64
    p = torch.tensor(c["p"], dtype=DT, requires_grad=True)
    Kv = torch.tensor(c["K"], dtype=DT, requires_grad=True)
    k = torch.tensor(c["k"], dtype=DT, requires_grad=True)
    cov = torch.tensor(c["cov"], dtype=DT, requires_grad=True)
    Ks = torch.zeros(1, 3, 3, dtype=DT)
    Ks = Ks + torch.zeros(1, 3, 3, dtype=DT)  # leaf-free copy
    Ks = torch.stack(
        [torch.stack([Kv[0], Kv.new_zeros(()), Kv[2]]),
         torch.stack([Kv.new_zeros(()), Kv[1], Kv[3]]),
         torch.tensor([0.0, 0.0, 1.0], dtype=DT)]
    )[None]
    if c["model"] == 1:
        m, cv = ti._kb_proj(p[None, None], cov[None, None], Ks, k[None], 1000, 1000)
    else:
        m, cv = ti._fisheye_proj(p[None, None], cov[None, None], Ks, 1000, 1000)
    m, cv = m[0, 0], cv[0, 0]
    v_mean = torch.tensor(c["v_mean"], dtype=DT)
    v_cov = torch.tensor(c["v_cov"], dtype=DT)
    L = (m * v_mean).sum() + (cv * v_cov).sum()
    gp, gcov, gK, gk = torch.autograd.grad(L, (p, cov, Kv, k), allow_unused=True)
    gk = torch.zeros(4, dtype=DT) if gk is None else gk
    return np.concatenate(
        [
            [1.0],
            m.detach().numpy(),
            [cv[0, 0].item(), cv[0, 1].item(), cv[1, 1].item()],
            gp.numpy(),
            gcov.numpy().reshape(-1),
            gK.numpy(),
            gk.numpy(),
        ]
    )


def _compare(got, ref, rtol):
    # per-group relative error, scaled by the largest magnitude of the group
    groups = {
        "mean2d": slice(1, 3), "cov2d": slice(3, 6), "v_mean3d": slice(6, 9),
        "v_cov3d": slice(9, 18), "v_K": slice(18, 22), "v_k": slice(22, 26),
    }
    errs = {}
    for name, sl in groups.items():
        scale = max(np.abs(ref[sl]).max(), 1e-12)
        errs[name] = float(np.abs(got[sl] - ref[sl]).max() / scale)
    bad = {n: e for n, e in errs.items() if e > rtol}
    return bad, errs


def test_kb_double_matches_reference(harness):
    cases = _cases()
    got = _run(harness, cases, "d")
    worst = {}
    for c, g in zip(cases, got):
        assert g[0] == 1, c
        bad, errs = _compare(g, _reference(c), 1e-9)
        assert not bad, (c["p"], c["k"], bad)
        for n, e in errs.items():
            worst[n] = max(worst.get(n, 0.0), e)
    print("double, worst relative error per output:", worst)


def test_kb_float_matches_reference(harness):
    cases = _cases()
    got = _run(harness, cases, "f")
    worst = {}
    for c, g in zip(cases, got):
        bad, errs = _compare(g, _reference(c), 1e-3)
        assert not bad, (c["p"], c["k"], bad)
        for n, e in errs.items():
            worst[n] = max(worst.get(n, 0.0), e)
    print("float, worst relative error per output:", worst)


def test_kb_culls_beyond_theta_limit(harness):
    c = _cases()[0]
    c = dict(c, p=sphere_point(1.65), k=KB_STRONG_K)  # theta_max of KB_STRONG_K ~ 1.607
    got = _run(harness, [c], "f")
    assert got[0, 0] == 0


def test_fisheye_intrinsics_gradient(harness):
    """The new v_K of the equidistant FISHEYE model; v_mean3d / v_cov3d as before."""
    cases = []
    for c in _cases():
        if c["k"] == (0, 0, 0, 0) and c["p"][2] > 0.05 and min(abs(c["p"][0]), abs(c["p"][1])) > 0.05:
            cases.append(dict(c, model=0))
    for c in [dict(c, p=sphere_point(t, 0.4)) for c in _cases()[:1] for t in [0.3, 0.9, 1.3]]:
        cases.append(dict(c, model=0, k=(0, 0, 0, 0)))
    assert len(cases) >= 3
    got = _run(harness, cases, "d")
    for c, g in zip(cases, got):
        ref = _reference(c)
        # gsplat's equidistant code regularises with eps = 1e-7 (x*x + eps,
        # |xy| + eps, z + eps), so it is not exactly the derivative of its own
        # forward; the intrinsics gradient is checked to that level.
        bad, errs = _compare(g, ref, 1e-4)
        assert not bad, (c["p"], bad, errs)
