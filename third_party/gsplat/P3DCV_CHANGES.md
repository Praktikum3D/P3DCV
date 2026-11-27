# gsplat fork vendored into P3DCV

This directory is a copy of the gsplat fork that E-RayZer uses, with changes
for the P3DCV project. gsplat is licensed under Apache-2.0 (see `LICENSE`); all
upstream copyright notices are kept, and every changed place in the source is
marked with `[P3DCV]`.

## Upstream

* Repository: <https://github.com/QitaoZhao/gsplat>
* Commit: `daad91d9e667cf49bab815d872aa65a5cda0a77e` (gsplat 1.4.0 plus
  "Add support for intrinsics gradient computation"), the commit that
  `docker/Dockerfile` installs.
* Downloaded as the GitHub archive
  `https://github.com/QitaoZhao/gsplat/archive/daad91d9e667cf49bab815d872aa65a5cda0a77e.tar.gz`.
* The git submodule `gsplat/cuda/csrc/third_party/glm` is pinned by the fork to
  glm commit `45008b225e28eb700fa0f7d3ff69b7c1db94fadf`
  (<https://github.com/g-truc/glm>, MIT / Happy Bunny licence, see
  `gsplat/cuda/csrc/third_party/glm/copying.txt`). It was downloaded as the
  archive of that commit and vendored as plain files.

## Left out of the copy

* `docs/` (24 MB of images and videos, not needed to build or test).
* `.gitmodules`, `.gitignore` and `.github/` of the fork (glm is plain files
  here, the repository's own `.gitignore` covers build artefacts, and the CI
  workflows do not apply).
* From glm only the header tree `glm/`, `copying.txt` and `readme.md` are kept
  (no `doc/`, `test/`, `util/`, `cmake/`, `CMakeLists.txt`, `manual.md`,
  `.github/`, `.gitignore`).

Everything else is byte-identical to the archive except the files below.

## Changes: Kannala-Brandt camera and intrinsics gradients

New camera model `camera_model="kb"` (C++ `CameraModelType::KANNALA_BRANDT`):
the OpenCV fisheye / Kannala-Brandt model with four radial coefficients,

    theta = atan2(|(x, y)|, z),  psi(theta) = theta (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8)
    u = fx psi x / r + cx,       v = fy psi y / r + cy.

Rendering through it is differentiable with respect to the Gaussians and to
fx, fy, cx, cy (`Ks`) and k1..k4 (`radial_coeffs`, [C, 4]). k = 0 is the
equidistant model of `camera_model="fisheye"`. The existing `"fisheye"` model
now also returns gradients for fx, fy, cx, cy (before: silently zero).

| File | Change |
| --- | --- |
| `gsplat/cuda/csrc/utils.cuh` | `fisheye_proj_vjp` gains an overload with `mat3 v_K` (v_mean3d / v_cov3d unchanged, the old signature forwards to it). New KB block: forward-mode dual number `KBDual<T, 7>` over (x, y, z, k1..k4), `kb_theta_limit` (first positive root of psi'), `kb_proj_normalized` (shared by forward and backward), `kb_proj` (forward, returns false to cull) and `kb_proj_vjp` (v_mean3d including dJ/dp, v_cov3d, v_K, v_k). |
| `gsplat/cuda/csrc/bindings.h`, `ext.cpp` | Enum value `KANNALA_BRANDT = 3`. The fused projection launchers take `radial_coeffs` (optional [C, 4]) and, in the forward, `kb_max_theta`; the backward launchers return `v_radial_coeffs` as a 7th tensor. |
| `fully_fused_projection_fwd.cu`, `fully_fused_projection_packed_fwd.cu` | KB case. For KB the near / far test and the returned `depths` use the distance to the camera centre instead of z (the camera sees points with z <= 0); other models are unchanged. Gaussians beyond `min(theta_max, kb_max_theta)` are culled. |
| `fully_fused_projection_bwd.cu`, `fully_fused_projection_packed_bwd.cu` | KB case with per-thread `v_k[4]`, warp sum per camera and `gpuAtomicAdd` into `v_radial_coeffs[cid]`, like `v_Ks`; `v_depths` goes through the distance for KB. FISHEYE passes `v_K`. `v_radial_coeffs` is always dense (also with `sparse_grad`). |
| `gsplat/cuda/_wrapper.py` | `KB_DEFAULT_MAX_THETA = 0.75 pi`, mapping `"kb"` to the enum. `fully_fused_projection(..., radial_coeffs=None, kb_max_theta=...)`, `_FullyFusedProjection(Packed)` pass them on and return `v_radial_coeffs` (gated on `ctx.needs_input_grad`). `proj()` asserts that `camera_model != "kb"`. |
| `gsplat/rendering.py` | `rasterization(..., camera_model="kb", radial_coeffs=None, kb_max_theta=None)`; the KB arguments are only passed on for `"kb"`, so all other calls are unchanged. Distributed mode gathers `radial_coeffs` with the cameras. |
| `gsplat/cuda/_torch_impl.py` | Pure PyTorch reference `_kb_proj`, `_kb_proj_normalized`, `kb_theta_limit`, `_kb_psi`, `_kb_dpsi`, mirroring the CUDA code branch by branch (works in float64); `_fully_fused_projection(..., camera_model="kb", radial_coeffs, kb_max_theta)`. |

### Numerical guards (CUDA and reference)

* `theta < 1e-3` (`GSPLAT_KB_SMALL_THETA`): s = psi / r, g and h = psi' / rho^2
  come from their series in q = r^2 / z^2 up to q^2 (truncation ~1e-18 at the
  threshold), which removes the 0/0 at r = 0 and stays differentiable there.
* Elsewhere g = (psi' sin cos - psi) / r^3 is evaluated as
  `-(w - sin w) / 2 + sum_i k_i t^i ((2i+1) sin cos - theta)` with w = 2 theta
  and sin cos = r z / rho^2; `w - sin w` uses its Taylor series for w < 1
  (`GSPLAT_KB_SERIES_W`). This avoids the catastrophic cancellation of the
  textbook g in float32. No sin / cos calls are needed in the kernel.
* `theta_max`: first positive root of psi' (64 samples on [0, kb_max_theta],
  then 30 bisection steps), computed per camera in the kernel. Gaussians with
  theta >= min(theta_max, kb_max_theta) are culled, so they get no gradient.
  `kb_max_theta` defaults to 0.75 pi (fields of view up to 270 degrees); the
  TartanAir fisheye (195 degrees, theta <= 1.70) is well inside.
* Per-camera gradients are warp-reduced before the atomic add.

### Not changed

* `proj()` / `proj_fwd.cu` / `proj_bwd.cu` (the non-fused projection) support
  neither KB nor any camera gradient, as before.
* ORTHO still returns no intrinsics gradient.
* 2DGS code paths are untouched.

## Tests

`tests/gsplat_kb/` in the P3DCV repository; build and run on the cluster with
`slurm/test_gsplat_kb.sh`.
