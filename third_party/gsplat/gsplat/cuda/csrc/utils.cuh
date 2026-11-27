#ifndef GSPLAT_CUDA_UTILS_H
#define GSPLAT_CUDA_UTILS_H

#include "helpers.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

#define FILTER_INV_SQUARE 2.0f

namespace gsplat {

template <typename T>
inline __device__ mat3<T> quat_to_rotmat(const vec4<T> quat) {
    T w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    // normalize
    T inv_norm = rsqrt(x * x + y * y + z * z + w * w);
    x *= inv_norm;
    y *= inv_norm;
    z *= inv_norm;
    w *= inv_norm;
    T x2 = x * x, y2 = y * y, z2 = z * z;
    T xy = x * y, xz = x * z, yz = y * z;
    T wx = w * x, wy = w * y, wz = w * z;
    return mat3<T>(
        (1.f - 2.f * (y2 + z2)),
        (2.f * (xy + wz)),
        (2.f * (xz - wy)), // 1st col
        (2.f * (xy - wz)),
        (1.f - 2.f * (x2 + z2)),
        (2.f * (yz + wx)), // 2nd col
        (2.f * (xz + wy)),
        (2.f * (yz - wx)),
        (1.f - 2.f * (x2 + y2)) // 3rd col
    );
}

template <typename T>
inline __device__ void
quat_to_rotmat_vjp(const vec4<T> quat, const mat3<T> v_R, vec4<T> &v_quat) {
    T w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    // normalize
    T inv_norm = rsqrt(x * x + y * y + z * z + w * w);
    x *= inv_norm;
    y *= inv_norm;
    z *= inv_norm;
    w *= inv_norm;
    vec4<T> v_quat_n = vec4<T>(
        2.f * (x * (v_R[1][2] - v_R[2][1]) + y * (v_R[2][0] - v_R[0][2]) +
               z * (v_R[0][1] - v_R[1][0])),
        2.f *
            (-2.f * x * (v_R[1][1] + v_R[2][2]) + y * (v_R[0][1] + v_R[1][0]) +
             z * (v_R[0][2] + v_R[2][0]) + w * (v_R[1][2] - v_R[2][1])),
        2.f * (x * (v_R[0][1] + v_R[1][0]) - 2.f * y * (v_R[0][0] + v_R[2][2]) +
               z * (v_R[1][2] + v_R[2][1]) + w * (v_R[2][0] - v_R[0][2])),
        2.f * (x * (v_R[0][2] + v_R[2][0]) + y * (v_R[1][2] + v_R[2][1]) -
               2.f * z * (v_R[0][0] + v_R[1][1]) + w * (v_R[0][1] - v_R[1][0]))
    );

    vec4<T> quat_n = vec4<T>(w, x, y, z);
    v_quat += (v_quat_n - glm::dot(v_quat_n, quat_n) * quat_n) * inv_norm;
}

template <typename T>
inline __device__ void quat_scale_to_covar_preci(
    const vec4<T> quat,
    const vec3<T> scale,
    // optional outputs
    mat3<T> *covar,
    mat3<T> *preci
) {
    mat3<T> R = quat_to_rotmat<T>(quat);
    if (covar != nullptr) {
        // C = R * S * S * Rt
        mat3<T> S =
            mat3<T>(scale[0], 0.f, 0.f, 0.f, scale[1], 0.f, 0.f, 0.f, scale[2]);
        mat3<T> M = R * S;
        *covar = M * glm::transpose(M);
    }
    if (preci != nullptr) {
        // P = R * S^-1 * S^-1 * Rt
        mat3<T> S = mat3<T>(
            1.0f / scale[0],
            0.f,
            0.f,
            0.f,
            1.0f / scale[1],
            0.f,
            0.f,
            0.f,
            1.0f / scale[2]
        );
        mat3<T> M = R * S;
        *preci = M * glm::transpose(M);
    }
}

template <typename T>
inline __device__ void quat_scale_to_covar_vjp(
    // fwd inputs
    const vec4<T> quat,
    const vec3<T> scale,
    // precompute
    const mat3<T> R,
    // grad outputs
    const mat3<T> v_covar,
    // grad inputs
    vec4<T> &v_quat,
    vec3<T> &v_scale
) {
    T w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    T sx = scale[0], sy = scale[1], sz = scale[2];

    // M = R * S
    mat3<T> S = mat3<T>(sx, 0.f, 0.f, 0.f, sy, 0.f, 0.f, 0.f, sz);
    mat3<T> M = R * S;

    // https://math.stackexchange.com/a/3850121
    // for D = W * X, G = df/dD
    // df/dW = G * XT, df/dX = WT * G
    // so
    // for D = M * Mt,
    // df/dM = df/dM + df/dMt = G * M + (Mt * G)t = G * M + Gt * M
    mat3<T> v_M = (v_covar + glm::transpose(v_covar)) * M;
    mat3<T> v_R = v_M * S;

    // grad for (quat, scale) from covar
    quat_to_rotmat_vjp<T>(quat, v_R, v_quat);

    v_scale[0] +=
        R[0][0] * v_M[0][0] + R[0][1] * v_M[0][1] + R[0][2] * v_M[0][2];
    v_scale[1] +=
        R[1][0] * v_M[1][0] + R[1][1] * v_M[1][1] + R[1][2] * v_M[1][2];
    v_scale[2] +=
        R[2][0] * v_M[2][0] + R[2][1] * v_M[2][1] + R[2][2] * v_M[2][2];
}

template <typename T>
inline __device__ void quat_scale_to_preci_vjp(
    // fwd inputs
    const vec4<T> quat,
    const vec3<T> scale,
    // precompute
    const mat3<T> R,
    // grad outputs
    const mat3<T> v_preci,
    // grad inputs
    vec4<T> &v_quat,
    vec3<T> &v_scale
) {
    T w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    T sx = 1.0f / scale[0], sy = 1.0f / scale[1], sz = 1.0f / scale[2];

    // M = R * S
    mat3<T> S = mat3<T>(sx, 0.f, 0.f, 0.f, sy, 0.f, 0.f, 0.f, sz);
    mat3<T> M = R * S;

    // https://math.stackexchange.com/a/3850121
    // for D = W * X, G = df/dD
    // df/dW = G * XT, df/dX = WT * G
    // so
    // for D = M * Mt,
    // df/dM = df/dM + df/dMt = G * M + (Mt * G)t = G * M + Gt * M
    mat3<T> v_M = (v_preci + glm::transpose(v_preci)) * M;
    mat3<T> v_R = v_M * S;

    // grad for (quat, scale) from preci
    quat_to_rotmat_vjp<T>(quat, v_R, v_quat);

    v_scale[0] +=
        -sx * sx *
        (R[0][0] * v_M[0][0] + R[0][1] * v_M[0][1] + R[0][2] * v_M[0][2]);
    v_scale[1] +=
        -sy * sy *
        (R[1][0] * v_M[1][0] + R[1][1] * v_M[1][1] + R[1][2] * v_M[1][2]);
    v_scale[2] +=
        -sz * sz *
        (R[2][0] * v_M[2][0] + R[2][1] * v_M[2][1] + R[2][2] * v_M[2][2]);
}

template <typename T>
inline __device__ void ortho_proj(
    // inputs
    const vec3<T> mean3d,
    const mat3<T> cov3d,
    const T fx,
    const T fy,
    const T cx,
    const T cy,
    const uint32_t width,
    const uint32_t height,
    // outputs
    mat2<T> &cov2d,
    vec2<T> &mean2d
) {
    T x = mean3d[0], y = mean3d[1], z = mean3d[2];

    // mat3x2 is 3 columns x 2 rows.
    mat3x2<T> J = mat3x2<T>(
        fx,
        0.f, // 1st column
        0.f,
        fy, // 2nd column
        0.f,
        0.f // 3rd column
    );
    cov2d = J * cov3d * glm::transpose(J);
    mean2d = vec2<T>({fx * x + cx, fy * y + cy});
}

template <typename T>
inline __device__ void ortho_proj_vjp(
    // fwd inputs
    const vec3<T> mean3d,
    const mat3<T> cov3d,
    const T fx,
    const T fy,
    const T cx,
    const T cy,
    const uint32_t width,
    const uint32_t height,
    // grad outputs
    const mat2<T> v_cov2d,
    const vec2<T> v_mean2d,
    // grad inputs
    vec3<T> &v_mean3d,
    mat3<T> &v_cov3d
) {
    T x = mean3d[0], y = mean3d[1], z = mean3d[2];

    // mat3x2 is 3 columns x 2 rows.
    mat3x2<T> J = mat3x2<T>(
        fx,
        0.f, // 1st column
        0.f,
        fy, // 2nd column
        0.f,
        0.f // 3rd column
    );

    // cov = J * V * Jt; G = df/dcov = v_cov
    // -> df/dV = Jt * G * J
    // -> df/dJ = G * J * Vt + Gt * J * V
    v_cov3d += glm::transpose(J) * v_cov2d * J;

    // df/dx = fx * df/dpixx
    // df/dy = fy * df/dpixy
    // df/dz = 0
    v_mean3d += vec3<T>(fx * v_mean2d[0], fy * v_mean2d[1], 0.f);
}

template <typename T>
inline __device__ void persp_proj(
    // inputs
    const vec3<T> mean3d,
    const mat3<T> cov3d,
    const T fx,
    const T fy,
    const T cx,
    const T cy,
    const uint32_t width,
    const uint32_t height,
    // outputs
    mat2<T> &cov2d,
    vec2<T> &mean2d
) {
    T x = mean3d[0], y = mean3d[1], z = mean3d[2];

    T tan_fovx = 0.5f * width / fx;
    T tan_fovy = 0.5f * height / fy;
    T lim_x_pos = (width - cx) / fx + 0.3f * tan_fovx;
    T lim_x_neg = cx / fx + 0.3f * tan_fovx;
    T lim_y_pos = (height - cy) / fy + 0.3f * tan_fovy;
    T lim_y_neg = cy / fy + 0.3f * tan_fovy;

    T rz = 1.f / z;
    T rz2 = rz * rz;
    T tx = z * min(lim_x_pos, max(-lim_x_neg, x * rz));
    T ty = z * min(lim_y_pos, max(-lim_y_neg, y * rz));

    // mat3x2 is 3 columns x 2 rows.
    mat3x2<T> J = mat3x2<T>(
        fx * rz,
        0.f, // 1st column
        0.f,
        fy * rz, // 2nd column
        -fx * tx * rz2,
        -fy * ty * rz2 // 3rd column
    );
    cov2d = J * cov3d * glm::transpose(J);
    mean2d = vec2<T>({fx * x * rz + cx, fy * y * rz + cy});
}

template <typename T>
inline __device__ void persp_proj_vjp(
    // fwd inputs
    const vec3<T> mean3d,
    const mat3<T> cov3d,
    const T fx,
    const T fy,
    const T cx,
    const T cy,
    const uint32_t width,
    const uint32_t height,
    // grad outputs
    const mat2<T> v_cov2d,
    const vec2<T> v_mean2d,
    // grad inputs
    vec3<T> &v_mean3d,
    mat3<T> &v_cov3d,
    mat3<T> &v_K
) {
    T x = mean3d[0], y = mean3d[1], z = mean3d[2];

    T tan_fovx = 0.5f * width / fx;
    T tan_fovy = 0.5f * height / fy;
    T lim_x_pos = (width - cx) / fx + 0.3f * tan_fovx;
    T lim_x_neg = cx / fx + 0.3f * tan_fovx;
    T lim_y_pos = (height - cy) / fy + 0.3f * tan_fovy;
    T lim_y_neg = cy / fy + 0.3f * tan_fovy;

    T rz = 1.f / z;
    T rz2 = rz * rz;
    T tx = z * min(lim_x_pos, max(-lim_x_neg, x * rz));
    T ty = z * min(lim_y_pos, max(-lim_y_neg, y * rz));

    // mat3x2 is 3 columns x 2 rows.
    mat3x2<T> J = mat3x2<T>(
        fx * rz,
        0.f, // 1st column
        0.f,
        fy * rz, // 2nd column
        -fx * tx * rz2,
        -fy * ty * rz2 // 3rd column
    );

    // cov = J * V * Jt; G = df/dcov = v_cov
    // -> df/dV = Jt * G * J
    // -> df/dJ = G * J * Vt + Gt * J * V
    v_cov3d += glm::transpose(J) * v_cov2d * J;

    // df/dx = fx * rz * df/dpixx
    // df/dy = fy * rz * df/dpixy
    // df/dz = - fx * mean.x * rz2 * df/dpixx - fy * mean.y * rz2 * df/dpixy
    v_mean3d += vec3<T>(
        fx * rz * v_mean2d[0],
        fy * rz * v_mean2d[1],
        -(fx * x * v_mean2d[0] + fy * y * v_mean2d[1]) * rz2
    );

    // === Grad wrt intrinsics from projected means ===
    v_K[0][0] += x * rz * v_mean2d[0];         // dL/dfx from x
    v_K[1][1] += y * rz * v_mean2d[1];         // dL/dfy from y
    v_K[2][0] += v_mean2d[0];                  // dL/dcx
    v_K[2][1] += v_mean2d[1];                  // dL/dcy

    // df/dx = -fx * rz2 * df/dJ_02
    // df/dy = -fy * rz2 * df/dJ_12
    // df/dz = -fx * rz2 * df/dJ_00 - fy * rz2 * df/dJ_11
    //         + 2 * fx * tx * rz3 * df/dJ_02 + 2 * fy * ty * rz3
    T rz3 = rz2 * rz;
    mat3x2<T> v_J = v_cov2d * J * glm::transpose(cov3d) +
                    glm::transpose(v_cov2d) * J * cov3d;

    // fov clipping
    if (x * rz <= lim_x_pos && x * rz >= -lim_x_neg) {
        v_mean3d.x += -fx * rz2 * v_J[2][0];
        v_K[0][0] -= tx * rz2 * v_J[2][0];
    } else {
        v_mean3d.z += -fx * rz3 * v_J[2][0] * tx;
        v_K[2][0] += rz * v_J[2][0];
    }
    if (y * rz <= lim_y_pos && y * rz >= -lim_y_neg) {
        v_mean3d.y += -fy * rz2 * v_J[2][1];
        v_K[1][1] -= ty * rz2 * v_J[2][1];
    } else {
        v_mean3d.z += -fy * rz3 * v_J[2][1] * ty;
        v_K[2][1] += rz * v_J[2][1];
    }
    v_mean3d.z += -fx * rz2 * v_J[0][0] - fy * rz2 * v_J[1][1] +
                  2.f * fx * tx * rz3 * v_J[2][0] +
                  2.f * fy * ty * rz3 * v_J[2][1];

    v_K[0][0] += rz * v_J[0][0];  // fx
    v_K[1][1] += rz * v_J[1][1];  // fy
}

template <typename T>
inline __device__ void fisheye_proj(
    // inputs
    const vec3<T> mean3d,
    const mat3<T> cov3d,
    const T fx,
    const T fy,
    const T cx,
    const T cy,
    const uint32_t width,
    const uint32_t height,
    // outputs
    mat2<T> &cov2d,
    vec2<T> &mean2d
) {
    T x = mean3d[0], y = mean3d[1], z = mean3d[2];

    T eps = 0.0000001f;
    T xy_len = glm::length(glm::vec2({x, y})) + eps;
    T theta = glm::atan(xy_len, z + eps);
    mean2d =
        vec2<T>({x * fx * theta / xy_len + cx, y * fy * theta / xy_len + cy});

    T x2 = x * x + eps;
    T y2 = y * y;
    T xy = x * y;
    T x2y2 = x2 + y2;
    T x2y2z2_inv = 1.f / (x2y2 + z * z);

    T b = glm::atan(xy_len, z) / xy_len / x2y2;
    T a = z * x2y2z2_inv / (x2y2);
    mat3x2<T> J = mat3x2<T>(
        fx * (x2 * a + y2 * b),
        fy * xy * (a - b),
        fx * xy * (a - b),
        fy * (y2 * a + x2 * b),
        -fx * x * x2y2z2_inv,
        -fy * y * x2y2z2_inv
    );
    cov2d = J * cov3d * glm::transpose(J);
}

// [P3DCV] This overload also accumulates the intrinsics gradient v_K (same
// layout as persp_proj_vjp: v_K[0][0] = fx, v_K[1][1] = fy, v_K[2][0] = cx,
// v_K[2][1] = cy). v_mean3d and v_cov3d are computed exactly as before.
template <typename T>
inline __device__ void fisheye_proj_vjp(
    // fwd inputs
    const vec3<T> mean3d,
    const mat3<T> cov3d,
    const T fx,
    const T fy,
    const T cx,
    const T cy,
    const uint32_t width,
    const uint32_t height,
    // grad outputs
    const mat2<T> v_cov2d,
    const vec2<T> v_mean2d,
    // grad inputs
    vec3<T> &v_mean3d,
    mat3<T> &v_cov3d,
    mat3<T> &v_K
) {
    T x = mean3d[0], y = mean3d[1], z = mean3d[2];

    const T eps = 0.0000001f;
    T x2 = x * x + eps;
    T y2 = y * y;
    T xy = x * y;
    T x2y2 = x2 + y2;
    T len_xy = length(glm::vec2({x, y})) + eps;
    const T x2y2z2 = x2y2 + z * z;
    T x2y2z2_inv = 1.f / x2y2z2;
    T b = glm::atan(len_xy, z) / len_xy / x2y2;
    T a = z * x2y2z2_inv / (x2y2);
    v_mean3d += vec3<T>(
        fx * (x2 * a + y2 * b) * v_mean2d[0] + fy * xy * (a - b) * v_mean2d[1],
        fx * xy * (a - b) * v_mean2d[0] + fy * (y2 * a + x2 * b) * v_mean2d[1],
        -fx * x * x2y2z2_inv * v_mean2d[0] - fy * y * x2y2z2_inv * v_mean2d[1]
    );

    const T theta = glm::atan(len_xy, z);
    const T J_b = theta / len_xy / x2y2;
    const T J_a = z * x2y2z2_inv / (x2y2);
    // mat3x2 is 3 columns x 2 rows.
    mat3x2<T> J = mat3x2<T>(
        fx * (x2 * J_a + y2 * J_b),
        fy * xy * (J_a - J_b), // 1st column
        fx * xy * (J_a - J_b),
        fy * (y2 * J_a + x2 * J_b), // 2nd column
        -fx * x * x2y2z2_inv,
        -fy * y * x2y2z2_inv // 3rd column
    );
    v_cov3d += glm::transpose(J) * v_cov2d * J;

    mat3x2<T> v_J = v_cov2d * J * glm::transpose(cov3d) +
                    glm::transpose(v_cov2d) * J * cov3d;

    // [P3DCV] Grad wrt intrinsics. The projected mean uses the same eps
    // regularisation as fisheye_proj; J is linear in fx (row 0) and fy (row 1),
    // so dL/dfx = sum_j v_J[j][0] * J[j][0] / fx, evaluated with the unscaled
    // row to avoid dividing by fx.
    {
        const T xy_len_fwd = len_xy;
        const T theta_fwd = glm::atan(xy_len_fwd, z + eps);
        v_K[0][0] += x * theta_fwd / xy_len_fwd * v_mean2d[0];
        v_K[1][1] += y * theta_fwd / xy_len_fwd * v_mean2d[1];
        v_K[2][0] += v_mean2d[0];
        v_K[2][1] += v_mean2d[1];
        v_K[0][0] += v_J[0][0] * (x2 * J_a + y2 * J_b) +
                     v_J[1][0] * xy * (J_a - J_b) -
                     v_J[2][0] * x * x2y2z2_inv;
        v_K[1][1] += v_J[0][1] * xy * (J_a - J_b) +
                     v_J[1][1] * (y2 * J_a + x2 * J_b) -
                     v_J[2][1] * y * x2y2z2_inv;
    }

    T l4 = x2y2z2 * x2y2z2;

    T E = -l4 * x2y2 * theta + x2y2z2 * x2y2 * len_xy * z;
    T F = 3 * l4 * theta - 3 * x2y2z2 * len_xy * z - 2 * x2y2 * len_xy * z;

    T A = x * (3 * E + x2 * F);
    T B = y * (E + x2 * F);
    T C = x * (E + y2 * F);
    T D = y * (3 * E + y2 * F);

    T S1 = x2 - y2 - z * z;
    T S2 = y2 - x2 - z * z;
    T inv1 = x2y2z2_inv * x2y2z2_inv;
    T inv2 = inv1 / (x2y2 * x2y2 * len_xy);

    T dJ_dx00 = fx * A * inv2;
    T dJ_dx01 = fx * B * inv2;
    T dJ_dx02 = fx * S1 * inv1;
    T dJ_dx10 = fy * B * inv2;
    T dJ_dx11 = fy * C * inv2;
    T dJ_dx12 = 2.f * fy * xy * inv1;

    T dJ_dy00 = dJ_dx01;
    T dJ_dy01 = fx * C * inv2;
    T dJ_dy02 = 2.f * fx * xy * inv1;
    T dJ_dy10 = dJ_dx11;
    T dJ_dy11 = fy * D * inv2;
    T dJ_dy12 = fy * S2 * inv1;

    T dJ_dz00 = dJ_dx02;
    T dJ_dz01 = dJ_dy02;
    T dJ_dz02 = 2.f * fx * x * z * inv1;
    T dJ_dz10 = dJ_dx12;
    T dJ_dz11 = dJ_dy12;
    T dJ_dz12 = 2.f * fy * y * z * inv1;

    T dL_dtx_raw = dJ_dx00 * v_J[0][0] + dJ_dx01 * v_J[1][0] +
                   dJ_dx02 * v_J[2][0] + dJ_dx10 * v_J[0][1] +
                   dJ_dx11 * v_J[1][1] + dJ_dx12 * v_J[2][1];
    T dL_dty_raw = dJ_dy00 * v_J[0][0] + dJ_dy01 * v_J[1][0] +
                   dJ_dy02 * v_J[2][0] + dJ_dy10 * v_J[0][1] +
                   dJ_dy11 * v_J[1][1] + dJ_dy12 * v_J[2][1];
    T dL_dtz_raw = dJ_dz00 * v_J[0][0] + dJ_dz01 * v_J[1][0] +
                   dJ_dz02 * v_J[2][0] + dJ_dz10 * v_J[0][1] +
                   dJ_dz11 * v_J[1][1] + dJ_dz12 * v_J[2][1];
    v_mean3d.x += dL_dtx_raw;
    v_mean3d.y += dL_dty_raw;
    v_mean3d.z += dL_dtz_raw;
}

// [P3DCV] Original signature without v_K, still used by proj_bwd.cu.
template <typename T>
inline __device__ void fisheye_proj_vjp(
    // fwd inputs
    const vec3<T> mean3d,
    const mat3<T> cov3d,
    const T fx,
    const T fy,
    const T cx,
    const T cy,
    const uint32_t width,
    const uint32_t height,
    // grad outputs
    const mat2<T> v_cov2d,
    const vec2<T> v_mean2d,
    // grad inputs
    vec3<T> &v_mean3d,
    mat3<T> &v_cov3d
) {
    mat3<T> v_K_unused(0.f);
    fisheye_proj_vjp<T>(
        mean3d,
        cov3d,
        fx,
        fy,
        cx,
        cy,
        width,
        height,
        v_cov2d,
        v_mean2d,
        v_mean3d,
        v_cov3d,
        v_K_unused
    );
}

/****************************************************************************
 * [P3DCV] Kannala-Brandt (OpenCV fisheye) projection with k1..k4.
 *
 *   r = |(x, y)|, rho = |(x, y, z)|, theta = atan2(r, z)
 *   psi(theta)  = theta (1 + k1 t + k2 t^2 + k3 t^3 + k4 t^4), t = theta^2
 *   s = psi / r,  u = fx s x + cx,  v = fy s y + cy
 *
 * EWA Jacobian, J = diag(fx, fy) * Jn with
 *   g  = psi' z / (r^2 rho^2) - psi / r^3,  h = psi' / rho^2
 *   Jn = [ s + x^2 g   x y g       -x h ]
 *        [ x y g       s + y^2 g   -y h ]
 *
 * All derivatives with respect to the camera-space mean and k1..k4 (including
 * the second derivatives of the projection needed for dJ/dp) come from a small
 * forward-mode dual number over the 7 tangents (x, y, z, k1, k2, k3, k4).
 * fx, fy, cx, cy enter linearly and are handled in closed form.
 *
 * Numerical guards:
 * - theta < GSPLAT_KB_SMALL_THETA: s, g and h are evaluated from their series
 *   in q = r^2 / z^2 (two terms beyond the leading one, truncation error
 *   O(q^3) ~ 1e-18 at the threshold). This removes the 0/0 at r = 0.
 * - Elsewhere g is evaluated as N / r^3 with
 *   N = psi' sin cos - psi = -(w - sin w) / 2 + sum_i k_i t^i ((2i+1) sin cos - theta)
 *   and w = 2 theta, where w - sin w uses its Taylor series for
 *   w < GSPLAT_KB_SERIES_W. This avoids the catastrophic cancellation of the
 *   textbook expression in float32.
 * - Gaussians with theta >= kb_theta_limit(k, max_theta), i.e. beyond the first
 *   positive root of psi' (where the projection folds over) or beyond the
 *   configurable maximum angle, are culled by the caller.
 ****************************************************************************/

#define GSPLAT_KB_SMALL_THETA 1e-3f
#define GSPLAT_KB_SERIES_W 1.0f
#define GSPLAT_KB_N_TANGENTS 7
#define GSPLAT_KB_LIMIT_SAMPLES 64
#define GSPLAT_KB_LIMIT_BISECT 30

template <typename T, int N> struct KBDual {
    T v;
    T d[N];
    __host__ __device__ KBDual() : v(T(0)) {
#pragma unroll
        for (int i = 0; i < N; i++)
            d[i] = T(0);
    }
    __host__ __device__ KBDual(const T c) : v(c) {
#pragma unroll
        for (int i = 0; i < N; i++)
            d[i] = T(0);
    }
    static __host__ __device__ KBDual seed(const T c, const int i) {
        KBDual r(c);
        r.d[i] = T(1);
        return r;
    }
    friend __host__ __device__ KBDual operator+(const KBDual &a, const KBDual &b) {
        KBDual r;
        r.v = a.v + b.v;
#pragma unroll
        for (int i = 0; i < N; i++)
            r.d[i] = a.d[i] + b.d[i];
        return r;
    }
    friend __host__ __device__ KBDual operator-(const KBDual &a, const KBDual &b) {
        KBDual r;
        r.v = a.v - b.v;
#pragma unroll
        for (int i = 0; i < N; i++)
            r.d[i] = a.d[i] - b.d[i];
        return r;
    }
    friend __host__ __device__ KBDual operator-(const KBDual &a) {
        KBDual r;
        r.v = -a.v;
#pragma unroll
        for (int i = 0; i < N; i++)
            r.d[i] = -a.d[i];
        return r;
    }
    friend __host__ __device__ KBDual operator*(const KBDual &a, const KBDual &b) {
        KBDual r;
        r.v = a.v * b.v;
#pragma unroll
        for (int i = 0; i < N; i++)
            r.d[i] = a.d[i] * b.v + a.v * b.d[i];
        return r;
    }
    friend __host__ __device__ KBDual operator/(const KBDual &a, const KBDual &b) {
        KBDual r;
        const T inv = T(1) / b.v;
        r.v = a.v * inv;
#pragma unroll
        for (int i = 0; i < N; i++)
            r.d[i] = (a.d[i] - r.v * b.d[i]) * inv;
        return r;
    }
};

template <typename T> inline __device__ T kb_value(const T a) { return a; }
template <typename T, int N>
inline __device__ T kb_value(const KBDual<T, N> &a) {
    return a.v;
}

template <typename T> inline __device__ T kb_sqrt(const T a) { return sqrt(a); }
template <typename T, int N>
inline __device__ KBDual<T, N> kb_sqrt(const KBDual<T, N> &a) {
    KBDual<T, N> r;
    r.v = sqrt(a.v);
    const T f = T(0.5) / r.v;
#pragma unroll
    for (int i = 0; i < N; i++)
        r.d[i] = a.d[i] * f;
    return r;
}

template <typename T> inline __device__ T kb_atan2(const T a, const T b) {
    return atan2(a, b);
}
template <typename T, int N>
inline __device__ KBDual<T, N>
kb_atan2(const KBDual<T, N> &a, const KBDual<T, N> &b) {
    KBDual<T, N> r;
    r.v = atan2(a.v, b.v);
    const T inv = T(1) / (a.v * a.v + b.v * b.v);
#pragma unroll
    for (int i = 0; i < N; i++)
        r.d[i] = (b.v * a.d[i] - a.v * b.d[i]) * inv;
    return r;
}

// psi'(theta) for the culling limit (plain scalar, no derivatives needed).
template <typename T>
inline __device__ T kb_dpsi(const T theta, const T *k) {
    const T t = theta * theta;
    return T(1) +
           t * (T(3) * k[0] +
                t * (T(5) * k[1] + t * (T(7) * k[2] + t * T(9) * k[3])));
}

// Largest valid incidence angle: the first positive root of psi' (found by
// sampling [0, max_theta] and bisecting the first sign change), or max_theta
// if psi' stays positive. Mirrored by kb_theta_limit() in _torch_impl.py.
template <typename T>
inline __device__ T kb_theta_limit(const T *k, const T max_theta) {
    T prev = T(0);
    for (int i = 1; i <= GSPLAT_KB_LIMIT_SAMPLES; i++) {
        const T th = max_theta * T(i) / T(GSPLAT_KB_LIMIT_SAMPLES);
        if (kb_dpsi(th, k) <= T(0)) {
            T lo = prev, hi = th;
            for (int it = 0; it < GSPLAT_KB_LIMIT_BISECT; it++) {
                const T mid = T(0.5) * (lo + hi);
                if (kb_dpsi(mid, k) > T(0)) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            return lo;
        }
        prev = th;
    }
    return max_theta;
}

// Normalised KB projection: U = (u - cx) / fx, V = (v - cy) / fy and the
// normalised Jacobian Jn (row-major: Jn[0..2] = dU/d(x,y,z),
// Jn[3..5] = dV/d(x,y,z)). S is either T or KBDual<T, N>.
template <typename S>
inline __device__ void kb_proj_normalized(
    const S x,
    const S y,
    const S z,
    const S k1,
    const S k2,
    const S k3,
    const S k4,
    S &U,
    S &V,
    S *Jn
) {
    // Non-integer constants are double literals so that a double build is
    // exact; S(double constant) is folded at compile time for float.
    const S r2 = x * x + y * y;
    const auto r2_v = kb_value(r2);
    const auto z_v = kb_value(z);
    const auto theta_v = atan2(sqrt(r2_v), z_v);

    S s, g, h;
    if (theta_v < GSPLAT_KB_SMALL_THETA) {
        // Series in q = r^2 / z^2 (z > 0 here).
        const S iz = S(1.f) / z;
        const S q = r2 * iz * iz;
        s = iz * (S(1.f) + q * ((k1 - S(1.0 / 3.0)) + q * (k2 - k1 + S(0.2))));
        g = iz * iz * iz *
            ((S(2.f) * k1 - S(2.0 / 3.0)) +
             q * ((S(4.f) * (k2 - k1) + S(0.8)) +
                  q * (S(5.6) * k1 - S(10.f) * k2 + S(6.f) * k3 -
                       S(6.0 / 7.0))));
        h = iz * iz *
            (S(1.f) + q * ((S(3.f) * k1 - S(1.f)) +
                           q * (S(5.f) * (k2 - k1) + S(1.f))));
    } else {
        const S r = kb_sqrt(r2);
        const S rho2 = r2 + z * z;
        const S theta = kb_atan2(r, z);
        const S t = theta * theta;
        const S psi =
            theta * (S(1.f) + t * (k1 + t * (k2 + t * (k3 + t * k4))));
        const S dpsi =
            S(1.f) + t * (S(3.f) * k1 +
                          t * (S(5.f) * k2 +
                               t * (S(7.f) * k3 + t * S(9.f) * k4)));
        const S sc = r * z / rho2; // sin(theta) cos(theta)
        // w - sin(w) with w = 2 theta, sin(2 theta) = 2 sc
        const S w = S(2.f) * theta;
        S wms;
        if (kb_value(w) < GSPLAT_KB_SERIES_W) {
            const S w2 = w * w;
            wms = w * w2 *
                  (S(1.0 / 6.0) -
                   w2 * (S(1.0 / 120.0) -
                         w2 * (S(1.0 / 5040.0) -
                               w2 * (S(1.0 / 362880.0) -
                                     w2 * (S(1.0 / 39916800.0) -
                                           w2 * S(1.0 / 6227020800.0))))));
        } else {
            wms = w - S(2.f) * sc;
        }
        const S n =
            S(-0.5f) * wms +
            t * (k1 * (S(3.f) * sc - theta) +
                 t * (k2 * (S(5.f) * sc - theta) +
                      t * (k3 * (S(7.f) * sc - theta) +
                           t * k4 * (S(9.f) * sc - theta))));
        s = psi / r;
        g = n / (r * r2);
        h = dpsi / rho2;
    }
    U = s * x;
    V = s * y;
    const S xyg = x * y * g;
    Jn[0] = s + x * x * g;
    Jn[1] = xyg;
    Jn[2] = -(x * h);
    Jn[3] = xyg;
    Jn[4] = s + y * y * g;
    Jn[5] = -(y * h);
}

// Forward. Returns false if the Gaussian has to be culled (theta beyond the
// limit). theta_limit comes from kb_theta_limit().
template <typename T>
inline __device__ bool kb_proj(
    // inputs
    const vec3<T> mean3d,
    const mat3<T> cov3d,
    const T fx,
    const T fy,
    const T cx,
    const T cy,
    const T *k, // [4]
    const T theta_limit,
    const uint32_t width,
    const uint32_t height,
    // outputs
    mat2<T> &cov2d,
    vec2<T> &mean2d
) {
    const T x = mean3d[0], y = mean3d[1], z = mean3d[2];
    const T theta = atan2(sqrt(x * x + y * y), z);
    if (!(theta < theta_limit)) {
        return false;
    }
    T U, V, Jn[6];
    kb_proj_normalized<T>(x, y, z, k[0], k[1], k[2], k[3], U, V, Jn);
    // mat3x2 is 3 columns x 2 rows.
    mat3x2<T> J = mat3x2<T>(
        fx * Jn[0],
        fy * Jn[3], // 1st column
        fx * Jn[1],
        fy * Jn[4], // 2nd column
        fx * Jn[2],
        fy * Jn[5] // 3rd column
    );
    cov2d = J * cov3d * glm::transpose(J);
    mean2d = vec2<T>({fx * U + cx, fy * V + cy});
    return true;
}

template <typename T>
inline __device__ void kb_proj_vjp(
    // fwd inputs
    const vec3<T> mean3d,
    const mat3<T> cov3d,
    const T fx,
    const T fy,
    const T cx,
    const T cy,
    const T *k, // [4]
    const uint32_t width,
    const uint32_t height,
    // grad outputs
    const mat2<T> v_cov2d,
    const vec2<T> v_mean2d,
    // grad inputs
    vec3<T> &v_mean3d,
    mat3<T> &v_cov3d,
    mat3<T> &v_K,
    T *v_k // [4]
) {
    typedef KBDual<T, GSPLAT_KB_N_TANGENTS> D;
    const D x = D::seed(mean3d[0], 0);
    const D y = D::seed(mean3d[1], 1);
    const D z = D::seed(mean3d[2], 2);
    const D k1 = D::seed(k[0], 3);
    const D k2 = D::seed(k[1], 4);
    const D k3 = D::seed(k[2], 5);
    const D k4 = D::seed(k[3], 6);
    D U, V, Jn[6];
    kb_proj_normalized<D>(x, y, z, k1, k2, k3, k4, U, V, Jn);

    // mat3x2 is 3 columns x 2 rows.
    mat3x2<T> J = mat3x2<T>(
        fx * Jn[0].v,
        fy * Jn[3].v, // 1st column
        fx * Jn[1].v,
        fy * Jn[4].v, // 2nd column
        fx * Jn[2].v,
        fy * Jn[5].v // 3rd column
    );

    // cov = J * V * Jt; G = df/dcov = v_cov
    // -> df/dV = Jt * G * J
    // -> df/dJ = G * J * Vt + Gt * J * V
    v_cov3d += glm::transpose(J) * v_cov2d * J;
    mat3x2<T> v_J = v_cov2d * J * glm::transpose(cov3d) +
                    glm::transpose(v_cov2d) * J * cov3d;

    // Weights of the normalised quantities (v_J[col][row]).
    const T wU = fx * v_mean2d[0];
    const T wV = fy * v_mean2d[1];
    T wJ[6];
    GSPLAT_PRAGMA_UNROLL
    for (int j = 0; j < 3; j++) {
        wJ[j] = fx * v_J[j][0];
        wJ[3 + j] = fy * v_J[j][1];
    }
    T grad[GSPLAT_KB_N_TANGENTS];
    GSPLAT_PRAGMA_UNROLL
    for (int i = 0; i < GSPLAT_KB_N_TANGENTS; i++) {
        T acc = wU * U.d[i] + wV * V.d[i];
        GSPLAT_PRAGMA_UNROLL
        for (int j = 0; j < 6; j++) {
            acc += wJ[j] * Jn[j].d[i];
        }
        grad[i] = acc;
    }
    v_mean3d += vec3<T>(grad[0], grad[1], grad[2]);
    v_k[0] += grad[3];
    v_k[1] += grad[4];
    v_k[2] += grad[5];
    v_k[3] += grad[6];

    // Intrinsics: u = fx U + cx, J row 0 = fx Jn[0..2]; same for v / row 1.
    v_K[0][0] += U.v * v_mean2d[0] + v_J[0][0] * Jn[0].v +
                 v_J[1][0] * Jn[1].v + v_J[2][0] * Jn[2].v;
    v_K[1][1] += V.v * v_mean2d[1] + v_J[0][1] * Jn[3].v +
                 v_J[1][1] * Jn[4].v + v_J[2][1] * Jn[5].v;
    v_K[2][0] += v_mean2d[0];
    v_K[2][1] += v_mean2d[1];
}

template <typename T>
inline __device__ void pos_world_to_cam(
    // [R, t] is the world-to-camera transformation
    const mat3<T> R,
    const vec3<T> t,
    const vec3<T> p,
    vec3<T> &p_c
) {
    p_c = R * p + t;
}

template <typename T>
inline __device__ void pos_world_to_cam_vjp(
    // fwd inputs
    const mat3<T> R,
    const vec3<T> t,
    const vec3<T> p,
    // grad outputs
    const vec3<T> v_p_c,
    // grad inputs
    mat3<T> &v_R,
    vec3<T> &v_t,
    vec3<T> &v_p
) {
    // for D = W * X, G = df/dD
    // df/dW = G * XT, df/dX = WT * G
    v_R += glm::outerProduct(v_p_c, p);
    v_t += v_p_c;
    v_p += glm::transpose(R) * v_p_c;
}

template <typename T>
inline __device__ void covar_world_to_cam(
    // [R, t] is the world-to-camera transformation
    const mat3<T> R,
    const mat3<T> covar,
    mat3<T> &covar_c
) {
    covar_c = R * covar * glm::transpose(R);
}

template <typename T>
inline __device__ void covar_world_to_cam_vjp(
    // fwd inputs
    const mat3<T> R,
    const mat3<T> covar,
    // grad outputs
    const mat3<T> v_covar_c,
    // grad inputs
    mat3<T> &v_R,
    mat3<T> &v_covar
) {
    // for D = W * X * WT, G = df/dD
    // df/dX = WT * G * W
    // df/dW
    // = G * (X * WT)T + ((W * X)T * G)T
    // = G * W * XT + (XT * WT * G)T
    // = G * W * XT + GT * W * X
    v_R += v_covar_c * R * glm::transpose(covar) +
           glm::transpose(v_covar_c) * R * covar;
    v_covar += glm::transpose(R) * v_covar_c * R;
}

template <typename T>
inline __device__ T inverse(const mat2<T> M, mat2<T> &Minv) {
    T det = M[0][0] * M[1][1] - M[0][1] * M[1][0];
    if (det <= 0.f) {
        return det;
    }
    T invDet = 1.f / det;
    Minv[0][0] = M[1][1] * invDet;
    Minv[0][1] = -M[0][1] * invDet;
    Minv[1][0] = Minv[0][1];
    Minv[1][1] = M[0][0] * invDet;
    return det;
}

template <typename T>
inline __device__ void inverse_vjp(const T Minv, const T v_Minv, T &v_M) {
    // P = M^-1
    // df/dM = -P * df/dP * P
    v_M += -Minv * v_Minv * Minv;
}

template <typename T>
inline __device__ T add_blur(const T eps2d, mat2<T> &covar, T &compensation) {
    T det_orig = covar[0][0] * covar[1][1] - covar[0][1] * covar[1][0];
    covar[0][0] += eps2d;
    covar[1][1] += eps2d;
    T det_blur = covar[0][0] * covar[1][1] - covar[0][1] * covar[1][0];
    compensation = sqrt(max(0.f, det_orig / det_blur));
    return det_blur;
}

template <typename T>
inline __device__ void add_blur_vjp(
    const T eps2d,
    const mat2<T> conic_blur,
    const T compensation,
    const T v_compensation,
    mat2<T> &v_covar
) {
    // comp = sqrt(det(covar) / det(covar_blur))

    // d [det(M)] / d M = adj(M)
    // d [det(M + aI)] / d M  = adj(M + aI) = adj(M) + a * I
    // d [det(M) / det(M + aI)] / d M
    // = (det(M + aI) * adj(M) - det(M) * adj(M + aI)) / (det(M + aI))^2
    // = adj(M) / det(M + aI) - adj(M + aI) / det(M + aI) * comp^2
    // = (adj(M) - adj(M + aI) * comp^2) / det(M + aI)
    // given that adj(M + aI) = adj(M) + a * I
    // = (adj(M + aI) - aI - adj(M + aI) * comp^2) / det(M + aI)
    // given that adj(M) / det(M) = inv(M)
    // = (1 - comp^2) * inv(M + aI) - aI / det(M + aI)
    // given det(inv(M)) = 1 / det(M)
    // = (1 - comp^2) * inv(M + aI) - aI * det(inv(M + aI))
    // = (1 - comp^2) * conic_blur - aI * det(conic_blur)

    T det_conic_blur = conic_blur[0][0] * conic_blur[1][1] -
                       conic_blur[0][1] * conic_blur[1][0];
    T v_sqr_comp = v_compensation * 0.5 / (compensation + 1e-6);
    T one_minus_sqr_comp = 1 - compensation * compensation;
    v_covar[0][0] += v_sqr_comp * (one_minus_sqr_comp * conic_blur[0][0] -
                                   eps2d * det_conic_blur);
    v_covar[0][1] += v_sqr_comp * (one_minus_sqr_comp * conic_blur[0][1]);
    v_covar[1][0] += v_sqr_comp * (one_minus_sqr_comp * conic_blur[1][0]);
    v_covar[1][1] += v_sqr_comp * (one_minus_sqr_comp * conic_blur[1][1] -
                                   eps2d * det_conic_blur);
}

template <typename T>
inline __device__ void compute_ray_transforms_aabb_vjp(
    const T *ray_transforms,
    const T *v_means2d,
    const vec3<T> v_normals,
    const mat3<T> W,
    const mat3<T> P,
    const vec3<T> cam_pos,
    const vec3<T> mean_c,
    const vec4<T> quat,
    const vec2<T> scale,
    mat3<T> &_v_ray_transforms,
    vec4<T> &v_quat,
    vec2<T> &v_scale,
    vec3<T> &v_mean
) {
    if (v_means2d[0] != 0 || v_means2d[1] != 0) {
        const T distance = ray_transforms[6] * ray_transforms[6] + ray_transforms[7] * ray_transforms[7] -
                           ray_transforms[8] * ray_transforms[8];
        const T f = 1 / (distance);
        const T dpx_dT00 = f * ray_transforms[6];
        const T dpx_dT01 = f * ray_transforms[7];
        const T dpx_dT02 = -f * ray_transforms[8];
        const T dpy_dT10 = f * ray_transforms[6];
        const T dpy_dT11 = f * ray_transforms[7];
        const T dpy_dT12 = -f * ray_transforms[8];
        const T dpx_dT30 = ray_transforms[0] * (f - 2 * f * f * ray_transforms[6] * ray_transforms[6]);
        const T dpx_dT31 = ray_transforms[1] * (f - 2 * f * f * ray_transforms[7] * ray_transforms[7]);
        const T dpx_dT32 = -ray_transforms[2] * (f + 2 * f * f * ray_transforms[8] * ray_transforms[8]);
        const T dpy_dT30 = ray_transforms[3] * (f - 2 * f * f * ray_transforms[6] * ray_transforms[6]);
        const T dpy_dT31 = ray_transforms[4] * (f - 2 * f * f * ray_transforms[7] * ray_transforms[7]);
        const T dpy_dT32 = -ray_transforms[5] * (f + 2 * f * f * ray_transforms[8] * ray_transforms[8]);

        _v_ray_transforms[0][0] += v_means2d[0] * dpx_dT00;
        _v_ray_transforms[0][1] += v_means2d[0] * dpx_dT01;
        _v_ray_transforms[0][2] += v_means2d[0] * dpx_dT02;
        _v_ray_transforms[1][0] += v_means2d[1] * dpy_dT10;
        _v_ray_transforms[1][1] += v_means2d[1] * dpy_dT11;
        _v_ray_transforms[1][2] += v_means2d[1] * dpy_dT12;
        _v_ray_transforms[2][0] += v_means2d[0] * dpx_dT30 + v_means2d[1] * dpy_dT30;
        _v_ray_transforms[2][1] += v_means2d[0] * dpx_dT31 + v_means2d[1] * dpy_dT31;
        _v_ray_transforms[2][2] += v_means2d[0] * dpx_dT32 + v_means2d[1] * dpy_dT32;
    }

    mat3<T> R = quat_to_rotmat(quat);
    mat3<T> v_M = P * glm::transpose(_v_ray_transforms);
    mat3<T> W_t = glm::transpose(W);
    mat3<T> v_RS = W_t * v_M;
    vec3<T> v_tn = W_t * v_normals;

    // dual visible
    vec3<T> tn = W * R[2];
    T cos = glm::dot(-tn, mean_c);
    T multiplier = cos > 0 ? 1 : -1;
    v_tn *= multiplier;

    mat3<T> v_R = mat3<T>(v_RS[0] * scale[0], v_RS[1] * scale[1], v_tn);

    quat_to_rotmat_vjp<T>(quat, v_R, v_quat);
    v_scale[0] += (T)glm::dot(v_RS[0], R[0]);
    v_scale[1] += (T)glm::dot(v_RS[1], R[1]);

    v_mean += v_RS[2];
}

} // namespace gsplat

#endif // GSPLAT_CUDA_UTILS_H
