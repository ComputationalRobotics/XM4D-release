#include <iostream>
#include <algorithm>
#include <cmath>
#include <Utils/memory.h>
#include <Dense/matmul.h>
#include <Dense/trace.h>
#include <Dense/transpose.h>
#include <Dense/matdiagmul.h>
#include <Dense/matdot.h>
#include <Dense/matdivide.h>
#include <Dense/batchedQR.h>

#include <chrono>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <limits>
#include <Optimization/optimization.h>

using namespace std::chrono;


// =============================================================
// Mixed manifold helpers: sphere on selected indices + positive manifold
// =============================================================

template <typename T>
__global__ void MaskedMulKernel(T* out, const T* a, const T* mask, size_s n) {
    size_s i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] * mask[i];
    }
}

template <typename T>
__global__ void MixedRgradKernel(T* out, const T* s, const T* grad, const T* mask, size_s n) {
    size_s i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (mask[i] > 0.5) {
            out[i] = grad[i];
        } else {
            out[i] = s[i] * s[i] * grad[i];
        }
    }
}

template <typename T>
__global__ void MixedSubtractSphereKernel(T* out, const T* s, const T* mask, double alpha, size_s n) {
    size_s i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && mask[i] > 0.5) {
        out[i] -= alpha * s[i];
    }
}

template <typename T>
__global__ void MixedRetractionKernel(T* out, const T* s, const T* step, const T* mask, size_s n, double lr) {
    size_s i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (mask[i] > 0.5) {
            out[i] = s[i] + lr * step[i];
        } else {
            out[i] = s[i] * exp(lr * step[i] / s[i]);
        }
    }
}

// Sphere-only retraction: x <- x + lr * eta for mask == 1.
template <typename T>
__global__ void SphereRetractionKernel(T* out, const T* s, const T* step, const T* mask, size_s n, double lr) {
    size_s i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && mask[i] > 0.5) {
        out[i] = s[i] + lr * step[i];
    }
}

// Positive-only retraction: x <- x .* exp(lr * eta ./ x) for mask == 0.
template <typename T>
__global__ void PositiveRetractionKernel(T* out, const T* s, const T* step, const T* mask, size_s n, double lr) {
    size_s i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && mask[i] <= 0.5) {
        out[i] = s[i] * exp(lr * step[i] / s[i]);
    }
}

template <typename T>
__global__ void MixedSphereScaleKernel(T* s, const T* mask, double scale, size_s n) {
    size_s i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && mask[i] > 0.5) {
        s[i] *= scale;
    }
}

template <typename T>
__global__ void ClampPositiveKernel(T* s, const T* mask, size_s n, double eps) {
    size_s i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && mask[i] <= 0.5) {
        if (s[i] < eps) {
            s[i] = eps;
        }
    }
}

template <typename T>
__global__ void ZeroIndexKernel(T* s, size_s n, size_s idx0) {
    if (idx0 < n && blockIdx.x == 0 && threadIdx.x == 0) {
        s[idx0] = 0;
    }
}

template <typename T>
__global__ void MixedPositiveEhessKernel(T* out, const T* s, const T* egrad, const T* ehess, const T* eta, const T* mask, size_s n) {
    size_s i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && mask[i] <= 0.5) {
        out[i] = s[i] * s[i] * ehess[i] + eta[i] * egrad[i] * s[i];
    }
}

// Positive-only Hessian (no mask): out = s^2 * ehess + eta * egrad * s.
template <typename T>
__global__ void PositiveEhessNoMaskKernel(T* out, const T* s, const T* egrad, const T* ehess, const T* eta, size_s n) {
    size_s i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = s[i] * s[i] * ehess[i] + eta[i] * egrad[i] * s[i];
    }
}

template <typename T>
__global__ void PositiveInnerPrepKernel(T* out_a, T* out_b, const T* Asv, const T* Bsv,
                                        const T* sv, const T* maskv, size_s nn) {
    size_s i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nn) {
        double keep = 1.0 - maskv[i];
        double inv = 1.0 / sv[i];
        out_a[i] = keep * Asv[i] * inv;
        out_b[i] = keep * Bsv[i] * inv;
    }
}

template <typename T>
__global__ void MixedDivideKernel(T* out, const T* in, const T* s, const T* mask, size_s n) {
    size_s i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (mask[i] > 0.5) {
            out[i] = in[i];
        } else {
            out[i] = in[i] / s[i];
        }
    }
}

// Divide only positive entries: out_i = in_i / s_i for mask == 0, else 0.
template <typename T>
__global__ void PositiveDivideKernel(T* out, const T* in, const T* s, const T* mask, size_s n) {
    size_s i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (mask[i] <= 0.5) {
            out[i] = in[i] / s[i];
        } else {
            out[i] = 0;
        }
    }
}

// Scale only positive-manifold entries: out_i = out_i * s_i^2 for mask == 0.
template <typename T>
__global__ void PositiveScaleKernel(T* out, const T* s, const T* mask, size_s n) {
    size_s i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && mask[i] <= 0.5) {
        out[i] = out[i] * s[i] * s[i];
    }
}

// Positive-only scale (no mask): out = out * s^2 for all entries.
template <typename T>
__global__ void PositiveScaleNoMaskKernel(T* out, const T* s, size_s n) {
    size_s i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = out[i] * s[i] * s[i];
    }
}

// Positive-manifold transport scaling: out_i *= s_num_i / s_den_i for mask == 0.
template <typename T>
__global__ void PositiveScaleRatioKernel(T* out, const T* s_num, const T* s_den, const T* mask, size_s n) {
    size_s i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && mask[i] <= 0.5) {
        out[i] = out[i] * (s_num[i] / s_den[i]);
    }
}

template <typename T>
__global__ void CopyKernel(T* out, const T* in, size_s n) {
    size_s i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[i];
    }
}

// Keep only sphere entries (mask == 1), zero elsewhere.
template <typename T>
__global__ void MaskedKeepSphereKernel(T* out, const T* in, const T* mask, size_s n) {
    size_s i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = (mask[i] > 0.5) ? in[i] : 0;
    }
}

// Keep only positive entries (mask == 0), zero elsewhere.
template <typename T>
__global__ void MaskedKeepPositiveKernel(T* out, const T* in, const T* mask, size_s n) {
    size_s i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = (mask[i] <= 0.5) ? in[i] : 0;
    }
}

template <typename T>
__global__ void MixedObjectiveLambdaKernel(T* out, const T* s, size_s n) {
    size_s i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        T v = s[i] * s[i] - (T)1;
        out[i] = v * v;
    }
}

template <typename T>
__global__ void MixedGradLambdaKernel(T* out, const T* s, size_s n) {
    size_s i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = (s[i] * s[i] - (T)1) * s[i];
    }
}

template <typename T>
__global__ void MixedHessLambdaKernel(T* out, const T* s, const T* su, size_s n) {
    size_s i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = ((T)3 * s[i] * s[i] - (T)1) * su[i];
    }
}

static inline double masked_dot(DeviceBlasHandle& h,
                                opt_var& tmp_a,
                                opt_var& tmp_b,
                                const opt_var& a,
                                const opt_var& b,
                                const opt_var& mask) {
    double result = 0.0;
    size_s n = a.total_size;
    MaskedMulKernel<<<(n + 255) / 256, 256>>>(tmp_a.vals, a.vals, mask.vals, n);
    MaskedMulKernel<<<(n + 255) / 256, 256>>>(tmp_b.vals, b.vals, mask.vals, n);
    CHECK_CUBLAS(cublasDdot(h.cublas_handle, n, tmp_a.vals, 1, tmp_b.vals, 1, &result));
    return result;
}

static inline void sphere_subset_normalize(DeviceBlasHandle& h,
                                           opt_var& s,
                                           const opt_var& mask,
                                           opt_var& tmp_a,
                                           opt_var& tmp_b,
                                           double sphere_norm) {
    double sTs = masked_dot(h, tmp_a, tmp_b, s, s, mask);
    if (sTs <= 0.0) return;
    double scale = sphere_norm / std::sqrt(sTs);
    size_s n = s.total_size;
    MixedSphereScaleKernel<<<(n + 255) / 256, 256>>>(s.vals, mask.vals, scale, n);
}

static inline double MixedProductManifoldInner(DeviceBlasHandle& h,
                                               opt_var& tmp_a,
                                               opt_var& tmp_b,
                                               const opt_var& AR,
                                               const opt_var& BR,
                                               const opt_var& As,
                                               const opt_var& Bs,
                                               const opt_var& s,
                                               const opt_var& mask,
                                               double sphere_norm) {
    double result1 = 0.0;
    CHECK_CUBLAS(cublasDdot(h.cublas_handle, AR.total_size, AR.vals, 1, BR.vals, 1, &result1));

    double sphere_part = masked_dot(h, tmp_a, tmp_b, As, Bs, mask);
    double sphere_scale = (sphere_norm > 0.0) ? (1.0 / (sphere_norm * sphere_norm)) : 1.0;
    sphere_part *= sphere_scale;

    // positive part: sum (As_i/s_i) * (Bs_i/s_i) where mask == 0
    size_s n = As.total_size;
    MaskedMulKernel<<<(n + 255) / 256, 256>>>(tmp_a.vals, As.vals, mask.vals, n);
    MaskedMulKernel<<<(n + 255) / 256, 256>>>(tmp_b.vals, Bs.vals, mask.vals, n);

    double pos_part = 0.0;
    PositiveInnerPrepKernel<<<(n + 255) / 256, 256>>>(tmp_a.vals, tmp_b.vals, As.vals, Bs.vals, s.vals, mask.vals, n);
    CHECK_CUBLAS(cublasDdot(h.cublas_handle, n, tmp_a.vals, 1, tmp_b.vals, 1, &pos_part));

    return result1 + sphere_part + pos_part;
}

void XMtrustregion_mixed(opt_var &C,
                         const opt_var &R0,
                         const opt_var &s0,
                         opt_var &R_result,
                         opt_var &s_result,
                         const double lam,
                         const size_s* sphere_indices,
                         size_s m,
                         double &gradtol,
                         double linesearch_step,
                         opt_var &v,
                         double* primal_value,
                         const double maxtime) {

    DeviceBlasHandle CUOPT_blas_handle;
    CUOPT_blas_handle.activate();

    size_s o = R0.dimensions[1];
    size_s n = C.dimensions[0] / 3;

    if (m < 1 || sphere_indices == nullptr) {
        m = 1;
    }
    if (m > n) {
        m = n;
    }
    std::vector<datatype> mask_h(n, 0.0);
    for (size_s i = 0; i < m; ++i) {
        const size_s idx = sphere_indices[i];
        if (idx < n) {
            mask_h[idx] = 1.0;
        }
    }
    double sphere_norm = std::sqrt(static_cast<double>(m));

    opt_var sphere_mask({n});
    sphere_mask.SynchronizeHostToDevice(mask_h.data());

    // Manifold dimension: Stiefel blocks + sphere S^{m-1} + positive entries
    size_s dim = n * (3 * o - 6) + (m - 1) + (n - m);
    double delta_bar = sqrt((double)dim);
    double delta = delta_bar / 8.0;

    opt_var R({3 * n, o});
    CHECK_CUDA(cudaMemcpy(R.vals, R0.vals, R.total_size * sizeof(datatype), cudaMemcpyDeviceToDevice));

    if ((size_s)s0.total_size != n) {
        std::cerr << "[XMtrustregion_mixed] ERROR: s0 must have length n = " << n
                  << " but got " << s0.total_size << std::endl;
        *primal_value = -1;
        return;
    }

    opt_var s({n});
    CHECK_CUDA(cudaMemcpy(s.vals, s0.vals, s.total_size * sizeof(datatype), cudaMemcpyDeviceToDevice));
    // Do not anchor the first frame to 1.

    opt_var tmp_a({n});
    opt_var tmp_b({n});
    sphere_subset_normalize(CUOPT_blas_handle, s, sphere_mask, tmp_a, tmp_b, sphere_norm);
    ClampPositiveKernel<<<(n + 255) / 256, 256>>>(s.vals, sphere_mask.vals, n, 1e-6);

    // s.print();
    // R.print();

    opt_var sR({3 * n, o});
    opt_var dfdsR({3 * n, o});
    dnmat_mul_spdiag_batch(sR, R, s, 3);

    opt_var grad_r({3 * n, o});
    opt_var grad_s({n});
    opt_var CsR({3 * n, o});
    opt_var slamobj({n});

    auto objc = [&C, &CUOPT_blas_handle, &CsR, &slamobj, &lam, n](opt_var& sR_point, opt_var& s_point) {
        double result = 0.0;
        DnMatDnMat(CUOPT_blas_handle, CsR, C, sR_point);
        CHECK_CUBLAS(cublasDdot(CUOPT_blas_handle.cublas_handle,
                                sR_point.total_size,
                                CsR.vals, 1,
                                sR_point.vals, 1,
                                &result));
        if (lam > 0.0) {
            MixedObjectiveLambdaKernel<<<(n + 255) / 256, 256>>>(slamobj.vals, s_point.vals, n);
            double result_lambda = 0.0;
            CHECK_CUBLAS(cublasDasum(CUOPT_blas_handle.cublas_handle,
                                     slamobj.total_size,
                                     slamobj.vals, 1,
                                     &result_lambda));
            result += lam * result_lambda;
        }
        return result;
    };

    opt_var slamgrad({n});
    auto grad = [&C, &CUOPT_blas_handle, &grad_r, &grad_s, &dfdsR, &slamgrad, &lam, n](opt_var& R_point,
                                                                                      opt_var& s_point,
                                                                                      opt_var& sR_point) {
        DnMatDnMat(CUOPT_blas_handle, dfdsR, C, sR_point, CUBLAS_OP_N, CUBLAS_OP_N, 2.0, 0.0);
        dnmat_mul_spdiag_batch(grad_r, dfdsR, s_point, 3);
        dnmat_Ddot_colomn_batch(grad_s, dfdsR, R_point, 3);
        if (lam > 0.0) {
            MixedGradLambdaKernel<<<(n + 255) / 256, 256>>>(slamgrad.vals, s_point.vals, n);
            double glam = 4.0 * lam;
            CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle,
                                     slamgrad.total_size,
                                     &glam,
                                     slamgrad.vals, 1,
                                     grad_s.vals, 1));
        }
    };

    opt_var hr({3 * n, o});
    opt_var hrT({o, 3 * n});
    opt_var hs({n});
    opt_var sRu({3 * n, o});
    opt_var suR({3 * n, o});
    opt_var CsRu({3 * n, o});
    opt_var suCsR({3 * n, o});
    opt_var sCsRu({3 * n, o});
    opt_var CsRudotR({n});
    opt_var CsRdotRu({n});
    opt_var slamhess({n});

    auto ehess = [&C, &CUOPT_blas_handle, &hr, &hs,
                  &sR, &sRu, &suR, &CsR, &CsRu,
                  &suCsR, &sCsRu, &CsRudotR, &CsRdotRu,
                  &slamhess, &lam, n](opt_var& R_point,
                                      opt_var& s_point,
                                      opt_var& Ru_point,
                                      opt_var& su_point) {
        dnmat_mul_spdiag_batch(sRu, Ru_point, s_point, 3);
        dnmat_mul_spdiag_batch(suR, R_point, su_point, 3);

        double alpha = 1.0;
        CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle,
                                 sRu.total_size, &alpha, suR.vals, 1, sRu.vals, 1));

        DnMatDnMat(CUOPT_blas_handle, CsRu, C, sRu, CUBLAS_OP_N, CUBLAS_OP_N, 2.0, 0.0);
        dnmat_mul_spdiag_batch(suCsR, CsR, su_point, 3);
        dnmat_mul_spdiag_batch(sCsRu, CsRu, s_point, 3);

        CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, hr.total_size, sCsRu.vals, 1, hr.vals, 1));
        CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, hr.total_size, &alpha, suCsR.vals, 1, hr.vals, 1));

        dnmat_Ddot_colomn_batch(CsRudotR, CsRu, R_point, 3);
        dnmat_Ddot_colomn_batch(CsRdotRu, CsR, Ru_point, 3);

        CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, hs.total_size, CsRudotR.vals, 1, hs.vals, 1));
        CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, hs.total_size, &alpha, CsRdotRu.vals, 1, hs.vals, 1));
        if (lam > 0.0) {
            MixedHessLambdaKernel<<<(n + 255) / 256, 256>>>(slamhess.vals, s_point.vals, su_point.vals, n);
            double hlam = 4.0 * lam;
            CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle,
                                     slamhess.total_size,
                                     &hlam,
                                     slamhess.vals, 1,
                                     hs.vals, 1));
        }
        // No anchoring of the first frame.
    };

    opt_var RTgradR({3, 3 * n});
    opt_var RTgradR_sym({3, 3 * n});
    opt_var rhr({o, 3 * n});
    opt_var rhs({n});
    opt_var RTrhr({3, 3 * n});
    opt_var RTrhr_sym({3, 3 * n});
    opt_var s_sphere_buf({n});
    opt_var s_pos_buf({n});
    opt_var pos_s({n});
    opt_var pos_egrad({n});
    opt_var pos_ehess({n});
    opt_var pos_eta({n});

    auto ehess2rhess = [&CUOPT_blas_handle,
                        &RTgradR, &RTgradR_sym,
                        &rhr, &rhs,
                        &RTrhr, &RTrhr_sym,
                        &sphere_mask, &tmp_a, &tmp_b, sphere_norm,
                        &s_sphere_buf, &s_pos_buf,
                        &pos_s, &pos_egrad, &pos_ehess, &pos_eta](opt_var& ehessR,
                                                    opt_var& ehesss,
                                                    opt_var& egradR,
                                                    opt_var& egrads,
                                                    opt_var& R_point,
                                                    opt_var& s_point,
                                                    opt_var& Ru_point,
                                                    opt_var& su_point) {

        size_s nb = R_point.dimensions[1] / 3;

        DnMatDnMatBatch(CUOPT_blas_handle, RTgradR, R_point, egradR,
                        3, R_point.dimensions[0], 3, nb, CUBLAS_OP_T, CUBLAS_OP_N);
        symBatched(RTgradR_sym.vals, RTgradR.vals, RTgradR.dimensions[0], nb);

        CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, ehessR.total_size, ehessR.vals, 1, rhr.vals, 1));
        CHECK_CUDA(cudaDeviceSynchronize());

        DnMatDnMatBatch(CUOPT_blas_handle, rhr, Ru_point, RTgradR_sym,
                        Ru_point.dimensions[0], 3, 3, nb, CUBLAS_OP_N, CUBLAS_OP_N, -1.0, 1.0);

        DnMatDnMatBatch(CUOPT_blas_handle, RTrhr, R_point, rhr,
                        3, R_point.dimensions[0], 3, nb, CUBLAS_OP_T, CUBLAS_OP_N);
        symBatched(RTrhr_sym.vals, RTrhr.vals, RTrhr.dimensions[0], nb);
        DnMatDnMatBatch(CUOPT_blas_handle, rhr, R_point, RTrhr_sym,
                        R_point.dimensions[0], 3, 3, nb, CUBLAS_OP_N, CUBLAS_OP_N, -1.0, 1.0);

        size_s nn = s_point.total_size;

        // ====================
        // Sphere-part (mask == 1)
        // Compute rhess using Manopt's unit-sphere formula, then map back to s-space.
        // x = s / r, u = su / r, g_x = r * egrads, H_x = r * ehesss
        // rhess_x = proj(x, H_x - (x^T g_x) * u), rhess_s = r * rhess_x
        // ====================
        CHECK_CUDA(cudaMemset(s_sphere_buf.vals, 0, nn * sizeof(datatype)));
        CHECK_CUDA(cudaMemset(pos_egrad.vals, 0, nn * sizeof(datatype)));
        CHECK_CUDA(cudaMemset(pos_ehess.vals, 0, nn * sizeof(datatype)));
        CHECK_CUDA(cudaMemset(pos_eta.vals, 0, nn * sizeof(datatype)));

        // H_x = r * ehesss (sphere part)
        MaskedKeepSphereKernel<<<(nn + 255) / 256, 256>>>(s_sphere_buf.vals, ehesss.vals, sphere_mask.vals, nn);
        MixedSphereScaleKernel<<<(nn + 255) / 256, 256>>>(s_sphere_buf.vals, sphere_mask.vals, sphere_norm, nn);

        // x = s / r
        MaskedKeepSphereKernel<<<(nn + 255) / 256, 256>>>(pos_ehess.vals, s_point.vals, sphere_mask.vals, nn);
        if (sphere_norm > 0.0) {
            MixedSphereScaleKernel<<<(nn + 255) / 256, 256>>>(pos_ehess.vals, sphere_mask.vals, 1.0 / sphere_norm, nn);
        }

        // u = su / r
        MaskedKeepSphereKernel<<<(nn + 255) / 256, 256>>>(pos_eta.vals, su_point.vals, sphere_mask.vals, nn);
        if (sphere_norm > 0.0) {
            MixedSphereScaleKernel<<<(nn + 255) / 256, 256>>>(pos_eta.vals, sphere_mask.vals, 1.0 / sphere_norm, nn);
        }

        // g_x = r * egrads
        MaskedKeepSphereKernel<<<(nn + 255) / 256, 256>>>(pos_egrad.vals, egrads.vals, sphere_mask.vals, nn);
        MixedSphereScaleKernel<<<(nn + 255) / 256, 256>>>(pos_egrad.vals, sphere_mask.vals, sphere_norm, nn);

        double xTg = masked_dot(CUOPT_blas_handle, tmp_a, tmp_b, pos_ehess, pos_egrad, sphere_mask);
        MixedSubtractSphereKernel<<<(nn + 255) / 256, 256>>>(s_sphere_buf.vals, pos_eta.vals, sphere_mask.vals, xTg, nn);

        double xTr = masked_dot(CUOPT_blas_handle, tmp_a, tmp_b, pos_ehess, s_sphere_buf, sphere_mask);
        MixedSubtractSphereKernel<<<(nn + 255) / 256, 256>>>(s_sphere_buf.vals, pos_ehess.vals, sphere_mask.vals, xTr, nn);

        // Map back to s-space: rhess_s = r * rhess_x
        MixedSphereScaleKernel<<<(nn + 255) / 256, 256>>>(s_sphere_buf.vals, sphere_mask.vals, sphere_norm, nn);

        // ====================
        // Positive-part (mask == 0)
        // Mirrors positivefactory.m:
        // rhess = X.*ehess.*X + (eta.*egrad.*X)
        // ====================
        CHECK_CUDA(cudaMemset(pos_s.vals, 0, nn * sizeof(datatype)));
        CHECK_CUDA(cudaMemset(pos_egrad.vals, 0, nn * sizeof(datatype)));
        CHECK_CUDA(cudaMemset(pos_ehess.vals, 0, nn * sizeof(datatype)));
        CHECK_CUDA(cudaMemset(pos_eta.vals, 0, nn * sizeof(datatype)));
        MaskedKeepPositiveKernel<<<(nn + 255) / 256, 256>>>(pos_s.vals, s_point.vals, sphere_mask.vals, nn);
        MaskedKeepPositiveKernel<<<(nn + 255) / 256, 256>>>(pos_egrad.vals, egrads.vals, sphere_mask.vals, nn);
        MaskedKeepPositiveKernel<<<(nn + 255) / 256, 256>>>(pos_ehess.vals, ehesss.vals, sphere_mask.vals, nn);
        MaskedKeepPositiveKernel<<<(nn + 255) / 256, 256>>>(pos_eta.vals, su_point.vals, sphere_mask.vals, nn);
        PositiveEhessNoMaskKernel<<<(nn + 255) / 256, 256>>>(s_pos_buf.vals, pos_s.vals, pos_egrad.vals, pos_ehess.vals, pos_eta.vals, nn);

        // Combine sphere + positive parts into rhs.
        CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, rhs.total_size, s_sphere_buf.vals, 1, rhs.vals, 1));
        double one = 1.0;
        CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, rhs.total_size, &one, s_pos_buf.vals, 1, rhs.vals, 1));
        // No anchoring of the first frame.
    };

    opt_var rgrad_r({o, 3 * n});
    opt_var rgrad_s({n});
    opt_var proj_R({o, 3 * n});
    opt_var proj_s({n});

    auto projection = [&CUOPT_blas_handle,
                       &RTgradR, &RTgradR_sym,
                       &rgrad_r, &rgrad_s,
                       &sphere_mask, &tmp_a, &tmp_b, sphere_norm,
                       &s_sphere_buf, &s_pos_buf,
                       &pos_s, &pos_egrad](opt_var& R_point,
                                                   opt_var& s_point,
                                                   opt_var& R_gradient,
                                                   opt_var& s_gradient) {

        size_s nb = R_point.dimensions[1] / 3;

        DnMatDnMatBatch(CUOPT_blas_handle, RTgradR, R_point, R_gradient,
                        3, R_point.dimensions[0], 3, nb, CUBLAS_OP_T, CUBLAS_OP_N);
        symBatched(RTgradR_sym.vals, RTgradR.vals, RTgradR.dimensions[0], nb);

        CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, R_gradient.total_size, R_gradient.vals, 1, rgrad_r.vals, 1));
        CHECK_CUDA(cudaDeviceSynchronize());

        DnMatDnMatBatch(CUOPT_blas_handle, rgrad_r, R_point, RTgradR_sym,
                        R_point.dimensions[0], 3, 3, nb, CUBLAS_OP_N, CUBLAS_OP_N, -1.0, 1.0);

        size_s nn = s_point.total_size;

        // ====================
        // Sphere-part (mask == 1)
        // Mirrors spherefactory.m: rgrad = proj(x, egrad)
        // ====================
        CHECK_CUDA(cudaMemset(s_sphere_buf.vals, 0, nn * sizeof(datatype)));
        MaskedKeepSphereKernel<<<(nn + 255) / 256, 256>>>(s_sphere_buf.vals, s_gradient.vals, sphere_mask.vals, nn);
        double sTs = masked_dot(CUOPT_blas_handle, tmp_a, tmp_b, s_point, s_point, sphere_mask);
        if (sTs > 0.0) {
            double sTg = masked_dot(CUOPT_blas_handle, tmp_a, tmp_b, s_point, s_gradient, sphere_mask);
            double alpha = sTg / sTs;
            MixedSubtractSphereKernel<<<(nn + 255) / 256, 256>>>(s_sphere_buf.vals, s_point.vals, sphere_mask.vals, alpha, nn);
        }
        double sphere_scale = sphere_norm * sphere_norm;
        MixedSphereScaleKernel<<<(nn + 255) / 256, 256>>>(s_sphere_buf.vals, sphere_mask.vals, sphere_scale, nn);

        // ====================
        // Positive-part (mask == 0)
        // Mirrors positivefactory.m: rgrad = X.*egrad.*X
        // ====================
        CHECK_CUDA(cudaMemset(pos_s.vals, 0, nn * sizeof(datatype)));
        CHECK_CUDA(cudaMemset(pos_egrad.vals, 0, nn * sizeof(datatype)));
        MaskedKeepPositiveKernel<<<(nn + 255) / 256, 256>>>(pos_s.vals, s_point.vals, sphere_mask.vals, nn);
        MaskedKeepPositiveKernel<<<(nn + 255) / 256, 256>>>(pos_egrad.vals, s_gradient.vals, sphere_mask.vals, nn);
        PositiveScaleNoMaskKernel<<<(nn + 255) / 256, 256>>>(pos_egrad.vals, pos_s.vals, nn);
        CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, s_pos_buf.total_size, pos_egrad.vals, 1, s_pos_buf.vals, 1));

        // Combine sphere + positive parts into rgrad_s.
        CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, rgrad_s.total_size, s_sphere_buf.vals, 1, rgrad_s.vals, 1));
        double one = 1.0;
        CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, rgrad_s.total_size, &one, s_pos_buf.vals, 1, rgrad_s.vals, 1));
    };

    auto tangent_project = [&CUOPT_blas_handle,
                             &RTgradR, &RTgradR_sym,
                             &sphere_mask, &tmp_a, &tmp_b,
                             &s_sphere_buf, &s_pos_buf](opt_var& R_point,
                                                          opt_var& s_point,
                                                          opt_var& R_gradient,
                                                          opt_var& s_gradient,
                                                          opt_var& out_R,
                                                          opt_var& out_s) {
        size_s nb = R_point.dimensions[1] / 3;

        DnMatDnMatBatch(CUOPT_blas_handle, RTgradR, R_point, R_gradient,
                        3, R_point.dimensions[0], 3, nb, CUBLAS_OP_T, CUBLAS_OP_N);
        symBatched(RTgradR_sym.vals, RTgradR.vals, RTgradR.dimensions[0], nb);

        CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, R_gradient.total_size, R_gradient.vals, 1, out_R.vals, 1));
        CHECK_CUDA(cudaDeviceSynchronize());

        DnMatDnMatBatch(CUOPT_blas_handle, out_R, R_point, RTgradR_sym,
                        R_point.dimensions[0], 3, 3, nb, CUBLAS_OP_N, CUBLAS_OP_N, -1.0, 1.0);

        size_s nn = s_point.total_size;

        // Sphere-part (mask == 1): proj(x, eta).
        CHECK_CUDA(cudaMemset(s_sphere_buf.vals, 0, nn * sizeof(datatype)));
        MaskedKeepSphereKernel<<<(nn + 255) / 256, 256>>>(s_sphere_buf.vals, s_gradient.vals, sphere_mask.vals, nn);
        double sTs = masked_dot(CUOPT_blas_handle, tmp_a, tmp_b, s_point, s_point, sphere_mask);
        if (sTs > 0.0) {
            double sTg = masked_dot(CUOPT_blas_handle, tmp_a, tmp_b, s_point, s_gradient, sphere_mask);
            double alpha = sTg / sTs;
            MixedSubtractSphereKernel<<<(nn + 255) / 256, 256>>>(s_sphere_buf.vals, s_point.vals, sphere_mask.vals, alpha, nn);
        }

        // Positive-part (mask == 0): projection is identity (positivefactory.m).
        CHECK_CUDA(cudaMemset(s_pos_buf.vals, 0, nn * sizeof(datatype)));
        MaskedKeepPositiveKernel<<<(nn + 255) / 256, 256>>>(s_pos_buf.vals, s_gradient.vals, sphere_mask.vals, nn);

        // Combine sphere + positive parts into out_s.
        CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, out_s.total_size, s_sphere_buf.vals, 1, out_s.vals, 1));
        double one = 1.0;
        CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, out_s.total_size, &one, s_pos_buf.vals, 1, out_s.vals, 1));
        // No anchoring of the first frame.
    };

    opt_var new_R_T({o, 3 * n});
    opt_var new_R({3 * n, o});
    opt_var new_s({n});

    auto retraction = [&CUOPT_blas_handle, &new_R_T, &new_s, &sphere_mask, &tmp_a, &tmp_b, sphere_norm](opt_var& R_point,
                                                                                                       opt_var& s_point,
                                                                                                       opt_var& rR_step,
                                                                                                       opt_var& rs_step,
                                                                                                       double lr) {
        CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, R_point.total_size, &lr, rR_step.vals, 1, R_point.vals, 1));
        batchedQR(new_R_T, R_point, 3, s_point.total_size);

        size_s nn = s_point.total_size;
        // Start from current point for product-manifold retraction.
        CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, new_s.total_size, s_point.vals, 1, new_s.vals, 1));
        SphereRetractionKernel<<<(nn + 255) / 256, 256>>>(new_s.vals, s_point.vals, rs_step.vals, sphere_mask.vals, nn, lr);
        PositiveRetractionKernel<<<(nn + 255) / 256, 256>>>(new_s.vals, s_point.vals, rs_step.vals, sphere_mask.vals, nn, lr);
        sphere_subset_normalize(CUOPT_blas_handle, new_s, sphere_mask, tmp_a, tmp_b, sphere_norm);
        ClampPositiveKernel<<<(nn + 255) / 256, 256>>>(new_s.vals, sphere_mask.vals, nn, 1e-6);
    };

    opt_var R_T({o, 3 * n});
    transpose(R_T.vals, R.vals, R.dimensions[0], R.dimensions[1]);

    opt_var grad_r_T({o, 3 * n});
    opt_var sR_new({3 * n, o});

    std::cout << "start linesearch" << std::endl;

    printf("%f",objc(sR, s));

    if (linesearch_step != 0) {
        double f0 = objc(sR, s);
        double alpha = linesearch_step;

        opt_var decent_dir_R({3 * n, o});
        CHECK_CUDA(cudaMemset(decent_dir_R.vals, 0, decent_dir_R.total_size * sizeof(datatype)));
        CHECK_CUDA(cudaMemcpyAsync(decent_dir_R.vals + 3 * n * (o - 1),
                                   v.vals, sizeof(datatype) * v.total_size,
                                   cudaMemcpyDeviceToDevice));

        opt_var decent_dir_R_T({o, 3 * n});
        transpose(decent_dir_R_T.vals, decent_dir_R.vals,
                  decent_dir_R.dimensions[0], decent_dir_R.dimensions[1]);

        opt_var decent_dir_s({n});
        CHECK_CUDA(cudaMemset(decent_dir_s.vals, 0, decent_dir_s.total_size * sizeof(datatype)));

        retraction(R_T, s, decent_dir_R_T, decent_dir_s, -alpha);

        transpose(R_T.vals, R.vals, R.dimensions[0], R.dimensions[1]);
        transpose(new_R.vals, new_R_T.vals, new_R_T.dimensions[0], new_R_T.dimensions[1]);

        dnmat_mul_spdiag_batch(sR_new, new_R, s, 3);
        double f = objc(sR_new, s);

        while (f > f0) {
            alpha = alpha / 2;
            retraction(R_T, s, decent_dir_R_T, decent_dir_s, -alpha);
            transpose(R_T.vals, R.vals, R.dimensions[0], R.dimensions[1]);
            transpose(new_R.vals, new_R_T.vals, new_R_T.dimensions[0], new_R_T.dimensions[1]);
            dnmat_mul_spdiag_batch(sR_new, new_R, s, 3);
            f = objc(sR_new, s);

            if (alpha < 1e-20) {
                printf("linesearch failed! BM stopped! \n");
                *primal_value = -1;
                return;
            }
        }

        if (f0 - f > 0) {
            printf("linesearch decrease %1.3e\n", f0 - f);
            CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, new_R_T.total_size, new_R_T.vals, 1, R_T.vals, 1));
            CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, new_R.total_size, new_R.vals, 1, R.vals, 1));
        } else {
            printf("linesearch failed! BM stopped! \n");
            *primal_value = -1;
            return;
        }
    }

    size_s max_inner_iter = 20000;
    size_s max_outer_iter = 1000;
    size_s totalite = 0;

    double* loss = new double[max_outer_iter + 1];
    double* gradnorm = new double[max_outer_iter + 1];
    loss[0] = objc(sR, s);

    int endreason = 6;
    int trstatus = 4;
    int shrink_count = 0;
    size_s neg_curv_count = 0;
    size_s tr_boundary_count = 0;
    size_s accept_count = 0;
    size_s reject_count = 0;

    opt_var r_R({o, 3 * n});
    opt_var r_s({n});
    opt_var p_R({o, 3 * n});
    opt_var p_s({n});
    opt_var p_R_T({3 * n, o});
    opt_var rgrad_r_T({3 * n, o});
    opt_var rsds({n});
    opt_var rhsds({n});
    opt_var vsds({n});
    opt_var hvsds({n});
    opt_var rgradsds({n});

    auto start = high_resolution_clock::now();


    for (size_s k = 0; k < max_outer_iter; ++k) {

        opt_var v_R({o, 3 * n});
        opt_var v_s({n});
        opt_var hv_R({o, 3 * n});
        opt_var hv_s({n});

        CHECK_CUDA(cudaMemset(v_R.vals, 0, v_R.total_size * sizeof(datatype)));
        CHECK_CUDA(cudaMemset(v_s.vals, 0, v_s.total_size * sizeof(datatype)));
        CHECK_CUDA(cudaMemset(hv_R.vals, 0, hv_R.total_size * sizeof(datatype)));
        CHECK_CUDA(cudaMemset(hv_s.vals, 0, hv_s.total_size * sizeof(datatype)));


        opt_var bestR(R);
        opt_var bests(s);
        double bestloss = loss[k];

        double* rdotr = new double[max_inner_iter];
        rdotr[0] = 0;

        grad(R, s, sR);

        // No anchoring of the first frame.

        transpose(grad_r_T.vals, grad_r.vals, grad_r.dimensions[0], grad_r.dimensions[1]);
        transpose(R_T.vals, R.vals, R.dimensions[0], R.dimensions[1]);

        projection(R_T, s, grad_r_T, grad_s);

        // No anchoring of the first frame.


        CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, rgrad_r.total_size, rgrad_r.vals, 1, r_R.vals, 1));
        CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, rgrad_s.total_size, rgrad_s.vals, 1, r_s.vals, 1));
        CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, rgrad_r.total_size, rgrad_r.vals, 1, p_R.vals, 1));
        CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, rgrad_s.total_size, rgrad_s.vals, 1, p_s.vals, 1));

        rdotr[0] = MixedProductManifoldInner(CUOPT_blas_handle, tmp_a, tmp_b, r_R, r_R, r_s, r_s, s, sphere_mask, sphere_norm);
        gradnorm[k] = sqrt(rdotr[0]);

        if (k == 0) {
            double cap = std::min(3.0 * delta_bar, 1000.0);
            double target = 0.001 * gradnorm[k];
            if (std::isfinite(target) && target > delta_bar) {
                delta_bar = std::min(target, cap);
                delta = std::min(delta_bar / 8.0, cap / 8.0);
            }
        }

        if (!std::isfinite(loss[k]) || !std::isfinite(gradnorm[k])) {
            printf("Terminate because of NaN/Inf in loss or gradient\n");
            break;
        }

        if (endreason == 5) break;

        if (gradnorm[k] < gradtol) {
            printf("Terminate because of small gradient norm\n");
            gradtol /= 10;
            break;
        }

        auto middle_end_time = high_resolution_clock::now();
        auto duration = duration_cast<seconds>(middle_end_time - start);
        if (duration.count() > maxtime) {
            printf("Terminate because of time limit\n");
            break;
        }

        endreason = 6;
        trstatus = 4;

        double* vdotv_record = new double[max_inner_iter];
        double* vdotp_record = new double[max_inner_iter];
        double* pdotp_record = new double[max_inner_iter];
        vdotv_record[0] = 0.0;
        vdotp_record[0] = 0.0;
        pdotp_record[0] = rdotr[0];

        DnMatDnMat(CUOPT_blas_handle, CsR, C, sR, CUBLAS_OP_N, CUBLAS_OP_N, 2.0, 0.0);
        CHECK_CUDA(cudaDeviceSynchronize());

        size_s i = 0;
        for (i = 0; i < max_inner_iter; ++i) {

            transpose(p_R_T.vals, p_R.vals, p_R.dimensions[0], p_R.dimensions[1]);

            ehess(R, s, p_R_T, p_s);

            transpose(hrT.vals, hr.vals, hr.dimensions[0], hr.dimensions[1]);
            ehess2rhess(hrT, hs, grad_r_T, grad_s, R_T, s, p_R, p_s);


            double denom = MixedProductManifoldInner(CUOPT_blas_handle, tmp_a, tmp_b, p_R, rhr, p_s, rhs, s, sphere_mask, sphere_norm);
            if (!std::isfinite(denom) || denom <= 0.0) {
                double vdotv = vdotv_record[i];
                double vdotp = vdotp_record[i];
                double pdotp = pdotp_record[i];
                double sqrt_val = sqrt(vdotp * vdotp + pdotp * (delta * delta - vdotv));
                double tau = (vdotp + sqrt_val) / pdotp;

                double neg_tau = -tau;
                CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, v_R.total_size, &neg_tau, p_R.vals, 1, v_R.vals, 1));
                CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, v_s.total_size, &neg_tau, p_s.vals, 1, v_s.vals, 1));

                CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, hv_R.total_size, &neg_tau, rhr.vals, 1, hv_R.vals, 1));
                CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, hv_s.total_size, &neg_tau, rhs.vals, 1, hv_s.vals, 1));

                endreason = 1;
                break;
            }
            double alpha = rdotr[i] / denom;
            double rp = MixedProductManifoldInner(CUOPT_blas_handle, tmp_a, tmp_b, r_R, p_R, r_s, p_s, s, sphere_mask, sphere_norm);
            if (!std::isfinite(alpha) || !std::isfinite(rp)) {
                endreason = 5;
                break;
            }

            double vdotv = vdotv_record[i];
            double pdotp = pdotp_record[i];
            double vdotp = vdotp_record[i];

            if (rdotr[i] < 1e-15) { endreason = 5; break; }

            if (alpha <= 0) {
                double sqrt_val = sqrt(vdotp * vdotp + pdotp * (delta * delta - vdotv));
                double tau = (vdotp + sqrt_val) / pdotp;

                double neg_tau = -tau;
                CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, v_R.total_size, &neg_tau, p_R.vals, 1, v_R.vals, 1));
                CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, v_s.total_size, &neg_tau, p_s.vals, 1, v_s.vals, 1));

                CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, hv_R.total_size, &neg_tau, rhr.vals, 1, hv_R.vals, 1));
                CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, hv_s.total_size, &neg_tau, rhs.vals, 1, hv_s.vals, 1));

                endreason = 1;
                break;
            }

            if (vdotv - 2 * alpha * vdotp + alpha * alpha * pdotp > delta * delta) {
                double sqrt_val = sqrt(vdotp * vdotp + pdotp * (delta * delta - vdotv));
                double tau = (vdotp + sqrt_val) / pdotp;

                double neg_tau = -tau;
                CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, v_R.total_size, &neg_tau, p_R.vals, 1, v_R.vals, 1));
                CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, v_s.total_size, &neg_tau, p_s.vals, 1, v_s.vals, 1));

                CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, hv_R.total_size, &neg_tau, rhr.vals, 1, hv_R.vals, 1));
                CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, hv_s.total_size, &neg_tau, rhs.vals, 1, hv_s.vals, 1));

                endreason = 2;
                break;
            }

            double neg_alpha = -alpha;
            CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, v_R.total_size, &neg_alpha, p_R.vals, 1, v_R.vals, 1));
            CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, v_s.total_size, &neg_alpha, p_s.vals, 1, v_s.vals, 1));

            CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, r_R.total_size, &neg_alpha, rhr.vals, 1, r_R.vals, 1));
            CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, r_s.total_size, &neg_alpha, rhs.vals, 1, r_s.vals, 1));

            CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, hv_R.total_size, &neg_alpha, rhr.vals, 1, hv_R.vals, 1));
            CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, hv_s.total_size, &neg_alpha, rhs.vals, 1, hv_s.vals, 1));

            rdotr[i + 1] = MixedProductManifoldInner(CUOPT_blas_handle, tmp_a, tmp_b, r_R, r_R, r_s, r_s, s, sphere_mask, sphere_norm);

            if (sqrt(rdotr[i + 1]) < gradnorm[k] * std::min(gradnorm[k], 0.1)) {
                endreason = 3;
                break;
            }

            double beta = rdotr[i + 1] / rdotr[i];

            CHECK_CUBLAS(cublasDscal(CUOPT_blas_handle.cublas_handle, p_R.total_size, &beta, p_R.vals, 1));
            CHECK_CUBLAS(cublasDscal(CUOPT_blas_handle.cublas_handle, p_s.total_size, &beta, p_s.vals, 1));

            double one = 1.0;
            CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, p_R.total_size, &one, r_R.vals, 1, p_R.vals, 1));
            CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, p_s.total_size, &one, r_s.vals, 1, p_s.vals, 1));

            // Project search direction back to tangent space (Manopt mdelta = tangent(mdelta))
            tangent_project(R_T, s, p_R, p_s, proj_R, proj_s);
            CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, p_R.total_size, proj_R.vals, 1, p_R.vals, 1));
            CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, p_s.total_size, proj_s.vals, 1, p_s.vals, 1));

            vdotv_record[i + 1] = vdotv - 2 * alpha * vdotp + alpha * alpha * pdotp;
            vdotp_record[i + 1] = beta * (vdotp - alpha * pdotp);
            pdotp_record[i + 1] = beta * beta * pdotp + rdotr[i + 1];
        }

        size_s inner_iter_used = i + 1;
        if (inner_iter_used > max_inner_iter) {
            inner_iter_used = max_inner_iter;
        }

        totalite += i + 1;

        // Note: Manopt returns the boundary/negative-curvature step directly; no fallback.

         double loss_qu = MixedProductManifoldInner(CUOPT_blas_handle, tmp_a, tmp_b, v_R, hv_R, v_s, hv_s, s, sphere_mask, sphere_norm) / 2.0
             + MixedProductManifoldInner(CUOPT_blas_handle, tmp_a, tmp_b, v_R, rgrad_r, v_s, rgrad_s, s, sphere_mask, sphere_norm);

        retraction(R_T, s, v_R, v_s, 1.0);

        CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, new_R_T.total_size, new_R_T.vals, 1, R_T.vals, 1));
        CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, new_s.total_size, new_s.vals, 1, s.vals, 1));

        transpose(R.vals, R_T.vals, R_T.dimensions[0], R_T.dimensions[1]);
        dnmat_mul_spdiag_batch(sR, R, s, 3);

        if (k + 1 >= max_outer_iter + 1) break;
        loss[k + 1] = objc(sR, s);

        double rhonum = loss[k] - loss[k + 1];
        double rhoden = -loss_qu;
        // Manopt regularization of rho for numerical stability.
        const double rho_regularization = 1e3;
        double rho_reg_offset = std::max(1.0, std::abs(loss[k])) * std::numeric_limits<double>::epsilon() * rho_regularization;
        rhonum += rho_reg_offset;
        rhoden += rho_reg_offset;

        bool model_decreased = (rhoden >= 0.0);
        double rou = rhonum / rhoden;
        if (!std::isfinite(rou)) {
            rou = std::numeric_limits<double>::quiet_NaN();
        }

        // // If the model is poor or loss did not decrease, try a short gradient-descent line search.
        // bool ls_success = false;
        // if (loss[k + 1] >= loss[k] || rou < 0.1) {
        //     double grad_norm = MixedProductManifoldInner(CUOPT_blas_handle, tmp_a, tmp_b,
        //                                                  rgrad_r, rgrad_r, rgrad_s, rgrad_s, s, sphere_mask, sphere_norm);
        //     grad_norm = std::sqrt(std::max(grad_norm, 1e-30));
        //     double alpha_ls = std::min(1e-3, delta / grad_norm);
        //     if (!std::isfinite(alpha_ls) || alpha_ls <= 0.0) {
        //         alpha_ls = 0.0;
        //     }

        //     for (int ls = 0; ls < 10 && alpha_ls > 0.0; ++ls) {
        //         CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, R.total_size, bestR.vals, 1, R.vals, 1));
        //         CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, s.total_size, bests.vals, 1, s.vals, 1));
        //         transpose(R_T.vals, R.vals, R.dimensions[0], R.dimensions[1]);

        //         CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, rgrad_r.total_size, rgrad_r.vals, 1, v_R.vals, 1));
        //         CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, rgrad_s.total_size, rgrad_s.vals, 1, v_s.vals, 1));
        //         double neg_alpha = -alpha_ls;
        //         CHECK_CUBLAS(cublasDscal(CUOPT_blas_handle.cublas_handle, v_R.total_size, &neg_alpha, v_R.vals, 1));
        //         CHECK_CUBLAS(cublasDscal(CUOPT_blas_handle.cublas_handle, v_s.total_size, &neg_alpha, v_s.vals, 1));

        //         retraction(R_T, s, v_R, v_s, 1.0);

        //         CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, new_R_T.total_size, new_R_T.vals, 1, R_T.vals, 1));
        //         CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, new_s.total_size, new_s.vals, 1, s.vals, 1));

        //         transpose(R.vals, R_T.vals, R_T.dimensions[0], R_T.dimensions[1]);
        //         dnmat_mul_spdiag_batch(sR, R, s, 3);
        //         double ls_loss = objc(sR, s);

        //         if (std::isfinite(ls_loss) && ls_loss < loss[k]) {
        //             loss[k + 1] = ls_loss;
        //             trstatus = 2;
        //             ls_success = true;
        //             break;
        //         }
        //         alpha_ls *= 0.5;
        //     }

        //     if (!ls_success) {
        //         CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, R.total_size, bestR.vals, 1, R.vals, 1));
        //         CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, s.total_size, bests.vals, 1, s.vals, 1));
        //         transpose(R_T.vals, R.vals, R.dimensions[0], R.dimensions[1]);
        //         dnmat_mul_spdiag_batch(sR, R, s, 3);
        //         loss[k + 1] = loss[k];
        //     }
        // }

        if (k > 0) {
            switch (endreason) {
                case 1: std::cout << "   nagative curvature" << std::endl; break;
                case 2: std::cout << "   exceed trust region" << std::endl; break;
                case 3: std::cout << "   reached norm tolerance" << std::endl; break;
                case 5: std::cout << "   numerical issue" << std::endl; break;
                case 6: std::cout << "   max iteration" << std::endl; break;
                case 7: std::cout << "   model increased" << std::endl; break;
            }
        } else {
            printf("\n");
        }

        bool rou_nan = !std::isfinite(rou);
        if (rou < 0.25 || !model_decreased || rou_nan) {
            delta = delta * 0.25;
            trstatus = 1;
            shrink_count++;
        } else if (rou > 0.75 && endreason == 2) {
            delta = std::min(delta * 2, delta_bar);
            trstatus = 2;
            shrink_count = 0;
        } else {
            shrink_count = 0;
        }

        if (endreason == 1) {
            neg_curv_count++;
        } else if (endreason == 2) {
            tr_boundary_count++;
        }

        if (delta < 1e-20) {
            if (gradnorm[k] > gradtol * 10) {
                double cap = std::min(3.0 * delta_bar, 1000.0);
                double target = 0.001 * gradnorm[k];
                delta = std::min(std::max(delta_bar, target), cap);
                delta_bar = std::max(delta_bar, delta);
                printf("delta reset to %1.3e due to nonzero gradient\n", delta);
            } else {
                printf("delta is too small, BM stopped!\n");
                break;
            }
        }
        const double rho_prime = 0.1;
        bool accept = model_decreased && (rou > rho_prime) && !rou_nan;
        if (!accept) {
            CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, R.total_size, bestR.vals, 1, R.vals, 1));
            CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, s.total_size, bests.vals, 1, s.vals, 1));
            loss[k + 1] = bestloss;
            dnmat_mul_spdiag_batch(sR, R, s, 3);
            trstatus = 3;
            reject_count++;
        } else {
            accept_count++;
        }

        if (k > 0) {
            switch (trstatus) {
                case 1: std::cout << "TR- "; break;
                case 2: std::cout << "TR+ "; break;
                case 3: std::cout << "REJ "; break;
                case 4: std::cout << "TR  "; break;
            }
        }

        printf("%d   %d   %1.3e   %1.3e   %1.3e   %1.3e", (int)k, (int)inner_iter_used, loss[k], gradnorm[k], rou, delta);

        delete[] rdotr;
        delete[] vdotv_record;
        delete[] vdotp_record;
        delete[] pdotp_record;
    }

    std::cout << std::endl << "Total iteration:     " << totalite << std::endl;
    std::cout << "Accept/Reject: " << accept_count << "/" << reject_count
              << " | NegCurv: " << neg_curv_count
              << " | TR-boundary: " << tr_boundary_count << std::endl;

    auto end = high_resolution_clock::now();
    std::cout << "Time taken by function1: "
              << duration_cast<milliseconds>(end - start).count() << " ms" << std::endl;

    *primal_value = objc(sR, s);

    CHECK_CUDA(cudaMemcpy(R_result.vals, R.vals, R.total_size * sizeof(datatype), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(s_result.vals, s.vals, s.total_size * sizeof(datatype), cudaMemcpyDeviceToDevice));

    delete[] loss;
    delete[] gradnorm;
}
