#include <iostream>
#include <algorithm>   // <-- for std::min / std::max
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
#include <Optimization/optimization.h>

using namespace std::chrono;

// =============================================================
// Sphere manifold helpers for s: ||s||_2 = sphere_norm
// =============================================================

template <typename T>
__global__ void ObjectiveLambdaKernal(T* out, const T* s, size_s n) {
    size_s i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        T v = s[i] * s[i] - (T)1;
        out[i] = v * v;
    }
}

template <typename T>
__global__ void GradLambdaKernal(T* out, const T* s, size_s n) {
    size_s i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        T v = s[i] * s[i] - (T)1;
        out[i] = v * s[i];
    }
}

template <typename T>
__global__ void HessLambdaKernal(T* out, const T* s, const T* su, size_s n) {
    size_s i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = ((T)3 * s[i] * s[i] - (T)1) * su[i];
    }
}

static inline void SphereNormalize(DeviceBlasHandle& h, opt_var& s, double sphere_norm) {
    double nrm = 0.0;
    CHECK_CUBLAS(cublasDnrm2(h.cublas_handle, s.total_size, s.vals, 1, &nrm));
    if (nrm <= 0.0) return;
    double target = sphere_norm > 0.0 ? sphere_norm : std::sqrt((double)s.total_size);
    double scale = target / nrm;
    CHECK_CUBLAS(cublasDscal(h.cublas_handle, s.total_size, &scale, s.vals, 1));
}

// out = v - s*(s^T v)
static inline void SphereProject(DeviceBlasHandle& h, opt_var& out, const opt_var& s, const opt_var& v) {
    CHECK_CUBLAS(cublasDcopy(h.cublas_handle, v.total_size, v.vals, 1, out.vals, 1));
    double sTv = 0.0;
    CHECK_CUBLAS(cublasDdot(h.cublas_handle, s.total_size, s.vals, 1, v.vals, 1, &sTv));
    double sTs = 0.0;
    CHECK_CUBLAS(cublasDdot(h.cublas_handle, s.total_size, s.vals, 1, s.vals, 1, &sTs));
    if (sTs <= 0.0) return;
    double alpha = -sTv / sTs;
    CHECK_CUBLAS(cublasDaxpy(h.cublas_handle, s.total_size, &alpha, s.vals, 1, out.vals, 1));
}

double ProductManifoldInner(DeviceBlasHandle& CUOPT_blas_handle,
                            opt_var &AR, opt_var &BR,
                            opt_var &As, opt_var &Bs) {
    double result1 = 0.0;
    double result2 = 0.0;
    CHECK_CUBLAS(cublasDdot(CUOPT_blas_handle.cublas_handle, AR.total_size, AR.vals, 1, BR.vals, 1, &result1));
    CHECK_CUBLAS(cublasDdot(CUOPT_blas_handle.cublas_handle, As.total_size, As.vals, 1, Bs.vals, 1, &result2));
    return result1 + result2;
}

void XMtrustregion(opt_var &C,
                   const opt_var &R0,
                   const opt_var &s0,
                   opt_var &R_result,
                   opt_var &s_result,
                   const double lam,
                   const double sphere_norm,  // radius of sphere for s
                   double &gradtol,
                   double linesearch_step,
                   opt_var &v,
                   double* primal_value,
                   const double maxtime) {

    DeviceBlasHandle CUOPT_blas_handle;
    CUOPT_blas_handle.activate();

    size_s o = R0.dimensions[1];
    size_s n = C.dimensions[0] / 3;

    // Manifold dimension: Stiefel blocks + sphere S^{n-1}
    size_s dim = n * (3 * o - 6) + (n - 1);

    double delta_bar = sqrt((double)dim);
    double delta = delta_bar / 8.0;

    // copy R
    opt_var R({3 * n, o});
    CHECK_CUDA(cudaMemcpy(R.vals, R0.vals, R.total_size * sizeof(datatype), cudaMemcpyDeviceToDevice));

    // s on sphere: length n
    if ((size_s)s0.total_size != n) {
        std::cerr << "[XMtrustregion] ERROR: s0 must have length n = " << n
                  << " but got " << s0.total_size << std::endl;
        *primal_value = -1;
        return;
    }

    opt_var s({n});
    CHECK_CUDA(cudaMemcpy(s.vals, s0.vals, s.total_size * sizeof(datatype), cudaMemcpyDeviceToDevice));
    SphereNormalize(CUOPT_blas_handle, s, sphere_norm);

    // sR = R .* kron(s,ones(3,1))
    opt_var sR({3 * n, o});
    opt_var dfdsR({3 * n, o});
    dnmat_mul_spdiag_batch(sR, R, s, 3);

    opt_var grad_r({3 * n, o});
    opt_var grad_s({n});
    opt_var CsR({3 * n, o});

    opt_var slamobj({n});

    // objective: f = <C*sR, sR> + lam * sum_i (s_i^2 - 1)^2
    auto objc = [&C, &CUOPT_blas_handle, &CsR, &slamobj, &lam, n](opt_var& sR_point, opt_var& s_point) {
        double result = 0.0;
        DnMatDnMat(CUOPT_blas_handle, CsR, C, sR_point);
        CHECK_CUBLAS(cublasDdot(CUOPT_blas_handle.cublas_handle,
                                sR_point.total_size,
                                CsR.vals, 1,
                                sR_point.vals, 1,
                                &result));
        if (lam > 0.0) {
            ObjectiveLambdaKernal<<<(n + 255) / 256, 256>>>(slamobj.vals, s_point.vals, n);
            double result_lambda = 0.0;
            CHECK_CUBLAS(cublasDasum(CUOPT_blas_handle.cublas_handle, slamobj.total_size, slamobj.vals, 1, &result_lambda));
            result += lam * result_lambda;
        }
        return result;
    };

    // Euclidean gradients
    opt_var slamgrad({n});
    auto grad = [&C, &CUOPT_blas_handle, &grad_r, &grad_s, &dfdsR, &slamgrad, &lam, n](opt_var& R_point,
                                                                                      opt_var& s_point,
                                                                                      opt_var& sR_point) {
        // df/d(sR) = 2*C*sR
        DnMatDnMat(CUOPT_blas_handle, dfdsR, C, sR_point, CUBLAS_OP_N, CUBLAS_OP_N, 2.0, 0.0);

        // d/dR: gradR = dfdsR .* kron(s,ones(3,1))
        dnmat_mul_spdiag_batch(grad_r, dfdsR, s_point, 3);

        // d/ds: grads(i) = sum( dfdsR_block_i .* R_block_i )
        dnmat_Ddot_colomn_batch(grad_s, dfdsR, R_point, 3);

        if (lam > 0.0) {
            GradLambdaKernal<<<(n + 255) / 256, 256>>>(slamgrad.vals, s_point.vals, n);
            double glam = 4.0 * lam;
            CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, slamgrad.total_size, &glam, slamgrad.vals, 1, grad_s.vals, 1));
        }
    };

    // Euclidean Hessian action (embedded) for (Ru, su)
    opt_var hr({3 * n, o});
    opt_var hrT({o, 3 * n});
    opt_var hs({n});
    opt_var slamhess({n});

    opt_var sRu({3 * n, o});
    opt_var suR({3 * n, o});
    opt_var CsRu({3 * n, o});
    opt_var suCsR({3 * n, o});
    opt_var sCsRu({3 * n, o});
    opt_var CsRudotR({n});
    opt_var CsRdotRu({n});

    auto ehess = [&C, &CUOPT_blas_handle, &hr, &hs,
                  &sR, &sRu, &suR, &CsR, &CsRu,
                  &suCsR, &sCsRu, &CsRudotR, &CsRdotRu, &slamhess, &lam, n](opt_var& R_point,
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
            HessLambdaKernal<<<(n + 255) / 256, 256>>>(slamhess.vals, s_point.vals, su_point.vals, n);
            double hlam = 4.0 * lam;
            CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, slamhess.total_size, &hlam, slamhess.vals, 1, hs.vals, 1));
        }
    };

    // Riemannian Hessian conversion: Stiefel unchanged, sphere uses:
    // rhs = P_s(ehesss) - (s^T egrads)*su
    opt_var RTgradR({3, 3 * n});
    opt_var RTgradR_sym({3, 3 * n});
    opt_var rhr({o, 3 * n});
    opt_var rhs({n});
    opt_var RTrhr({3, 3 * n});
    opt_var RTrhr_sym({3, 3 * n});

    auto ehess2rhess = [&CUOPT_blas_handle,
                        &RTgradR, &RTgradR_sym,
                        &rhr, &rhs,
                        &RTrhr, &RTrhr_sym](opt_var& ehessR,
                                            opt_var& ehesss,
                                            opt_var& egradR,
                                            opt_var& egrads,
                                            opt_var& R_point,
                                            opt_var& s_point,
                                            opt_var& Ru_point,
                                            opt_var& su_point) {

        size_s nb = R_point.dimensions[1] / 3;

        // --- Stiefel part (unchanged) ---
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

        // --- Sphere part ---
        CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, ehesss.total_size, ehesss.vals, 1, rhs.vals, 1));

        // rhs <- rhs - s*(s^T ehesss)
        double sTeh = 0.0;
        CHECK_CUBLAS(cublasDdot(CUOPT_blas_handle.cublas_handle, s_point.total_size,
                                s_point.vals, 1, ehesss.vals, 1, &sTeh));
        double sTs = 0.0;
        CHECK_CUBLAS(cublasDdot(CUOPT_blas_handle.cublas_handle, s_point.total_size,
                                s_point.vals, 1, s_point.vals, 1, &sTs));
        if (sTs > 0.0) {
            double a1 = -sTeh / sTs;
            CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, s_point.total_size,
                                     &a1, s_point.vals, 1, rhs.vals, 1));
        }

        // rhs <- rhs - (s^T egrads) * su
        double sTg = 0.0;
        CHECK_CUBLAS(cublasDdot(CUOPT_blas_handle.cublas_handle, s_point.total_size,
                                s_point.vals, 1, egrads.vals, 1, &sTg));
        if (sTs > 0.0) {
            double a2 = -sTg / sTs;
            CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, s_point.total_size,
                                     &a2, su_point.vals, 1, rhs.vals, 1));
        }

        // Project again to tangent to reduce drift
        double sTr = 0.0;
        CHECK_CUBLAS(cublasDdot(CUOPT_blas_handle.cublas_handle, s_point.total_size,
                                s_point.vals, 1, rhs.vals, 1, &sTr));
        if (sTs > 0.0) {
            double a3 = -sTr / sTs;
            CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, s_point.total_size,
                                     &a3, s_point.vals, 1, rhs.vals, 1));
        }
    };

    // Projection of Euclidean gradient to Riemannian gradient:
    opt_var rgrad_r({o, 3 * n});
    opt_var rgrad_s({n});

    auto projection = [&CUOPT_blas_handle,
                       &RTgradR, &RTgradR_sym,
                       &rgrad_r, &rgrad_s](opt_var& R_point,
                                           opt_var& s_point,
                                           opt_var& R_gradient,
                                           opt_var& s_gradient) {

        size_s nb = R_point.dimensions[1] / 3;

        // --- Stiefel projection ---
        DnMatDnMatBatch(CUOPT_blas_handle, RTgradR, R_point, R_gradient,
                        3, R_point.dimensions[0], 3, nb, CUBLAS_OP_T, CUBLAS_OP_N);
        symBatched(RTgradR_sym.vals, RTgradR.vals, RTgradR.dimensions[0], nb);

        CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, R_gradient.total_size, R_gradient.vals, 1, rgrad_r.vals, 1));
        CHECK_CUDA(cudaDeviceSynchronize());

        DnMatDnMatBatch(CUOPT_blas_handle, rgrad_r, R_point, RTgradR_sym,
                        R_point.dimensions[0], 3, 3, nb, CUBLAS_OP_N, CUBLAS_OP_N, -1.0, 1.0);

        // --- Sphere projection ---
        CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, s_gradient.total_size, s_gradient.vals, 1, rgrad_s.vals, 1));
        double sTg = 0.0;
        CHECK_CUBLAS(cublasDdot(CUOPT_blas_handle.cublas_handle, s_point.total_size,
                                s_point.vals, 1, s_gradient.vals, 1, &sTg));
        double sTs = 0.0;
        CHECK_CUBLAS(cublasDdot(CUOPT_blas_handle.cublas_handle, s_point.total_size,
                                s_point.vals, 1, s_point.vals, 1, &sTs));
        if (sTs > 0.0) {
            double alpha = -sTg / sTs;
            CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, s_point.total_size,
                                     &alpha, s_point.vals, 1, rgrad_s.vals, 1));
        }
    };

    // Retraction:
    opt_var new_R_T({o, 3 * n});
    opt_var new_R({3 * n, o});
    opt_var new_s({n});

    auto retraction = [&CUOPT_blas_handle, &new_R_T, &new_s, sphere_norm](opt_var& R_point,
                                                            opt_var& s_point,
                                                            opt_var& rR_step,
                                                            opt_var& rs_step,
                                                            double lr) {

        CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, R_point.total_size, &lr, rR_step.vals, 1, R_point.vals, 1));
        batchedQR(new_R_T, R_point, 3, s_point.total_size);

        CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, s_point.total_size, s_point.vals, 1, new_s.vals, 1));
        CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, s_point.total_size, &lr, rs_step.vals, 1, new_s.vals, 1));
        SphereNormalize(CUOPT_blas_handle, new_s, sphere_norm);
    };

    opt_var R_T({o, 3 * n});
    transpose(R_T.vals, R.vals, R.dimensions[0], R.dimensions[1]);

    opt_var grad_r_T({o, 3 * n});
    opt_var sR_new({3 * n, o});

    std::cout << "start linesearch" << std::endl;

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

    size_s max_inner_iter = 1000;
    size_s max_outer_iter = 1000;
    size_s totalite = 0;

    // allocate +1 because we write loss[k+1]
    double* loss = new double[max_outer_iter + 1];
    double* gradnorm = new double[max_outer_iter + 1];
    loss[0] = objc(sR, s);

    int endreason = 6;
    int trstatus = 4;
    int shrink_count = 0;

    opt_var r_R({o, 3 * n});
    opt_var r_s({n});
    opt_var p_R({o, 3 * n});
    opt_var p_s({n});
    opt_var p_R_T({3 * n, o});

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

        transpose(grad_r_T.vals, grad_r.vals, grad_r.dimensions[0], grad_r.dimensions[1]);
        transpose(R_T.vals, R.vals, R.dimensions[0], R.dimensions[1]);

        projection(R_T, s, grad_r_T, grad_s);

        CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, rgrad_r.total_size, rgrad_r.vals, 1, r_R.vals, 1));
        CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, rgrad_s.total_size, rgrad_s.vals, 1, r_s.vals, 1));
        CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, rgrad_r.total_size, rgrad_r.vals, 1, p_R.vals, 1));
        CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, rgrad_s.total_size, rgrad_s.vals, 1, p_s.vals, 1));

        double neg_one = -1.0;
        CHECK_CUBLAS(cublasDscal(CUOPT_blas_handle.cublas_handle, p_R.total_size, &neg_one, p_R.vals, 1));
        CHECK_CUBLAS(cublasDscal(CUOPT_blas_handle.cublas_handle, p_s.total_size, &neg_one, p_s.vals, 1));

        rdotr[0] = ProductManifoldInner(CUOPT_blas_handle, r_R, r_R, r_s, r_s);
        gradnorm[k] = sqrt(rdotr[0]);

        if (k > 0) {
            switch (trstatus) {
                case 1: std::cout << "TR- "; break;
                case 2: std::cout << "TR+ "; break;
                case 3: std::cout << "REJ "; break;
                case 4: std::cout << "TR  "; break;
            }
        }

        printf("%d   %d   %1.3e   %1.3e", (int)k, 1, loss[k], gradnorm[k]);

        if (k > 0) {
            switch (endreason) {
                case 1: std::cout << "   nagative curvature" << std::endl; break;
                case 2: std::cout << "   exceed trust region" << std::endl; break;
                case 3: std::cout << "   reached norm tolerance" << std::endl; break;
                case 5: std::cout << "   numerical issue" << std::endl; break;
                case 6: std::cout << "   max iteration" << std::endl; break;
            }
        } else {
            printf("\n");
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

            double denom = ProductManifoldInner(CUOPT_blas_handle, p_R, rhr, p_s, rhs);
            double alpha = rdotr[i] / denom;

            double vdotv = vdotv_record[i];
            double pdotp = pdotp_record[i];
            double vdotp = vdotp_record[i];

            if (rdotr[i] < 1e-15) { endreason = 5; break; }

            if (alpha <= 0) {
                double sqrt_val = sqrt(vdotp * vdotp + pdotp * (delta * delta - vdotv));
                double tau = (-vdotp + sqrt_val) / pdotp;

                CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, v_R.total_size, &tau, p_R.vals, 1, v_R.vals, 1));
                CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, v_s.total_size, &tau, p_s.vals, 1, v_s.vals, 1));

                CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, hv_R.total_size, &tau, rhr.vals, 1, hv_R.vals, 1));
                CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, hv_s.total_size, &tau, rhs.vals, 1, hv_s.vals, 1));

                endreason = 1;
                break;
            }

            if (vdotv + 2 * alpha * vdotp + alpha * alpha * pdotp > delta * delta) {
                double sqrt_val = sqrt(vdotp * vdotp + pdotp * (delta * delta - vdotv));
                double tau = (-vdotp + sqrt_val) / pdotp;

                CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, v_R.total_size, &tau, p_R.vals, 1, v_R.vals, 1));
                CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, v_s.total_size, &tau, p_s.vals, 1, v_s.vals, 1));

                CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, hv_R.total_size, &tau, rhr.vals, 1, hv_R.vals, 1));
                CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, hv_s.total_size, &tau, rhs.vals, 1, hv_s.vals, 1));

                endreason = 2;
                break;
            }

            CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, v_R.total_size, &alpha, p_R.vals, 1, v_R.vals, 1));
            CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, v_s.total_size, &alpha, p_s.vals, 1, v_s.vals, 1));

            CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, r_R.total_size, &alpha, rhr.vals, 1, r_R.vals, 1));
            CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, r_s.total_size, &alpha, rhs.vals, 1, r_s.vals, 1));

            CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, hv_R.total_size, &alpha, rhr.vals, 1, hv_R.vals, 1));
            CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, hv_s.total_size, &alpha, rhs.vals, 1, hv_s.vals, 1));

            rdotr[i + 1] = ProductManifoldInner(CUOPT_blas_handle, r_R, r_R, r_s, r_s);

            if (sqrt(rdotr[i + 1]) < gradnorm[k] * std::min(gradnorm[k], 0.1)) {
                endreason = 3;
                break;
            }

            double beta = rdotr[i + 1] / rdotr[i];

            CHECK_CUBLAS(cublasDscal(CUOPT_blas_handle.cublas_handle, p_R.total_size, &beta, p_R.vals, 1));
            CHECK_CUBLAS(cublasDscal(CUOPT_blas_handle.cublas_handle, p_s.total_size, &beta, p_s.vals, 1));

            CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, p_R.total_size, &neg_one, r_R.vals, 1, p_R.vals, 1));
            CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, p_s.total_size, &neg_one, r_s.vals, 1, p_s.vals, 1));

            vdotv_record[i + 1] = vdotv + 2 * alpha * vdotp + alpha * alpha * pdotp;
            vdotp_record[i + 1] = beta * (vdotp + alpha * pdotp);
            pdotp_record[i + 1] = beta * beta * pdotp + rdotr[i + 1];
        }

        totalite += i + 1;

        double loss_qu = ProductManifoldInner(CUOPT_blas_handle, v_R, hv_R, v_s, hv_s) / 2.0
                       + ProductManifoldInner(CUOPT_blas_handle, v_R, rgrad_r, v_s, rgrad_s);

        if (loss_qu >= 0) {
            printf("error! loss_qu is larger than 0\n");
            break;
        }

        retraction(R_T, s, v_R, v_s, 1.0);

        CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, new_R_T.total_size, new_R_T.vals, 1, R_T.vals, 1));
        CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, new_s.total_size, new_s.vals, 1, s.vals, 1));

        transpose(R.vals, R_T.vals, R_T.dimensions[0], R_T.dimensions[1]);
        dnmat_mul_spdiag_batch(sR, R, s, 3);

        // if we're at the last k, don't write loss[k+1]
        if (k + 1 >= max_outer_iter + 1) break;

        loss[k + 1] = objc(sR, s);

        double rou = (loss[k + 1] - loss[k]) / loss_qu;

        if (rou < 0.25) {
            delta = delta * 0.25;
            trstatus = 1;
            shrink_count++;
        } else if (rou > 0.75 && endreason <= 2) {
            delta = std::min(delta * 2, delta_bar);
            trstatus = 2;
            shrink_count = 0;
        } else {
            shrink_count = 0;
        }

        if (shrink_count > 3) {
            delta = delta * 1e-3;
            shrink_count = 0;
            printf("delta shrinked to %1.3e\n", delta);
            if (delta < 1e-20) {
                printf("delta is too small, BM stopped!\n");
                break;
            }
        }

        if (loss[k + 1] > bestloss || rou < 0.1) {
            CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, R.total_size, bestR.vals, 1, R.vals, 1));
            CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, s.total_size, bests.vals, 1, s.vals, 1));
            loss[k + 1] = bestloss;
            dnmat_mul_spdiag_batch(sR, R, s, 3);
            trstatus = 3;
        }

        delete[] rdotr;
        delete[] vdotv_record;
        delete[] vdotp_record;
        delete[] pdotp_record;
    }

    std::cout << std::endl << "Total iteration:     " << totalite << std::endl;
    auto end = high_resolution_clock::now();
    std::cout << "Time taken by function1: "
              << duration_cast<milliseconds>(end - start).count() << " ms" << std::endl;

    *primal_value = objc(sR, s);

    CHECK_CUDA(cudaMemcpy(R_result.vals, R.vals, R.total_size * sizeof(datatype), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(s_result.vals, s.vals, s.total_size * sizeof(datatype), cudaMemcpyDeviceToDevice));

    delete[] loss;
    delete[] gradnorm;
}
