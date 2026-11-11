#include <iostream>
#include <algorithm>   // std::min / std::max
#include <Utils/memory.h>
#include <Dense/matmul.h>
#include <Dense/trace.h>
#include <Dense/transpose.h>
#include <Dense/matdiagmul.h>
#include <Dense/matdot.h>
#include <Dense/matdivide.h>
#include <Dense/batchedQR.h>
#include <limits>
#include <cmath>

#include <chrono>
#include <vector>
#include <fstream>
#include <Optimization/optimization.h>

using namespace std::chrono;

// =============================================================
// Rotation-only (product Stiefel) trust-region for R
// Objective: f(R) = tr(R^T C R) = <C R, R>
// No scale manifold. Each node block is handled via Stiefel QR (columns=3).
// =============================================================

static inline double StiefelInner(DeviceBlasHandle& h, const opt_var& A, const opt_var& B) {
    double result = 0.0;
    CHECK_CUBLAS(cublasDdot(h.cublas_handle, A.total_size, A.vals, 1, B.vals, 1, &result));
    return result;
}

void XMtrustregion(opt_var &C,
                   const opt_var &R0,
                   opt_var &R_result,
                   double &gradtol,
                   double linesearch_step,
                   opt_var &v,
                   double* primal_value,
                   const double maxtime) {

    DeviceBlasHandle CUOPT_blas_handle;
    CUOPT_blas_handle.activate();

    size_s o = R0.dimensions[1];
    size_s n = C.dimensions[0] / 3;   // N

    // Sanity checks
    if (o < 3) {
        std::cerr << "[XMtrustregion] ERROR: expected R0 to have at least 3 columns (o>=3), got o=" << o << std::endl;
        *primal_value = std::numeric_limits<double>::quiet_NaN();  // out-of-band failure sentinel (objective can be legitimately negative)
        return;
    }
    if (R0.dimensions[0] != 3 * n) {
        std::cerr << "[XMtrustregion] ERROR: R0 rows must be 3N = " << 3 * n
                  << " but got " << R0.dimensions[0] << std::endl;
        *primal_value = std::numeric_limits<double>::quiet_NaN();  // out-of-band failure sentinel (objective can be legitimately negative)
        return;
    }

    size_s dim = n * (3 * o - 6); // = 3N when o=3
    double delta_bar = sqrt((double)dim);
    double delta = delta_bar / 8.0;

    // Copy R
    opt_var R({3 * n, o});
    CHECK_CUDA(cudaMemcpy(R.vals, R0.vals, R.total_size * sizeof(datatype), cudaMemcpyDeviceToDevice));

    // Scratch buffers
    opt_var CR({3 * n, o});        // C*R
    opt_var grad_r({3 * n, o});    // Euclidean grad (3N x 3)

    // Objective: f(R) = <C R, R>
    auto objc = [&C, &CUOPT_blas_handle, &CR](opt_var& R_point) {
        double result = 0.0;
        DnMatDnMat(CUOPT_blas_handle, CR, C, R_point);
        CHECK_CUBLAS(cublasDdot(CUOPT_blas_handle.cublas_handle,
                                R_point.total_size,
                                CR.vals, 1,
                                R_point.vals, 1,
                                &result));
        return result;
    };

    // Euclidean gradient: df/dR = 2*C*R
    auto grad = [&C, &CUOPT_blas_handle, &grad_r](opt_var& R_point) {
        DnMatDnMat(CUOPT_blas_handle, grad_r, C, R_point,
                   CUBLAS_OP_N, CUBLAS_OP_N, 2.0, 0.0);
    };

    // Euclidean Hessian action: H[Ru] = 2*C*Ru
    opt_var hr({3 * n, o});
    auto ehess = [&C, &CUOPT_blas_handle, &hr](opt_var& Ru_point) {
        DnMatDnMat(CUOPT_blas_handle, hr, C, Ru_point,
                   CUBLAS_OP_N, CUBLAS_OP_N, 2.0, 0.0);
    };

    // --- Stiefel projection helpers (on R_T = (3 x 3N)) ---
    opt_var RTgradR({3, 3 * n});
    opt_var RTgradR_sym({3, 3 * n});
    opt_var rgrad_r({o, 3 * n});     // (3 x 3N) when o=3

    auto projection = [&CUOPT_blas_handle,
                       &RTgradR, &RTgradR_sym,
                       &rgrad_r](opt_var& R_T_point,
                                 opt_var& R_T_gradient) {

        size_s nb = R_T_point.dimensions[1] / 3; // N

        // RTgradR = R^T * grad (blockwise 3x3)
        DnMatDnMatBatch(CUOPT_blas_handle, RTgradR, R_T_point, R_T_gradient,
                        3, R_T_point.dimensions[0], 3, nb, CUBLAS_OP_T, CUBLAS_OP_N);
        symBatched(RTgradR_sym.vals, RTgradR.vals, RTgradR.dimensions[0], nb);

        // rgrad = grad - R*sym(R^T grad)
        CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle,
                                 R_T_gradient.total_size,
                                 R_T_gradient.vals, 1,
                                 rgrad_r.vals, 1));
        CHECK_CUDA(cudaDeviceSynchronize());

        DnMatDnMatBatch(CUOPT_blas_handle, rgrad_r, R_T_point, RTgradR_sym,
                        R_T_point.dimensions[0], 3, 3, nb,
                        CUBLAS_OP_N, CUBLAS_OP_N, -1.0, 1.0);
    };

    // --- Euclidean Hessian -> Riemannian Hessian (Stiefel) ---
    opt_var rhr({o, 3 * n});
    opt_var RTrhr({3, 3 * n});
    opt_var RTrhr_sym({3, 3 * n});

    auto ehess2rhess = [&CUOPT_blas_handle,
                        &RTgradR, &RTgradR_sym,
                        &rhr,
                        &RTrhr, &RTrhr_sym](opt_var& ehessR_T,
                                            opt_var& egradR_T,
                                            opt_var& R_T_point,
                                            opt_var& Ru_T_point) {

        size_s nb = R_T_point.dimensions[1] / 3;

        // RTgradR = R^T * egradR
        DnMatDnMatBatch(CUOPT_blas_handle, RTgradR, R_T_point, egradR_T,
                        3, R_T_point.dimensions[0], 3, nb, CUBLAS_OP_T, CUBLAS_OP_N);
        symBatched(RTgradR_sym.vals, RTgradR.vals, RTgradR.dimensions[0], nb);

        // rhr = ehessR - Ru * sym(R^T egradR)
        CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle,
                                 ehessR_T.total_size,
                                 ehessR_T.vals, 1,
                                 rhr.vals, 1));
        CHECK_CUDA(cudaDeviceSynchronize());

        DnMatDnMatBatch(CUOPT_blas_handle, rhr, Ru_T_point, RTgradR_sym,
                        Ru_T_point.dimensions[0], 3, 3, nb,
                        CUBLAS_OP_N, CUBLAS_OP_N, -1.0, 1.0);

        // RTrhr = R^T * rhr
        DnMatDnMatBatch(CUOPT_blas_handle, RTrhr, R_T_point, rhr,
                        3, R_T_point.dimensions[0], 3, nb, CUBLAS_OP_T, CUBLAS_OP_N);
        symBatched(RTrhr_sym.vals, RTrhr.vals, RTrhr.dimensions[0], nb);

        // rhr = rhr - R * sym(R^T rhr)
        DnMatDnMatBatch(CUOPT_blas_handle, rhr, R_T_point, RTrhr_sym,
                        R_T_point.dimensions[0], 3, 3, nb,
                        CUBLAS_OP_N, CUBLAS_OP_N, -1.0, 1.0);
    };

    // --- Retraction via batched QR ---
    opt_var new_R_T({o, 3 * n});
    opt_var new_R({3 * n, o});

    auto retraction = [&CUOPT_blas_handle, &new_R_T, n](opt_var& R_T_point,
                                                        opt_var& rR_step,
                                                        double lr) {
        // R_T_point += lr * rR_step
        CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle,
                                 R_T_point.total_size,
                                 &lr,
                                 rR_step.vals, 1,
                                 R_T_point.vals, 1));

        // Project each 3x3 block back to orthonormal
        batchedQR(new_R_T, R_T_point, 3, n);
    };

    // Build transpose representations
    opt_var R_T({o, 3 * n});
    transpose(R_T.vals, R.vals, R.dimensions[0], R.dimensions[1]);

    opt_var grad_r_T({o, 3 * n});

    std::cout << "start linesearch" << std::endl;

    // --------------------------
    // Optional line search (kept structurally same)
    // --------------------------
    if (linesearch_step != 0) {
        double f0 = objc(R);
        double alpha = linesearch_step;

        opt_var decent_dir_R({3 * n, o});
        CHECK_CUDA(cudaMemset(decent_dir_R.vals, 0, decent_dir_R.total_size * sizeof(datatype)));
        CHECK_CUDA(cudaMemcpyAsync(decent_dir_R.vals + 3 * n * (o - 1),
                                   v.vals, sizeof(datatype) * v.total_size,
                                   cudaMemcpyDeviceToDevice));

        opt_var decent_dir_R_T({o, 3 * n});
        transpose(decent_dir_R_T.vals, decent_dir_R.vals,
                  decent_dir_R.dimensions[0], decent_dir_R.dimensions[1]);

        // trial step
        retraction(R_T, decent_dir_R_T, -alpha);

        // restore current R_T from accepted R
        transpose(R_T.vals, R.vals, R.dimensions[0], R.dimensions[1]);

        // build trial R from new_R_T
        transpose(new_R.vals, new_R_T.vals, new_R_T.dimensions[0], new_R_T.dimensions[1]);

        double f = objc(new_R);

        while (f > f0) {
            alpha *= 0.5;

            retraction(R_T, decent_dir_R_T, -alpha);

            transpose(R_T.vals, R.vals, R.dimensions[0], R.dimensions[1]);
            transpose(new_R.vals, new_R_T.vals, new_R_T.dimensions[0], new_R_T.dimensions[1]);

            f = objc(new_R);

            if (alpha < 1e-5) {
                printf("linesearch failed! BM stopped! \n");
                *primal_value = std::numeric_limits<double>::quiet_NaN();  // out-of-band failure sentinel (objective can be legitimately negative)
                return;
            }
        }

        if (f0 - f > 0) {
            printf("linesearch decrease %1.3e\n", f0 - f);
            CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle,
                                     new_R_T.total_size, new_R_T.vals, 1,
                                     R_T.vals, 1));
            CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle,
                                     new_R.total_size, new_R.vals, 1,
                                     R.vals, 1));
        } else {
            // No strict decrease found: give up the rank climb like the alpha-underflow
            // branch does (continuing TR from the lifted saddle just re-converges to the
            // same point and emits a duplicate certificate).
            printf("linesearch failed! BM stopped! \n");
            *primal_value = std::numeric_limits<double>::quiet_NaN();  // out-of-band failure sentinel (objective can be legitimately negative)
            return;
        }
    }

    // -------------------------------------------------------------
    // Trust-region outer / inner iterations
    // -------------------------------------------------------------
    size_s max_inner_iter = 1000;
    size_s max_outer_iter = 1000;
    size_s totalite = 0;

    std::vector<double> loss(max_outer_iter + 1, 0.0);
    std::vector<double> gradnorm(max_outer_iter + 1, 0.0);
    loss[0] = objc(R);

    int endreason = 6;
    int trstatus = 4;
    int shrink_count = 0;

    opt_var r_R({o, 3 * n});
    opt_var p_R({o, 3 * n});
    opt_var p_R_T({3 * n, o});

    auto start = high_resolution_clock::now();

    for (size_s k = 0; k < max_outer_iter; ++k) {

        opt_var v_R({o, 3 * n});
        opt_var hv_R({o, 3 * n});

        CHECK_CUDA(cudaMemset(v_R.vals, 0, v_R.total_size * sizeof(datatype)));
        CHECK_CUDA(cudaMemset(hv_R.vals, 0, hv_R.total_size * sizeof(datatype)));

        // best backup
        opt_var bestR({3 * n, o});
        CHECK_CUDA(cudaMemcpy(bestR.vals, R.vals, R.total_size * sizeof(datatype), cudaMemcpyDeviceToDevice));
        double bestloss = loss[k];

        std::vector<double> rdotr(max_inner_iter + 1, 0.0);

        // compute Euclidean gradient
        grad(R);

        // transpose to R_T-space
        transpose(grad_r_T.vals, grad_r.vals, grad_r.dimensions[0], grad_r.dimensions[1]);
        transpose(R_T.vals, R.vals, R.dimensions[0], R.dimensions[1]);

        // project to tangent: rgrad_r
        projection(R_T, grad_r_T);

        // r_R = rgrad, p_R = -r_R
        CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, rgrad_r.total_size, rgrad_r.vals, 1, r_R.vals, 1));
        CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, rgrad_r.total_size, rgrad_r.vals, 1, p_R.vals, 1));

        double neg_one = -1.0;
        CHECK_CUBLAS(cublasDscal(CUOPT_blas_handle.cublas_handle, p_R.total_size, &neg_one, p_R.vals, 1));

        rdotr[0] = StiefelInner(CUOPT_blas_handle, r_R, r_R);
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

        std::vector<double> vdotv_record(max_inner_iter + 1, 0.0);
        std::vector<double> vdotp_record(max_inner_iter + 1, 0.0);
        std::vector<double> pdotp_record(max_inner_iter + 1, 0.0);

        vdotv_record[0] = 0.0;
        vdotp_record[0] = 0.0;
        pdotp_record[0] = rdotr[0];

        size_s i = 0;
        for (i = 0; i < max_inner_iter; ++i) {

            // p_R_T = transpose(p_R) => (3N x 3)
            transpose(p_R_T.vals, p_R.vals, p_R.dimensions[0], p_R.dimensions[1]);

            // Euclidean Hessian action: hr = 2*C*p_R_T
            ehess(p_R_T);

            // hrT = transpose(hr) => (3 x 3N)
            opt_var hrT({o, 3 * n});
            transpose(hrT.vals, hr.vals, hr.dimensions[0], hr.dimensions[1]);

            // Riemannian Hessian action: rhr
            ehess2rhess(hrT, grad_r_T, R_T, p_R);

            double denom = StiefelInner(CUOPT_blas_handle, p_R, rhr);
            double alpha = rdotr[i] / denom;

            double vdotv = vdotv_record[i];
            double pdotp = pdotp_record[i];
            double vdotp = vdotp_record[i];

            if (rdotr[i] < 1e-15) { endreason = 5; break; }

            if (alpha <= 0) {
                double sqrt_val = sqrt(vdotp * vdotp + pdotp * (delta * delta - vdotv));
                double tau = (-vdotp + sqrt_val) / pdotp;

                CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, v_R.total_size, &tau, p_R.vals, 1, v_R.vals, 1));
                CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, hv_R.total_size, &tau, rhr.vals, 1, hv_R.vals, 1));

                endreason = 1;
                break;
            }

            if (vdotv + 2 * alpha * vdotp + alpha * alpha * pdotp > delta * delta) {
                double sqrt_val = sqrt(vdotp * vdotp + pdotp * (delta * delta - vdotv));
                double tau = (-vdotp + sqrt_val) / pdotp;

                CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, v_R.total_size, &tau, p_R.vals, 1, v_R.vals, 1));
                CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, hv_R.total_size, &tau, rhr.vals, 1, hv_R.vals, 1));

                endreason = 2;
                break;
            }

            // v += alpha p
            CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, v_R.total_size, &alpha, p_R.vals, 1, v_R.vals, 1));

            // r += alpha Hp
            CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, r_R.total_size, &alpha, rhr.vals, 1, r_R.vals, 1));

            // hv += alpha Hp
            CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, hv_R.total_size, &alpha, rhr.vals, 1, hv_R.vals, 1));

            rdotr[i + 1] = StiefelInner(CUOPT_blas_handle, r_R, r_R);

            if (sqrt(rdotr[i + 1]) < gradnorm[k] * std::min(gradnorm[k], 0.1)) {
                endreason = 3;
                break;
            }

            double beta = rdotr[i + 1] / rdotr[i];

            // p = beta p - r
            CHECK_CUBLAS(cublasDscal(CUOPT_blas_handle.cublas_handle, p_R.total_size, &beta, p_R.vals, 1));
            CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, p_R.total_size, &neg_one, r_R.vals, 1, p_R.vals, 1));

            vdotv_record[i + 1] = vdotv + 2 * alpha * vdotp + alpha * alpha * pdotp;
            vdotp_record[i + 1] = beta * (vdotp + alpha * pdotp);
            pdotp_record[i + 1] = beta * beta * pdotp + rdotr[i + 1];
        }

        totalite += i + 1;

        // quadratic model value at v: g^T v + 1/2 v^T H v
        double loss_qu = StiefelInner(CUOPT_blas_handle, v_R, hv_R) / 2.0
                       + StiefelInner(CUOPT_blas_handle, v_R, rgrad_r);

        if (loss_qu >= 0) {
            printf("error! loss_qu is larger than 0\n");
            break;
        }

        // retract
        retraction(R_T, v_R, 1.0);

        // accept new_R_T into R_T, then update R
        CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle,
                                 new_R_T.total_size, new_R_T.vals, 1,
                                 R_T.vals, 1));

        transpose(R.vals, R_T.vals, R_T.dimensions[0], R_T.dimensions[1]);

        if (k + 1 >= max_outer_iter + 1) break;
        loss[k + 1] = objc(R);

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
            CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle,
                                     R.total_size, bestR.vals, 1,
                                     R.vals, 1));
            loss[k + 1] = bestloss;
            trstatus = 3;
        }
    }

    std::cout << std::endl << "Total iteration:     " << totalite << std::endl;
    auto end = high_resolution_clock::now();
    std::cout << "Time taken by function1: "
              << duration_cast<milliseconds>(end - start).count() << " ms" << std::endl;

    *primal_value = objc(R);

    CHECK_CUDA(cudaMemcpy(R_result.vals, R.vals, R.total_size * sizeof(datatype), cudaMemcpyDeviceToDevice));
}
