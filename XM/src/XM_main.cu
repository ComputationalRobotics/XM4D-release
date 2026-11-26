#include <iostream>
#include <algorithm>
#include <XM/trustregion.h>
#include <XM/trustregion_4d.h>
#include <XM/trustregion_mixed.h>
#include <XM/checkeig.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

template <typename T>
__global__ void DecentDirectionKernal(T* v, T* s, size_s n){
    size_s i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<n){
        v[3*i] /= s[i] ;
        v[3*i+1] /= s[i] ;
        v[3*i+2] /= s[i] ;
    }  
}

void loadCMatrixFromBin(const std::string& filename, std::vector<double>& matrix, size_s& n) {
    // Load the matrix from a binary file
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "cannot open file" << std::endl;
        return;
    }
    int rows, cols;
    file.read(reinterpret_cast<char*>(&rows), sizeof(int));
    file.read(reinterpret_cast<char*>(&cols), sizeof(int));
    n = rows/3;
    matrix.resize(rows * cols);
    file.read(reinterpret_cast<char*>(matrix.data()), sizeof(double) * rows * cols);
    printf("rows: %d, cols: %d\n", rows, cols);
    file.close();
}

void solve_Ronly(const std::string& dataset_path,
           size_s max_rank = 10,
           double tol = 1e-6,
           double /*lam*/ = 0.0,     // unused now
           double max_time = 1000) {

    std::cout << "+++++++++++++++++++++++++++++++++" << std::endl;
    std::cout << "Begin XM (rotation-only)" << std::endl;
    std::cout << "+++++++++++++++++++++++++++++++++" << std::endl;

    std::string Q_matrix_filename = dataset_path + "/Q.bin";
    std::string output_path = dataset_path + "/";
    size_s n = 0;

    // load Q (= C) matrix on host
    std::vector<datatype> Q_h;
    loadCMatrixFromBin(Q_matrix_filename, Q_h, n);

    opt_var C({3 * n, 3 * n});
    C.SynchronizeHostToDevice(Q_h.data());

    // init
    size_s o = 3;
    opt_var v({3 * n});     // used by checkeig + direction
    opt_var R0({3 * n, o}); // current iterate / warm start

    double gradtol = tol;
    double primal = 0;

    while (o <= max_rank) {

        std::cout << "+++++++++++++++++++++++++++++++++" << std::endl;
        std::cout << "Solve TR with Rank   " << o << std::endl;
        std::cout << "+++++++++++++++++++++++++++++++++" << std::endl;

        opt_var R({3 * n, o}); // output for current rank

        if (o == 3) {
            // Initialize R0 to identity blocks
            std::vector<datatype> R0_h(3 * o * n, 0);
            for (size_l i = 0; i < n; ++i) {
                // column 0
                R0_h[3 * i] = 1.0;
                // column 1
                R0_h[3 * i + 3 * n + 1] = 1.0;
                // column 2
                R0_h[3 * i + 6 * n + 2] = 1.0;
            }
            R0.SynchronizeHostToDevice(R0_h.data());

            // rotation-only XMtrustregion signature
            XMtrustregion(C, R0, R, gradtol, 0, v, &primal, max_time);
        } else {
            XMtrustregion(C, R0, R, gradtol, 1, v, &primal, max_time);
        }

        if (std::isnan(primal)) {   // NaN = linesearch-failure sentinel; negative objective is valid
            o += 1;
            break;
        }

        R.SynchronizeDevicetoHost();



        std::cout << "+++++++++++++++++++++++++++++++++" << std::endl;
        std::cout << "Check Eigen value" << std::endl;
        std::cout << "+++++++++++++++++++++++++++++++++" << std::endl;

        // IMPORTANT: now we check using R directly (equivalent to s = 1)
        if (checkeig_noscale(C, R, /*lam=*/0.0, v, primal, output_path)) {
            o += 1;

            // warm start next rank / save best
            CHECK_CUDA(cudaMemcpyAsync(R0.vals, R.vals,
                                       sizeof(datatype) * R.total_size,
                                       cudaMemcpyDeviceToDevice));
            break;
        } else if (o < max_rank) {
            // expand rank (add one column)
            R0.dimensions[1] = o + 1;
            R0.total_size = 3 * n * (o + 1);
            R0.allocate();
            CHECK_CUDA(cudaMemset(R0.vals, 0, sizeof(datatype) * R0.total_size));

            // copy current R into the first 3N*o entries of expanded R0
            CHECK_CUDA(cudaMemcpyAsync(R0.vals, R.vals,
                                       sizeof(datatype) * R.total_size,
                                       cudaMemcpyDeviceToDevice));

            // Rotation-only solver: s == 1, so the descent direction is v itself.
            // (The old DecentDirectionKernal divide-by-ones here read a stale static
            // device buffer holding zeros/garbage, corrupting v -> escape always failed.)
        } else {
            CHECK_CUDA(cudaMemcpyAsync(R0.vals, R.vals,
                                       sizeof(datatype) * R.total_size,
                                       cudaMemcpyDeviceToDevice));
        }

        o += 1;
    }

    if (o > max_rank) {
        std::cout << "BM stoped because max rank" << std::endl;
    }

    // save R (same as before)
    std::vector<datatype> R_h(3 * n * (o - 1));
    CHECK_CUDA(cudaMemcpy(R_h.data(), R0.vals,
                          sizeof(datatype) * R0.total_size,
                          cudaMemcpyDeviceToHost));

    std::ofstream fileR(output_path + "R.bin", std::ios::binary);
    int temp1 = (int)(3 * n);
    int temp2 = (int)(o - 1);
    fileR.write(reinterpret_cast<char*>(&temp1), sizeof(int));
    fileR.write(reinterpret_cast<char*>(&temp2), sizeof(int));
    fileR.write(reinterpret_cast<char*>(R_h.data()), sizeof(double) * 3 * n * (o - 1));
    fileR.close();

    std::cout << "saved R" << std::endl;

    // No more s to save.
    return;
}


void solve(const std::string& dataset_path,
           size_s max_rank = 10,
           double tol = 1e-6,
           double lam = 0.0,
           double max_time = 1000) {

    std::cout << "+++++++++++++++++++++++++++++++++" << std::endl;
    std::cout << "Begin XM" << std::endl;
    std::cout << "+++++++++++++++++++++++++++++++++" << std::endl;

    std::string Q_matrix_filename = dataset_path + "/Q.bin";
    std::string output_path = dataset_path + "/";
    size_s n = 0;

    // load Q matrix on host
    std::vector<datatype> Q_h;
    loadCMatrixFromBin(Q_matrix_filename, Q_h, n);

    opt_var C({3 * n, 3 * n});
    C.SynchronizeHostToDevice(Q_h.data());

    // init
    size_s o = 3;
    opt_var v({3 * n});
    opt_var R0({3 * n, o});

    // -------------------------------
    // NEW: s0 is full length n (sphere)
    // -------------------------------
    std::vector<datatype> s0_h(n, (datatype)1);

    opt_var s0({n});   // full n-dim vector
    s0.SynchronizeHostToDevice(s0_h.data());

    opt_var s({n});    // full n-dim vector (output buffer too)
    s.SynchronizeHostToDevice(s0_h.data());
    // Note: XMtrustregion() should normalize s0 onto the sphere internally.

    double gradtol = tol;
    double primal = 0;

    while (o <= max_rank) {

        std::cout << "+++++++++++++++++++++++++++++++++" << std::endl;
        std::cout << "Solve TR with Rank   " << o << std::endl;
        std::cout << "+++++++++++++++++++++++++++++++++" << std::endl;

        opt_var R({3 * n, o});

        if (o == 3) {
            std::vector<datatype> R0_h(3 * o * n, 0);
            for (size_l i = 0; i < n; ++i) {
                R0_h[3 * i] = 1.0;
                R0_h[3 * i + 3 * n + 1] = 1.0;
                R0_h[3 * i + 6 * n + 2] = 1.0;
            }
            R0.SynchronizeHostToDevice(R0_h.data());

            double sphere_norm = std::sqrt((double)n);
            XMtrustregion(C, R0, s0, R, s, lam, sphere_norm, gradtol, 0, v, &primal, max_time);
        } else {
            double sphere_norm = std::sqrt((double)n);
            XMtrustregion(C, R0, s0, R, s, lam, sphere_norm, gradtol, 1, v, &primal, max_time);
        }

        if (primal < 0) {
            o += 1;
            break;
        }

        // build sR using full s (length n)
        opt_var sR({3 * n, o});
        dnmat_mul_spdiag_batch(sR, R, s, 3);

        sR.SynchronizeDevicetoHost();

        std::cout << "+++++++++++++++++++++++++++++++++" << std::endl;
        std::cout << "Check Eigen value" << std::endl;
        std::cout << "+++++++++++++++++++++++++++++++++" << std::endl;

        if (checkeig(C, sR, lam, v, primal, output_path, std::sqrt((double)n), nullptr, 0)) {
            o += 1;
            CHECK_CUDA(cudaMemcpyAsync(R0.vals, R.vals,
                                       sizeof(datatype) * R.total_size,
                                       cudaMemcpyDeviceToDevice));
            CHECK_CUDA(cudaMemcpyAsync(s0.vals, s.vals,
                                       sizeof(datatype) * s.total_size,
                                       cudaMemcpyDeviceToDevice));
            break;
        } else if (o < max_rank) {
            // expand rank
            R0.dimensions[1] = o + 1;
            R0.total_size = 3 * n * (o + 1);
            R0.allocate();

            CHECK_CUDA(cudaMemcpyAsync(R0.vals, R.vals,
                                       sizeof(datatype) * R.total_size,
                                       cudaMemcpyDeviceToDevice));
            CHECK_CUDA(cudaMemcpyAsync(s0.vals, s.vals,
                                       sizeof(datatype) * s.total_size,
                                       cudaMemcpyDeviceToDevice));

            // was: DecentDirectionKernal(..., s_ex.vals, n);
            // now: use s (full length n)
            DecentDirectionKernal<<<(3 * n + 255) / 256, 256>>>(v.vals, s.vals, n);

        } else {
            CHECK_CUDA(cudaMemcpyAsync(R0.vals, R.vals,
                                       sizeof(datatype) * R.total_size,
                                       cudaMemcpyDeviceToDevice));
            CHECK_CUDA(cudaMemcpyAsync(s0.vals, s.vals,
                                       sizeof(datatype) * s.total_size,
                                       cudaMemcpyDeviceToDevice));
        }

        o += 1;
    }

    if (o > max_rank) {
        std::cout << "BM stoped because max rank" << std::endl;
    }

    // save R
    std::vector<datatype> R_h(3 * n * (o - 1));
    CHECK_CUDA(cudaMemcpy(R_h.data(), R0.vals,
                          sizeof(datatype) * R0.total_size,
                          cudaMemcpyDeviceToHost));

    std::ofstream fileR(output_path + "R.bin", std::ios::binary);
    int temp1 = (int)(3 * n);
    int temp2 = (int)(o - 1);
    fileR.write(reinterpret_cast<char*>(&temp1), sizeof(int));
    fileR.write(reinterpret_cast<char*>(&temp2), sizeof(int));
    fileR.write(reinterpret_cast<char*>(R_h.data()), sizeof(double) * 3 * n * (o - 1));
    fileR.close();

    std::cout << "saved R" << std::endl;

    // save s (full length n)
    std::vector<datatype> s_h(n);
    CHECK_CUDA(cudaMemcpy(s_h.data(), s0.vals,
                          sizeof(datatype) * s0.total_size,
                          cudaMemcpyDeviceToHost));

    std::ofstream files(output_path + "s.bin", std::ios::binary);
    int nn = (int)n;
    files.write(reinterpret_cast<char*>(&nn), sizeof(int));
    int one = 1;
    files.write(reinterpret_cast<char*>(&one), sizeof(int));
    files.write(reinterpret_cast<char*>(s_h.data()), sizeof(double) * n);
    files.close();

    return;
}


void solve_mixed(const std::string& dataset_path,
                 size_s max_rank = 10,
                 double tol = 1e-6,
                 double lam = 0.0,
                 double max_time = 1000,
                 double separate_ratio = 0.1) {

    std::cout << "+++++++++++++++++++++++++++++++++" << std::endl;
    std::cout << "Begin XM (mixed manifold)" << std::endl;
    std::cout << "+++++++++++++++++++++++++++++++++" << std::endl;

    std::string Q_matrix_filename = dataset_path + "/Q.bin";
    std::string output_path = dataset_path + "/";
    size_s n = 0;

    // load Q matrix on host
    std::vector<datatype> Q_h;
    loadCMatrixFromBin(Q_matrix_filename, Q_h, n);

    opt_var C({3 * n, 3 * n});
    C.SynchronizeHostToDevice(Q_h.data());

    const size_s rows = 3 * n;
    std::vector<std::pair<double, size_s>> traces;
    traces.reserve(n);
    for (size_s i = 0; i < n; ++i) {
        double trace = 0.0;
        for (size_s k = 0; k < 3; ++k) {
            size_s idx = (3 * i + k) + (3 * i + k) * rows;
            trace += Q_h[idx];
        }
        traces.emplace_back(trace, i);
    }
    std::sort(traces.begin(), traces.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });
    size_s m = std::max<size_s>(1, static_cast<size_s>(std::ceil(separate_ratio * n)));
    std::vector<size_s> sphere_indices(m);
    for (size_s i = 0; i < m; ++i) {
        sphere_indices[i] = traces[i].second;
    }

    // init
    size_s o = 3;
    opt_var v({3 * n});
    opt_var R0({3 * n, o});

    std::vector<datatype> s0_h(n, (datatype)1);
    opt_var s0({n});
    s0.SynchronizeHostToDevice(s0_h.data());

    opt_var s({n});
    s.SynchronizeHostToDevice(s0_h.data());

    double gradtol = tol;
    double primal = 0;

    while (o <= max_rank) {

        std::cout << "+++++++++++++++++++++++++++++++++" << std::endl;
        std::cout << "Solve TR with Rank   " << o << std::endl;
        std::cout << "+++++++++++++++++++++++++++++++++" << std::endl;

        opt_var R({3 * n, o});

        if (o == 3) {
            std::vector<datatype> R0_h(3 * o * n, 0);
            for (size_l i = 0; i < n; ++i) {
                R0_h[3 * i] = 1.0;
                R0_h[3 * i + 3 * n + 1] = 1.0;
                R0_h[3 * i + 6 * n + 2] = 1.0;
            }
            R0.SynchronizeHostToDevice(R0_h.data());

            double sphere_norm = std::sqrt((double)m);
            XMtrustregion_mixed(C, R0, s0, R, s, lam,
                                sphere_indices.data(), m,
                                gradtol, 0, v, &primal, max_time);
        } else {
            double sphere_norm = std::sqrt((double)m);
            XMtrustregion_mixed(C, R0, s0, R, s, lam,
                                sphere_indices.data(), m,
                                gradtol, 1, v, &primal, max_time);
        }

        if (primal < 0) {
            o += 1;
            break;
        }

        opt_var sR({3 * n, o});
        dnmat_mul_spdiag_batch(sR, R, s, 3);
        sR.SynchronizeDevicetoHost();

        std::cout << "+++++++++++++++++++++++++++++++++" << std::endl;
        std::cout << "Check Eigen value" << std::endl;
        std::cout << "+++++++++++++++++++++++++++++++++" << std::endl;

        if (checkeig(C, sR, lam, v, primal, output_path, std::sqrt((double)m), sphere_indices.data(), m)) {
            o += 1;
            CHECK_CUDA(cudaMemcpyAsync(R0.vals, R.vals,
                                       sizeof(datatype) * R.total_size,
                                       cudaMemcpyDeviceToDevice));
            CHECK_CUDA(cudaMemcpyAsync(s0.vals, s.vals,
                                       sizeof(datatype) * s.total_size,
                                       cudaMemcpyDeviceToDevice));
            break;
        } else if (o < max_rank) {
            R0.dimensions[1] = o + 1;
            R0.total_size = 3 * n * (o + 1);
            R0.allocate();

            CHECK_CUDA(cudaMemcpyAsync(R0.vals, R.vals,
                                       sizeof(datatype) * R.total_size,
                                       cudaMemcpyDeviceToDevice));
            CHECK_CUDA(cudaMemcpyAsync(s0.vals, s.vals,
                                       sizeof(datatype) * s.total_size,
                                       cudaMemcpyDeviceToDevice));

            DecentDirectionKernal<<<(3 * n + 255) / 256, 256>>>(v.vals, s.vals, n);
        } else {
            CHECK_CUDA(cudaMemcpyAsync(R0.vals, R.vals,
                                       sizeof(datatype) * R.total_size,
                                       cudaMemcpyDeviceToDevice));
            CHECK_CUDA(cudaMemcpyAsync(s0.vals, s.vals,
                                       sizeof(datatype) * s.total_size,
                                       cudaMemcpyDeviceToDevice));
        }

        o += 1;
    }

    if (o > max_rank) {
        std::cout << "BM stoped because max rank" << std::endl;
    }

    std::vector<datatype> R_h(3 * n * (o - 1));
    CHECK_CUDA(cudaMemcpy(R_h.data(), R0.vals,
                          sizeof(datatype) * R0.total_size,
                          cudaMemcpyDeviceToHost));

    std::ofstream fileR(output_path + "R.bin", std::ios::binary);
    int temp1 = (int)(3 * n);
    int temp2 = (int)(o - 1);
    fileR.write(reinterpret_cast<char*>(&temp1), sizeof(int));
    fileR.write(reinterpret_cast<char*>(&temp2), sizeof(int));
    fileR.write(reinterpret_cast<char*>(R_h.data()), sizeof(double) * 3 * n * (o - 1));
    fileR.close();

    std::cout << "saved R" << std::endl;

    std::vector<datatype> s_h(n);
    CHECK_CUDA(cudaMemcpy(s_h.data(), s0.vals,
                          sizeof(datatype) * s0.total_size,
                          cudaMemcpyDeviceToHost));

    std::ofstream files(output_path + "s.bin", std::ios::binary);
    int nn = (int)n;
    files.write(reinterpret_cast<char*>(&nn), sizeof(int));
    int one = 1;
    files.write(reinterpret_cast<char*>(&one), sizeof(int));
    files.write(reinterpret_cast<char*>(s_h.data()), sizeof(double) * n);
    files.close();

    return;
}


void solve_rank3(const std::string& dataset_path, size_s max_rank = 10, double tol = 1e-6, double lam = 0.0, double max_time = 1000) {
    std::cout << "+++++++++++++++++++++++++++++++++" << std::endl;
    std::cout << "Begin XM" << std::endl;
    std::cout << "+++++++++++++++++++++++++++++++++" << std::endl;

    std::string Q_matrix_filename = dataset_path + "/Q.bin";
    std::string output_path = dataset_path + "/";
    size_s n = 0;
    // load Q matrix on host
    std::vector<datatype> Q_h;
    loadCMatrixFromBin(Q_matrix_filename, Q_h, n); 
    opt_var C({3*n,3*n});
    C.SynchronizeHostToDevice(Q_h.data());

    // init
    // order of R is 3*n*o
    size_s o = 3;
    opt_var v({3*n});
    opt_var R0({3*n,o});
    
    // can be replaced by given initial guess
    std::vector<datatype> s0_ex_h(n,1);

    opt_var s0_ex({n}); 
    s0_ex.SynchronizeHostToDevice(s0_ex_h.data());
    opt_var s0; //because we reserve the first element to be 1
    s0.vals = s0_ex.vals+1;
    s0.num_dims = 1;
    s0.dimensions = new size_s[1];
    s0.dimensions[0] = n-1;
    s0.total_size = n-1;

    opt_var s_ex({n}); 
    s_ex.SynchronizeHostToDevice(s0_ex_h.data());
    opt_var s; //because we reserve the first element to be 1
    s.vals = s_ex.vals+1;
    s.num_dims = 1;
    s.dimensions = new size_s[1];
    s.dimensions[0] = n-1;
    s.total_size = n-1;

    double gradtol = tol;
    double primal = 0;

    std::cout << "+++++++++++++++++++++++++++++++++" << std::endl;
    std::cout << "Solve TR with Rank   " << o <<std::endl;
    std::cout << "+++++++++++++++++++++++++++++++++" << std::endl;

    opt_var R({3*n,o});

    std::vector<datatype> R0_h(3*o*n,0);
    for(size_l i = 0; i<n; ++i){
        R0_h[3*i] = 1.0;
        R0_h[3*i+3*n+1] = 1.0;
        R0_h[3*i+6*n+2] = 1.0;
    }
    R0.SynchronizeHostToDevice(R0_h.data());
    XMtrustregion(C,R0,s0,R,s,lam,std::sqrt((double)n),gradtol,0,v,&primal, max_time);

    CHECK_CUDA(cudaMemcpyAsync(R0.vals, R.vals, sizeof(datatype) * R.total_size, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpyAsync(s0.vals, s.vals, sizeof(datatype) * s.total_size, cudaMemcpyDeviceToDevice));
    
    // save R and s
    std::vector<datatype> R_h(3*n*o);
    CHECK_CUDA(cudaMemcpy(R_h.data(), R0.vals, sizeof(datatype) * R0.total_size, cudaMemcpyDeviceToHost));
    std::ofstream fileR(output_path + "R.bin", std::ios::binary);
    // write row and col
    int temp1 = 3 * n;
    fileR.write(reinterpret_cast<char*>(&temp1), sizeof(int));  
    int temp2 = o;
    fileR.write(reinterpret_cast<char*>(&temp2), sizeof(int));

    fileR.write(reinterpret_cast<char*>(R_h.data()), sizeof(double) * 3 * n * o);
    fileR.close();

    std::cout << "saved R" << std::endl;

    std::vector<datatype> s_h(n);
    CHECK_CUDA(cudaMemcpy(s_h.data(), s0_ex.vals, sizeof(datatype) * s0_ex.total_size, cudaMemcpyDeviceToHost));
    std::ofstream files(output_path + "s.bin", std::ios::binary);
    files.write(reinterpret_cast<char*>(&n), sizeof(int));
    int one = 1;
    files.write(reinterpret_cast<char*>(&one), sizeof(int));
    files.write(reinterpret_cast<char*>(s_h.data()), sizeof(double) * n);
    files.close();

    s0.vals = nullptr;
    s.vals = nullptr;
    return;
}

PYBIND11_MODULE(XM, m) {
    m.doc() = "pybind11 for XM"; 
    m.def("solve", &solve, "XM main function");
    m.def("solve_mixed", &solve_mixed, "XM mixed manifold function");
    m.def("solve_Ronly", &solve_Ronly, "Only rotation XM main function");
    m.def("solve_rank3", &solve_rank3, "XM main function for rank 3 only");
}