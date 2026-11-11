#ifdef __INTEL_COMPILER
#pragma diag_suppress 20011
#endif

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
#include <Dense/eig.h>
#include <Sparse/spmatmul.h>
#include <Sparse/sparseformat.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>

#include <chrono>
#include <vector>
#include <fstream>
#include <Optimization/optimization.h>
#include <cusolverSp.h>

using namespace std::chrono;

static inline void write_scalar_bin(const std::string& path, double value) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "[checkeig] Failed to open " << path << " for writing." << std::endl;
        return;
    }
    int rows = 1;
    int cols = 1;
    file.write(reinterpret_cast<char*>(&rows), sizeof(int));
    file.write(reinterpret_cast<char*>(&cols), sizeof(int));
    file.write(reinterpret_cast<char*>(&value), sizeof(double));
    file.close();
}

template <typename T>
__global__ void ConstructZmatrixKernal(T* Z, T* sR, size_s n,size_s o,double lam){
    size_s i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<n){
        double norm = 0;
        for(int j = 0; j<o; ++j){
            norm += sR[j*3*n+3*i] * sR[j*3*n+3*i];
        }
        Z[3*i*3*n+3*i] += 2 * lam * (norm - 1);
    }  
}

bool checkeig(opt_var &C,
              const opt_var &sR,
              const double lam,
              opt_var &v,
              double primal_value,
              const std::string& output_path,
              double sphere_norm,
              const size_s* sphere_indices,
              size_s m_sphere){
    // initialize
    DeviceBlasHandle CUOPT_blas_handle;
    CUOPT_blas_handle.activate();
    DeviceSolverSpHandle CUOPT_cusloversp_handle;
    CUOPT_cusloversp_handle.activate();
    DeviceSolverDnHandle CUOPT_cusloverdn_handle;
    CUOPT_cusloverdn_handle.activate();
    DeviceSparseHandle CUOPT_sparse_handle;
    CUOPT_sparse_handle.activate();

    size_s n = C.dimensions[0] / 3;
    size_s o = sR.dimensions[1];

    // Select sphere indices (default: all nodes)
    std::vector<size_s> sphere_idx;
    if (sphere_indices == nullptr || m_sphere == 0 || m_sphere > n) {
        sphere_idx.resize(n);
        for (size_s i = 0; i < n; ++i) sphere_idx[i] = i;
    } else {
        sphere_idx.assign(sphere_indices, sphere_indices + m_sphere);
    }

    const size_s m = 5*n + 1;

    const size_l nnz_cap = (3*n*o) + (10*n*o) + 64;

    std::vector<int>      Acellrow(nnz_cap, 0);
    std::vector<int>      Acellcol(nnz_cap, 0);
    std::vector<int>      AcellcolCSC(m + 1, 0);   // = 5*n+2
    std::vector<datatype> Acellval(nnz_cap, 0);


    // CONSTRUCTING THE LEFT MATRIX 
    // Left = [];
    // % new try
    // for i = 1:order
    //     V = Acell'*sR_est(i,:)';
    //     V = reshape(V,3*n,[]);
    //     Left = [Left;V];
    // end
    size_l count = 0;
    size_l nnz = 0;

    for (size_s k = 0; k < o; ++k) {
        for (size_s r_idx = 0; r_idx < sphere_idx.size(); ++r_idx) {
            size_s r = sphere_idx[r_idx];
            Acellrow[nnz] = 3 * r + k*3*n;         // row in (3*o*n)
            Acellcol[nnz] = count;             // col 0
            Acellval[nnz] = sR.vals_host[3 * r + k*3*n];
            nnz++;
        }
    }
    count++;
    AcellcolCSC[count] = (int)nnz;

    for(size_s i = 0; i < n; ++i){
        for(size_s k = 0; k<o; ++k){
            Acellrow[nnz] = 3*i + k*3*n;
            Acellcol[nnz] = count;
            Acellval[nnz] = sR.vals_host[3*i + k*3*n] * 0.5;
            nnz += 1;
            Acellrow[nnz] = 3*i+1 + k*3*n;
            Acellcol[nnz] = count;
            Acellval[nnz] = sR.vals_host[3*i+1 + k*3*n] * -0.5;
            nnz += 1;
        }
        count += 1;
        AcellcolCSC[count] = nnz;
        for(size_s k = 0; k<o; ++k){
            Acellrow[nnz] = 3*i+1 + k*3*n;
            Acellcol[nnz] = count;
            Acellval[nnz] = sR.vals_host[3*i+1 + k*3*n] * 0.5;
            nnz += 1;
            Acellrow[nnz] = 3*i+2 + k*3*n;
            Acellcol[nnz] = count;
            Acellval[nnz] = sR.vals_host[3*i+2 + k*3*n] * -0.5;
            nnz += 1;
        }
        count += 1;
        AcellcolCSC[count] = nnz;
        for(size_s k = 0; k<o; ++k){
            Acellrow[nnz] = 3*i + k*3*n;
            Acellcol[nnz] = count;
            Acellval[nnz] = sR.vals_host[3*i+1 + k*3*n] * 0.5;
            nnz += 1;
            Acellrow[nnz] = 3*i+1 + k*3*n;
            Acellcol[nnz] = count;
            Acellval[nnz] = sR.vals_host[3*i + k*3*n] * 0.5;
            nnz += 1;
        }
        count += 1;
        AcellcolCSC[count] = nnz;
        for(size_s k = 0; k<o; ++k){
            Acellrow[nnz] = 3*i + k*3*n;
            Acellcol[nnz] = count;
            Acellval[nnz] = sR.vals_host[3*i+2 + k*3*n] * 0.5;
            nnz += 1;
            Acellrow[nnz] = 3*i+2 + k*3*n;
            Acellcol[nnz] = count;
            Acellval[nnz] = sR.vals_host[3*i + k*3*n] * 0.5;
            nnz += 1;
        }
        count += 1;
        AcellcolCSC[count] = nnz;
        for(size_s k = 0; k<o; ++k){
            Acellrow[nnz] = 3*i+1 + k*3*n;
            Acellcol[nnz] = count;
            Acellval[nnz] = sR.vals_host[3*i+2 + k*3*n] * 0.5;
            nnz += 1;
            Acellrow[nnz] = 3*i+2 + k*3*n;
            Acellcol[nnz] = count;
            Acellval[nnz] = sR.vals_host[3*i+1 + k*3*n] * 0.5;
            nnz += 1;
        }
        count += 1;
        AcellcolCSC[count] = nnz;
    }
    

    // for(int i = count + 1; i<5*n+1; ++i){
    //     AcellcolCSC[i] = nnz;
    // }

    DeviceSpMatCSC<double> AcellCSC(0,3*o*n,5*n+1,nnz);
    DeviceSpMatCSR<double> AcellCSR(0,3*o*n,5*n+1,nnz);
    AcellCSC.SynchronizeHostToDevice(AcellcolCSC.data(),Acellrow.data(),Acellval.data());
    CSC2CSR(CUOPT_sparse_handle,AcellCSC,AcellCSR);

    opt_var Z(C);
    ConstructZmatrixKernal<<<(n + 1024 - 1) / 1024,1024>>>(Z.vals,sR.vals,n,o,lam);
    opt_var Right({3*n,o});
    DnMatDnMat(CUOPT_blas_handle,Right,Z,sR);
    Z.SynchronizeDevicetoHost();
    Right.SynchronizeDevicetoHost();

    // test: sparse multiplication ATA on CUDA
    
    // test: directly use eigen

    Eigen::SparseMatrix<double> Acell_Eigen(3 * o * n, 5 * n + 1);
    Eigen::VectorXd b(3 * o * n), y;

    memcpy(b.data(), Right.vals_host, 3 * o * n * sizeof(double));

    Acell_Eigen.resizeNonZeros(nnz);
    memcpy(Acell_Eigen.valuePtr(), Acellval.data(), nnz * sizeof(double));
    memcpy(Acell_Eigen.innerIndexPtr(), Acellrow.data(), nnz * sizeof(int));
    memcpy(Acell_Eigen.outerIndexPtr(), AcellcolCSC.data(), (5 * n + 2) * sizeof(int));

    Acell_Eigen.makeCompressed();

    Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double>> solver;

    auto start_decomp = std::chrono::high_resolution_clock::now();
    solver.compute(Acell_Eigen);
    auto end_decomp = std::chrono::high_resolution_clock::now();

    if (solver.info() != Eigen::Success) {
        std::cerr << "Decomposition failed" << std::endl;
        return -1;
    }

    auto start_solve = std::chrono::high_resolution_clock::now();
    y = solver.solve(b);
    auto end_solve = std::chrono::high_resolution_clock::now();

    if (solver.info() != Eigen::Success) {
        std::cerr << "Solving failed" << std::endl;
        return -1;
    }

    auto decomp_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_decomp - start_decomp);
    auto solve_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_solve - start_solve);

    std::cout << "Decomposition time: " << decomp_duration.count() << " microseconds" << std::endl;
    std::cout << "Solve time: " << solve_duration.count() << " microseconds" << std::endl;

    
    // opt_var y({5*n+1});
    // DeviceDnTen<int> p({5*n+1});
    // cusparseMatDescr_t descrAcell;
    // cusparseCreateMatDescr(&descrAcell);
    // CHECK_CUSPARSE( cusparseSetMatIndexBase(descrAcell, CUSPARSE_INDEX_BASE_ZERO) );
    // CHECK_CUSPARSE( cusparseSetMatType(descrAcell, CUSPARSE_MATRIX_TYPE_GENERAL) );
    // CHECK_CUSPARSE( cusparseSetMatDiagType(descrAcell,CUSPARSE_DIAG_TYPE_NON_UNIT) );
    // // TURN all vraiable to host
    // double *h_vals = (double*) malloc(nnz * sizeof(double));
    // int *h_row_ptrs = (int*) malloc((3*o*n + 1) * sizeof(int));
    // int *h_col_ids = (int*) malloc(nnz * sizeof(int));
    // double *h_right_vals = (double*) malloc(3*o*n * sizeof(double));
    // // Copy AcellCSR.vals (now avell)
    // cudaMemcpy(h_vals, AcellCSR.vals, nnz * sizeof(double), cudaMemcpyDeviceToHost);

    // // Copy AcellCSR.row_ptrs
    // cudaMemcpy(h_row_ptrs, AcellCSR.row_ptrs, (3*o*n + 1) * sizeof(int), cudaMemcpyDeviceToHost);

    // // Copy AcellCSR.col_ids
    // cudaMemcpy(h_col_ids, AcellCSR.col_ids, nnz * sizeof(int), cudaMemcpyDeviceToHost);

    // // Copy Right.vals
    // cudaMemcpy(h_right_vals, Right.vals, 3*o*n * sizeof(double), cudaMemcpyDeviceToHost);

    // std::vector<double> y(5*n+1,0);
    // std::vector<int> p(5*n+1,0);

    // auto start = std::chrono::high_resolution_clock::now();
    // CHECK_CUSOLVER( cusolverSpDcsrlsqvqrHost(CUOPT_cusloversp_handle.cusolver_sp_handle,3*o*n,5*n+1,nnz,descrAcell,h_vals,h_row_ptrs,h_col_ids,h_right_vals,1e-15,&rankLeft,y.data(),p.data(),&minnorm));
    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // std::cout << "cusolverSpDcsrlsqvqrHost execution time: " << duration.count() << " microseconds" << std::endl;

    // std::cout<<"Minnorm of Least-square is: "<<minnorm<<std::endl;
    
    count = 0;

    for (size_s d_idx = 0; d_idx < sphere_idx.size(); ++d_idx) {
        size_s d = sphere_idx[d_idx];
        Z.vals_host[3 * d + 3 * d * 3 * n] -= y[count];
    }
    count += 1;

    for(size_s i = 0; i < n; ++i){
        Z.vals_host[3*i*3*n + 3*i] -= y[count] * 0.5;
        Z.vals_host[(3*i+1)*3*n + (3*i+1)] -= y[count] * -0.5;
        count += 1;
        
        Z.vals_host[(3*i+1)*3*n + (3*i+1)] -= y[count] * 0.5;
        Z.vals_host[(3*i+2)*3*n + (3*i+2)] -= y[count] * -0.5;
        count += 1;
        
        Z.vals_host[3*i*3*n + 3*i+1] -= y[count] * 0.5;
        Z.vals_host[(3*i+1)*3*n + 3*i] -= y[count] * 0.5;
        count += 1;

        Z.vals_host[3*i*3*n + 3*i+2] -= y[count] * 0.5;
        Z.vals_host[(3*i+2)*3*n + 3*i] -= y[count] * 0.5;
        count += 1;

        Z.vals_host[(3*i+1)*3*n + 3*i+2] -= y[count] * 0.5;
        Z.vals_host[(3*i+2)*3*n + 3*i+1] -= y[count] * 0.5;
        count += 1;
    }
    Z.SynchronizeHostToDevice();

    // eigen value of z
    SingleEigParam eigparam;
    size_t buffer_size_eig = 0;
    size_t buffer_size_host_eig = 0;
    opt_var W({3*n});
    single_eig_get_buffersize_cusolver(CUOPT_cusloverdn_handle,eigparam,Z,W,3*n,&buffer_size_eig,&buffer_size_host_eig);
    //printf("buffer_size_eig: %ld\n",buffer_size_eig);
    void* buffer_eig = nullptr;
    void* buffer_host_eig = (void*)malloc(buffer_size_host_eig);
    int* info = nullptr;
    CHECK_CUDA(cudaMalloc(&buffer_eig,buffer_size_eig));
    CHECK_CUDA(cudaMalloc(&info,sizeof(int)));
    
    single_eig_cusolver(CUOPT_cusloverdn_handle,eigparam,Z,W,buffer_eig,buffer_host_eig,info,3*n,buffer_size_eig,buffer_size_host_eig);
    W.SynchronizeDevicetoHost();
    printf("The min eig is: %1.3e \n",W.vals_host[0]);
    CHECK_CUDA(cudaMemcpy(v.vals,Z.vals,v.total_size * sizeof(datatype),cudaMemcpyDeviceToDevice));
    
    // calculating optimility gap
    std::cout<<"Primal value: "<<primal_value<<std::endl;
    double dual_value = sphere_norm * sphere_norm * y[0];
    std::cout<<"Dual value: "<<dual_value<<std::endl;
    std::vector<double> xii(n,0);
    for(size_s i = 0; i<n; ++i){
        for(size_s j = 0; j<o; ++j){
            xii[i] += sR.vals_host[3*i+3*j*n] * sR.vals_host[3*i+3*j*n];
        }
    }
    for(size_s i = 0; i<n; ++i){
        dual_value += (1 - xii[i]*xii[i])*lam;
    }
    std::cout<<"new dual"<<dual_value<<std::endl;

    CHECK_CUDA(cudaFree(buffer_eig));
    free(buffer_host_eig);
    CHECK_CUDA(cudaFree(info));
    // free(h_vals);
    // free(h_row_ptrs);
    // free(h_col_ids);
    // free(h_right_vals);

    // cusparseDestroyMatDescr(descrAcell);

    double min_eig_bound = 1e-8;
    if(n > 2000){
        min_eig_bound = 1e-3;
    }
    else if(n > 5000){
        min_eig_bound = 1e-1;
    }
    else if(n > 10000){
        min_eig_bound = 100;
    }

    std::vector<char> sphere_seen(n, 0);
    size_s sphere_unique_count = 0;
    for (size_s idx : sphere_idx) {
        if (idx < n && !sphere_seen[idx]) {
            sphere_seen[idx] = 1;
            sphere_unique_count++;
        }
    }

    const bool full_sphere_trace_bound = (sphere_unique_count == n);
    const double min_eig = (double)W.vals_host[0];
    double gap = primal_value - dual_value;
    if (full_sphere_trace_bound) {
        const double K = 3.0 * sphere_norm * sphere_norm;
        gap -= K * std::min(0.0, min_eig);
    } else if (min_eig < -min_eig_bound) {
        std::cout << "[checkeig] Mixed positive scales have no finite trace bound; "
                  << "dual gap requires PSD Z." << std::endl;
    }
    std::cout<<"Optimility gap: "<<gap<<std::endl;

    const bool dual_psd = min_eig > -min_eig_bound;
    const bool gap_valid = full_sphere_trace_bound || dual_psd;
    if((gap_valid && gap/primal_value < 1e-3) || dual_psd){
        std::cout<<"BM finished with rank "<< o <<std::endl;
        return true;
    }
    else{
        std::cout<<"BM order plus one"<<std::endl;
        CHECK_CUDA(cudaMemcpy(v.vals,Z.vals,v.total_size * sizeof(datatype),cudaMemcpyDeviceToDevice));
        return false;
    }
    
    // opt_var X({5*n+1});
    // std::vector<datatype> X_h(5*n+1,0);
    // //create rand dense vector
    // for(size_s i = 0; i<5*n+1; ++i){
    //     X_h[i] = i + 1.0;
    // }
    // X.SynchronizeHostToDevice(X_h.data());
    // X.print();
    // opt_var Y({3*o*n});

    // size_t buffer_size = x`_get_buffersize_cusparse(CUOPT_sparse_handle,AcellCSR,X,Y,1.0,0.0);
    // void* buffer = nullptr;  
    // cudaMalloc(&buffer,buffer_size);
    // SpMatDnVec(CUOPT_sparse_handle,AcellCSR,X,Y,1.0,0.0,buffer);
    // CHECK_CUDA(cudaDeviceSynchronize());
    // Y.print();

    
}

bool checkeig_noscale(opt_var &C,
              const opt_var &sR,
              const double lam,
              opt_var &v,
              double primal_value,
              const std::string& output_path) {

    DeviceBlasHandle CUOPT_blas_handle;      CUOPT_blas_handle.activate();
    DeviceSolverSpHandle CUOPT_cusloversp_handle; CUOPT_cusloversp_handle.activate();
    DeviceSolverDnHandle CUOPT_cusloverdn_handle; CUOPT_cusloverdn_handle.activate();
    DeviceSparseHandle CUOPT_sparse_handle;  CUOPT_sparse_handle.activate();

    size_s n = C.dimensions[0] / 3;
    size_s o = sR.dimensions[1];

    // NEW: 6 constraints per node i: (x11=1,x22=1,x33=1,x12=0,x13=0,x23=0)
    const size_s m = 6 * n;

    // nnz per node:
    // 3 diag columns -> each has o entries => 3*o
    // 3 offdiag columns -> each has 2*o entries => 6*o
    // total nnz = 9*n*o
    const size_l nnz_cap = (9 * (size_l)n * (size_l)o) + 64;

    std::vector<int>      Acellrow(nnz_cap, 0);
    std::vector<int>      Acellcol(nnz_cap, 0);
    std::vector<int>      AcellcolCSC(m + 1, 0);   // outer ptr, size m+1
    std::vector<datatype> Acellval(nnz_cap, 0);

    // ------------------------------
    // Build Acell (CSC) with m=6n cols
    // Column ordering per node i:
    //   c0: x11
    //   c1: x22
    //   c2: x33
    //   c3: x12
    //   c4: x13
    //   c5: x23
    // ------------------------------
    size_l count = 0;
    size_l nnz   = 0;

    for (size_s i = 0; i < n; ++i) {

        // (1) x11 = sum_k sR(3i+0,k)^2  -> column uses entries sR(3i+0,k)
        for (size_s k = 0; k < o; ++k) {
            Acellrow[nnz] = (3 * i + 0) + k * 3 * n;
            Acellcol[nnz] = (int)count;
            Acellval[nnz] = sR.vals_host[(3 * i + 0) + k * 3 * n];
            nnz++;
        }
        count++;
        AcellcolCSC[count] = (int)nnz;

        // (2) x22
        for (size_s k = 0; k < o; ++k) {
            Acellrow[nnz] = (3 * i + 1) + k * 3 * n;
            Acellcol[nnz] = (int)count;
            Acellval[nnz] = sR.vals_host[(3 * i + 1) + k * 3 * n];
            nnz++;
        }
        count++;
        AcellcolCSC[count] = (int)nnz;

        // (3) x33
        for (size_s k = 0; k < o; ++k) {
            Acellrow[nnz] = (3 * i + 2) + k * 3 * n;
            Acellcol[nnz] = (int)count;
            Acellval[nnz] = sR.vals_host[(3 * i + 2) + k * 3 * n];
            nnz++;
        }
        count++;
        AcellcolCSC[count] = (int)nnz;

        // (4) x12 = sum_k sR0*sR1  -> symmetric 0.5 trick
        for (size_s k = 0; k < o; ++k) {
            Acellrow[nnz] = (3 * i + 0) + k * 3 * n;
            Acellcol[nnz] = (int)count;
            Acellval[nnz] = sR.vals_host[(3 * i + 1) + k * 3 * n] * (datatype)0.5;
            nnz++;

            Acellrow[nnz] = (3 * i + 1) + k * 3 * n;
            Acellcol[nnz] = (int)count;
            Acellval[nnz] = sR.vals_host[(3 * i + 0) + k * 3 * n] * (datatype)0.5;
            nnz++;
        }
        count++;
        AcellcolCSC[count] = (int)nnz;

        // (5) x13
        for (size_s k = 0; k < o; ++k) {
            Acellrow[nnz] = (3 * i + 0) + k * 3 * n;
            Acellcol[nnz] = (int)count;
            Acellval[nnz] = sR.vals_host[(3 * i + 2) + k * 3 * n] * (datatype)0.5;
            nnz++;

            Acellrow[nnz] = (3 * i + 2) + k * 3 * n;
            Acellcol[nnz] = (int)count;
            Acellval[nnz] = sR.vals_host[(3 * i + 0) + k * 3 * n] * (datatype)0.5;
            nnz++;
        }
        count++;
        AcellcolCSC[count] = (int)nnz;

        // (6) x23
        for (size_s k = 0; k < o; ++k) {
            Acellrow[nnz] = (3 * i + 1) + k * 3 * n;
            Acellcol[nnz] = (int)count;
            Acellval[nnz] = sR.vals_host[(3 * i + 2) + k * 3 * n] * (datatype)0.5;
            nnz++;

            Acellrow[nnz] = (3 * i + 2) + k * 3 * n;
            Acellcol[nnz] = (int)count;
            Acellval[nnz] = sR.vals_host[(3 * i + 1) + k * 3 * n] * (datatype)0.5;
            nnz++;
        }
        count++;
        AcellcolCSC[count] = (int)nnz;
    }

    if ((size_s)count != m) {
        std::cerr << "[checkeig] ERROR: constraint count mismatch, count=" << count
                  << " expected m=" << m << std::endl;
        return false;
    }
    if (nnz > nnz_cap) {
        std::cerr << "[checkeig] ERROR: nnz overflow, nnz=" << nnz
                  << " nnz_cap=" << nnz_cap << std::endl;
        return false;
    }

    // Build device sparse
    DeviceSpMatCSC<double> AcellCSC(0, 3 * o * n, m, nnz);
    DeviceSpMatCSR<double> AcellCSR(0, 3 * o * n, m, nnz);
    AcellCSC.SynchronizeHostToDevice(AcellcolCSC.data(), Acellrow.data(), Acellval.data());
    CSC2CSR(CUOPT_sparse_handle, AcellCSC, AcellCSR);

    // Z = C (+ optional augmentation), Right = Z*sR, b = vec(Right)
    opt_var Z(C);


    opt_var Right({3 * n, o});
    DnMatDnMat(CUOPT_blas_handle, Right, Z, sR);

    Z.SynchronizeDevicetoHost();
    Right.SynchronizeDevicetoHost();

    // Solve least squares on host via Eigen
    Eigen::SparseMatrix<double> Acell_Eigen(3 * o * n, m);
    Eigen::VectorXd b(3 * o * n), y;

    memcpy(b.data(), Right.vals_host, (size_t)(3 * o * n) * sizeof(double));

    Acell_Eigen.resizeNonZeros(nnz);
    memcpy(Acell_Eigen.valuePtr(),      Acellval.data(),   (size_t)nnz * sizeof(double));
    memcpy(Acell_Eigen.innerIndexPtr(), Acellrow.data(),   (size_t)nnz * sizeof(int));
    memcpy(Acell_Eigen.outerIndexPtr(), AcellcolCSC.data(), (size_t)(m + 1) * sizeof(int));
    Acell_Eigen.makeCompressed();

    Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double>> solver;

    auto start_decomp = std::chrono::high_resolution_clock::now();
    solver.compute(Acell_Eigen);
    auto end_decomp = std::chrono::high_resolution_clock::now();

    if (solver.info() != Eigen::Success) {
        std::cerr << "Decomposition failed" << std::endl;
        return false;
    }

    auto start_solve = std::chrono::high_resolution_clock::now();
    y = solver.solve(b);
    auto end_solve = std::chrono::high_resolution_clock::now();

    if (solver.info() != Eigen::Success) {
        std::cerr << "Solving failed" << std::endl;
        return false;
    }

    auto decomp_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_decomp - start_decomp);
    auto solve_duration  = std::chrono::duration_cast<std::chrono::microseconds>(end_solve - start_solve);

    std::cout << "Decomposition time: " << decomp_duration.count() << " microseconds" << std::endl;
    std::cout << "Solve time: " << solve_duration.count() << " microseconds" << std::endl;

    // Apply dual variables y to Z on host:
    // Z = Z - sum_j y_j * A_j  (in matrix form)
    size_l idx = 0;
    for (size_s i = 0; i < n; ++i) {
        // diag multipliers (x11, x22, x33)
        Z.vals_host[(3 * i + 0) + (3 * i + 0) * 3 * n] -= y[idx++]; // x11
        Z.vals_host[(3 * i + 1) + (3 * i + 1) * 3 * n] -= y[idx++]; // x22
        Z.vals_host[(3 * i + 2) + (3 * i + 2) * 3 * n] -= y[idx++]; // x33

        // off-diag multipliers (x12, x13, x23) with 0.5 symmetric split
        double y12 = y[idx++], y13 = y[idx++], y23 = y[idx++];

        Z.vals_host[(3 * i + 0) + (3 * i + 1) * 3 * n] -= 0.5 * y12;
        Z.vals_host[(3 * i + 1) + (3 * i + 0) * 3 * n] -= 0.5 * y12;

        Z.vals_host[(3 * i + 0) + (3 * i + 2) * 3 * n] -= 0.5 * y13;
        Z.vals_host[(3 * i + 2) + (3 * i + 0) * 3 * n] -= 0.5 * y13;

        Z.vals_host[(3 * i + 1) + (3 * i + 2) * 3 * n] -= 0.5 * y23;
        Z.vals_host[(3 * i + 2) + (3 * i + 1) * 3 * n] -= 0.5 * y23;
    }

    if (idx != (size_l)m) {
        std::cerr << "[checkeig] ERROR: y indexing mismatch idx=" << idx << " m=" << m << std::endl;
        return false;
    }

    Z.SynchronizeHostToDevice();

    // eigen value of Z
    SingleEigParam eigparam;
    size_t buffer_size_eig = 0;
    size_t buffer_size_host_eig = 0;
    opt_var W({3 * n});

    single_eig_get_buffersize_cusolver(CUOPT_cusloverdn_handle, eigparam, Z, W, 3 * n,
                                       &buffer_size_eig, &buffer_size_host_eig);

    void* buffer_eig = nullptr;
    void* buffer_host_eig = (void*)malloc(buffer_size_host_eig);
    int* info = nullptr;

    CHECK_CUDA(cudaMalloc(&buffer_eig, buffer_size_eig));
    CHECK_CUDA(cudaMalloc(&info, sizeof(int)));

    single_eig_cusolver(CUOPT_cusloverdn_handle, eigparam, Z, W,
                        buffer_eig, buffer_host_eig, info,
                        3 * n, buffer_size_eig, buffer_size_host_eig);

    W.SynchronizeDevicetoHost();
    printf("The min eig is: %1.3e \n", W.vals_host[0]);

    CHECK_CUDA(cudaMemcpy(v.vals, Z.vals, v.total_size * sizeof(datatype), cudaMemcpyDeviceToDevice));

    // ---- Dual value / gap ----
    std::cout << "Primal value: " << primal_value << std::endl;

    // RHS constants: x11=1, x22=1, x33=1 for each node; off-diag RHS=0
    double dual_value = 0.0;
    for (size_s i = 0; i < n; ++i) {
        dual_value += y[6 * i + 0] + y[6 * i + 1] + y[6 * i + 2];
    }
    std::cout << "Dual value: " << dual_value << std::endl;

    std::vector<double> xii(n, 0.0);
    for (size_s i = 0; i < n; ++i) {
        for (size_s j = 0; j < o; ++j) {
            double a = sR.vals_host[3 * i + 3 * j * n];
            xii[i] += a * a;
        }
    }
    for (size_s i = 0; i < n; ++i) {
        dual_value += (1.0 - xii[i] * xii[i]) * lam;
    }
    std::cout << "new dual " << dual_value << std::endl;

    double bar_s = 1.0;
    double K = 3.0 * n * bar_s * bar_s;
    double gap = primal_value - dual_value - K * std::min(0.0, (double)W.vals_host[0]);
    std::cout << "Optimility gap: " << gap << std::endl;

    CHECK_CUDA(cudaFree(buffer_eig));
    free(buffer_host_eig);
    CHECK_CUDA(cudaFree(info));

    double min_eig_bound = 1e-8;
    if (n > 2000) {
        min_eig_bound = 1e-3;
    } else if (n > 5000) {
        min_eig_bound = 1e-1;
    } else if (n > 10000) {
        min_eig_bound = 100;
    }

    if (std::abs(gap) / std::max(std::abs(primal_value), 1e-300) < 1e-3 || W.vals_host[0] > -min_eig_bound) {
        std::cout << "BM finished with rank " << o << std::endl;
        return true;
    } else {
        std::cout << "BM order plus one" << std::endl;
        CHECK_CUDA(cudaMemcpy(v.vals, Z.vals, v.total_size * sizeof(datatype), cudaMemcpyDeviceToDevice));        return false;
    }
}
