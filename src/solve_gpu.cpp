#include <iostream>     // std::cerr
#include <cstdlib>      // std::exit, EXIT_FAILURE

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cuComplex.h>
#include <cublas_v2.h>

void solve_gpu(cuDoubleComplex* A_host, cuDoubleComplex* b_host, int N) {
    cusolverDnHandle_t handle;
    cuDoubleComplex *A_dev = nullptr, *b_dev = nullptr, *work_dev = nullptr;
    int *pivot_dev = nullptr, *info_dev = nullptr;
    int work_size = 0, info_host = 0;

    // Create cuSolver handle
    if (cusolverDnCreate(&handle) != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "solve_gpu: Failed to create cuSolver handle.\n";
        std::exit(EXIT_FAILURE);
    }

    // Allocate device memory
    cudaMalloc(&A_dev, sizeof(cuDoubleComplex) * N * N);
    cudaMalloc(&b_dev, sizeof(cuDoubleComplex) * N);
    cudaMalloc(&pivot_dev, sizeof(int) * N);
    cudaMalloc(&info_dev, sizeof(int));

    // Copy inputs to device
    cudaMemcpy(A_dev, A_host, sizeof(cuDoubleComplex) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(b_dev, b_host, sizeof(cuDoubleComplex) * N, cudaMemcpyHostToDevice);

    // Query workspace size and allocate
    cusolverDnZgetrf_bufferSize(handle, N, N, A_dev, N, &work_size);
    cudaMalloc(&work_dev, sizeof(cuDoubleComplex) * work_size);

    // Perform LU factorization and solve
    cusolverDnZgetrf(handle, N, N, A_dev, N, work_dev, pivot_dev, info_dev);
    cusolverDnZgetrs(handle, CUBLAS_OP_N, N, 1, A_dev, N, pivot_dev, b_dev, N, info_dev);

    // Copy solution back to host
    cudaMemcpy(b_host, b_dev, sizeof(cuDoubleComplex) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(&info_host, info_dev, sizeof(int), cudaMemcpyDeviceToHost);

    if (info_host != 0) {
        std::cerr << "solve_gpu: solver returned error code " << info_host << "\n";
        std::exit(EXIT_FAILURE);
    }

    // Free resources
    cudaFree(A_dev);
    cudaFree(b_dev);
    cudaFree(pivot_dev);
    cudaFree(info_dev);
    cudaFree(work_dev);
    cusolverDnDestroy(handle);
}

void invert_matrix_gpu(cuDoubleComplex* A_host, int N) {
    cusolverDnHandle_t handle;
    cublasHandle_t cublas_handle;
    cuDoubleComplex *A_dev = nullptr, *work_dev = nullptr;
    int *pivot_dev = nullptr, *info_dev = nullptr;
    int work_size = 0, info_host = 0;

    // Create handles
    if (cusolverDnCreate(&handle) != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "invert_matrix_gpu: Failed to create cuSolver handle.\n";
        std::exit(EXIT_FAILURE);
    }
    if (cublasCreate(&cublas_handle) != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "invert_matrix_gpu: Failed to create cuBLAS handle.\n";
        std::exit(EXIT_FAILURE);
    }

    // Allocate device memory
    cudaMalloc(&A_dev, sizeof(cuDoubleComplex) * N * N);
    cudaMalloc(&pivot_dev, sizeof(int) * N);
    cudaMalloc(&info_dev, sizeof(int));

    // Copy matrix to device
    cudaMemcpy(A_dev, A_host, sizeof(cuDoubleComplex) * N * N, cudaMemcpyHostToDevice);

    // Get workspace size and allocate
    cusolverDnZgetrf_bufferSize(handle, N, N, A_dev, N, &work_size);
    cudaMalloc(&work_dev, sizeof(cuDoubleComplex) * work_size);

    // Compute LU factorization
    cusolverDnZgetrf(handle, N, N, A_dev, N, work_dev, pivot_dev, info_dev);
    cudaMemcpy(&info_host, info_dev, sizeof(int), cudaMemcpyDeviceToHost);
    if (info_host != 0) {
        std::cerr << "invert_matrix_gpu: LU factorization failed with error " << info_host << "\n";
        std::exit(EXIT_FAILURE);
    }

    // Compute inverse using LU factorization
    cusolverDnZgetri(handle, N, A_dev, N, pivot_dev, work_dev, work_size, info_dev);
    cudaMemcpy(&info_host, info_dev, sizeof(int), cudaMemcpyDeviceToHost);
    if (info_host != 0) {
        std::cerr << "invert_matrix_gpu: Matrix inversion failed with error " << info_host << "\n";
        std::exit(EXIT_FAILURE);
    }

    // Copy result back to host
    cudaMemcpy(A_host, A_dev, sizeof(cuDoubleComplex) * N * N, cudaMemcpyHostToDevice);

    // Free resources
    cudaFree(A_dev);
    cudaFree(pivot_dev);
    cudaFree(info_dev);
    cudaFree(work_dev);
    cusolverDnDestroy(handle);
    cublasDestroy(cublas_handle);
}
