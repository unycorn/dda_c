#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cuComplex.h>

void solve_gpu(cuDoubleComplex* A_host, cuDoubleComplex* b_host, int N) {
    cusolverDnHandle_t handle;
    cuDoubleComplex *A_dev = nullptr, *b_dev = nullptr, *work_dev = nullptr;
    int *pivot_dev = nullptr, *info_dev = nullptr;
    int work_size = 0, info_host = 0;

    cusolverDnCreate(&handle);

    cudaMalloc(&A_dev, sizeof(cuDoubleComplex) * N * N);
    cudaMalloc(&b_dev, sizeof(cuDoubleComplex) * N);
    cudaMalloc(&pivot_dev, sizeof(int) * N);
    cudaMalloc(&info_dev, sizeof(int));

    cudaMemcpy(A_dev, A_host, sizeof(cuDoubleComplex) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(b_dev, b_host, sizeof(cuDoubleComplex) * N, cudaMemcpyHostToDevice);

    cusolverDnZgetrf_bufferSize(handle, N, N, A_dev, N, &work_size);
    cudaMalloc(&work_dev, sizeof(cuDoubleComplex) * work_size);

    cusolverDnZgetrf(handle, N, N, A_dev, N, work_dev, pivot_dev, info_dev);
    cusolverDnZgetrs(handle, CUBLAS_OP_N, N, 1, A_dev, N, pivot_dev, b_dev, N, info_dev);

    cudaMemcpy(b_host, b_dev, sizeof(cuDoubleComplex) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(&info_host, info_dev, sizeof(int), cudaMemcpyDeviceToHost);

    if (info_host != 0) {
        std::cerr << "solve_gpu: solver returned error code " << info_host << "\n";
        std::exit(EXIT_FAILURE);
    }

    cudaFree(A_dev);
    cudaFree(b_dev);
    cudaFree(pivot_dev);
    cudaFree(info_dev);
    cudaFree(work_dev);
    cusolverDnDestroy(handle);
}