#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cuComplex.h>

cuDoubleComplex* load_matrix(const char* filename, int& N) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) { std::cerr << "Failed to open " << filename << "\n"; std::exit(1); }

    in.read(reinterpret_cast<char*>(&N), sizeof(int));
    cuDoubleComplex* data = new cuDoubleComplex[N * N];
    in.read(reinterpret_cast<char*>(data), sizeof(cuDoubleComplex) * N * N);
    in.close();
    return data;
}

cuDoubleComplex* load_vector(const char* filename, int& N) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) { std::cerr << "Failed to open " << filename << "\n"; std::exit(1); }

    in.read(reinterpret_cast<char*>(&N), sizeof(int));
    cuDoubleComplex* data = new cuDoubleComplex[N];
    in.read(reinterpret_cast<char*>(data), sizeof(cuDoubleComplex) * N);
    in.close();
    return data;
}

void write_polarizations(const char* filename, cuDoubleComplex* p, int N) {
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Error: could not open " << filename << " for writing\n";
        std::exit(1);
    }

    // CSV header
    out << "Re_px,Im_px,Re_py,Im_py,Re_pz,Im_pz\n";

    out << std::scientific << std::setprecision(6);
    for (int j = 0; j < N; ++j) {
        int idx = 3 * j;
        out
            << cuCreal(p[idx + 0]) << ',' << cuCimag(p[idx + 0]) << ','  // x
            << cuCreal(p[idx + 1]) << ',' << cuCimag(p[idx + 1]) << ','  // y
            << cuCreal(p[idx + 2]) << ',' << cuCimag(p[idx + 2]) << '\n'; // z
    }

    out.close();
}

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

int main() {
    auto start_time = std::chrono::high_resolution_clock::now();

    const int N_width = 100;
    const int N_height = 100;
    const int N = N_width * N_width;

    int dimension = 3*N;
    
    cuDoubleComplex* A = load_matrix("A.bin", dimension);
    cuDoubleComplex* b = load_vector("b.bin", dimension);

    solve_gpu(A, b, dimension);

    // Write solution out to a file.
    // Assume b is overwritten with solution
    write_polarizations("output/output.txt", b, N);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto ms_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Elapsed: " << ms_duration.count() << " ms\n";
    return 0;
}
