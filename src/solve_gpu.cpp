#include <iostream>
#include <complex>
#include <vector>
#include <stdexcept>  // Add this for std::runtime_error
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cuComplex.h>

extern "C" {
    void zgetrf_(int* m, int* n, std::complex<double>* a, int* lda, int* ipiv, int* info);
    void zgetrs_(char* trans, int* n, int* nrhs, std::complex<double>* a, int* lda, 
                 int* ipiv, std::complex<double>* b, int* ldb, int* info);
}

// Helper function to check CUDA errors
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

// Helper function to check cuSOLVER errors
#define CHECK_CUSOLVER(call) \
    do { \
        cusolverStatus_t err = call; \
        if (err != CUSOLVER_STATUS_SUCCESS) { \
            std::cerr << "CUSOLVER error in " << __FILE__ << ":" << __LINE__ << ": " \
                      << err << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

// Helper function to check cuBLAS errors
#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t err = call; \
        if (err != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "CUBLAS error in " << __FILE__ << ":" << __LINE__ << ": " \
                      << err << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

void init_cuda_device() {
    cudaFree(0);  // Force CUDA context initialization
}

// Helper to check available GPU memory
bool check_gpu_memory(size_t required_bytes, const char* allocation_type = "") {
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    return free_bytes >= required_bytes;
}

// Cleanup helper
void cleanup_gpu_resources(cusolverDnHandle_t* handle, cublasHandle_t* cublas_handle,
                         cuDoubleComplex* A_dev, cuDoubleComplex* b_dev,
                         cuDoubleComplex* work_dev, int* pivot_dev, int* info_dev) {
    if (handle) cusolverDnDestroy(*handle);
    if (cublas_handle) cublasDestroy(*cublas_handle);
    if (A_dev) cudaFree(A_dev);
    if (b_dev) cudaFree(b_dev);
    if (work_dev) cudaFree(work_dev);
    if (pivot_dev) cudaFree(pivot_dev);
    if (info_dev) cudaFree(info_dev);
}

void solve_gpu(cuDoubleComplex* A_device, cuDoubleComplex* b_host, int N) {
    // Initialize CUDA device first
    init_cuda_device();

    std::cout << "Starting linear system solve for N = " << N << "\n";

    // Calculate memory requirements
    size_t vector_size = (size_t)N * sizeof(cuDoubleComplex);
    size_t pivot_size = (size_t)N * sizeof(int);

    // Get workspace size
    cusolverDnHandle_t handle = nullptr;
    CHECK_CUSOLVER(cusolverDnCreate(&handle));
    
    int work_size = 0;
    CHECK_CUSOLVER(cusolverDnZgetrf_bufferSize(handle, N, N, A_device, N, &work_size));

    // Print memory requirements
    size_t workspace_size = (size_t)work_size * sizeof(cuDoubleComplex);
    std::cout << "Memory requirements:\n";
    std::cout << "- Solution vector: " << vector_size/(1024.0*1024.0*1024.0) << " GB\n";
    std::cout << "- Pivot array: " << pivot_size/(1024.0*1024.0*1024.0) << " GB\n";
    std::cout << "- Solver workspace: " << workspace_size/(1024.0*1024.0*1024.0) << " GB\n";

    // Allocate minimum required memory
    cuDoubleComplex* b_dev = nullptr;
    int* pivot_dev = nullptr;
    int* info_dev = nullptr;
    cuDoubleComplex* work_dev = nullptr;
    int info_host = 0;

    try {
        CHECK_CUDA(cudaMalloc(&b_dev, vector_size));
        CHECK_CUDA(cudaMalloc(&pivot_dev, pivot_size));
        CHECK_CUDA(cudaMalloc(&info_dev, sizeof(int)));
        CHECK_CUDA(cudaMalloc(&work_dev, workspace_size));
        
        // Copy right-hand side to device
        CHECK_CUDA(cudaMemcpy(b_dev, b_host, vector_size, cudaMemcpyHostToDevice));

        // LU factorization (modifies A_device in-place)
        CHECK_CUSOLVER(cusolverDnZgetrf(handle, N, N, A_device, N, work_dev, pivot_dev, info_dev));
        CHECK_CUDA(cudaMemcpy(&info_host, info_dev, sizeof(int), cudaMemcpyDeviceToHost));
        if (info_host != 0) {
            std::cerr << "LU factorization failed with error " << info_host << std::endl;
            throw std::runtime_error("LU factorization failed");
        }

        // Solve system using factored matrix
        CHECK_CUSOLVER(cusolverDnZgetrs(handle, CUBLAS_OP_N, N, 1, A_device, N, pivot_dev, b_dev, N, info_dev));
        CHECK_CUDA(cudaMemcpy(&info_host, info_dev, sizeof(int), cudaMemcpyDeviceToHost));
        if (info_host != 0) {
            std::cerr << "Back substitution failed with error " << info_host << std::endl;
            throw std::runtime_error("Back substitution failed");
        }

        // Copy solution back to host
        CHECK_CUDA(cudaMemcpy(b_host, b_dev, vector_size, cudaMemcpyDeviceToHost));

    } catch (...) {
        cleanup_gpu_resources(&handle, nullptr, nullptr, b_dev, work_dev, pivot_dev, info_dev);
        throw;
    }

    cleanup_gpu_resources(&handle, nullptr, nullptr, b_dev, work_dev, pivot_dev, info_dev);
}

void solve_gpu_multiple_rhs(cuDoubleComplex* A_device, cuDoubleComplex* B_host, int N, int nrhs) {
    // Initialize CUDA device first
    init_cuda_device();

    // Get workspace size
    cusolverDnHandle_t handle = nullptr;
    CHECK_CUSOLVER(cusolverDnCreate(&handle));
    
    int work_size = 0;
    CHECK_CUSOLVER(cusolverDnZgetrf_bufferSize(handle, N, N, A_device, N, &work_size));

    // Calculate memory requirements
    size_t matrix_size = (size_t)N * nrhs * sizeof(cuDoubleComplex);
    size_t pivot_size = (size_t)N * sizeof(int);
    size_t workspace_size = (size_t)work_size * sizeof(cuDoubleComplex);

    // Allocate memory
    cuDoubleComplex* B_dev = nullptr;
    int* pivot_dev = nullptr;
    int* info_dev = nullptr;
    cuDoubleComplex* work_dev = nullptr;
    int info_host = 0;

    try {
        CHECK_CUDA(cudaMalloc(&B_dev, matrix_size));
        CHECK_CUDA(cudaMalloc(&pivot_dev, pivot_size));
        CHECK_CUDA(cudaMalloc(&info_dev, sizeof(int)));
        CHECK_CUDA(cudaMalloc(&work_dev, workspace_size));
        
        // Copy right-hand side matrix to device
        CHECK_CUDA(cudaMemcpy(B_dev, B_host, matrix_size, cudaMemcpyHostToDevice));

        // LU factorization (modifies A_device in-place)
        CHECK_CUSOLVER(cusolverDnZgetrf(handle, N, N, A_device, N, work_dev, pivot_dev, info_dev));
        CHECK_CUDA(cudaMemcpy(&info_host, info_dev, sizeof(int), cudaMemcpyDeviceToHost));
        if (info_host != 0) {
            std::cerr << "LU factorization failed with error " << info_host << std::endl;
            throw std::runtime_error("LU factorization failed");
        }

        // Solve system using factored matrix for multiple RHS
        CHECK_CUSOLVER(cusolverDnZgetrs(handle, CUBLAS_OP_N, N, nrhs, A_device, N, pivot_dev, B_dev, N, info_dev));
        CHECK_CUDA(cudaMemcpy(&info_host, info_dev, sizeof(int), cudaMemcpyDeviceToHost));
        if (info_host != 0) {
            std::cerr << "Back substitution failed with error " << info_host << std::endl;
            throw std::runtime_error("Back substitution failed");
        }

        // Copy solution matrix back to host
        CHECK_CUDA(cudaMemcpy(B_host, B_dev, matrix_size, cudaMemcpyDeviceToHost));

    } catch (...) {
        cleanup_gpu_resources(&handle, nullptr, nullptr, B_dev, work_dev, pivot_dev, info_dev);
        throw;
    }

    cleanup_gpu_resources(&handle, nullptr, nullptr, B_dev, work_dev, pivot_dev, info_dev);
}

void invert_6x6_matrix_lapack(cuDoubleComplex* matrix) {
    const int N = 6;
    std::vector<std::complex<double>> A(N * N);
    std::vector<int> ipiv(N);
    int info = 0;
    int n = N;  // Non-const version for LAPACK
    
    // Convert cuDoubleComplex to std::complex<double>
    for (int i = 0; i < N * N; i++) {
        A[i] = std::complex<double>(matrix[i].x, matrix[i].y);
    }

    // Perform LU factorization using LAPACK
    // Note: We're using non-const variables for LAPACK parameters
    zgetrf_(&n, &n, A.data(), &n, ipiv.data(), &info);
    if (info != 0) {
        std::cerr << "LAPACK zgetrf_ failed with error " << info << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Create identity matrix as RHS
    std::vector<std::complex<double>> B(N * N, 0.0);
    for (int i = 0; i < N; i++) {
        B[i * N + i] = 1.0;
    }

    // Solve the system using the LU factorization
    char trans = 'N';
    // Note: Using non-const variables for LAPACK parameters
    zgetrs_(&trans, &n, &n, A.data(), &n, ipiv.data(), B.data(), &n, &info);
    if (info != 0) {
        std::cerr << "LAPACK zgetrs_ failed with error " << info << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Copy result back to cuDoubleComplex format
    for (int i = 0; i < N * N; i++) {
        matrix[i].x = std::real(B[i]);
        matrix[i].y = std::imag(B[i]);
    }
}
