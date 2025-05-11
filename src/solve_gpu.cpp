#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cuComplex.h>

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
        cusolverStatus_t status = call; \
        if (status != CUSOLVER_STATUS_SUCCESS) { \
            std::cerr << "cuSOLVER error in " << __FILE__ << ":" << __LINE__ << ": " \
                      << status << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

// Helper function to check cuBLAS errors
#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error in " << __FILE__ << ":" << __LINE__ << ": " \
                      << status << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

// Initialize CUDA device
void init_cuda_device() {
    static bool initialized = false;
    if (!initialized) {
        int deviceCount;
        CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
        
        // Get the device ID from CUDA_VISIBLE_DEVICES if set
        const char* visibleDevices = std::getenv("CUDA_VISIBLE_DEVICES");
        int deviceId = 0;
        if (visibleDevices != nullptr) {
            deviceId = std::atoi(visibleDevices);
        }

        // Get device properties
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, deviceId));
        
        // Print device info
        std::cout << "Using GPU: " << prop.name << "\n";
        std::cout << "Total GPU memory: " << prop.totalGlobalMem / (1024*1024*1024) << " GB\n";
        
        CHECK_CUDA(cudaSetDevice(deviceId));
        initialized = true;
    }
}

// Add this function to check if we have enough memory
bool check_gpu_memory(size_t required_bytes) {
    size_t free_bytes, total_bytes;
    CHECK_CUDA(cudaMemGetInfo(&free_bytes, &total_bytes));
    std::cout << "GPU Memory - Free: " << free_bytes / (1024*1024*1024) 
              << " GB, Required: " << required_bytes / (1024*1024*1024) 
              << " GB\n";
    return free_bytes >= required_bytes;
}

void solve_gpu(cuDoubleComplex* A_host, cuDoubleComplex* b_host, int N) {
    // Initialize CUDA device first
    init_cuda_device();

    cusolverDnHandle_t handle;
    cuDoubleComplex *A_dev = nullptr, *b_dev = nullptr, *work_dev = nullptr;
    int *pivot_dev = nullptr, *info_dev = nullptr;
    int work_size = 0, info_host = 0;

    // Create cuSolver handle with error checking
    CHECK_CUSOLVER(cusolverDnCreate(&handle));

    // Allocate device memory with error checking
    CHECK_CUDA(cudaMalloc(&A_dev, sizeof(cuDoubleComplex) * N * N));
    CHECK_CUDA(cudaMalloc(&b_dev, sizeof(cuDoubleComplex) * N));
    CHECK_CUDA(cudaMalloc(&pivot_dev, sizeof(int) * N));
    CHECK_CUDA(cudaMalloc(&info_dev, sizeof(int)));

    // Copy data to device with error checking
    CHECK_CUDA(cudaMemcpy(A_dev, A_host, sizeof(cuDoubleComplex) * N * N, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(b_dev, b_host, sizeof(cuDoubleComplex) * N, cudaMemcpyHostToDevice));

    // Get workspace size
    CHECK_CUSOLVER(cusolverDnZgetrf_bufferSize(handle, N, N, A_dev, N, &work_size));
    CHECK_CUDA(cudaMalloc(&work_dev, sizeof(cuDoubleComplex) * work_size));

    // Perform LU factorization and solve
    CHECK_CUSOLVER(cusolverDnZgetrf(handle, N, N, A_dev, N, work_dev, pivot_dev, info_dev));
    CHECK_CUSOLVER(cusolverDnZgetrs(handle, CUBLAS_OP_N, N, 1, A_dev, N, pivot_dev, b_dev, N, info_dev));

    // Copy solution back to host
    CHECK_CUDA(cudaMemcpy(b_host, b_dev, sizeof(cuDoubleComplex) * N, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&info_host, info_dev, sizeof(int), cudaMemcpyDeviceToHost));

    if (info_host != 0) {
        std::cerr << "solve_gpu: solver returned error code " << info_host << "\n";
        std::exit(EXIT_FAILURE);
    }

    // Free resources
    CHECK_CUDA(cudaFree(A_dev));
    CHECK_CUDA(cudaFree(b_dev));
    CHECK_CUDA(cudaFree(pivot_dev));
    CHECK_CUDA(cudaFree(info_dev));
    CHECK_CUDA(cudaFree(work_dev));
    CHECK_CUSOLVER(cusolverDnDestroy(handle));
}

void invert_matrix_gpu(cuDoubleComplex* A_host, int N) {
    // Initialize CUDA device first
    init_cuda_device();

    // Calculate required memory
    size_t matrix_size = N * N * sizeof(cuDoubleComplex);
    size_t total_required = matrix_size * 3;  // Main matrix + Identity + Work space

    if (!check_gpu_memory(total_required)) {
        std::cerr << "Not enough GPU memory for matrix size " << N << "x" << N << "\n";
        std::exit(EXIT_FAILURE);
    }

    cusolverDnHandle_t handle;
    cublasHandle_t cublas_handle;
    cuDoubleComplex *A_dev = nullptr, *work_dev = nullptr;
    cuDoubleComplex *identity_dev = nullptr;
    int *pivot_dev = nullptr, *info_dev = nullptr;
    int work_size = 0, info_host = 0;

    // Create handles with error checking
    CHECK_CUSOLVER(cusolverDnCreate(&handle));
    CHECK_CUBLAS(cublasCreate(&cublas_handle));

    // Allocate device memory with error checking
    CHECK_CUDA(cudaMalloc(&A_dev, sizeof(cuDoubleComplex) * N * N));
    CHECK_CUDA(cudaMalloc(&identity_dev, sizeof(cuDoubleComplex) * N * N));
    CHECK_CUDA(cudaMalloc(&pivot_dev, sizeof(int) * N));
    CHECK_CUDA(cudaMalloc(&info_dev, sizeof(int)));

    // Copy matrix to device
    CHECK_CUDA(cudaMemcpy(A_dev, A_host, sizeof(cuDoubleComplex) * N * N, cudaMemcpyHostToDevice));

    // Create identity matrix on device
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int idx = i * N + j;
            cuDoubleComplex value = (i == j) ? make_cuDoubleComplex(1.0, 0.0) : make_cuDoubleComplex(0.0, 0.0);
            CHECK_CUDA(cudaMemcpy(&identity_dev[idx], &value, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        }
    }

    // Get workspace size for LU factorization
    CHECK_CUSOLVER(cusolverDnZgetrf_bufferSize(handle, N, N, A_dev, N, &work_size));
    CHECK_CUDA(cudaMalloc(&work_dev, sizeof(cuDoubleComplex) * work_size));

    // Perform LU factorization
    CHECK_CUSOLVER(cusolverDnZgetrf(handle, N, N, A_dev, N, work_dev, pivot_dev, info_dev));
    CHECK_CUDA(cudaMemcpy(&info_host, info_dev, sizeof(int), cudaMemcpyDeviceToHost));
    if (info_host != 0) {
        std::cerr << "invert_matrix_gpu: LU factorization failed with error " << info_host << "\n";
        std::exit(EXIT_FAILURE);
    }

    // Solve N systems of equations to get inverse (A * X = I)
    CHECK_CUSOLVER(cusolverDnZgetrs(handle, CUBLAS_OP_N, N, N, A_dev, N, pivot_dev, identity_dev, N, info_dev));
    CHECK_CUDA(cudaMemcpy(&info_host, info_dev, sizeof(int), cudaMemcpyDeviceToHost));
    if (info_host != 0) {
        std::cerr << "invert_matrix_gpu: Matrix inversion failed with error " << info_host << "\n";
        std::exit(EXIT_FAILURE);
    }

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(A_host, identity_dev, sizeof(cuDoubleComplex) * N * N, cudaMemcpyDeviceToHost));

    // Free resources with error checking
    CHECK_CUDA(cudaFree(A_dev));
    CHECK_CUDA(cudaFree(identity_dev));
    CHECK_CUDA(cudaFree(pivot_dev));
    CHECK_CUDA(cudaFree(info_dev));
    CHECK_CUDA(cudaFree(work_dev));
    CHECK_CUSOLVER(cusolverDnDestroy(handle));
    CHECK_CUBLAS(cublasDestroy(cublas_handle));
}
