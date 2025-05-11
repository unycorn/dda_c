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
bool check_gpu_memory(size_t required_bytes, const char* allocation_type = "") {
    size_t free_bytes, total_bytes;
    CHECK_CUDA(cudaMemGetInfo(&free_bytes, &total_bytes));
    
    // Only print memory info if required memory is significant (>1MB)
    if (required_bytes > 1024*1024) {
        std::cout << allocation_type << " Memory - Free: " << free_bytes / (1024.0*1024*1024.0) 
                  << " GB, Required: " << required_bytes / (1024*1024*1024.0) 
                  << " GB\n";
    }
    return free_bytes >= required_bytes;
}

// Add cleanup helper
void cleanup_gpu_resources(cusolverDnHandle_t* handle, cublasHandle_t* cublas_handle,
                         cuDoubleComplex* A_dev, cuDoubleComplex* b_dev,
                         cuDoubleComplex* work_dev, int* pivot_dev, int* info_dev) {
    if (handle) {
        cusolverDnDestroy(*handle);
    }
    if (cublas_handle) {
        cublasDestroy(*cublas_handle);
    }
    if (A_dev) cudaFree(A_dev);
    if (b_dev) cudaFree(b_dev);
    if (work_dev) cudaFree(work_dev);
    if (pivot_dev) cudaFree(pivot_dev);
    if (info_dev) cudaFree(info_dev);
    
    // Reset any errors
    cudaGetLastError();
}

// Update the function signature to clarify that A is on device
void solve_gpu(cuDoubleComplex* A_device, cuDoubleComplex* b_host, int N) {
    // Initialize CUDA device first
    init_cuda_device();

    std::cout << "Starting linear system solve for N = " << N << "\n";

    // Calculate additional memory needed beyond the matrix (which is already on GPU)
    size_t vector_size = (size_t)N * sizeof(cuDoubleComplex);
    size_t pivot_size = (size_t)N * sizeof(int);

    // Create temporary handle just to get workspace size
    cusolverDnHandle_t temp_handle = nullptr;
    int work_size = 0;

    std::cout << "Getting workspace size...\n";
    // Get workspace size using the existing matrix on GPU
    CHECK_CUSOLVER(cusolverDnCreate(&temp_handle));
    CHECK_CUSOLVER(cusolverDnZgetrf_bufferSize(temp_handle, N, N, A_device, N, &work_size));
    cusolverDnDestroy(temp_handle);

    size_t workspace_size = (size_t)work_size * sizeof(cuDoubleComplex);
    // Only count additional memory needed beyond matrix
    size_t additional_required = vector_size + pivot_size + sizeof(int) + workspace_size;

    std::cout << "Additional solver workspace required: " << additional_required/(1024.0*1024.0*1024.0) << " GB\n";

    if (!check_gpu_memory(additional_required, "Additional Solver")) {
        std::cerr << "Not enough GPU memory for solver workspace\n";
        std::cerr << "Additional memory required: " << additional_required/(1024.0*1024.0*1024.0) << " GB\n";
        std::exit(EXIT_FAILURE);
    }

    // Reset pointers for actual solve
    cusolverDnHandle_t handle = nullptr;
    cuDoubleComplex *b_dev = nullptr, *work_dev = nullptr;
    int *pivot_dev = nullptr, *info_dev = nullptr;
    int info_host = 0;
    
    try {
        std::cout << "Creating solver handle...\n";
        // Create handle and allocate memory
        CHECK_CUSOLVER(cusolverDnCreate(&handle));
        CHECK_CUDA(cudaMalloc(&b_dev, vector_size));
        CHECK_CUDA(cudaMalloc(&pivot_dev, pivot_size));
        CHECK_CUDA(cudaMalloc(&info_dev, sizeof(int)));
        CHECK_CUDA(cudaMalloc(&work_dev, workspace_size));

        std::cout << "Copying b vector to device...\n";
        // Copy b vector to device (matrix A is already on device)
        CHECK_CUDA(cudaMemcpy(b_dev, b_host, vector_size, cudaMemcpyHostToDevice));

        std::cout << "Starting LU factorization...\n";
        // Perform LU factorization and solve using the matrix already on GPU
        CHECK_CUSOLVER(cusolverDnZgetrf(handle, N, N, A_device, N, work_dev, pivot_dev, info_dev));
        CHECK_CUDA(cudaMemcpy(&info_host, info_dev, sizeof(int), cudaMemcpyDeviceToHost));

        if (info_host != 0) {
            std::cerr << "solve_gpu: LU factorization failed with error " << info_host << "\n";
            cleanup_gpu_resources(&handle, nullptr, nullptr, b_dev, work_dev, pivot_dev, info_dev);
            std::exit(EXIT_FAILURE);
        }

        std::cout << "LU factorization complete, solving system...\n";
        CHECK_CUSOLVER(cusolverDnZgetrs(handle, CUBLAS_OP_N, N, 1, A_device, N, pivot_dev, b_dev, N, info_dev));
        CHECK_CUDA(cudaMemcpy(&info_host, info_dev, sizeof(int), cudaMemcpyDeviceToHost));

        if (info_host != 0) {
            std::cerr << "solve_gpu: solver returned error code " << info_host << "\n";
            cleanup_gpu_resources(&handle, nullptr, nullptr, b_dev, work_dev, pivot_dev, info_dev);
            std::exit(EXIT_FAILURE);
        }

        std::cout << "Solve complete, copying solution back to host...\n";
        // Copy solution back to host
        CHECK_CUDA(cudaMemcpy(b_host, b_dev, vector_size, cudaMemcpyDeviceToHost));
        std::cout << "Solution copied to host successfully\n";

        // Clean up (but don't free A_device since it was passed in)
        cleanup_gpu_resources(&handle, nullptr, nullptr, b_dev, work_dev, pivot_dev, info_dev);
        
    } catch (...) {
        cleanup_gpu_resources(&handle, nullptr, nullptr, b_dev, work_dev, pivot_dev, info_dev);
        throw;
    }
}

void invert_matrix_gpu(cuDoubleComplex* A_host, int N) {
    // Initialize CUDA device first
    init_cuda_device();

    cusolverDnHandle_t handle = nullptr;
    cublasHandle_t cublas_handle = nullptr;
    cuDoubleComplex *A_dev = nullptr, *work_dev = nullptr;
    cuDoubleComplex *identity_dev = nullptr;
    int *pivot_dev = nullptr, *info_dev = nullptr;
    int work_size = 0, info_host = 0;

    try {
        // Create handles
        CHECK_CUSOLVER(cusolverDnCreate(&handle));
        CHECK_CUBLAS(cublasCreate(&cublas_handle));

        // Calculate required memory
        size_t matrix_size = (size_t)N * (size_t)N * sizeof(cuDoubleComplex);
        size_t pivot_size = (size_t)N * sizeof(int);
        
        // Get workspace size first
        CHECK_CUDA(cudaMalloc(&A_dev, matrix_size));
        CHECK_CUSOLVER(cusolverDnZgetrf_bufferSize(handle, N, N, A_dev, N, &work_size));
        
        size_t workspace_size = (size_t)work_size * sizeof(cuDoubleComplex);
        size_t total_required = matrix_size * 2 + pivot_size + sizeof(int) + workspace_size;

        if (!check_gpu_memory(total_required)) {
            std::cerr << "Not enough GPU memory for matrix inversion " << N << "x" << N << "\n";
            std::cerr << "Matrix size: " << matrix_size/(1024.0*1024.0*1024.0) << " GB\n";
            std::cerr << "Workspace size: " << workspace_size/(1024.0*1024.0*1024.0) << " GB\n";
            std::cerr << "Total required: " << total_required/(1024.0*1024.0*1024.0) << " GB\n";
            cleanup_gpu_resources(&handle, &cublas_handle, A_dev, nullptr, nullptr, nullptr, nullptr);
            std::exit(EXIT_FAILURE);
        }

        // Allocate remaining resources
        CHECK_CUDA(cudaMalloc(&identity_dev, matrix_size));
        CHECK_CUDA(cudaMalloc(&pivot_dev, pivot_size));
        CHECK_CUDA(cudaMalloc(&info_dev, sizeof(int)));
        CHECK_CUDA(cudaMalloc(&work_dev, workspace_size));

        // Copy matrix to device and create identity matrix
        CHECK_CUDA(cudaMemcpy(A_dev, A_host, matrix_size, cudaMemcpyHostToDevice));
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                int idx = i * N + j;
                cuDoubleComplex value = (i == j) ? make_cuDoubleComplex(1.0, 0.0) : make_cuDoubleComplex(0.0, 0.0);
                CHECK_CUDA(cudaMemcpy(&identity_dev[idx], &value, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
            }
        }

        // Perform LU factorization
        CHECK_CUSOLVER(cusolverDnZgetrf(handle, N, N, A_dev, N, work_dev, pivot_dev, info_dev));
        CHECK_CUDA(cudaMemcpy(&info_host, info_dev, sizeof(int), cudaMemcpyDeviceToHost));
        if (info_host != 0) {
            std::cerr << "invert_matrix_gpu: LU factorization failed with error " << info_host << "\n";
            cleanup_gpu_resources(&handle, &cublas_handle, A_dev, nullptr, work_dev, pivot_dev, info_dev);
            std::exit(EXIT_FAILURE);
        }

        // Solve N systems of equations to get inverse (A * X = I)
        CHECK_CUSOLVER(cusolverDnZgetrs(handle, CUBLAS_OP_N, N, N, A_dev, N, pivot_dev, identity_dev, N, info_dev));
        CHECK_CUDA(cudaMemcpy(&info_host, info_dev, sizeof(int), cudaMemcpyDeviceToHost));
        if (info_host != 0) {
            std::cerr << "invert_matrix_gpu: Matrix inversion failed with error " << info_host << "\n";
            cleanup_gpu_resources(&handle, &cublas_handle, A_dev, identity_dev, work_dev, pivot_dev, info_dev);
            std::exit(EXIT_FAILURE);
        }

        // Copy result back to host
        CHECK_CUDA(cudaMemcpy(A_host, identity_dev, matrix_size, cudaMemcpyDeviceToHost));

        // Clean up
        cleanup_gpu_resources(&handle, &cublas_handle, A_dev, nullptr, work_dev, pivot_dev, info_dev);
        if (identity_dev) cudaFree(identity_dev);

    } catch (...) {
        cleanup_gpu_resources(&handle, &cublas_handle, A_dev, nullptr, work_dev, pivot_dev, info_dev);
        if (identity_dev) cudaFree(identity_dev);
        throw;
    }
}
