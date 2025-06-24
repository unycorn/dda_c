#pragma once

#include <cuComplex.h>

enum class SolverType {
    GPU,
    CPU
};

void init_cuda_device();
void solve_gpu(cuDoubleComplex* A_device, cuDoubleComplex* b_host, int N);
void solve_cpu(const cuDoubleComplex* A, cuDoubleComplex* b, int N);
void invert_6x6_matrix_lapack(cuDoubleComplex* matrix);

// Convenience function that chooses between GPU and CPU solvers
inline void solve_system(cuDoubleComplex* A, cuDoubleComplex* b, int N, SolverType solver = SolverType::GPU) {
    if (solver == SolverType::GPU) {
        solve_gpu(A, b, N);
    } else {
        solve_cpu(A, b, N);
    }
}
