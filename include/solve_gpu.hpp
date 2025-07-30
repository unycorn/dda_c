#pragma once

#include <cuComplex.h>

void init_cuda_device();
void solve_gpu(cuDoubleComplex* A_device, cuDoubleComplex* b_host, int N);
void solve_gpu_multiple_rhs(cuDoubleComplex* A_device, cuDoubleComplex* B_host, int N, int nrhs);
void invert_6x6_matrix_lapack(cuDoubleComplex* matrix);
