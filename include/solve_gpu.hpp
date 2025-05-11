#pragma once

#include <cuComplex.h>

void init_cuda_device();
void solve_gpu(cuDoubleComplex* A_device, cuDoubleComplex* b_host, int N);
void invert_matrix_gpu(cuDoubleComplex* A_host, int N);
