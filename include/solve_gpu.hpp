#ifndef SOLVE_GPU_HPP
#define SOLVE_GPU_HPP

#include <cuComplex.h> // for cuDoubleComplex

// Solves A x = b for x using LU decomposition on GPU.
// A_host: (N x N) matrix in row-major order
// b_host: RHS vector of size N (modified in-place to contain solution)
void solve_gpu(cuDoubleComplex* A_host, cuDoubleComplex* b_host, int N);

#endif // SOLVE_GPU_HPP
