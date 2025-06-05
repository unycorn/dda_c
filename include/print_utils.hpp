#ifndef PRINT_UTILS_HPP
#define PRINT_UTILS_HPP

#include <cuda_runtime.h>
#include <cuComplex.h>

void print_complex_matrix(const char* label, const cuDoubleComplex* matrix, int n);
void print_complex_vector(const char* label, const cuDoubleComplex* vector, int n);

#endif