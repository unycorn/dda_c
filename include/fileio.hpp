#ifndef FILEIO_HPP
#define FILEIO_HPP

#include <cuComplex.h>
#include <complex>
#include <vector>
#include "vector3.hpp"

// Reads a square complex matrix from a binary file.
// First 4 bytes: int (N), then N*N cuDoubleComplex elements.
cuDoubleComplex* load_matrix(const char* filename, int& N);

// Reads a complex vector from a binary file.
// First 4 bytes: int (N), then N cuDoubleComplex elements.
cuDoubleComplex* load_vector(const char* filename, int& N);

// Writes polarizations (6N-vector of cuDoubleComplex) and full polarizability matrices to CSV text file.
void write_polarizations(const char* filename, std::complex<double>* p, std::vector<vec3> positions, const std::vector<std::complex<double>[6][6]>& alpha, int N);

#endif // FILEIO_HPP
