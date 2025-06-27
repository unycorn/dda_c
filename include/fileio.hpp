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
void write_polarizations(
    const char* filename, 
    std::complex<double>* p, 
    std::vector<vec3> positions, 
    const std::vector<std::complex<double>[6][6]>& alpha, 
    int N
);

// Overload for scalar (2x2) polarizability matrices
void write_polarizations(
    const char* filename, 
    std::complex<double>* p, 
    std::vector<vec3> positions, 
    const std::vector<std::complex<double>[2][2]>& alpha, 
    int N
);

// Writes polarizations (2N-vector of cuDoubleComplex) and scalar polarizability matrices to binary file.
void write_polarizations_binary(
    const char* filename,
    std::complex<double>* p,
    std::vector<vec3> positions,
    const std::vector<std::complex<double>[2][2]>& alpha,
    int N
);

// Reads polarizations from a binary file. Returns vector of complex doubles and sets N.
std::vector<std::complex<double>> read_polarizations_binary(const char* filename, int& N);

#endif // FILEIO_HPP
