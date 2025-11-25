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
    int N,
    double frequency,
    double absorption = 0.0
);

// Writes polarizations for plane wave sweep with multiple k-vectors and polarizations
void write_PW_sweep_polarization_binary(
    const char* filename,
    const std::vector<std::complex<double>>& polarizations,  // All polarizations [N_dipoles * 2 * N_k_points * N_polarizations]
    const std::vector<vec3>& positions,
    const std::vector<std::complex<double>[2][2]>& alpha,
    const std::vector<double>& kx_values,
    const std::vector<double>& ky_values,
    const std::vector<std::string>& polarization_types,  // "TE" or "TM"
    int N,
    double frequency
);

// Reads polarizations from a binary file. Returns vector of complex doubles and sets N.
std::vector<std::complex<double>> read_polarizations_binary(const char* filename, int& N, double& frequency);

#endif // FILEIO_HPP
