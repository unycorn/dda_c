#pragma once

#include <complex>
#include "vector3.hpp"
#include <cuComplex.h>

// Computes each component of the 6x6 Green's function tensor
void green_E_E_dipole(std::complex<double> out[3][3], vec3 r_j, vec3 r_k, double k);
void green_H_E_dipole(std::complex<double> out[3][3], vec3 r_j, vec3 r_k, double k);
void green_E_M_dipole(std::complex<double> out[3][3], vec3 r_j, vec3 r_k, double k);
void green_H_M_dipole(std::complex<double> out[3][3], vec3 r_j, vec3 r_k, double k);

// Computes the full 6x6 bianisotropic Green's function tensor
void biani_green_matrix(
    std::complex<double>* out,    // Output matrix of size (6x6)
    vec3 r_j,                     // Observer position
    vec3 r_k,                     // Source position
    double k                      // Wavenumber
);

// Now returns the device pointer to the matrix for use in solve_gpu
cuDoubleComplex* get_full_interaction_matrix(
    std::complex<double>* A_host,                      // Host matrix of size (6N x 6N), row-major
    const vec3* positions,                             // Positions of N dipoles
    const std::complex<double> (*polarizability)[6][6], // Full 6x6 polarizability matrix for each dipole
    int N,                                             // Number of dipoles
    double k                                           // Wavenumber
);
