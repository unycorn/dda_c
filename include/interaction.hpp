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

// Computes the scalar Green's function tensor (2x2 matrix) for rotated dipoles
void biani_green_matrix_scalar(
    std::complex<double>* out,    // Output matrix of size (2x2)
    vec3 r_j,                     // Observer position
    vec3 r_k,                     // Source position
    double theta_j,               // Rotation angle of observer dipole
    double theta_k,               // Rotation angle of source dipole
    double k                      // Wavenumber
);

// Returns a pointer to the matrix on the GPU device for use in solve_gpu
cuDoubleComplex* get_full_interaction_matrix(
    std::complex<double>* A_host,                      // Host matrix of size (6N x 6N), row-major
    const vec3* positions,                             // Positions of N dipoles
    const std::complex<double> (*polarizability)[6][6], // Full 6x6 polarizability matrix for each dipole
    int N,                                             // Number of dipoles
    double k                                           // Wavenumber
);

// Returns a pointer to the scalar matrix on the GPU device for use in solve_gpu
cuDoubleComplex* get_full_interaction_matrix_scalar(
    std::complex<double>* A_host,                     // Host matrix of size (2N x 2N), row-major
    const vec3* positions,                            // Positions of N dipoles
    const std::complex<double> (*pol_2x2)[2][2],     // 2x2 polarizability matrix for each dipole
    const double* thetas,                             // Rotation angles for each dipole
    int N,                                            // Number of dipoles
    double k                                          // Wavenumber
);
