#ifndef INTERACTION_HPP
#define INTERACTION_HPP

#include <complex>
#include "vector3.hpp"

// Computes the dyadic Green's function block from dipole j to dipole k
void pair_interaction_matrix(std::complex<double> out[3][3], vec3 r_j, vec3 r_k, double k);

// Builds the full 3N x 3N interaction matrix for N dipoles
void get_full_interaction_matrix(
    std::complex<double>* A,                            // Output matrix of size (3N x 3N), row-major
    const vec3* positions,                              // Positions of N dipoles
    const std::complex<double> (*alpha_inv)[3][3],      // Inverse polarizabilities for each dipole
    int N,                                              // Number of dipoles
    double k                                            // Wavenumber
);

#endif // INTERACTION_HPP
