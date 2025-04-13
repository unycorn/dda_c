// interaction.h
#pragma once
#include <complex.h>
#include "vector3.h"

void pair_interaction_matrix(double complex out[3][3], vec3 r_j, vec3 r_k, double k);

void get_full_interaction_matrix(
    double complex *A,       // Output matrix of size 3N x 3N
    const vec3 *positions,   // Array of N dipole positions
    const double complex (*alpha_inv)[3][3],  // Inverse polarizability tensors
    int N,
    double k
);
