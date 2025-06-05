#ifndef ROTATION_HPP
#define ROTATION_HPP

#include <complex>

// Functions to create rotation matrices
void create_rotation_matrix_2x2(std::complex<double> out[2][2], double theta);
void create_rotation_matrix(std::complex<double> out[6][6], double theta);

// Functions to rotate polarizability matrices
void rotate_polarizability_matrix_2x2(std::complex<double> alpha[2][2], double theta);
void rotate_polarizability_matrix_6x6(std::complex<double> alpha[6][6], double theta);

#endif // ROTATION_HPP