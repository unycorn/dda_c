#include "rotation.hpp"
#include "matrix_ops.hpp"
#include <cmath>

void create_rotation_matrix_2x2(std::complex<double> out[2][2], double theta) {
    out[0][0] = std::cos(theta);
    out[0][1] = -std::sin(theta);
    out[1][0] = std::sin(theta);
    out[1][1] = std::cos(theta);
}

void create_rotation_matrix(std::complex<double> out[6][6], double theta) {
    // Clear the matrix
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            out[i][j] = 0.0;
        }
    }
    
    // Fill in rotation matrix elements for electric part (top-left 3x3)
    out[0][0] = std::cos(theta);
    out[0][1] = -std::sin(theta);
    out[1][0] = std::sin(theta);
    out[1][1] = std::cos(theta);
    out[2][2] = 1.0;  // z-component unchanged

    // Fill in rotation matrix elements for magnetic part (bottom-right 3x3)
    out[3][3] = std::cos(theta);
    out[3][4] = -std::sin(theta);
    out[4][3] = std::sin(theta);
    out[4][4] = std::cos(theta);
    out[5][5] = 1.0;
}

void rotate_polarizability_matrix_2x2(std::complex<double> alpha[2][2], double theta) {
    std::complex<double> rotation[2][2];
    std::complex<double> rotation_T[2][2];
    std::complex<double> temp[2][2];
    
    create_rotation_matrix_2x2(rotation, theta);
    create_rotation_matrix_2x2(rotation_T, -theta);
    
    matrix_multiply_2x2(temp, rotation, alpha);
    matrix_multiply_2x2(alpha, temp, rotation_T);
}

void rotate_polarizability_matrix_6x6(std::complex<double> alpha[6][6], double theta) {
    std::complex<double> rotation[6][6];
    std::complex<double> rotation_T[6][6];
    std::complex<double> temp[6][6];
    
    create_rotation_matrix(rotation, theta);
    create_rotation_matrix(rotation_T, -theta);
    
    matrix_multiply(temp, rotation, alpha);
    matrix_multiply(alpha, temp, rotation_T);
}