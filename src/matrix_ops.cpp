#include "matrix_ops.hpp"

void matrix_multiply_2x2(std::complex<double> result[2][2], 
                    const std::complex<double> a[2][2], 
                    const std::complex<double> b[2][2]) {
    std::complex<double> temp[2][2];
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            temp[i][j] = 0;
            for (int k = 0; k < 2; ++k) {
                temp[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            result[i][j] = temp[i][j];
        }
    }
}

void matrix_multiply(std::complex<double> result[6][6], 
                    const std::complex<double> a[6][6], 
                    const std::complex<double> b[6][6]) {
    std::complex<double> temp[6][6];
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            temp[i][j] = 0;
            for (int k = 0; k < 6; ++k) {
                temp[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            result[i][j] = temp[i][j];
        }
    }
}