#ifndef MATRIX_OPS_HPP
#define MATRIX_OPS_HPP

#include <complex>

// Matrix operation functions for 2x2 and 6x6 matrices
void matrix_multiply_2x2(std::complex<double> result[2][2], 
                        const std::complex<double> a[2][2], 
                        const std::complex<double> b[2][2]);

void matrix_multiply(std::complex<double> result[6][6], 
                    const std::complex<double> a[6][6], 
                    const std::complex<double> b[6][6]);

#endif // MATRIX_OPS_HPP