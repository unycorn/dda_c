#include "print_utils.hpp"
#include <iostream>

void print_complex_matrix(const char* label, const cuDoubleComplex* matrix, int n) {
    std::cout << "\n" << label << ":\n";
    std::cout << "np.array([";
    for (int i = 0; i < n; ++i) {
        std::cout << "[";
        for (int j = 0; j < n; ++j) {
            // Handle real and imaginary parts
            if (j > 0) std::cout << ", ";
            if (matrix[i*n + j].y == 0) {
                std::cout << matrix[i*n + j].x;
            } else if (matrix[i*n + j].x == 0) {
                std::cout << matrix[i*n + j].y << "j";
            } else {
                std::cout << matrix[i*n + j].x;
                if (matrix[i*n + j].y > 0) std::cout << "+";
                std::cout << matrix[i*n + j].y << "j";
            }
        }
        std::cout << "]" << (i < n-1 ? "," : "") << "\n";
    }
    std::cout << "])" << std::endl;
}

void print_complex_vector(const char* label, const cuDoubleComplex* vector, int n) {
    std::cout << "\n" << label << ":\n";
    std::cout << "np.array([";
    for (int i = 0; i < n; ++i) {
        if (i > 0) std::cout << ", ";
        if (vector[i].y == 0) {
            std::cout << vector[i].x;
        } else if (vector[i].x == 0) {
            std::cout << vector[i].y << "j";
        } else {
            std::cout << vector[i].x;
            if (vector[i].y > 0) std::cout << "+";
            std::cout << vector[i].y << "j";
        }
    }
    std::cout << "])" << std::endl;
}