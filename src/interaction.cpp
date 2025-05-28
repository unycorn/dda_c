#include <complex>
#include <cmath>
#include <iostream>
#include <vector>  // Add missing include
#include <cuda_runtime.h>
#include <cuComplex.h>
#include "solve_gpu.hpp"
#include "constants.hpp"
#include "vector3.hpp"

void print_complex_matrix(const char* label, const cuDoubleComplex* matrix, int n) {
    std::cout << "\n" << label << ":\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << "(" << matrix[i*n + j].x << "," << matrix[i*n + j].y << ") ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

// Helper function to check CUDA errors
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

constexpr std::complex<double> I(0.0, 1.0);

// Helper function for cross product matrix
void cross_matrix(std::complex<double> out[3][3], const vec3& r) {
    out[0][0] = 0;        out[0][1] = r.z;      out[0][2] = -r.y;
    out[1][0] = -r.z;     out[1][1] = 0;        out[1][2] = r.x;
    out[2][0] = r.y;      out[2][1] = -r.x;     out[2][2] = 0;
}

// Electric field from electric dipole
void green_E_E_dipole(std::complex<double> out[3][3], vec3 r_j, vec3 r_k, double k) {
    vec3 r = vec3_sub(r_j, r_k);
    double r_len = vec3_norm(r);

    if (r_len == 0) {
        std::cerr << "Error: self-interaction\n";
        return;
    }

    vec3 r_hat = vec3_unit(r);
    std::complex<double> expikr = std::exp(I * k * r_len);
    std::complex<double> prefac = 1.0/(4*M_PI*EPSILON_0) * expikr / r_len;

    std::complex<double> term1 = k * k;
    std::complex<double> term2 = (I * k * r_len - 1.0) / (r_len * r_len);

    std::complex<double> dyad[3][3];
    outer_product(dyad, r_hat, r_hat);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double delta_ij = (i == j) ? 1.0 : 0.0;
            out[i][j] = prefac * (term1 * (dyad[i][j] - delta_ij) + term2 * (3.0 * dyad[i][j] - delta_ij));
        }
    }
}

// Magnetic field from electric dipole
void green_H_E_dipole(std::complex<double> out[3][3], vec3 r_j, vec3 r_k, double k) {
    vec3 r = vec3_sub(r_j, r_k);
    double r_len = vec3_norm(r);

    if (r_len == 0) {
        std::cerr << "Error: self-interaction\n";
        return;
    }

    std::complex<double> expikr = std::exp(I * k * r_len);
    double omega = k * C_LIGHT;
    std::complex<double> prefac = -I * omega * expikr / (4 * M_PI * r_len * r_len);
    std::complex<double> term = (1.0/r_len - I*k);

    std::complex<double> cross[3][3];
    cross_matrix(cross, r);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            out[i][j] = -prefac * term * cross[i][j];
        }
    }
}

// Electric field from magnetic dipole
void green_E_M_dipole(std::complex<double> out[3][3], vec3 r_j, vec3 r_k, double k) {
    vec3 r = vec3_sub(r_j, r_k);
    double r_len = vec3_norm(r);

    if (r_len == 0) {
        std::cerr << "Error: self-interaction\n";
        return;
    }

    std::complex<double> expikr = std::exp(I * k * r_len);
    double omega = k * C_LIGHT;
    std::complex<double> prefac = I * omega * MU_0 * expikr / (4 * M_PI * r_len * r_len);
    std::complex<double> term = (1.0/r_len - I*k);

    std::complex<double> cross[3][3];
    cross_matrix(cross, r);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            out[i][j] = -prefac * term * cross[i][j];
        }
    }
}

// Magnetic field from magnetic dipole
void green_H_M_dipole(std::complex<double> out[3][3], vec3 r_j, vec3 r_k, double k) {
    vec3 r = vec3_sub(r_j, r_k);
    double r_len = vec3_norm(r);

    if (r_len == 0) {
        std::cerr << "Error: self-interaction\n";
        return;
    }

    vec3 r_hat = vec3_unit(r);
    std::complex<double> expikr = std::exp(I * k * r_len);
    std::complex<double> prefac = 1.0/(4*M_PI) * expikr / r_len;

    std::complex<double> term1 = k * k;
    std::complex<double> term2 = (I * k * r_len - 1.0) / (r_len * r_len);

    std::complex<double> dyad[3][3];
    outer_product(dyad, r_hat, r_hat);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double delta_ij = (i == j) ? 1.0 : 0.0;
            out[i][j] = prefac * (term1 * (dyad[i][j] - delta_ij) + term2 * (3.0 * dyad[i][j] - delta_ij));
        }
    }
}

// Builds the full 6x6 bianisotropic Green's function tensor
void biani_green_matrix(std::complex<double>* out, vec3 r_j, vec3 r_k, double k) {
    std::complex<double> EE[3][3], HE[3][3], EM[3][3], HM[3][3];
    
    green_E_E_dipole(EE, r_j, r_k, k);
    green_H_E_dipole(HE, r_j, r_k, k);
    green_E_M_dipole(EM, r_j, r_k, k);
    green_H_M_dipole(HM, r_j, r_k, k);

    // Fill the 6x6 matrix in blocks
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            out[i*6 + j] = EE[i][j];           // Top-left block
            out[i*6 + (j+3)] = EM[i][j];       // Top-right block
            out[(i+3)*6 + j] = HE[i][j];       // Bottom-left block
            out[(i+3)*6 + (j+3)] = HM[i][j];   // Bottom-right block
        }
    }
}

// Builds the full 6N x 6N interaction matrix
cuDoubleComplex* get_full_interaction_matrix(
    std::complex<double>* A_host,
    const vec3* positions,
    const std::complex<double> (*polarizability)[6][6],
    int N,
    double k
) {
    // Calculate total memory needed
    size_t matrix_size = (size_t)(6 * N) * (size_t)(6 * N) * sizeof(cuDoubleComplex);
    
    // Calculate total elements and initialize progress tracking
    size_t total_elements = static_cast<size_t>(N) * static_cast<size_t>(N);
    size_t elements_processed = 0;
    int last_percent = -1;

    // Allocate CPU buffer for matrix construction
    std::vector<cuDoubleComplex> A_cpu(6 * N * 6 * N);

    // Main construction loop
    for (int j = 0; j < N; ++j) {
        for (int k_idx = 0; k_idx < N; ++k_idx) {
            // Update progress
            elements_processed++;
            double progress = (static_cast<double>(elements_processed) * 100.0) / 
                            static_cast<double>(total_elements);
            int percent_complete = static_cast<int>(progress);
            
            if (percent_complete != last_percent && percent_complete % 10 == 0) {
                std::cout << "Matrix construction: " << percent_complete << "% complete\n";
                last_percent = percent_complete;
            }

            int row_offset = j * 6;
            int col_offset = k_idx * 6;

            if (j == k_idx) {
                // Convert and invert polarizability on CPU
                std::vector<cuDoubleComplex> polarizability_block(36);
                for (int i = 0; i < 6; ++i) {
                    for (int m = 0; m < 6; ++m) {
                        polarizability_block[i*6 + m] = make_cuDoubleComplex(
                            std::real(polarizability[j][i][m]),
                            std::imag(polarizability[j][i][m])
                        );
                    }
                }

                print_complex_matrix("Polarizability matrix before inversion", polarizability_block.data(), 6);
    
                // Invert the 6x6 polarizability matrix
                invert_6x6_matrix_lapack(polarizability_block.data());

                print_complex_matrix("Polarizability matrix after inversion", polarizability_block.data(), 6);

                // Copy inverted matrix to the big interaction matrix
                for (int i = 0; i < 6; ++i) {
                    for (int m = 0; m < 6; ++m) {
                        A_cpu[(row_offset + i) * 6 * N + (col_offset + m)] = 
                            polarizability_block[i*6 + m];
                    }
                }
            } else {
                // Off-diagonal blocks store Green's function tensor
                std::complex<double> block[36];  // 6x6 block
                biani_green_matrix(block, positions[j], positions[k_idx], k);
                
                // Copy block to interaction matrix
                for (int i = 0; i < 6; ++i) {
                    for (int m = 0; m < 6; ++m) {
                        A_cpu[(row_offset + i) * 6 * N + (col_offset + m)] = make_cuDoubleComplex(
                            std::real(block[i*6 + m]),
                            std::imag(block[i*6 + m])
                        );
                    }
                }
            }
        }
    }

    std::cout << "\nMatrix construction complete, transferring to GPU...\n";

    // Allocate and transfer matrix to GPU in one operation
    cuDoubleComplex* A_dev = nullptr;
    CHECK_CUDA(cudaMalloc(&A_dev, matrix_size));
    CHECK_CUDA(cudaMemcpy(A_dev, A_cpu.data(), matrix_size, cudaMemcpyHostToDevice));
    
    // Copy to host buffer if provided
    if (A_host != nullptr) {
        size_t total_elements = static_cast<size_t>(6 * N) * static_cast<size_t>(6 * N);
        for (size_t i = 0; i < total_elements; ++i) {
            A_host[i] = std::complex<double>(A_cpu[i].x, A_cpu[i].y);
        }
    }

    std::cout << "Matrix transferred to GPU successfully\n";
    return A_dev;
}
