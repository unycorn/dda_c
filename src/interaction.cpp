#include <complex>
#include <cmath>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include "solve_gpu.hpp"
#include "constants.hpp"
#include "vector3.hpp"
#include "print_utils.hpp"
#include <chrono>
#include <omp.h>

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
    std::complex<double> term2 = (1.0 - I * k * r_len) / (r_len * r_len);

    std::complex<double> dyad[3][3];
    outer_product(dyad, r_hat, r_hat);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double delta_ij = (i == j) ? 1.0 : 0.0;
            out[i][j] = prefac * (term2 * (3.0 * dyad[i][j] - delta_ij) + term1 * (delta_ij - dyad[i][j]));
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
    std::complex<double> term2 = (1.0 - I * k * r_len) / (r_len * r_len);

    std::complex<double> dyad[3][3];
    outer_product(dyad, r_hat, r_hat);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double delta_ij = (i == j) ? 1.0 : 0.0;
            out[i][j] = prefac * (term2 * (3.0 * dyad[i][j] - delta_ij) + term1 * (delta_ij - dyad[i][j]));
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

// void biani_green_matrix_scalar(std::complex<double>* out, vec3 r_j, vec3 r_k, double theta_j, double theta_k, double k) {
//     // Calculate shared geometric terms
//     vec3 r = vec3_sub(r_j, r_k);
//     double r_len = vec3_norm(r);

//     if (r_len == 0) {
//         std::cerr << "Error: self-interaction\n";
//         return;
//     }
    
//     // Unit vectors for dipole orientations
//     vec3 u_e_j = {cos(theta_j), sin(theta_j), 0.0};
//     vec3 u_e_k = {cos(theta_k), sin(theta_k), 0.0};
//     vec3 u_m_j = {0.0, 0.0, 1.0};  // z-direction
    
//     // Precompute shared terms
//     vec3 r_hat = vec3_unit(r);
//     std::complex<double> expikr = std::exp(I * k * r_len);
//     double omega = k * C_LIGHT;
    
//     // Dot products we'll need multiple times
//     double r_dot_ej = vec3_dot(r_hat, u_e_j);
//     double r_dot_ek = vec3_dot(r_hat, u_e_k);
//     double ej_dot_ek = vec3_dot(u_e_j, u_e_k);
    
//     // EE block (electric-electric coupling)
//     std::complex<double> ee_prefac = expikr / (4*M_PI*EPSILON_0*r_len);
//     std::complex<double> ee_term1 = k * k;
//     std::complex<double> ee_term2 = (I * k * r_len - 1.0) / (r_len * r_len);
//     std::complex<double> ee_scalar = ee_prefac * (
//         ee_term1 * (3.0 * r_dot_ej * r_dot_ek - ej_dot_ek) +
//         ee_term2 * (3.0 * r_dot_ej * r_dot_ek - ej_dot_ek)
//     );
    
//     // HM block (magnetic-magnetic coupling) - similar structure to EE
//     // Since u_m is in z direction, many terms simplify
//     double r_dot_m = r_hat.z;  // dot product with z unit vector
//     std::complex<double> hm_scalar = ee_prefac * (
//         ee_term1 * (3.0 * r_dot_m * r_dot_m - 1.0) +
//         ee_term2 * (3.0 * r_dot_m * r_dot_m - 1.0)
//     );
    
//     // HE and EM blocks (cross coupling)
//     // These involve cross products with r_hat
//     std::complex<double> cross_prefac = -I * omega * expikr / (4 * M_PI * r_len * r_len);
//     std::complex<double> cross_term = (1.0/r_len - I*k);
    
//     // For z cross r_hat cross u_e
//     vec3 r_cross_ue_k = {
//         -r_hat.y * cos(theta_k),
//         r_hat.x * cos(theta_k),
//         0.0
//     };
//     vec3 r_cross_ue_j = {
//         -r_hat.y * cos(theta_j),
//         r_hat.x * cos(theta_j),
//         0.0
//     };
    
//     std::complex<double> he_scalar = cross_prefac * cross_term * r_cross_ue_k.z;  // Only z component matters for u_m_j
//     std::complex<double> em_scalar = cross_prefac * cross_term * r_cross_ue_j.z;  // Only z component matters for u_m_k
    
//     // Fill output matrix (2x2 scalar result)
//     out[0] = ee_scalar;  // EE block
//     out[1] = em_scalar;  // EM block
//     out[2] = he_scalar;  // HE block
//     out[3] = hm_scalar;  // HM block

// }

void biani_green_matrix_scalar(std::complex<double>* out, vec3 r_j, vec3 r_k, double theta_j, double theta_k, double k) {
    std::complex<double> EE[3][3], HE[3][3], EM[3][3], HM[3][3];
    
    // Calculate the Green's function tensors
    green_E_E_dipole(EE, r_j, r_k, k);
    green_H_E_dipole(HE, r_j, r_k, k);

    // green_E_M_dipole(EM, r_j, r_k, k);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EM[i][j] = -MU_0 * HE[i][j];
        }
    }

    // green_H_M_dipole(HM, r_j, r_k, k);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            HM[i][j] = EPSILON_0 * EE[i][j];
        }
    }

    // Print all four matrices
    // std::cout << "\nElectric-Electric (EE) Green's function:\n";
    // for(int i = 0; i < 3; i++) {
    //     for(int j = 0; j < 3; j++) {
    //         std::cout << "(" << EE[i][j].real() << "," << EE[i][j].imag() << ")\t";
    //     }
    //     std::cout << "\n";
    // }

    // std::cout << "\nMagnetic-Electric (HE) Green's function:\n";
    // for(int i = 0; i < 3; i++) {
    //     for(int j = 0; j < 3; j++) {
    //         std::cout << "(" << HE[i][j].real() << "," << HE[i][j].imag() << ")\t";
    //     }
    //     std::cout << "\n";
    // }

    // std::cout << "\nElectric-Magnetic (EM) Green's function:\n";
    // for(int i = 0; i < 3; i++) {
    //     for(int j = 0; j < 3; j++) {
    //         std::cout << "(" << EM[i][j].real() << "," << EM[i][j].imag() << ")\t";
    //     }
    //     std::cout << "\n";
    // }

    // std::cout << "\nMagnetic-Magnetic (HM) Green's function:\n";
    // for(int i = 0; i < 3; i++) {
    //     for(int j = 0; j < 3; j++) {
    //         std::cout << "(" << HM[i][j].real() << "," << HM[i][j].imag() << ")\t";
    //     }
    //     std::cout << "\n";
    // }

    // Define the unit vectors
    vec3 u_e_j = {cos(theta_j), sin(theta_j), 0.0};
    vec3 u_e_k = {cos(theta_k), sin(theta_k), 0.0};
    vec3 u_m_j = {0.0, 0.0, 1.0};
    vec3 u_m_k = {0.0, 0.0, 1.0};

    // Calculate the scalar products for each 3x3 block
    std::complex<double> ee_scalar = 0.0;
    std::complex<double> he_scalar = 0.0;
    std::complex<double> em_scalar = 0.0;
    std::complex<double> hm_scalar = 0.0;

    // Calculate u_j * EE * u_k etc using array indexing
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            ee_scalar += u_e_j[i] * EE[i][j] * u_e_k[j];
            he_scalar += u_m_j[i] * HE[i][j] * u_e_k[j];
            em_scalar += u_e_j[i] * EM[i][j] * u_m_k[j];
            hm_scalar += u_m_j[i] * HM[i][j] * u_m_k[j];
        }
    }

    // Print the scalar values
    // std::cout << "\nGreen's function scalar components:\n";
    // std::cout << "EE scalar: " << ee_scalar << "\n";
    // std::cout << "EM scalar: " << em_scalar << "\n";
    // std::cout << "HE scalar: " << he_scalar << "\n";
    // std::cout << "HM scalar: " << hm_scalar << "\n";

    // Fill output matrix (2x2 scalar result)
    out[0] = ee_scalar;  // EE block
    out[1] = em_scalar;  // EM block
    out[2] = he_scalar;  // HE block
    out[3] = hm_scalar;  // HM block

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
    
    // Allocate CPU buffer for matrix construction
    std::vector<cuDoubleComplex> A_cpu(6 * N * 6 * N);

    // Main construction loop
    for (int j = 0; j < N; ++j) {
        for (int k_idx = 0; k_idx < N; ++k_idx) {
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

                // Invert the 6x6 polarizability matrix
                invert_6x6_matrix_lapack(polarizability_block.data());

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

    return A_dev;
}

void print_progress(int tid, size_t elements_processed, size_t total_elements) {
    double progress = (static_cast<double>(elements_processed) * 100.0) / static_cast<double>(total_elements);
    int percent_complete = static_cast<int>(progress);
    #pragma omp critical
    {
        std::cout << "Thread " << tid << ": " << percent_complete << "% complete\n";
    }
}

// Builds the full 2N x 2N interaction matrix
cuDoubleComplex* get_full_interaction_matrix_scalar(
    std::complex<double>* A_host,
    const vec3* positions,
    const std::complex<double> (*pol_2x2)[2][2],
    const double* thetas,
    int N,
    double k
) {
    size_t matrix_size = (size_t)(2 * N) * (size_t)(2 * N) * sizeof(cuDoubleComplex);
    
    // Progress tracking
    size_t total_elements = static_cast<size_t>(N) * static_cast<size_t>(N);
    size_t elements_processed = 0;
    int last_percent = -1;

    // Allocate CPU buffer for matrix construction
    std::vector<cuDoubleComplex> A_cpu(2 * N * 2 * N);

    // Main construction loop
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int last_percent = -1;
        
        #pragma omp for schedule(dynamic)
        for (int j = 0; j < N; ++j) {
            size_t thread_elements = 0;
            for (int k_idx = 0; k_idx < N; ++k_idx) {
                thread_elements++;
                int row_offset = j * 2;
                int col_offset = k_idx * 2;

                if (j == k_idx) {
                    // Invert the 2x2 matrix
                    std::complex<double> det = pol_2x2[j][0][0] * pol_2x2[j][1][1] - pol_2x2[j][0][1] * pol_2x2[j][1][0];
                    std::complex<double> inv_2x2[4];
                    inv_2x2[0] = pol_2x2[j][1][1] / det;
                    inv_2x2[1] = -pol_2x2[j][0][1] / det;
                    inv_2x2[2] = -pol_2x2[j][1][0] / det;
                    inv_2x2[3] = pol_2x2[j][0][0] / det;

                    // Copy inverted 2x2 matrix to interaction matrix
                    for (int i = 0; i < 2; ++i) {
                        for (int m = 0; m < 2; ++m) {
                            A_cpu[(row_offset + i) * 2 * N + (col_offset + m)] = make_cuDoubleComplex(
                                std::real(inv_2x2[i*2 + m]),
                                std::imag(inv_2x2[i*2 + m])
                            );
                        }
                    }
                } else {
                    // Get 2x2 Green's function using scalar version
                    std::complex<double> block[4];
                    biani_green_matrix_scalar(block, positions[j], positions[k_idx], 
                                            thetas[j], thetas[k_idx], k);
                    
                    // Copy 2x2 block to interaction matrix -- and NEGATE it!!
                    for (int i = 0; i < 2; ++i) {
                        for (int m = 0; m < 2; ++m) {
                            A_cpu[(col_offset + i) * 2 * N + (row_offset + m)] = make_cuDoubleComplex(
                                -std::real(block[i*2 + m]),
                                -std::imag(block[i*2 + m])
                            );
                        }
                    }
                }

                // Print progress every 10% for each thread
                // if (thread_elements % (N/10) == 0) {
                //     print_progress(tid, thread_elements, N);
                // }
            }
        }
    }

    std::cout << "Matrix construction complete.\n";

    // Allocate and transfer matrix to GPU
    cuDoubleComplex* A_dev = nullptr;
    CHECK_CUDA(cudaMalloc(&A_dev, matrix_size));
    CHECK_CUDA(cudaMemcpy(A_dev, A_cpu.data(), matrix_size, cudaMemcpyHostToDevice));
    
    // Copy to host buffer if provided
    if (A_host != nullptr) {
        size_t total_elements = static_cast<size_t>(2 * N) * static_cast<size_t>(2 * N);
        for (size_t i = 0; i < total_elements; ++i) {
            A_host[i] = std::complex<double>(A_cpu[i].x, A_cpu[i].y);
        }
    }

    std::cout << "Matrix transferred to GPU successfully\n";
    return A_dev;
}