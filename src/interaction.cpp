#include <complex>
#include <cmath>
#include <iostream>
#include <vector>  // Add missing include
#include <cuda_runtime.h>
#include <cuComplex.h>
#include "solve_gpu.hpp"
#include "constants.hpp"
#include "vector3.hpp"
#include "print_utils.hpp"
#include <chrono>  // Add at top with other includes
#include <omp.h>  // Add at top with other includes
#include <atomic> // Add at top with other includes

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

void biani_green_matrix_scalar(std::complex<double>* out, vec3 r_j, vec3 r_k, double theta_j, double theta_k, double k) {
    // Cache vector operations
    vec3 r = vec3_sub(r_j, r_k);
    double r_len = vec3_norm(r);
    
    if (r_len == 0) {
        std::cerr << "Error: self-interaction\n";
        return;
    }

    // Pre-compute expensive values
    double inv_r = 1.0/r_len;
    double inv_r2 = inv_r * inv_r;
    std::complex<double> expikr = std::exp(I * k * r_len);
    
    // Pre-compute trig values
    double cos_j = cos(theta_j);
    double sin_j = sin(theta_j);
    double cos_k = cos(theta_k);
    double sin_k = sin(theta_k);
    
    // Pre-compute unit vectors
    vec3 u_e_j = {cos_j, sin_j, 0.0};
    vec3 u_e_k = {cos_k, sin_k, 0.0};
    const vec3 u_m = {0.0, 0.0, 1.0}; // Same for both j and k
    
    // Calculate common terms for Green's functions
    vec3 r_hat = vec3_scale(r, inv_r);
    std::complex<double> prefac = 1.0/(4*M_PI*EPSILON_0) * expikr * inv_r;
    std::complex<double> k2 = k * k;
    std::complex<double> ikr = I * k * r_len;
    std::complex<double> term1 = k2;
    std::complex<double> term2 = (ikr - 1.0) * inv_r2;

    // Compute Green's tensors with cached values
    std::complex<double> EE[3][3], HE[3][3], EM[3][3], HM[3][3];
    
    // Calculate the Green's function tensors using cached values
    // EE tensor
    std::complex<double> dyad[3][3];
    outer_product(dyad, r_hat, r_hat);
    
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double delta_ij = (i == j) ? 1.0 : 0.0;
            EE[i][j] = prefac * (term1 * (dyad[i][j] - delta_ij) + term2 * (3.0 * dyad[i][j] - delta_ij));
        }
    }
    
    // HE tensor (optimize cross product)
    std::complex<double> he_prefac = -I * k * C_LIGHT * prefac * inv_r * (1.0 - ikr);
    cross_matrix(HE, r_hat);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            HE[i][j] *= he_prefac;
        }
    }
    
    // EM and HM tensors (reuse calculations)
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EM[i][j] = -HE[i][j]; // Due to duality
            HM[i][j] = EE[i][j];  // Due to duality
        }
    }

    // Calculate scalar products efficiently
    std::complex<double> ee_scalar = 0.0;
    std::complex<double> he_scalar = 0.0;
    std::complex<double> em_scalar = 0.0;
    std::complex<double> hm_scalar = 0.0;

    // Unroll small loops for better optimization
    ee_scalar = u_e_j[0] * (EE[0][0] * u_e_k[0] + EE[0][1] * u_e_k[1] + EE[0][2] * u_e_k[2]) +
                u_e_j[1] * (EE[1][0] * u_e_k[0] + EE[1][1] * u_e_k[1] + EE[1][2] * u_e_k[2]) +
                u_e_j[2] * (EE[2][0] * u_e_k[0] + EE[2][1] * u_e_k[1] + EE[2][2] * u_e_k[2]);

    he_scalar = u_m[2] * (HE[2][0] * u_e_k[0] + HE[2][1] * u_e_k[1]);  // z component only
    em_scalar = u_e_j[0] * EM[0][2] + u_e_j[1] * EM[1][2];  // z component only
    hm_scalar = HM[2][2];  // z-z component only due to u_m

    // Fill output matrix (2x2 scalar result)
    out[0] = ee_scalar;
    out[1] = em_scalar;
    out[2] = he_scalar;
    out[3] = hm_scalar;
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

                // print_complex_matrix("Polarizability matrix before inversion", polarizability_block.data(), 6);
    
                // Invert the 6x6 polarizability matrix
                invert_6x6_matrix_lapack(polarizability_block.data());

                // print_complex_matrix("Polarizability matrix after inversion", polarizability_block.data(), 6);

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

    // Some Debug Stuff to make sure the interaction matrix is correct
    // Print positions
    // std::cout << "\nPositions (in meters):\n";
    // std::cout << "positions = np.array([";
    // for (int i = 0; i < N; ++i) {
    //     std::cout << "[" << positions[i].x << ", " << positions[i].y << ", " << positions[i].z << "]";
    //     if (i < N-1) std::cout << ",\n";
    // }
    // std::cout << "])\n";

    // // Print wavenumber
    // std::cout << "\nWavenumber k (in m^-1):\n";
    // std::cout << "k = " << k << "\n";

    // // Print full interaction matrix
    // std::cout << "\nFull interaction matrix:\n";
    // print_complex_matrix("A", A_cpu.data(), 6*N);

    return A_dev;
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
    auto total_start = std::chrono::high_resolution_clock::now();
    size_t matrix_size = (size_t)(2 * N) * (size_t)(2 * N) * sizeof(cuDoubleComplex);
    
    // Progress tracking (non-atomic since OpenMP will handle synchronization)
    size_t elements_processed = 0;
    int last_percent = -1;
    size_t total_elements = static_cast<size_t>(N) * static_cast<size_t>(N);

    // Timing counters (use regular variables with OpenMP reduction)
    double green_function_time = 0;
    double matrix_copy_time = 0;
    double inversion_time = 0;

    // Allocate CPU buffer for matrix construction
    std::vector<cuDoubleComplex> A_cpu(2 * N * 2 * N);

    // Main construction loop with OpenMP parallelization and reduction
    #pragma omp parallel for collapse(2) schedule(dynamic) \
        reduction(+:green_function_time,matrix_copy_time,inversion_time)
    for (int j = 0; j < N; ++j) {
        for (int k_idx = 0; k_idx < N; ++k_idx) {
            // Update progress atomically
            size_t current_element = ++elements_processed;
            int row_offset = j * 2;
            int col_offset = k_idx * 2;

            if (j == k_idx) {
                auto inv_start = std::chrono::high_resolution_clock::now();
                // Invert the 2x2 matrix
                std::complex<double> det = pol_2x2[j][0][0] * pol_2x2[j][1][1] - pol_2x2[j][0][1] * pol_2x2[j][1][0];
                std::complex<double> inv_2x2[4];
                inv_2x2[0] = pol_2x2[j][1][1] / det;
                inv_2x2[1] = -pol_2x2[j][0][1] / det;
                inv_2x2[2] = -pol_2x2[j][1][0] / det;
                inv_2x2[3] = pol_2x2[j][0][0] / det;
                auto inv_end = std::chrono::high_resolution_clock::now();
                inversion_time += std::chrono::duration_cast<std::chrono::nanoseconds>(inv_end - inv_start).count();

                auto copy_start = std::chrono::high_resolution_clock::now();
                // Copy inverted 2x2 matrix to interaction matrix
                for (int i = 0; i < 2; ++i) {
                    for (int m = 0; m < 2; ++m) {
                        A_cpu[(row_offset + i) * 2 * N + (col_offset + m)] = make_cuDoubleComplex(
                            std::real(inv_2x2[i*2 + m]),
                            std::imag(inv_2x2[i*2 + m])
                        );
                    }
                }
                auto copy_end = std::chrono::high_resolution_clock::now();
                matrix_copy_time += std::chrono::duration_cast<std::chrono::nanoseconds>(copy_end - copy_start).count();
            } else {
                auto green_start = std::chrono::high_resolution_clock::now();
                // Get 2x2 Green's function using scalar version
                std::complex<double> block[4];
                biani_green_matrix_scalar(block, positions[j], positions[k_idx], 
                                        thetas[j], thetas[k_idx], k);
                auto green_end = std::chrono::high_resolution_clock::now();
                green_function_time += std::chrono::duration_cast<std::chrono::nanoseconds>(green_end - green_start).count();
            }

            // Print progress (only from one thread)
            #pragma omp critical
            {
                double progress = (static_cast<double>(current_element) * 100.0) / static_cast<double>(total_elements);
                int percent_complete = static_cast<int>(progress);
                if (percent_complete != last_percent && percent_complete % 10 == 0) {
                    auto current = std::chrono::high_resolution_clock::now();
                    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current - total_start);
                    std::cout << "Matrix construction: " << percent_complete << "% complete\n";
                    std::cout << "Time spent in green function: " << green_function_time/1e9 << "s\n";
                    std::cout << "Time spent in matrix copy: " << matrix_copy_time/1e9 << "s\n";
                    std::cout << "Time spent in matrix inversion: " << inversion_time/1e9 << "s\n";
                    std::cout << "Total time so far: " << elapsed.count() << "s\n";
                    std::cout << "Using " << omp_get_num_threads() << " threads\n\n";
                    last_percent = percent_complete;
                }
            }
        }
    }

    auto gpu_start = std::chrono::high_resolution_clock::now();
    std::cout << "\nMatrix construction complete, transferring to GPU...\n";

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