#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <complex>
#include <sstream>
#include <cuda_runtime.h>
#include "constants.hpp"
#include "vector3.hpp"
#include "interaction.hpp"
#include "fileio.hpp"
#include "solve_gpu.hpp"

using mat6x6 = std::complex<double>[6][6];
constexpr std::complex<double> I(0.0, 1.0);

// Function declarations
void create_rotation_matrix(std::complex<double> out[6][6], double theta);
void matrix_multiply(std::complex<double> result[6][6], 
                    const std::complex<double> a[6][6], 
                    const std::complex<double> b[6][6]);

void run_simulation(
    double f_start,
    double f_end,
    int num_freqs,
    const std::vector<vec3>& positions,
    int N,
    double spacing,
    double disorder,
    double f0_disorder,
    double angle_disorder,
    unsigned int seed);

// Function to create a rotation matrix for the 6x6 polarizability tensor
void create_rotation_matrix(std::complex<double> out[6][6], double theta) {
    // Typically, rotation matrices are 3x3, but here we need a 6x6 matrix
    // which rotates all four tensors in the 6x6 "supertensor"
    // this matrix is like a tensor product of the 3x3 rotation matrix with the identity matrix
    //         R6x6 = R(3x3) ⊗ I_2 =  | R(3x3)  0      |
    //                                | 0       R(3x3) |

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

// Function to multiply 6x6 complex matrices: result = a * b (or a @ b in Python notation)
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

// Fills a 2D grid of dipoles in the xy-plane (z = 0)
void generate_positions(vec3* positions, int N_width, int N_height, double spacing) {
    int i = 0;
    for (int i_x = 0; i_x < N_width; ++i_x) {
        for (int i_y = 0; i_y < N_height; ++i_y) {
            positions[i++] = vec3{
                i_x * spacing,
                i_y * spacing,
                0.0
            };
        }
    }
}

// Generates a disordered dipole grid in the xy-plane with optional RNG seed
void generate_disordered_positions(vec3* positions, int N_width, int N_height, double spacing, double rms_displacement, unsigned int seed = 0) {
    std::default_random_engine rng(seed);  // seed controls reproducibility
    std::normal_distribution<double> normal(0.0, rms_displacement);  // standard deviation = RMS displacement

    int i = 0;
    for (int i_x = 0; i_x < N_width; ++i_x) {
        for (int i_y = 0; i_y < N_height; ++i_y) {
            positions[i] = vec3{
                i_x * spacing + normal(rng),
                i_y * spacing + normal(rng),
                0.0
            };
            i += 1;
        }
    }
}

// Lorentzian polarizability function in Hz
std::complex<double> lorentz_alpha(double f) {
    std::complex<double> denom = (F0 * F0 - f * f) - I * f * GAMMA_PARAM;
    std::complex<double> norm_alpha = A_PARAM / denom + B_PARAM + C_PARAM * f;
    return norm_alpha * EPSILON_0;
}

// Disordered Lorentzian polarizability function with pre-generated F0 value
std::complex<double> disordered_lorentz_alpha(double f, double disordered_f0) {
    std::complex<double> denom = (disordered_f0 * disordered_f0 - f * f) - I * f * GAMMA_PARAM;
    std::complex<double> norm_alpha = A_PARAM / denom + B_PARAM + C_PARAM * f;
    return norm_alpha * EPSILON_0;
}

void run_simulation(
    double f_start,
    double f_end,
    int num_freqs,
    const std::vector<vec3>& positions,
    int N,
    double spacing,
    double disorder,
    double f0_disorder,
    double angle_disorder,
    unsigned int seed
) {
    // Create RNG once for the whole simulation
    std::default_random_engine rng(seed);
    
    // Generate disordered F0 values for each dipole once
    std::vector<double> disordered_f0s(N);
    std::normal_distribution<double> normal_f0(0.0, f0_disorder);
    for (int j = 0; j < N; ++j) {
        disordered_f0s[j] = F0 + normal_f0(rng);
    }
    
    // Generate random rotation angles for each dipole once
    std::vector<double> rotation_angles(N);
    std::normal_distribution<double> normal_angle(0.0, angle_disorder);
    for (int j = 0; j < N; ++j) {
        rotation_angles[j] = normal_angle(rng);
    }

    for (int i = 0; i < num_freqs; ++i) {
        double freq = f_start + i * (f_end - f_start) / (num_freqs - 1);
        
        // Check if output file already exists
        std::ostringstream filename;
        filename << "output/output_(" << std::scientific << std::setprecision(2)
                 << freq << ")_(" << disorder * 1e9 << "nm)_(" << f0_disorder << "Hz)_(" 
                 << angle_disorder << "rad)_seed" << seed << ".csv";
        
        if (std::ifstream(filename.str()).good()) {
            std::cout << "Skipping frequency " << freq << " Hz - output file already exists\n";
            continue;
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        double wavelength = C_LIGHT / freq;
        double k = 2.0 * M_PI / wavelength;

        std::vector<mat6x6> alpha(N);
        
        for (int j = 0; j < N; ++j) {
            // Calculate polarizabilities (not inverted)
            auto alpha_x = disordered_lorentz_alpha(freq, disordered_f0s[j]);
            auto alpha_yz = lorentz_alpha(200e12);
            
            // Create diagonal polarizability matrix (6x6)
            for (int ii = 0; ii < 6; ++ii) {
                for (int jj = 0; jj < 6; ++jj) {
                    alpha[j][ii][jj] = 0.0;
                }
            }
            // Set electric-electric components (top-left 3x3)
            alpha[j][0][0] = alpha_x;
            alpha[j][1][1] = alpha_yz;
            alpha[j][2][2] = alpha_yz;

            // Set magnetic-magnetic components (bottom-right 3x3) to small non-zero values
            // These values are small enough to not affect the physics significantly
            alpha[j][3][3] = alpha_x * 1e-6;
            alpha[j][4][4] = alpha_yz * 1e-6;
            alpha[j][5][5] = alpha_yz * 1e-6;
            
            // Create rotation matrix and its transpose for 6x6
            std::complex<double> rotation[6][6];
            std::complex<double> rotation_T[6][6];
            create_rotation_matrix(rotation, rotation_angles[j]);
            create_rotation_matrix(rotation_T, -rotation_angles[j]);
            
            // Rotate the polarizability matrix: R * alpha * R^T
            std::complex<double> temp[6][6];
            matrix_multiply(temp, rotation, alpha[j]);
            matrix_multiply(alpha[j], temp, rotation_T);
        }

        std::vector<std::complex<double>> A_host(6 * N * 6 * N, std::complex<double>(0.0, 0.0));
        cuDoubleComplex* A_device = get_full_interaction_matrix(A_host.data(), positions.data(), alpha.data(), N, k);
        std::cout << "freq " << freq << ": Finished Computing Interaction Matrix!\n";

        // Initialize incident field (now 6N components, but only E-field is non-zero)
        std::vector<std::complex<double>> inc_field(6 * N, std::complex<double>(0.0, 0.0));
        for (int j = 0; j < N; ++j) {
            double phase = k * positions[j].z;
            auto val = std::exp(I * phase);
            inc_field[6 * j] = val;  // Only x-component of E-field is non-zero
        }

        std::vector<std::complex<double>> b(6 * N);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < 6; ++j) {
                b[i * 6 + j] = inc_field[i * 6 + j];
            }
        }

        std::vector<cuDoubleComplex> b_cuda(6 * N);
        for (int i = 0; i < 6 * N; ++i) {
            b_cuda[i] = make_cuDoubleComplex(std::real(b[i]), std::imag(b[i]));
        }

        solve_gpu(A_device, b_cuda.data(), 6 * N);

        for (int i = 0; i < 6 * N; ++i) {
            b[i] = std::complex<double>(cuCreal(b_cuda[i]), cuCimag(b_cuda[i]));
        }

        cudaFree(A_device);

        // For output, we only use the electric part of the response
        write_polarizations(filename.str().c_str(), b.data(), positions, alpha[0][0][0], inc_field, N);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto ms_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "Elapsed: " << ms_duration.count() << " ms\n";
    }
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <disorder_nm> <f0_disorder_Hz> <angle_disorder_rad> <seed>\n";
        std::cerr << "  disorder_nm: RMS displacement in nanometers\n";
        std::cerr << "  f0_disorder_Hz: RMS disorder in F0 frequency (Hz)\n";
        std::cerr << "  angle_disorder_rad: RMS disorder in rotation angle (radians)\n";
        std::cerr << "  seed: Random number generator seed\n";
        return 1;
    }

    double disorder = std::stod(argv[1]) * 1e-9;
    double f0_disorder = std::stod(argv[2]);
    double angle_disorder = std::stod(argv[3]);
    unsigned int seed = static_cast<unsigned int>(std::stoul(argv[4]));

    const int N_width = 80;
    const int N_height = 80;
    const int N = N_width * N_height;
    const double spacing = 300e-9;

    std::vector<vec3> positions(N);
    generate_disordered_positions(positions.data(), N_width, N_height, spacing, disorder, seed);

    run_simulation(201e12, 240e12, 10, positions, N, spacing, disorder, f0_disorder, angle_disorder, seed);
    // run_simulation(100e12, 500e12, 50, positions, N, spacing, disorder, f0_disorder, angle_disorder, seed);
    // run_simulation(151e12, 251e12, 20, positions, N, spacing, disorder, f0_disorder, angle_disorder, seed);

    return 0;
}
