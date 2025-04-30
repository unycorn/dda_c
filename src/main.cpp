#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <complex>
#include <sstream>
#include <random>
#include "constants.hpp"
#include "vector3.hpp"
#include "interaction.hpp"
#include "fileio.hpp"
#include "solve_gpu.hpp"

using mat3x3 = std::complex<double>[3][3];
constexpr std::complex<double> I(0.0, 1.0);

// Function to print a 3x3 complex matrix
void print_matrix(const std::complex<double> matrix[3][3], const std::string& label) {
    std::cout << "\n" << label << ":\n";
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            std::cout << std::setw(15) << std::real(matrix[i][j]) 
                     << " + " << std::setw(15) << std::imag(matrix[i][j]) << "i  ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

// Function to create a 2D rotation matrix in the xy-plane
void create_rotation_matrix(std::complex<double> out[3][3], double theta) {
    // Clear the matrix
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            out[i][j] = 0.0;
        }
    }
    
    // Fill in rotation matrix elements
    out[0][0] = std::cos(theta);
    out[0][1] = -std::sin(theta);
    out[1][0] = std::sin(theta);
    out[1][1] = std::cos(theta);
    out[2][2] = 1.0;  // z-component unchanged
}

// Function to multiply 3x3 complex matrices: result = a * b
void matrix_multiply(std::complex<double> result[3][3], 
                    const std::complex<double> a[3][3], 
                    const std::complex<double> b[3][3]) {
    std::complex<double> temp[3][3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            temp[i][j] = 0;
            for (int k = 0; k < 3; ++k) {
                temp[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
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
        // std::cout << "f0 shift " << disordered_f0s[j]-F0 << "\n";
    }
    
    // Generate random rotation angles for each dipole once
    std::vector<double> rotation_angles(N);
    std::normal_distribution<double> normal_angle(0.0, angle_disorder);
    for (int j = 0; j < N; ++j) {
        rotation_angles[j] = normal_angle(rng);
        // std::cout << "angle shift " << rotation_angles[j] << "\n";
    }

    for (int i = 0; i < num_freqs; ++i) {
        auto start_time = std::chrono::high_resolution_clock::now();

        double freq = f_start + i * (f_end - f_start) / (num_freqs - 1);
        double wavelength = C_LIGHT / freq;
        double k = 2.0 * M_PI / wavelength;

        std::vector<mat3x3> alpha_inv(N);
        
        for (int j = 0; j < N; ++j) {
            // Calculate inverted polarizabilities
            auto alpha_x = disordered_lorentz_alpha(freq, disordered_f0s[j]);
            auto alpha_x_inv_scalar = 1.0 / alpha_x;
            auto alpha_yz = lorentz_alpha(200e12);
            auto alpha_yz_inv_scalar = 1.0 / alpha_yz;
            
            // Create diagonal inverted polarizability matrix
            for (int ii = 0; ii < 3; ++ii) {
                for (int jj = 0; jj < 3; ++jj) {
                    alpha_inv[j][ii][jj] = 0.0;
                }
            }
            alpha_inv[j][0][0] = alpha_x_inv_scalar;
            alpha_inv[j][1][1] = alpha_yz_inv_scalar;
            alpha_inv[j][2][2] = alpha_yz_inv_scalar;
            
            if (j == 0) { // Print for first dipole only to avoid cluttering output
                print_matrix(alpha_inv[j], "Alpha inverse before rotation");
            }
            
            // Create rotation matrix and its transpose
            std::complex<double> rotation[3][3];
            std::complex<double> rotation_T[3][3];
            create_rotation_matrix(rotation, rotation_angles[j]);
            create_rotation_matrix(rotation_T, -rotation_angles[j]);  // Transpose = inverse for rotation matrices
            
            // Rotate the inverted polarizability matrix: R * alpha_inv * R^T
            std::complex<double> temp[3][3];
            matrix_multiply(temp, rotation, alpha_inv[j]);
            matrix_multiply(alpha_inv[j], temp, rotation_T);
            
            if (j == 0) { // Print for first dipole only to avoid cluttering output
                print_matrix(alpha_inv[j], "Alpha inverse after rotation");
            }
        }

        std::vector<std::complex<double>> A(3 * N * 3 * N, std::complex<double>(0.0, 0.0));
        std::cout << "freq " << freq << ": Computing Interaction Matrix...\n";
        get_full_interaction_matrix(A.data(), positions.data(), alpha_inv.data(), N, k);
        std::cout << "freq " << freq << ": Finished Computing Interaction Matrix!\n";

        std::vector<std::complex<double>> E_inc(3 * N, std::complex<double>(0.0, 0.0));
        for (int j = 0; j < N; ++j) {
            double phase = k * positions[j].z;
            auto val = std::exp(I * phase);
            E_inc[3 * j] = val;
        }

        std::vector<std::complex<double>> polarizations = E_inc;
        std::complex<double>* b = polarizations.data();
        int dimension = 3 * N;

        solve_gpu(
            reinterpret_cast<cuDoubleComplex*>(A.data()),
            reinterpret_cast<cuDoubleComplex*>(b),
            dimension
        );

        std::ostringstream filename;
        filename << "output/output_(" << std::scientific << std::setprecision(2)
                 << freq << ")_(" << disorder * 1e9 << "nm)_(" << f0_disorder << "Hz)_(" 
                 << angle_disorder << "rad)_seed" << seed << ".csv";

        write_polarizations(filename.str().c_str(), b, positions, 1.0 / alpha_inv[0][0][0], E_inc, N);

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

    const int N_width = 100;
    const int N_height = 100;
    const int N = N_width * N_height;
    const double spacing = 300e-9;

    std::vector<vec3> positions(N);
    generate_disordered_positions(positions.data(), N_width, N_height, spacing, disorder, seed);

    run_simulation(201e12, 240e12, 10, positions, N, spacing, disorder, f0_disorder, angle_disorder, seed);
    // run_simulation(100e12, 500e12, 30, positions, N, spacing, disorder, f0_disorder, angle_disorder, seed);
    // run_simulation(151e12, 251e12, 20, positions, N, spacing, disorder, f0_disorder, angle_disorder, seed);

    return 0;
}
