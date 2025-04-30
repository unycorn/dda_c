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

// Disordered Lorentzian polarizability function with random F0
std::complex<double> disordered_lorentz_alpha(double f, double f0_rms_disorder, unsigned int& seed, std::default_random_engine& rng) {
    // Use the provided RNG to maintain consistency with position disorder
    std::normal_distribution<double> normal(0.0, f0_rms_disorder);
    
    // Add disorder to F0
    double disordered_f0 = F0 + normal(rng);
    
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
    unsigned int seed
) {
    // Create RNG once for the whole simulation to maintain consistent disorder
    std::default_random_engine rng(seed);

    for (int i = 0; i < num_freqs; ++i) {
        auto start_time = std::chrono::high_resolution_clock::now();

        double freq = f_start + i * (f_end - f_start) / (num_freqs - 1);
        double wavelength = C_LIGHT / freq;
        double k = 2.0 * M_PI / wavelength;

        std::vector<mat3x3> alpha_inv(N);
        for (int j = 0; j < N; ++j) {
            auto alpha_x = disordered_lorentz_alpha(freq, f0_disorder, seed, rng);
            auto alpha_x_inv_scalar = 1.0 / alpha_x;
            alpha_inv[j][0][0] = alpha_x_inv_scalar;

            // Keep y and z polarizabilities ordered
            auto alpha_yz = lorentz_alpha(200e12);
            auto alpha_yz_inv_scalar = 1.0 / alpha_yz;
            alpha_inv[j][1][1] = alpha_yz_inv_scalar;
            alpha_inv[j][2][2] = alpha_yz_inv_scalar;
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
        filename << "output/output_" << std::scientific << std::setprecision(2)
                 << freq << "_" << disorder * 1e9 << "nm_f0d" << f0_disorder << "_seed" << seed << ".csv";

        write_polarizations(filename.str().c_str(), b, positions, 1.0 / alpha_inv[0][0][0], E_inc, N);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto ms_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "Elapsed: " << ms_duration.count() << " ms\n";
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <disorder_nm> <f0_disorder_Hz> <seed>\n";
        std::cerr << "  disorder_nm: RMS displacement in nanometers\n";
        std::cerr << "  f0_disorder_Hz: RMS disorder in F0 frequency (Hz)\n";
        std::cerr << "  seed: Random number generator seed\n";
        return 1;
    }

    double disorder = std::stod(argv[1]) * 1e-9;
    double f0_disorder = std::stod(argv[2]);
    unsigned int seed = static_cast<unsigned int>(std::stoul(argv[3]));

    const int N_width = 100;
    const int N_height = 100;
    const int N = N_width * N_height;
    const double spacing = 300e-9;

    std::vector<vec3> positions(N);
    generate_disordered_positions(positions.data(), N_width, N_height, spacing, disorder, seed);

    run_simulation(201e12, 240e12, 10, positions, N, spacing, disorder, f0_disorder, seed);
    // run_simulation(100e12, 500e12, 30, positions, N, spacing, disorder, f0_disorder, seed);
    // run_simulation(151e12, 251e12, 20, positions, N, spacing, disorder, f0_disorder, seed);

    return 0;
}
