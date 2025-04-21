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


int main() {
    const int N_width = 100;
    const int N_height = 100;
    const int N = N_width * N_height;

    const double spacing = 300e-9;
    const int num_freqs = 30;
    const double f_start = 100e12;
    const double f_end = 500e12;

    // Translational disorder parameters
    double disorder = 100e-9;
    const unsigned int seed = 1;

    // Position array
    std::vector<vec3> positions(N);
    generate_disordered_positions(positions.data(), N_width, N_height, spacing, disorder, seed);

    for (int i = 0; i < num_freqs; ++i) {
        auto start_time = std::chrono::high_resolution_clock::now();

        double freq = f_start + i * (f_end - f_start) / (num_freqs - 1);
        double wavelength = C_LIGHT / freq;
        double k = 2.0 * M_PI / wavelength;

        // 3x3 inverse polarizability tensors per dipole
        using mat3x3 = std::complex<double>[3][3];
        std::vector<mat3x3> alpha_inv(N);
        for (int j = 0; j < N; ++j) {
            auto alpha = lorentz_alpha(freq);
            auto alpha_inv_scalar = 1.0 / alpha;
            for (int i = 0; i < 3; ++i)
                alpha_inv[j][i][i] = alpha_inv_scalar;
        }

        // Full interaction matrix A: size (3N x 3N)
        std::vector<std::complex<double>> A(3 * N * 3 * N, std::complex<double>(0.0, 0.0));
        std::cout << "freq " << freq << ": Computing Interaction Matrix...\n";
        get_full_interaction_matrix(A.data(), positions.data(), alpha_inv.data(), N, k);
        std::cout << "freq " << freq << ": Finished Computing Interaction Matrix!\n";

        // Incident field vector
        std::vector<std::complex<double>> E_inc(3 * N, std::complex<double>(0.0, 0.0));
        for (int j = 0; j < N; ++j) {
            double phase = k * positions[j].z;
            auto val = std::exp(I * phase);
            E_inc[3 * j] = val;
        }

        // Copy to polarizations (in-place LAPACK overwrite)
        std::vector<std::complex<double>> polarizations = E_inc;

        // You may need to declare and size ipiv, b, and dimension
        std::vector<int> ipiv(3 * N); // If needed by LAPACK
        std::complex<double>* b = polarizations.data(); // In-place solve
        int dimension = 3 * N;

        solve_gpu(
            reinterpret_cast<cuDoubleComplex*>(A.data()),
            reinterpret_cast<cuDoubleComplex*>(b),
            dimension
        ); // Solve modifies b in-place

        // Output
        std::ostringstream filename;
        filename << "output/output_" << std::scientific << std::setprecision(2) << freq << ".csv";

        write_polarizations(filename.str().c_str(), b, positions, 1.0/alpha_inv[0][0][0], E_inc, N);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto ms_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "Elapsed: " << ms_duration.count() << " ms\n";
    }

    return 0;
}
