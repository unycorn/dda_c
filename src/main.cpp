#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <complex>

#include "constants.hpp"
#include "vector3.hpp"
#include "fileio.hpp"
#include "solve_gpu.hpp"

constexpr std::complex<double> I(0.0, 1.0);

int main() {
    const int N_width = 100;
    const int N_height = 100;
    const int N = N_width * N_height;

    const double spacing = 300e-9;
    const int num_freqs = 2;
    const double f_start = 100e12;
    const double f_end = 500e12;

    // Position array
    std::vector<vec3> positions(N);
    generate_positions(positions.data(), N_width, N_height, spacing);

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
        write_polarizations("output/output.txt", reinterpret_cast<const cuDoubleComplex*>(b), N);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto ms_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "Elapsed: " << ms_duration.count() << " ms\n";
    }

    return 0;
}
