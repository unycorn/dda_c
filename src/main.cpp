#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <complex>

#include "vector3.hpp"
#include "fileio.hpp"
#include "solve_gpu.hpp"

int main() {

    const int N_width = 100;
    const int N_height = 100;
    const int N = N_width * N_height;

    const double spacing = 300e-9;
    const int num_freqs = 1;
    const double f_start = 100e12;
    const double f_end = 500e12;

    vec3 *positions = malloc(N * sizeof(vec3));
    generate_positions(positions, N_width, N_height, spacing);

    for (int i = 0; i < num_freqs; ++i) {
        auto start_time = std::chrono::high_resolution_clock::now();

        double freq = f_start; //+ i * (f_end - f_start) / (num_freqs - 1);
        double wavelength = C_LIGHT / freq;
        double k = 2.0 * M_PI / wavelength;

        double complex (*alpha_inv)[3][3] = malloc(N * sizeof(*alpha_inv));
        // Update polarizabilities for this frequency
        memset(alpha_inv, 0, N * sizeof(*alpha_inv));  // not setting all values to zero will result in SEGFAULT
        for (int j = 0; j < N; ++j) {
            double complex alpha = lorentz_alpha(freq); // same for all dipoles for now
            double complex alpha_inv_scalar = 1.0 / alpha;
            
            // Fill diagonal inverse polarizability tensor
            for (int i = 0; i < 3; ++i)
                alpha_inv[j][i][i] = alpha_inv_scalar;
        }

        double complex *A = calloc(3 * N * 3 * N, sizeof(double complex));
        printf("freq %.2f: Computing Interaction Matrix...\n", freq);
        get_full_interaction_matrix(A, positions, alpha_inv, N, k);
        printf("freq %.2f: Finished Computing Interaction Matrix!\n", freq);

        // Allocate incident field vector E_inc of size 3N (x,y,z for each dipole)
        double complex *E_inc = calloc(3 * N, sizeof(double complex));
        
        // Fill E_inc with a plane wave: E = e^(i k x)
        for (int j = 0; j < N; ++j) {
            double phase = k * positions[j].z;             // Wave is normally incident (along z-axis)
            double complex val = cexp(I * phase);          // Complex exponential phase

            E_inc[3 * j] = val;                        // Polarized along y-axis
        }

        solve_gpu(A, b, dimension);

        // Write solution out to a file.
        // Assume b is overwritten with solution
        write_polarizations("output/output.txt", b, N);
    
        auto end_time = std::chrono::high_resolution_clock::now();
        auto ms_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "Elapsed: " << ms_duration.count() << " ms\n";

        free(alpha_inv);
        free(E_inc);
        free(polarizations);
        free(ipiv);
        free(A);
    }

    free(positions);
    return 0;
}
