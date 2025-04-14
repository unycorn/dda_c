// main.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <omp.h>
#include "vector3.h"
#include "interaction.h"
#include "constants.h"

// LAPACK prototype for complex linear solver
extern void zgesv_(int *n, int *nrhs, double _Complex *a, int *lda,
                   int *ipiv, double _Complex *b, int *ldb, int *info);

// Example: uniformly spaced dipole patch in xy-plane
void generate_positions(vec3 *positions, int N_width, int N_height, double spacing) {
    int i = 0;
    for (int i_x = 0; i_x < N_width; ++i_x) {
        for (int i_y = 0; i_y < N_width; ++i_y) {
            positions[i].x = i_x * spacing;
            positions[i].y = i_y * spacing;
            positions[i].z = 0.0;

            i += 1;
        }
    }
}

double complex lorentz_alpha(double f) {
    double complex denom = (F0 * F0 - f * f) - I * f * GAMMA_PARAM;
    double complex norm_alpha = A_PARAM / denom + B_PARAM + C_PARAM * f;
    return norm_alpha * EPSILON_0;
}

int main() {
    const int N_width = 30;
    const int N_height = 30;
    const int N = N_width * N_width;

    const double spacing = 300e-9;
    const int num_freqs = 1;
    const double f_start = 100e12;
    const double f_end = 500e12;

    vec3 *positions = malloc(N * sizeof(vec3));
    generate_positions(positions, N_width, N_height, spacing);

    #pragma omp parallel for
    for (int i = 0; i < num_freqs; ++i) {
        double start = omp_get_wtime();

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
        printf("alpha %.6e %.6e \n", creal(lorentz_alpha(freq)), cimag(lorentz_alpha(freq)));


        double complex *A = calloc(3 * N * 3 * N, sizeof(double complex));
        printf("freq %.2f: Computing Interaction Matrix...\n", freq);
        get_full_interaction_matrix(A, positions, alpha_inv, N, k);
        printf("freq %.2f: Finished Computing Interaction Matrix!\n", freq);

        printf("freq %.6e A[3,0] %.6e %.6e \n", freq, creal(A[3,0]), cimag(A[3,0]));

        // Allocate incident field vector E_inc of size 3N (x,y,z for each dipole)
        double complex *E_inc = calloc(3 * N, sizeof(double complex));
        
        // Fill E_inc with a plane wave: E = e^(i k x)
        for (int j = 0; j < N; ++j) {
            double phase = k * positions[j].z;             // Wave is normally incident (along z-axis)
            double complex val = cexp(I * phase);          // Complex exponential phase

            E_inc[3 * j] = val;                        // Polarized along y-axis
        }

        // Allocate pivot index array for LAPACK solver
        int *ipiv = malloc(3 * N * sizeof(int));

        // Allocate polarization vector -- since LAPACK solves in-place, this starts 
        // out as the incident field E_inc but is immeidately overwritten with p
        double complex *polarizations = malloc(3 * N * sizeof(double complex));
        memcpy(polarizations, E_inc, 3 * N * sizeof(double complex));

        // LAPACK expects:
        // n     = dimension of square matrix (3N)
        // nrhs  = number of right-hand sides (1, since E_inc is a single vector)
        // A     = input matrix (3N x 3N, stored column-major)
        // ipiv  = output pivot indices for LU factorization
        // polarizations = right-hand side (begins as incident field E_inc and is overwritten with solution vector p)
        // info  = status flag (0 = success)
        int info, nrhs = 1;
        printf("freq %.2f: Solving Matrix Equation...\n", freq);
        zgesv_(&(int){3 * N}, &nrhs, A, &(int){3 * N}, ipiv, polarizations, &(int){3 * N}, &info);
        printf("freq %.2f: Finished Solving Matrix Equation!\n", freq);

        if (info != 0) {
            printf("freq %.2f: solve failed (info = %d)\n", freq, info);
        } else {
            // Write resulting dipole polarization vector p to file
            char fname[64];
            snprintf(fname, sizeof(fname), "output/output_freq_%.0f.txt", freq);
            FILE *f = fopen(fname, "w");
            if (!f) {
                fprintf(stderr, "Error: could not open %s for writing\n", fname);
                exit(1);
            }

            // CSV header
            fprintf(f, "Re_px,Im_px,Re_py,Im_py,Re_pz,Im_pz\n");

            for (int j = 0; j < N; ++j) {
                int idx = 3 * j;
                fprintf(f, "%.6e,%.6e,%.6e,%.6e,%.6e,%.6e\n",
                        creal(polarizations[idx + 0]), cimag(polarizations[idx + 0]),  // x
                        creal(polarizations[idx + 1]), cimag(polarizations[idx + 1]),  // y
                        creal(polarizations[idx + 2]), cimag(polarizations[idx + 2])); // z
            }

            fclose(f);
        }

        free(alpha_inv);
        free(E_inc);
        free(polarizations);
        free(ipiv);
        free(A);

        // print how much time it took for this thread to compute.
        double end = omp_get_wtime();
        printf("Thread %2d | freq = %.3e | time = %.6f s\n",
            omp_get_thread_num(), freq, end - start);
    }

    free(positions);
    return 0;
}
