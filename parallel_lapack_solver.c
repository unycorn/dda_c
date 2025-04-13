#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <omp.h>

// LAPACK complex solver
extern void zgesv_(int*, int*, double _Complex*, int*, int*,
                   double _Complex*, int*, int*);

// Simulate DDA matrix and incident field
void generate_complex_A_and_E(int N, double freq,
                              double _Complex *A, double _Complex *E_inc) {
    for (int i = 0; i < N * N; ++i) {
        A[i] = (i % (N + 1) == 0) ? (1.0 + freq) + 0.1*I : (0.01 * freq) + 0.02*I;
    }

    for (int i = 0; i < N; ++i) {
        E_inc[i] = 1.0 + 0.0*I;  // Real incident field
    }
}

// Solve and write polarization vector p to a file
void solve_complex_system(int N, double freq) {
    double _Complex *A = malloc(N * N * sizeof(double _Complex));
    double _Complex *E_inc = malloc(N * sizeof(double _Complex));
    int *ipiv = malloc(N * sizeof(int));
    int nrhs = 1, info;

    generate_complex_A_and_E(N, freq, A, E_inc);

    zgesv_(&N, &nrhs, A, &N, ipiv, E_inc, &N, &info);

    if (info == 0) {
        char filename[64];
        snprintf(filename, sizeof(filename), "output_freq_%.2f.txt", freq);
        FILE *fp = fopen(filename, "w");
        if (fp) {
            for (int i = 0; i < N; ++i)
                fprintf(fp, "%g %g\n", creal(E_inc[i]), cimag(E_inc[i]));
            fclose(fp);
            printf("freq=%.2f: wrote to %s\n", freq, filename);
        } else {
            printf("freq=%.2f: failed to open file\n", freq);
        }
    } else {
        printf("freq=%.2f: Solve failed (info=%d)\n", freq, info);
    }

    free(A); free(E_inc); free(ipiv);
}

int main() {
    const int N = 100;  // Number of dipoles
    const int num_freqs = 10;
    const double f_start = 0.1, f_end = 1.0;

    #pragma omp parallel for
    for (int i = 0; i < num_freqs; ++i) {
        double freq = f_start + i * (f_end - f_start) / (num_freqs - 1);
        solve_complex_system(N, freq);
    }

    return 0;
}
