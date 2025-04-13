#include "vector3.h"
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include "constants.h"

// Fills a 3x3 complex matrix with interaction from dipole j to k
void pair_interaction_matrix(double complex out[3][3], vec3 r_j, vec3 r_k, double k) {
    vec3 r = vec3_sub(r_j, r_k);
    double r_len = vec3_norm(r);

    if (r_len == 0) {
        fprintf(stderr, "Error: self-interaction\n");
        return;
    }

    vec3 r_hat = vec3_unit(r);
    double complex expikr = cexp(I * k * r_len);
    double complex prefac = COULOMBK * expikr / r_len;

    double complex term1 = k * k;
    double complex term2 = (I * k * r_len - 1.0) / (r_len * r_len);

    double complex dyad[3][3];
    outer_product(dyad, r_hat, r_hat);

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            out[i][j] = prefac * (term1 * (dyad[i][j] - (i == j)) + term2 * (3 * dyad[i][j] - (i == j)));
}

void get_full_interaction_matrix(
    double complex *A,       // Output matrix of size 3N x 3N
    const vec3 *positions,   // Array of N positions
    const double complex (*alpha_inv)[3][3],  // Same for all for now
    int N,
    double k
) {
    for (int j = 0; j < N; ++j) {
        for (int k_idx = 0; k_idx < N; ++k_idx) {
            int row_offset = j * 3;
            int col_offset = k_idx * 3;

            double complex block[3][3];

            if (j == k_idx) {
                const double complex (*alpha)[3] = alpha_inv[j];
                for (int i = 0; i < 3; ++i)
                    for (int m = 0; m < 3; ++m)
                        A[(row_offset + i) * 3*N + (col_offset + m)] = alpha[i][m];
            } else {
                pair_interaction_matrix(block, positions[j], positions[k_idx], k);
                for (int i = 0; i < 3; ++i)
                    for (int m = 0; m < 3; ++m)
                        A[(row_offset + i) * 3*N + (col_offset + m)] = block[i][m];
            }
        }
    }
}
