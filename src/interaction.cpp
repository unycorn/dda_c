#include <complex>
#include <cmath>
#include <iostream>

#include "constants.hpp"
#include "vector3.hpp"

constexpr std::complex<double> I(0.0, 1.0);

// Fills a 3x3 complex matrix with interaction from dipole j to k
void pair_interaction_matrix(std::complex<double> out[3][3], vec3 r_j, vec3 r_k, double k) {
    vec3 r = vec3_sub(r_j, r_k);
    double r_len = vec3_norm(r);

    if (r_len == 0) {
        std::cerr << "Error: self-interaction\n";
        return;
    }

    vec3 r_hat = vec3_unit(r);
    std::complex<double> expikr = std::exp(I * k * r_len);
    std::complex<double> prefac = COULOMBK * expikr / r_len;

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

void get_full_interaction_matrix(
    std::complex<double>* A,                       // Output matrix of size 3N x 3N
    vec3* positions,                               // Array of N positions
    std::complex<double> (*alpha_inv)[3][3],       // Inverse polarizability tensors
    int N,
    double k
) {
    for (int j = 0; j < N; ++j) {
        for (int k_idx = 0; k_idx < N; ++k_idx) {
            int row_offset = j * 3;
            int col_offset = k_idx * 3;

            std::complex<double> block[3][3];

            if (j == k_idx) {
                std::complex<double> (*alpha_inv_local)[3] = alpha_inv[j];
                for (int i = 0; i < 3; ++i)
                    for (int m = 0; m < 3; ++m)
                        A[(row_offset + i) * 3 * N + (col_offset + m)] = alpha_inv_local[i][m];
            } else {
                pair_interaction_matrix(block, positions[j], positions[k_idx], k);
                for (int i = 0; i < 3; ++i)
                    for (int m = 0; m < 3; ++m)
                        A[(row_offset + i) * 3 * N + (col_offset + m)] = block[i][m];
            }
        }
    }
}
