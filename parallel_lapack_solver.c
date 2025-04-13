#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// LAPACK prototype for dgesv (solves Ax = b)
extern void dgesv_(int *n, int *nrhs, double *a, int *lda,
                   int *ipiv, double *b, int *ldb, int *info);

void build_matrix_and_rhs(int size, double *A, double *b, int seed) {
    srand(seed);
    for (int i = 0; i < size * size; ++i)
        A[i] = 1.0 + rand() % 10;  // simple random matrix

    for (int i = 0; i < size; ++i)
        b[i] = 1.0 + rand() % 10;  // random RHS
}

int main() {
    const int num_systems = 8;   // how many independent systems to solve
    const int size = 4;          // dimension of each system

    #pragma omp parallel for
    for (int i = 0; i < num_systems; ++i) {
        double A[size * size];
        double b[size];
        int ipiv[size];
        int info;
        int nrhs = 1;

        build_matrix_and_rhs(size, A, b, i + 1);

        dgesv_((int *)&size, &nrhs, A, (int *)&size, ipiv, b, (int *)&size, &info);

        if (info == 0) {
            printf("Thread %d: Solution x = [", omp_get_thread_num());
            for (int j = 0; j < size; ++j)
                printf(" %g", b[j]);
            printf(" ]\n");
        } else {
            printf("Thread %d: Error solving system (info=%d)\n", omp_get_thread_num(), info);
        }
    }

    return 0;
}
