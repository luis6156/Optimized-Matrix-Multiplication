/*
 * Tema 2 ASC
 * 2022 Spring
 */
#include <string.h>

#include "cblas.h"
#include "utils.h"

/*
 * Add your BLAS implementation here
 */
double *my_solver(int N, double *A, double *B) {
    printf("BLAS SOLVER\n");

    double *partial_C = (double *)calloc(N * N, sizeof(double));

    memcpy(partial_C, B, N * N * sizeof(double));

    // Compute partial_C = B x A (A is upper triangular)
    cblas_dtrmm(CblasRowMajor, CblasRight, CblasUpper, CblasNoTrans,
                CblasNonUnit, N, N, 1, A, N, partial_C, N);
    // Compute partial_C = B x A x A^t (partial_C x A^t, where A^t is lower
    // triangular)
    cblas_dtrmm(CblasRowMajor, CblasRight, CblasUpper, CblasTrans, CblasNonUnit,
                N, N, 1, A, N, partial_C, N);
    // Compute partial_C = B^t x B + B x A x A^t
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, N, N, N, 1, B, N, B, N,
                1, partial_C, N);

    return partial_C;
}
