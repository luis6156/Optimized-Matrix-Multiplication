/*
 * Tema 2 ASC
 * 2022 Spring
 */
#include "utils.h"

/*
 * Add your unoptimized implementation here
 */
double *my_solver(int N, double *A, double *B) {
    printf("NEOPT SOLVER\n");

    double *partial_BA = (double *)calloc(N * N, sizeof(double));
    double *partial_BAA_t = (double *)calloc(N * N, sizeof(double));
    double *partial_B_tB = (double *)calloc(N * N, sizeof(double));
    double *C = (double *)calloc(N * N, sizeof(double));

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            /*
             * Compute partial_BA = B x A (A is triangular so we only need the
             * upper half of the matrix).
             * Compute partial_B_tB = B^t x B in the same loop
             */
            for (int k = 0; k <= j; ++k) {
                partial_BA[i * N + j] += B[i * N + k] * A[k * N + j];
                partial_B_tB[i * N + j] += B[k * N + i] * B[k * N + j];
            }
            /*
             * Compute the rest of the matrix partial_B_tB
             */
            for (int k = j + 1; k < N; ++k) {
                partial_B_tB[i * N + j] += B[k * N + i] * B[k * N + j];
            }
        }

        /*
         * This part could not be done in parallel with the other partial
         * matrices as partial_BA is not written to fast enough.
         */
        for (int j = 0; j < N; ++j) {
            /*
             * Compute the matrix partial_BAA_t = B x A x A^t (partial_BA x
             * A^t). A^t is lower triangular therefore we only need the lower
             * half for computations. Compute the final matrix C = B x A x A^t +
             * B^t x B (partial_BAA_t x partialB_tT).
             */
            for (int k = j; k < N; ++k) {
                partial_BAA_t[i * N + j] +=
                    partial_BA[i * N + k] * A[j * N + k];
            }
            C[i * N + j] = partial_BAA_t[i * N + j] + partial_B_tB[i * N + j];
        }
    }

    free(partial_B_tB);
    free(partial_BAA_t);
    free(partial_BA);

    return C;
}
