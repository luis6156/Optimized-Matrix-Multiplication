/*
 * Tema 2 ASC
 * 2022 Spring
 */
#include "utils.h"

/*
 * Add your optimized implementation here
 */
double *my_solver(int N, double *A, double *B) {
    printf("OPT SOLVER\n");

    int i, j, k;
    int size = N * N;

    double *partial_BA = (double *)calloc(size, sizeof(double));
    double *partial_BAA_t = (double *)calloc(size, sizeof(double));
    double *partial_B_tB = (double *)calloc(size, sizeof(double));
    double *C = (double *)calloc(size, sizeof(double));

    register double *B_ik = B;
    register double *BA_ij = partial_BA;
    register double *BA_i = partial_BA;
    register double *BAA_t_ij = partial_BAA_t;
    register double *B_tB_ij = partial_B_tB;
    register double *C_ij = C;

    for (i = 0; i < N; ++i) {
        register double *A_kj = A;
        register double *B_ki = B + i;
        register double *B_kj = B;
        register double *A_jk = A;

        for (k = 0; k < N; ++k) {
            register double *BA_ij_copy = BA_ij;
            register double *B_tB_ij_copy = B_tB_ij;

            /*
             * Compute the first half of the matrix partial_B_tB
             */
            for (j = 0; j < k; ++j) {
                *B_tB_ij_copy += (*B_ki) * (*B_kj);

                ++B_tB_ij_copy;
                ++B_kj;
            }

            A_kj += k;

            /*
             * Compute partial_BA = B x A (A is triangular so we only need the
             * upper half of the matrix).
             * Compute the rest of partial_B_tB = B^t x B in the same loop
             */
            for (j = k; j < N; ++j) {
                *BA_ij_copy += (*B_ik) * (*A_kj);
                *B_tB_ij_copy += (*B_ki) * (*B_kj);

                ++B_tB_ij_copy;
                ++BA_ij_copy;
                ++B_kj;
                ++A_kj;
            }

            B_ki += N;
            ++BA_ij;
            ++B_ik;
        }

        /*
         * This part could not be done in parallel with the other partial
         * matrices as partial_BA is not written to fast enough.
         */
        for (j = 0; j < N; ++j) {
            register double *BA_i_copy = BA_i + j;
            register double *A_jk_copy = A_jk + j;

            /*
             * Compute the matrix partial_BAA_t = B x A x A^t (partial_BA x
             * A^t). A^t is lower triangular therefore we only need the lower
             * half for computations. Compute the final matrix C = B x A x A^t +
             * B^t x B (partial_BAA_t x partialB_tT).
             */
            for (int k = j; k < N; ++k) {
                *BAA_t_ij += (*BA_i_copy) * (*A_jk_copy);
                ++BA_i_copy;
                ++A_jk_copy;
            }

            *C_ij = *BAA_t_ij + *B_tB_ij;

            ++C_ij;
            ++B_tB_ij;
            ++BAA_t_ij;
            A_jk += N;
        }

        BA_i += N;
    }

    free(partial_B_tB);
    free(partial_BAA_t);
    free(partial_BA);

    return C;
}
