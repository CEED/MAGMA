/*
 *   -- MAGMA (version 1.1) --
 *      Univ. of Tennessee, Knoxville
 *      Univ. of California, Berkeley
 *      Univ. of Colorado, Denver
 *      November 2011
 *
 * @precisions normal z -> s d c
 */

#ifndef MAGMA_ZBULGEINC_H
#define MAGMA_ZBULGEINC_H

#define PRECISION_z

#ifdef __cplusplus
extern "C" {
#endif

magma_int_t magma_zhetrd_he2hb(
    char uplo, magma_int_t n, magma_int_t NB,
    magmaDoubleComplex *a, magma_int_t lda,
    magmaDoubleComplex *tau, magmaDoubleComplex *work, magma_int_t lwork,
    magmaDoubleComplex *dT, magma_int_t threads,
    magma_int_t *info);

magma_int_t magma_zhetrd_hb2st(
    magma_int_t threads, char uplo, magma_int_t n, magma_int_t nb, magma_int_t Vblksiz,
    magmaDoubleComplex *A, magma_int_t lda,
    double *D, double *E,
    magmaDoubleComplex *V, magma_int_t ldv,
    magmaDoubleComplex *TAU, magma_int_t compT,
    magmaDoubleComplex *T, magma_int_t ldt);

magma_int_t magma_zbulge_back(
    magma_int_t threads, char uplo, magma_int_t n, magma_int_t nb, magma_int_t ne, magma_int_t Vblksiz,
    magmaDoubleComplex *Z, magma_int_t ldz,
    magmaDoubleComplex *dZ, magma_int_t lddz,
    magmaDoubleComplex *V, magma_int_t ldv,
    magmaDoubleComplex *TAU,
    magmaDoubleComplex *T, magma_int_t ldt,
    magma_int_t* info);

magma_int_t magma_zunmqr_gpu_2stages(
    char side, char trans, magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex *dA, magma_int_t ldda,
    magmaDoubleComplex *dC, magma_int_t lddc,
    magmaDoubleComplex *dT, magma_int_t nb,
    magma_int_t *info);

magma_int_t magma_zbulge_get_lq2(magma_int_t n);

magma_int_t magma_zbulge_get_Vblksiz(magma_int_t n, magma_int_t nb);

#ifdef __cplusplus
}
#endif

#endif
