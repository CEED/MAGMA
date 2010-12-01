/*
 *   -- MAGMA (version 1.0) --
 *      Univ. of Tennessee, Knoxville
 *      Univ. of California, Berkeley
 *      Univ. of Colorado, Denver
 *      November 2010
 *
 * @precisions mixed zc -> ds
 */

#ifndef _MAGMA_ZC_H_
#define _MAGMA_ZC_H_

#ifdef __cplusplus
extern "C" {
#endif

/* Mixed precision */
magma_int_t magma_zcgetrs_gpu( magma_int_t n, magma_int_t nrhs, cuFloatComplex *a, magma_int_t lda, 
                               magma_int_t *ipiv, cuFloatComplex *x, cuDoubleComplex *b, magma_int_t ldb, magma_int_t *info);
magma_int_t magma_zcgesv_gpu( magma_int_t N, magma_int_t NRHS, 
                              cuDoubleComplex *A, magma_int_t LDA, 
                              magma_int_t *IPIV, magma_int_t *DIPIV,
                              cuDoubleComplex *B, magma_int_t LDB, 
                              cuDoubleComplex *X, magma_int_t LDX, 
                              cuDoubleComplex *WORK, cuFloatComplex *SWORK,
                              magma_int_t *ITER, magma_int_t *INFO);
magma_int_t magma_zcposv_gpu( char uplo, magma_int_t n, magma_int_t nrhs, 
                              cuDoubleComplex *A, magma_int_t lda, 
                              cuDoubleComplex *B, magma_int_t ldb, 
                              cuDoubleComplex *X, magma_int_t ldx, 
                              cuDoubleComplex *dworkd, cuFloatComplex *dworks,
                              magma_int_t *iter, magma_int_t *info);
magma_int_t magma_zcgeqrsv_gpu(magma_int_t M, magma_int_t N, magma_int_t NRHS, cuDoubleComplex *A, magma_int_t LDA, cuDoubleComplex *B, 
			       magma_int_t LDB, cuDoubleComplex *X,magma_int_t LDX, cuDoubleComplex *WORK, cuFloatComplex *SWORK, 
			       magma_int_t *ITER, magma_int_t *INFO, cuFloatComplex *tau, magma_int_t lwork, cuFloatComplex *h_work,
			       cuFloatComplex *d_work, cuDoubleComplex *tau_d, magma_int_t lwork_d, cuDoubleComplex *h_work_d,
			       cuDoubleComplex *d_work_d);
  

#ifdef __cplusplus
}
#endif

#endif /* _MAGMA_Z_H_ */
