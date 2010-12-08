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
magma_int_t magma_zcgesv_gpu(  magma_int_t N, magma_int_t NRHS, 
			       cuDoubleComplex *dA, magma_int_t ldda, 
			       magma_int_t *IPIV, magma_int_t *dIPIV,
			       cuDoubleComplex *dB, magma_int_t lddb, 
			       cuDoubleComplex *dX, magma_int_t lddx, 
			       cuDoubleComplex *dworkd, cuFloatComplex *dworks,
			       magma_int_t *iter, magma_int_t *info);

magma_int_t magma_zcgetrs_gpu( magma_int_t n, magma_int_t nrhs, 
			       cuFloatComplex  *dA, magma_int_t ldda,
                               magma_int_t *ipiv, 
			       cuDoubleComplex *dB, magma_int_t lddb,
			       cuFloatComplex  *dX, magma_int_t lddx, 
			       magma_int_t *info );

magma_int_t magma_zcposv_gpu( char uplo, magma_int_t n, magma_int_t nrhs, 
                              cuDoubleComplex *dA, magma_int_t ldda, 
                              cuDoubleComplex *dB, magma_int_t lddb, 
                              cuDoubleComplex *dX, magma_int_t lddx, 
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
