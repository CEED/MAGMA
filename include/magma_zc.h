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
magma_int_t magma_zcgetrs_gpu(magma_int_t n, magma_int_t nrhs, float2 *a, magma_int_t lda, 
			      magma_int_t *ipiv, float2 *x, double2 *b, magma_int_t ldb, magma_int_t *info);


magma_int_t magma_zcgesv_gpu(magma_int_t N, magma_int_t NRHS, double2 *A, magma_int_t LDA, magma_int_t *IPIV, double2 *B, 
			     magma_int_t LDB, double2 *X, magma_int_t LDX, double2 *WORK, float2 *SWORK,
			     magma_int_t *ITER, magma_int_t *INFO, float2 *H_SWORK, double2 *H_WORK,
			     magma_int_t *DIPIV);


magma_int_t magma_zcposv_gpu(char UPLO, magma_int_t N, magma_int_t NRHS, double2 *A, magma_int_t LDA, double2 *B, 
			     magma_int_t LDB, double2 *X, magma_int_t LDX, double2 *WORK, float2 *SWORK,
			     magma_int_t *ITER, magma_int_t *INFO, float2 *H_SWORK, double2 *H_WORK);

magma_int_t magma_zcgeqrsv_gpu(magma_int_t M, magma_int_t N, magma_int_t NRHS, double2 *A, magma_int_t LDA, double2 *B, 
			       magma_int_t LDB, double2 *X,magma_int_t LDX, double2 *WORK, float2 *SWORK, 
			       magma_int_t *ITER, magma_int_t *INFO, float2 *tau, magma_int_t lwork, float2 *h_work,
			       float2 *d_work, double2 *tau_d, magma_int_t lwork_d, double2 *h_work_d,
			       double2 *d_work_d);
  

#ifdef __cplusplus
}
#endif

#endif /* _MAGMA_Z_H_ */
