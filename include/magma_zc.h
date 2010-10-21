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
magma_int_t magma_czgetrs_gpu(magma_int_t n, magma_int_t *nrhs, float2 *a, magma_int_t lda, 
			      magma_int_t *ipiv, float2 *x, double2 *b, magma_int_t ldb, magma_int_t *info);


#ifdef __cplusplus
}
#endif

#endif /* _MAGMA_Z_H_ */
