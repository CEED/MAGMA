/*
 *   -- MAGMA (version 1.0) --
 *      Univ. of Tennessee, Knoxville
 *      Univ. of California, Berkeley
 *      Univ. of Colorado, Denver
 *      November 2010
 *
 * @precisions mixed zc -> ds
 */

#ifndef _MAGMABLAS_ZC_H_
#define _MAGMABLAS_ZC_H_

#include "cublas.h"
#include "cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

  /* Mixed precision */
void magmablas_zcaxpycp(float2 *, double2 *, magma_int_t, magma_int_t, magma_int_t, double2 *, double2 *);
void magmablas_zclaswp(magma_int_t, double2 *, magma_int_t, float2 *, magma_int_t, magma_int_t *);
void magmablas_zlag2c(magma_int_t M, magma_int_t N, const double2 *A, magma_int_t lda,  float2 *SA, magma_int_t ldsa, magma_int_t *info);
void magmablas_clag2z(magma_int_t M, magma_int_t N, const float2 *SA, magma_int_t ldsa, double2 *A, magma_int_t lda,  magma_int_t *info);
void magmablas_zlat2c(char, magma_int_t, double2 *, magma_int_t, float2 *, magma_int_t, magma_int_t *);

#ifdef __cplusplus
}
#endif

#endif
