/*
 *   -- MAGMA (version 1.0) --
 *      Univ. of Tennessee, Knoxville
 *      Univ. of California, Berkeley
 *      Univ. of Colorado, Denver
 *      November 2010
 *
 * @precisions normal z -> s d c
 */

#ifndef _MAGMABLAS_Z_H_
#define _MAGMABLAS_Z_H_

#include "cublas.h"
#include "cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

double cpu_gpu_zdiff(int M, int N, double2 * a, int lda, double2 *da, int ldda);


  /* Maybe these functions should be in magma_z because they are not blas */
double magma_zlange(char norm, magma_int_t m, magma_int_t n, double2 *A, magma_int_t lda, double2 *work);
double magma_zlansy(char norm, char uplo, magma_int_t n, double2 *A, magma_int_t lda, double2 *work);
void   magma_zlacpy(magma_int_t, magma_int_t, double2 *, magma_int_t, double2 *, magma_int_t);

  /* to make prescision generation easier, we should add a prefix like magma_ to these functions */
void   zzero_32x32_block(double2 *, magma_int_t);
void   zzero_nbxnb_block(magma_int_t, double2 *, magma_int_t);

void magmablas_zlaset(magma_int_t m, magma_int_t n, double2 *A, magma_int_t lda);
void magmablas_zlacpy(magma_int_t m, magma_int_t n, double2 *SA, magma_int_t ldsa ,
		      double2 *A, magma_int_t lda);
double magmablas_zlanhe( char norm, char uplo, magma_int_t n, double2 *A, 
			 magma_int_t lda, double2 *WORK);
double magmablas_zlansy( char norm, char uplo, magma_int_t n, double2 *A, 
			 magma_int_t lda, double2 *WORK);

void magmablas_zinplace_transpose(double2 *, magma_int_t, magma_int_t);
void magmablas_zpermute_long(double2 *, magma_int_t, magma_int_t *, magma_int_t, magma_int_t);
void magmablas_zpermute_long2(double2 *, magma_int_t, magma_int_t *, magma_int_t, magma_int_t);
void magmablas_ztranspose(double2 *, magma_int_t, double2 *, magma_int_t, magma_int_t, magma_int_t);
void magmablas_ztranspose2(double2 *, magma_int_t, double2 *, magma_int_t, magma_int_t, magma_int_t);

void magmablas_zlaswp( magma_int_t N, cuDoubleComplex *dAT, magma_int_t lda, 
                       magma_int_t i1, magma_int_t i2, magma_int_t *ipiv, int inci );
void magmablas_zlaswpx( magma_int_t N, cuDoubleComplex *dAT, magma_int_t ldx, magma_int_t ldy,
                        magma_int_t i1, magma_int_t i2, magma_int_t *ipiv, int inci );
void magmablas_zswap( int n, cuDoubleComplex *dA1, int lda1, cuDoubleComplex *dA2, int lda2);

void magmablas_zswapblk( int n, cuDoubleComplex *dA1T, int ldx1, int ldy1, 
                         cuDoubleComplex *dA2T, int ldx2, int ldy2,
                         int i1, int i2, int *ipiv, int inci, int offset );

void magmablas_zgemm(char transA, char transB, magma_int_t m, magma_int_t n, magma_int_t k, 
		     double2 alpha, const double2 *A, magma_int_t lda, 
		     const double2 *B, magma_int_t ldb, 
		     double2 beta, double2 *C, magma_int_t ldc);
void magmablas_zgemv_tesla(magma_int_t M, magma_int_t N, double2 *A,
			   magma_int_t lda, double2 *X, double2 *);
void magmablas_zherk(char, char, magma_int_t, magma_int_t, double, double2 *, magma_int_t, double, double2 *, magma_int_t);
void magmablas_zsyrk(char, char, magma_int_t, magma_int_t, double2, double2 *, magma_int_t, double2, double2 *, magma_int_t);
void magmablas_zhemv(char, magma_int_t, double2, double2 *, magma_int_t, double2 *, magma_int_t, double2, double2 *, magma_int_t);
void magmablas_zsymv(char, magma_int_t, double2, double2 *, magma_int_t, double2 *, magma_int_t, double2, double2 *, magma_int_t);
void magmablas_zher2k(char, char, magma_int_t, magma_int_t, double2, const double2 *, magma_int_t, const double2 *, magma_int_t, double, double2 *, magma_int_t);
void magmablas_zsyr2k(char, char, magma_int_t, magma_int_t, double2, const double2 *, magma_int_t, const double2 *, magma_int_t, double2, double2 *, magma_int_t);
void magmablas_ztrmm(char, char, char, char, magma_int_t, magma_int_t, double2, double2 *, magma_int_t, double2 *, magma_int_t);
void magmablas_ztrsm(char, char, char, char, magma_int_t, magma_int_t, double2, double2 *, magma_int_t, double2 *, magma_int_t);


  /* Should not be in this file */
void magmablas_zgemm_kernel_a_0(double2 *, const double2 *, const double2 *, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, double2, double2);
void magmablas_zgemm_kernel_ab_0(double2 *, const double2 *, const double2 *, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, double2, double2);
void magmablas_zgemm_kernel_N_N_64_16_16_16_4(double2 *, const double2 *, const double2 *, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, double2, double2);
void magmablas_zgemm_kernel_N_N_64_16_16_16_4_special(double2 *, const double2 *, const double2 *, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, double2, double2);
void magmablas_zgemm_kernel_N_T_64_16_4_16_4(double2 *, const double2 *, const double2 *, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, double2, double2);
void magmablas_zgemm_kernel_T_N_32_32_8_8_8(double2 *, const double2 *, const double2 *, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, double2, double2);
void magmablas_zgemm_kernel_T_T_64_16_16_16_4(double2 *, const double2 *, const double2 *, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, double2, double2);
void magmablas_zgemm_kernel_T_T_64_16_16_16_4_v2(double2 *, const double2 *, const double2 *, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, double2, double2);


  /* Maybe all theses routines don't need to be in this file either */
void magmablas_zgemv_MLU( magma_int_t, magma_int_t, double2 *, 
                          magma_int_t, double2 *, double2 *);
void magmablas_zgemv32_tesla(char, magma_int_t, magma_int_t, double2, double2 *, 
			     magma_int_t, double2 *, double2 *);
void magmablas_zgemvt1_tesla(magma_int_t,magma_int_t,double2,double2 *,
			     magma_int_t,double2 *,double2 *);
void magmablas_zgemvt2_tesla(magma_int_t,magma_int_t,double2,double2 *,
			     magma_int_t,double2 *,double2 *);
void  magmablas_zgemvt_tesla(magma_int_t,magma_int_t,double2,double2 *,
			     magma_int_t,double2 *,double2 *);

void magmablas_zsymv6(char, magma_int_t, double2, double2 *, magma_int_t, double2 *, magma_int_t, double2, double2 *, magma_int_t, double2 *, magma_int_t);

#ifdef __cplusplus
}
#endif

#endif
