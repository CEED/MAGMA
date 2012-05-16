/*
 *   -- MAGMA (version 1.1) --
 *      Univ. of Tennessee, Knoxville
 *      Univ. of California, Berkeley
 *      Univ. of Colorado, Denver
 *      November 2011
 *
 * @precisions normal z -> s d c
 */

#ifndef _MAGMABLAS_Z_H_
#define _MAGMABLAS_Z_H_

#define PRECISION_z

#ifdef __cplusplus
extern "C" {
#endif

  /*
   * Interface to clean
   */
double cpu_gpu_zdiff(             int M, int N, 
                  cuDoubleComplex * a, int lda, 
                  cuDoubleComplex *da, int ldda);
void zzero_32x32_block(           cuDoubleComplex *, magma_int_t);
void zzero_nbxnb_block(           magma_int_t, cuDoubleComplex *, magma_int_t);
void magmablas_zinplace_transpose(cuDoubleComplex *, magma_int_t, magma_int_t);
void magmablas_zpermute_long(     cuDoubleComplex *, magma_int_t, 
                  magma_int_t *, magma_int_t, magma_int_t);
void magmablas_zpermute_long2(    cuDoubleComplex *, magma_int_t, 
                  magma_int_t *, magma_int_t, magma_int_t);
void magmablas_zpermute_long3( cuDoubleComplex *dAT, int lda, 
                               int *ipiv, int nb, int ind );
void magmablas_ztranspose(        cuDoubleComplex *, magma_int_t, 
                  cuDoubleComplex *, magma_int_t, 
                  magma_int_t, magma_int_t);
void magmablas_ztranspose2(       cuDoubleComplex *, magma_int_t, 
                  cuDoubleComplex *, magma_int_t, 
                  magma_int_t, magma_int_t);
void magmablas_ztranspose2s(cuDoubleComplex *odata, int ldo,
                       cuDoubleComplex *idata, int ldi,
                       int m, int n, cudaStream_t *stream );

void magmablas_zgetmatrix_transpose(  int m, int n,
                                      cuDoubleComplex *dat, int ldda,
                                      cuDoubleComplex  *ha, int lda,
                                      cuDoubleComplex  *dB, int lddb, int nb );
void magmablas_zgetmatrix_transpose2( int m, int n,
                                      cuDoubleComplex **dat, int *ldda,
                                      cuDoubleComplex  *ha,  int  lda,
                                      cuDoubleComplex **dB,  int  lddb, int nb,
                                      int num_gpus, cudaStream_t stream[][2] );
void magmablas_zsetmatrix_transpose(  int m, int n,
                                      cuDoubleComplex  *ha, int lda, 
                                      cuDoubleComplex *dat, int ldda,
                                      cuDoubleComplex  *dB, int lddb, int nb );
void magmablas_zsetmatrix_transpose2( int m, int n,
                                      cuDoubleComplex  *ha,  int  lda, 
                                      cuDoubleComplex **dat, int *ldda,
                                      cuDoubleComplex **dB,  int  lddb, int nb,
                                      int num_gpus, cudaStream_t stream[][2] );
void magmablas_zgetmatrix_1D_bcyclic( int m, int n,
                                      cuDoubleComplex  *da[], int ldda,
                                      cuDoubleComplex  *ha, int lda,
                                      int num_gpus, int nb );
void magmablas_zsetmatrix_1D_bcyclic( int m, int n,
                                      cuDoubleComplex  *ha, int lda,
                                      cuDoubleComplex  *da[], int ldda,
                                      int num_gpus, int nb );

  /*
   * LAPACK auxiliary functions
   */
void   magmablas_zlacpy( char uplo, 
             magma_int_t m, magma_int_t n, 
             cuDoubleComplex *A, magma_int_t lda, 
             cuDoubleComplex *B, magma_int_t ldb);
double magmablas_zlange( char norm, 
             magma_int_t m, magma_int_t n, 
             cuDoubleComplex *A, magma_int_t lda, double *WORK);
double magmablas_zlanhe( char norm, char uplo, 
             magma_int_t n,
             cuDoubleComplex *A, magma_int_t lda, double *WORK);
double magmablas_zlansy( char norm, char uplo,
             magma_int_t n, 
             cuDoubleComplex *A, magma_int_t lda, double *WORK);
void   magmablas_zlascl( char type, int kl, int ku,
             double cfrom, double cto,
             int m, int n,
             cuDoubleComplex *A, int lda, int *info );
void   magmablas_zlaset( char uplo, magma_int_t m, magma_int_t n,
             cuDoubleComplex *A, magma_int_t lda);
void   magmablas_zlaset_identity(
             magma_int_t m, magma_int_t n,
             cuDoubleComplex *A, magma_int_t lda);
void   magmablas_zlaswp( magma_int_t N, 
             cuDoubleComplex *dAT, magma_int_t lda, 
             magma_int_t i1,  magma_int_t i2, 
             magma_int_t *ipiv, magma_int_t inci );
void   magmablas_zlaswpx(magma_int_t N, 
             cuDoubleComplex *dAT, magma_int_t ldx, magma_int_t ldy, 
             magma_int_t i1, magma_int_t i2,
             magma_int_t *ipiv, magma_int_t inci );

  /*
   * Level 1 BLAS
   */
void   magmablas_zswap(   magma_int_t N, 
              cuDoubleComplex *dA1, magma_int_t lda1, 
              cuDoubleComplex *dA2, magma_int_t lda2 );
void   magmablas_zswapblk(char storev, 
              magma_int_t N, 
              cuDoubleComplex *dA1, magma_int_t lda1, 
              cuDoubleComplex *dA2, magma_int_t lda2,
              magma_int_t i1, magma_int_t i2, 
              magma_int_t *ipiv, magma_int_t inci, 
              magma_int_t offset);
void magmablas_zswapdblk(magma_int_t n, magma_int_t nb,
             cuDoubleComplex *dA1, magma_int_t ldda1, magma_int_t inca1,
             cuDoubleComplex *dA2, magma_int_t ldda2, magma_int_t inca2 );

  /*
   * Level 2 BLAS
   */
void magmablas_zgemv(char t, magma_int_t M, magma_int_t N, 
             cuDoubleComplex alpha,
             cuDoubleComplex *A, magma_int_t lda, 
             cuDoubleComplex * X, magma_int_t incX, 
             cuDoubleComplex beta, 
             cuDoubleComplex *Y, magma_int_t incY);
#if defined(PRECISION_z) || defined(PRECISION_c)
magma_int_t magmablas_zhemv(char u, magma_int_t N, 
                            cuDoubleComplex alpha, 
                            cuDoubleComplex *A, magma_int_t lda, 
                            cuDoubleComplex *X, magma_int_t incX, 
                            cuDoubleComplex beta, 
                            cuDoubleComplex *Y, magma_int_t incY);
#endif
magma_int_t magmablas_zsymv(char u, magma_int_t N, 
                            cuDoubleComplex alpha, 
                            cuDoubleComplex *A, magma_int_t lda, 
                            cuDoubleComplex *X, magma_int_t incX, 
                            cuDoubleComplex beta, 
                            cuDoubleComplex *Y, magma_int_t incY);

  /*
   * Level 3 BLAS
   */
void magmablas_zgemm(char tA, char tB,
             magma_int_t m, magma_int_t n, magma_int_t k, 
             cuDoubleComplex alpha,
             const cuDoubleComplex *A, magma_int_t lda, 
             const cuDoubleComplex *B, magma_int_t ldb, 
             cuDoubleComplex beta,
             cuDoubleComplex *C, magma_int_t ldc);
void magmablas_zgemm_fermi80(char tA, char tB, 
                 magma_int_t m, magma_int_t n, magma_int_t k,
                 cuDoubleComplex alpha, 
                 const cuDoubleComplex *A, magma_int_t lda, 
                 const cuDoubleComplex *B, magma_int_t ldb,
                 cuDoubleComplex beta, 
                 cuDoubleComplex *C, magma_int_t ldc);
void magmablas_zgemm_fermi64(char tA, char tB, 
                 magma_int_t m, magma_int_t n, magma_int_t k,
                 cuDoubleComplex alpha, 
                 const cuDoubleComplex *A, magma_int_t lda, 
                 const cuDoubleComplex *B, magma_int_t ldb, 
                 cuDoubleComplex beta, 
                 cuDoubleComplex *C, magma_int_t ldc);
void magmablas_zhemm(char s, char u,          
             magma_int_t m, magma_int_t n,
             cuDoubleComplex alpha, 
             const cuDoubleComplex *A, magma_int_t lda,
             const cuDoubleComplex *B, magma_int_t ldb,
             cuDoubleComplex beta, 
             cuDoubleComplex *C, magma_int_t ldc);
void magmablas_zsymm(char s, char u,
             magma_int_t m, magma_int_t n,
             cuDoubleComplex alpha, 
             const cuDoubleComplex *A, magma_int_t lda, 
             const cuDoubleComplex *B, magma_int_t ldb,
             cuDoubleComplex beta,
             cuDoubleComplex *C, magma_int_t ldc);
void magmablas_zsyrk(char u, char t,
             magma_int_t n, magma_int_t k, 
             cuDoubleComplex alpha, 
             const cuDoubleComplex *A, magma_int_t lda,
             cuDoubleComplex beta,
             cuDoubleComplex *C, magma_int_t ldc);
void magmablas_zherk(char u, char t,
             magma_int_t n, magma_int_t k, 
             double  alpha, 
             const cuDoubleComplex *A, magma_int_t lda,
             double  beta, 
             cuDoubleComplex *C, magma_int_t ldc);
void magmablas_zsyr2k(char u, char t,
              magma_int_t n, magma_int_t k,
              cuDoubleComplex alpha, 
              const cuDoubleComplex *A, magma_int_t lda,
              const cuDoubleComplex *B, magma_int_t ldb, 
              cuDoubleComplex beta, 
              cuDoubleComplex *C, magma_int_t ldc);
void magmablas_zher2k(char u, char t,
              magma_int_t n, magma_int_t k, 
              cuDoubleComplex alpha, 
              const cuDoubleComplex *A, magma_int_t lda, 
              const cuDoubleComplex *B, magma_int_t ldb,
              double  beta,
              cuDoubleComplex *C, magma_int_t ldc);
void magmablas_ztrmm(char s, char u, char t,  char d, 
             magma_int_t m, magma_int_t n,
             cuDoubleComplex alpha,
             const cuDoubleComplex *A, magma_int_t lda,
             cuDoubleComplex *B, magma_int_t ldb);
void magmablas_ztrsm(char s, char u, char t, char d,
             magma_int_t m, magma_int_t n,
             cuDoubleComplex alpha,
             /*const*/ cuDoubleComplex *A, magma_int_t lda,
             cuDoubleComplex *B, magma_int_t ldb);


  /*
   * Wrappers for platform independence.
   * These wrap CUBLAS or AMD OpenCL BLAS functions.
   */

// ========================================
// copying vectors
// set copies host to device
// get copies device to host

void magma_zsetvector(
    magma_int_t n,
    cuDoubleComplex const *hx_src, magma_int_t incx,
    cuDoubleComplex       *dy_dst, magma_int_t incy );

void magma_zgetvector(
    magma_int_t n,
    cuDoubleComplex const *dx_src, magma_int_t incx,
    cuDoubleComplex       *hy_dst, magma_int_t incy );

void magma_zsetvector_async(
    magma_int_t n,
    cuDoubleComplex const *hx_src, magma_int_t incx,
    cuDoubleComplex       *dy_dst, magma_int_t incy,
    magma_stream_t stream );

void magma_zgetvector_async(
    magma_int_t n,
    cuDoubleComplex const *dx_src, magma_int_t incx,
    cuDoubleComplex       *hy_dst, magma_int_t incy,
    magma_stream_t stream );


// ========================================
// copying sub-matrices (contiguous columns)
// set copies host to device
// get copies device to host
// cpy copies device to device (with CUDA unified addressing, can be same or different devices)

void magma_zsetmatrix(
    magma_int_t m, magma_int_t n,
    cuDoubleComplex const *hA_src, magma_int_t lda,
    cuDoubleComplex       *dB_dst, magma_int_t ldb );

void magma_zgetmatrix(
    magma_int_t m, magma_int_t n,
    cuDoubleComplex const *dA_src, magma_int_t lda,
    cuDoubleComplex       *hB_dst, magma_int_t ldb );

void magma_zsetmatrix_async(
    magma_int_t m, magma_int_t n,
    cuDoubleComplex const *hA_src, magma_int_t lda,
    cuDoubleComplex       *dB_dst, magma_int_t ldb,
    magma_stream_t stream );

void magma_zgetmatrix_async(
    magma_int_t m, magma_int_t n,
    cuDoubleComplex const *dA_src, magma_int_t lda,
    cuDoubleComplex       *hB_dst, magma_int_t ldb,
    magma_stream_t stream );

void magma_zcopymatrix(
    magma_int_t m, magma_int_t n,
    cuDoubleComplex const *dA_src, magma_int_t lda,
    cuDoubleComplex       *dB_dst, magma_int_t ldb );

void magma_zcopymatrix_async(
    magma_int_t m, magma_int_t n,
    cuDoubleComplex const *dA_src, magma_int_t lda,
    cuDoubleComplex       *dB_dst, magma_int_t ldb,
    magma_stream_t stream );


// ========================================
// Level 1 BLAS

void magma_zswap(
    magma_int_t n,
    cuDoubleComplex *dx, magma_int_t incx,
    cuDoubleComplex *dy, magma_int_t incy );

magma_int_t magma_izamax(
    magma_int_t n,
    cuDoubleComplex *dx, magma_int_t incx );

// ========================================
// Level 2 BLAS

void magma_zgemv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    cuDoubleComplex alpha, cuDoubleComplex const *dA, magma_int_t lda,
                           cuDoubleComplex const *dx, magma_int_t incx,
    cuDoubleComplex beta,  cuDoubleComplex       *dy, magma_int_t incy );

void magma_zhemv(
    magma_uplo_t uplo,
    magma_int_t n,
    cuDoubleComplex alpha, cuDoubleComplex const *dA, magma_int_t lda,
                           cuDoubleComplex const *dx, magma_int_t incx,
    cuDoubleComplex beta,  cuDoubleComplex       *dy, magma_int_t incy );

void magma_ztrsv(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag, 
    magma_int_t n, 
    cuDoubleComplex const *dA, magma_int_t lda, 
    cuDoubleComplex       *dx, magma_int_t incx );

// ========================================
// Level 3 BLAS

void magma_zgemm(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    cuDoubleComplex alpha, cuDoubleComplex const *dA, magma_int_t lda,
                           cuDoubleComplex const *dB, magma_int_t ldb,
    cuDoubleComplex beta,  cuDoubleComplex       *dC, magma_int_t ldc );

void magma_zhemm(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    cuDoubleComplex alpha, cuDoubleComplex const *dA, magma_int_t lda,
                           cuDoubleComplex const *dB, magma_int_t ldb,
    cuDoubleComplex beta,  cuDoubleComplex       *dC, magma_int_t ldc );

void magma_zherk(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    double alpha, cuDoubleComplex const *dA, magma_int_t lda,
    double beta,  cuDoubleComplex       *dC, magma_int_t ldc );

void magma_zher2k(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    cuDoubleComplex alpha, cuDoubleComplex const *dA, magma_int_t lda,
                           cuDoubleComplex const *dB, magma_int_t ldb,
    double beta,           cuDoubleComplex       *dC, magma_int_t ldc );

void magma_ztrmm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    cuDoubleComplex alpha, cuDoubleComplex const *dA, magma_int_t lda,
                           cuDoubleComplex       *dB, magma_int_t ldb );

void magma_ztrsm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    cuDoubleComplex alpha, cuDoubleComplex const *dA, magma_int_t lda,
                           cuDoubleComplex       *dB, magma_int_t ldb );

#ifdef __cplusplus
}
#endif

#undef PRECISION_z
#endif
