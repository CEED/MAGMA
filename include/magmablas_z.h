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
double cpu_gpu_zdiff(             magma_int_t M, magma_int_t N, 
                  cuDoubleComplex * a, magma_int_t lda, 
                  cuDoubleComplex *da, magma_int_t ldda);

void zzero_32x32_block(           cuDoubleComplex *, magma_int_t);

void zzero_nbxnb_block(           magma_int_t, cuDoubleComplex *, magma_int_t);

void magmablas_zpermute_long(     cuDoubleComplex *, magma_int_t, 
                  magma_int_t *, magma_int_t, magma_int_t);

void magmablas_zpermute_long2(magma_int_t n, cuDoubleComplex *, magma_int_t, 
                  magma_int_t *, magma_int_t, magma_int_t);

void magmablas_zpermute_long3( cuDoubleComplex *dAT, magma_int_t lda, 
                               magma_int_t *ipiv, magma_int_t nb, magma_int_t ind );

  /*
   * Transpose functions
   */
void magmablas_zinplace_transpose(cuDoubleComplex *, magma_int_t, magma_int_t);

void magmablas_ztranspose(        cuDoubleComplex *, magma_int_t, 
                  cuDoubleComplex *, magma_int_t, 
                  magma_int_t, magma_int_t);

void magmablas_ztranspose2(       cuDoubleComplex *, magma_int_t, 
                  cuDoubleComplex *, magma_int_t, 
                  magma_int_t, magma_int_t);

void magmablas_ztranspose2s(cuDoubleComplex *odata, magma_int_t ldo,
                       cuDoubleComplex *idata, magma_int_t ldi,
                       magma_int_t m, magma_int_t n, cudaStream_t *stream );

void magmablas_zgetmatrix_transpose(  magma_int_t m, magma_int_t n,
                                      cuDoubleComplex *dat, magma_int_t ldda,
                                      cuDoubleComplex  *ha, magma_int_t lda,
                                      cuDoubleComplex  *dB, magma_int_t lddb, magma_int_t nb );
void magmablas_zsetmatrix_transpose(  magma_int_t m, magma_int_t n,
                                      cuDoubleComplex  *ha, magma_int_t lda, 
                                      cuDoubleComplex *dat, magma_int_t ldda,
                                      cuDoubleComplex  *dB, magma_int_t lddb, magma_int_t nb );

  /*
   * Multi-GPU functions
   */
void magmablas_zgetmatrix_transpose_mgpu(
                  magma_int_t num_gpus, cudaStream_t **stream0,
                  cuDoubleComplex **dat, magma_int_t ldda,
                  cuDoubleComplex   *ha, magma_int_t lda,
                  cuDoubleComplex  **dB, magma_int_t lddb,
                  magma_int_t m, magma_int_t n, magma_int_t nb);

void magmablas_zsetmatrix_transpose_mgpu(
                  magma_int_t num_gpus, cudaStream_t **stream0,
                  cuDoubleComplex  *ha,  magma_int_t lda, 
                  cuDoubleComplex **dat, magma_int_t ldda, magma_int_t starti,
                  cuDoubleComplex **dB,  magma_int_t lddb,
                  magma_int_t m, magma_int_t n, magma_int_t nb);

void magmablas_zgetmatrix_1D_bcyclic( magma_int_t m, magma_int_t n,
                                      cuDoubleComplex  *da[], magma_int_t ldda,
                                      cuDoubleComplex  *ha, magma_int_t lda,
                                      magma_int_t num_gpus, magma_int_t nb );

void magmablas_zsetmatrix_1D_bcyclic( magma_int_t m, magma_int_t n,
                                      cuDoubleComplex  *ha, magma_int_t lda,
                                      cuDoubleComplex  *da[], magma_int_t ldda,
                                      magma_int_t num_gpus, magma_int_t nb );


void magmablas_zhemm_1gpu_old(
    char side, char uplo, magma_int_t m, magma_int_t n,
    cuDoubleComplex alpha, cuDoubleComplex *dA[], magma_int_t ldda,  magma_int_t offset,
                           cuDoubleComplex *dB[], magma_int_t lddb,
    cuDoubleComplex beta,  cuDoubleComplex *dC[], magma_int_t lddc,
                           cuDoubleComplex *C,    magma_int_t ldc,
    magma_int_t ngpu, magma_int_t nb, cudaStream_t streams[][20], magma_int_t nstream );
void magmablas_zhemm_1gpu(
    char side, char uplo, magma_int_t m, magma_int_t n,
    cuDoubleComplex alpha, cuDoubleComplex *dA[], magma_int_t ldda,  magma_int_t offset,
                           cuDoubleComplex *dB[], magma_int_t lddb,
    cuDoubleComplex beta,  cuDoubleComplex *dC[], magma_int_t lddc,
                           cuDoubleComplex *C,    magma_int_t ldc,
    magma_int_t ngpu, magma_int_t nb, cudaStream_t streams[][20], magma_int_t nstream );
void magmablas_zhemm_mgpu(
    char side, char uplo, magma_int_t m, magma_int_t n,
    cuDoubleComplex alpha, cuDoubleComplex *dA[], magma_int_t ldda,  magma_int_t offset,
                           cuDoubleComplex *dB[], magma_int_t lddb,
    cuDoubleComplex beta,  cuDoubleComplex *dC[], magma_int_t lddc,
                           cuDoubleComplex *dwork[],    magma_int_t lddwork,
                           cuDoubleComplex *C,    magma_int_t ldc,
    magma_int_t ngpu, magma_int_t nb, cudaStream_t streams[][20], magma_int_t nstream );

void magmablas_zher2k_mgpu2(
    char uplo, char trans, magma_int_t n, magma_int_t k,
    cuDoubleComplex alpha, cuDoubleComplex *dA[], magma_int_t lda,
                           cuDoubleComplex *dB[], magma_int_t ldb,
    double beta,           cuDoubleComplex *dC[], magma_int_t ldc,  magma_int_t offset,
    magma_int_t ngpu, magma_int_t nb, cudaStream_t streams[][20], magma_int_t nstream );

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

void   magmablas_zlascl( char type, magma_int_t kl, magma_int_t ku,
             double cfrom, double cto,
             magma_int_t m, magma_int_t n,
             cuDoubleComplex *A, magma_int_t lda, magma_int_t *info );

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

void   magmablas_zsymmetrize(
             char uplo, int m, cuDoubleComplex *dA, int ldda );

void   magmablas_zsymmetrize_tiles(
             char uplo, int m, cuDoubleComplex *dA, int ldda,
             int ntile, int mstride, int nstride );

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
// Add the function, file, and line for error-reporting purposes.

#define magma_zsetvector(           n, hx_src, incx, dy_dst, incy ) \
        magma_zsetvector_internal(  n, hx_src, incx, dy_dst, incy, __func__, __FILE__, __LINE__ )
                           
#define magma_zgetvector(           n, dx_src, incx, hy_dst, incy ) \
        magma_zgetvector_internal(  n, dx_src, incx, hy_dst, incy, __func__, __FILE__, __LINE__ )

#define magma_zsetvector_async(           n, hx_src, incx, dy_dst, incy, stream ) \
        magma_zsetvector_async_internal(  n, hx_src, incx, dy_dst, incy, stream, __func__, __FILE__, __LINE__ )
                                 
#define magma_zgetvector_async(           n, dx_src, incx, hy_dst, incy, stream ) \
        magma_zgetvector_async_internal(  n, dx_src, incx, hy_dst, incy, stream, __func__, __FILE__, __LINE__ )

void magma_zsetvector_internal(
    magma_int_t n,
    cuDoubleComplex const *hx_src, magma_int_t incx,
    cuDoubleComplex       *dy_dst, magma_int_t incy,
    const char* func, const char* file, int line );

void magma_zgetvector_internal(
    magma_int_t n,
    cuDoubleComplex const *dx_src, magma_int_t incx,
    cuDoubleComplex       *hy_dst, magma_int_t incy,
    const char* func, const char* file, int line );

void magma_zsetvector_async_internal(
    magma_int_t n,
    cuDoubleComplex const *hx_src, magma_int_t incx,
    cuDoubleComplex       *dy_dst, magma_int_t incy,
    magma_stream_t stream,
    const char* func, const char* file, int line );

void magma_zgetvector_async_internal(
    magma_int_t n,
    cuDoubleComplex const *dx_src, magma_int_t incx,
    cuDoubleComplex       *hy_dst, magma_int_t incy,
    magma_stream_t stream,
    const char* func, const char* file, int line );


// ========================================
// copying sub-matrices (contiguous columns)
// set  copies host to device
// get  copies device to host
// copy copies device to device (with CUDA unified addressing, can be same or different devices)
// Add the function, file, and line for error-reporting purposes.

#define magma_zsetmatrix(           m, n, hA_src, lda, dB_dst, ldb ) \
        magma_zsetmatrix_internal(  m, n, hA_src, lda, dB_dst, ldb, __func__, __FILE__, __LINE__ )

#define magma_zgetmatrix(           m, n, dA_src, lda, hB_dst, ldb ) \
        magma_zgetmatrix_internal(  m, n, dA_src, lda, hB_dst, ldb, __func__, __FILE__, __LINE__ )

#define magma_zcopymatrix(          m, n, dA_src, lda, dB_dst, ldb ) \
        magma_zcopymatrix_internal( m, n, dA_src, lda, dB_dst, ldb, __func__, __FILE__, __LINE__ )

#define magma_zsetmatrix_async(           m, n, hA_src, lda, dB_dst, ldb, stream ) \
        magma_zsetmatrix_async_internal(  m, n, hA_src, lda, dB_dst, ldb, stream, __func__, __FILE__, __LINE__ )
                                 
#define magma_zgetmatrix_async(           m, n, dA_src, lda, hB_dst, ldb, stream ) \
        magma_zgetmatrix_async_internal(  m, n, dA_src, lda, hB_dst, ldb, stream, __func__, __FILE__, __LINE__ )
                                 
#define magma_zcopymatrix_async(          m, n, dA_src, lda, dB_dst, ldb, stream ) \
        magma_zcopymatrix_async_internal( m, n, dA_src, lda, dB_dst, ldb, stream, __func__, __FILE__, __LINE__ )

void magma_zsetmatrix_internal(
    magma_int_t m, magma_int_t n,
    cuDoubleComplex const *hA_src, magma_int_t lda,
    cuDoubleComplex       *dB_dst, magma_int_t ldb,
    const char* func, const char* file, int line );

void magma_zgetmatrix_internal(
    magma_int_t m, magma_int_t n,
    cuDoubleComplex const *dA_src, magma_int_t lda,
    cuDoubleComplex       *hB_dst, magma_int_t ldb,
    const char* func, const char* file, int line );

void magma_zcopymatrix_internal(
    magma_int_t m, magma_int_t n,
    cuDoubleComplex const *dA_src, magma_int_t lda,
    cuDoubleComplex       *dB_dst, magma_int_t ldb,
    const char* func, const char* file, int line );

void magma_zsetmatrix_async_internal(
    magma_int_t m, magma_int_t n,
    cuDoubleComplex const *hA_src, magma_int_t lda,
    cuDoubleComplex       *dB_dst, magma_int_t ldb,
    magma_stream_t stream,
    const char* func, const char* file, int line );

void magma_zgetmatrix_async_internal(
    magma_int_t m, magma_int_t n,
    cuDoubleComplex const *dA_src, magma_int_t lda,
    cuDoubleComplex       *hB_dst, magma_int_t ldb,
    magma_stream_t stream,
    const char* func, const char* file, int line );

void magma_zcopymatrix_async_internal(
    magma_int_t m, magma_int_t n,
    cuDoubleComplex const *dA_src, magma_int_t lda,
    cuDoubleComplex       *dB_dst, magma_int_t ldb,
    magma_stream_t stream,
    const char* func, const char* file, int line );


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
