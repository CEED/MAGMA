/*
 *   -- MAGMA (version 0.1) --
 *      Univ. of Tennessee, Knoxville
 *      Univ. of California, Berkeley
 *      Univ. of Colorado, Denver
 *      April 2012
 *
 * @author Mark Gates
 * @precisions normal z -> s d c
 */

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "magma.h"

#ifdef HAVE_CUBLAS

// For now, magma constants are the same as cublas v1 constants (character).
// This will change in the future.
#define cublas_side_const(  x )  (x)
#define cublas_uplo_const(  x )  (x)
#define cublas_trans_const( x )  (x)
#define cublas_diag_const(  x )  (x)

// ========================================
// copying vectors
void magma_zsetvector(
    magma_int_t n,
    cuDoubleComplex const* hx_src, magma_int_t inchx,
    cuDoubleComplex*       dx_dst, magma_int_t incdx )
{
    cublasStatus_t status;
    status = cublasSetVector(
        n, sizeof(cuDoubleComplex),
        hx_src, inchx,
        dx_dst, incdx );
    assert( status == CUBLAS_STATUS_SUCCESS );
}

// --------------------
void magma_zgetvector(
    magma_int_t n,
    cuDoubleComplex const* dx_src, magma_int_t incdx,
    cuDoubleComplex*       hx_dst, magma_int_t inchx )
{
    cublasStatus_t status;
    status = cublasGetVector(
        n, sizeof(cuDoubleComplex),
        dx_src, incdx,
        hx_dst, inchx );
    assert( status == CUBLAS_STATUS_SUCCESS );
}

// --------------------
void magma_zsetvector_async(
    magma_int_t n,
    cuDoubleComplex const* hx_src, magma_int_t inchx,
    cuDoubleComplex*       dx_dst, magma_int_t incdx,
    cudaStream_t stream )
{
    cublasStatus_t status;
    status = cublasSetVectorAsync(
        n, sizeof(cuDoubleComplex),
        hx_src, inchx,
        dx_dst, incdx, stream );
    assert( status == CUBLAS_STATUS_SUCCESS );
}

// --------------------
void magma_zgetvector_async(
    magma_int_t n,
    cuDoubleComplex const* dx_src, magma_int_t incdx,
    cuDoubleComplex*       hx_dst, magma_int_t inchx,
    cudaStream_t stream )
{
    cublasStatus_t status;
    status = cublasGetVectorAsync(
        n, sizeof(cuDoubleComplex),
        dx_src, incdx,
        hx_dst, inchx, stream );
    assert( status == CUBLAS_STATUS_SUCCESS );
}


// ========================================
// copying sub-matrices (contiguous columns)
void magma_zsetmatrix(
    magma_int_t m, magma_int_t n,
    cuDoubleComplex const* hA_src, magma_int_t ldha,
    cuDoubleComplex*       dA_dst, magma_int_t ldda )
{
    cublasStatus_t status;
    status = cublasSetMatrix(
        m, n, sizeof(cuDoubleComplex),
        hA_src, ldha,
        dA_dst, ldda );
    assert( status == CUBLAS_STATUS_SUCCESS );
}

// --------------------
void magma_zgetmatrix(
    magma_int_t m, magma_int_t n,
    cuDoubleComplex const* dA_src, magma_int_t ldda,
    cuDoubleComplex*       hA_dst, magma_int_t ldha )
{
    cublasStatus_t status;
    status = cublasGetMatrix(
        m, n, sizeof(cuDoubleComplex),
        dA_src, ldda,
        hA_dst, ldha );
    assert( status == CUBLAS_STATUS_SUCCESS );
}

// --------------------
void magma_zsetmatrix_async(
    magma_int_t m, magma_int_t n,
    cuDoubleComplex const* hA_src, magma_int_t ldha,
    cuDoubleComplex*       dA_dst, magma_int_t ldda,
    cudaStream_t stream )
{
    cublasStatus_t status;
    status = cublasSetMatrixAsync(
        m, n, sizeof(cuDoubleComplex),
        hA_src, ldha,
        dA_dst, ldda, stream );
    assert( status == CUBLAS_STATUS_SUCCESS );
}

// --------------------
void magma_zgetmatrix_async(
    magma_int_t m, magma_int_t n,
    cuDoubleComplex const* dA_src, magma_int_t ldda,
    cuDoubleComplex*       hA_dst, magma_int_t ldha,
    cudaStream_t stream )
{
    cublasStatus_t status;
    status = cublasGetMatrixAsync(
        m, n, sizeof(cuDoubleComplex),
        dA_src, ldda,
        hA_dst, ldha, stream );
    assert( status == CUBLAS_STATUS_SUCCESS );
}


// ========================================
// Level 1 BLAS

// --------------------
void magma_zswap(
    magma_int_t n,
    cuDoubleComplex *dx, magma_int_t incx,
    cuDoubleComplex *dy, magma_int_t incy )
{
    cublasZswap( n, dx, incx, dy, incy );
}

// --------------------
magma_int_t magma_izamax(
    magma_int_t n,
    cuDoubleComplex *dx, magma_int_t incx )
{
    return cublasIzamax( n, dx, incx );
}

// ========================================
// Level 2 BLAS

// --------------------
void magma_zgemv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    cuDoubleComplex alpha, cuDoubleComplex const* dA, magma_int_t lda,
                           cuDoubleComplex const* dx, magma_int_t incx,
    cuDoubleComplex beta,  cuDoubleComplex*       dy, magma_int_t incy )
{
    cublasZgemv(
        cublas_trans_const( transA ),
        m, n,
        alpha, dA, lda,
               dx, incx,
        beta,  dy, incy );
}

// --------------------
void magma_zhemv(
    magma_uplo_t uplo,
    magma_int_t n,
    cuDoubleComplex alpha, cuDoubleComplex const* dA, magma_int_t lda,
                           cuDoubleComplex const* dx, magma_int_t incx,
    cuDoubleComplex beta,  cuDoubleComplex*       dy, magma_int_t incy )
{
    cublasZhemv(
        cublas_uplo_const( uplo ),
        n,
        alpha, dA, lda,
               dx, incx,
        beta,  dy, incy );
}

// --------------------
void magma_ztrsv(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag, 
    magma_int_t n, 
    cuDoubleComplex const *dA, magma_int_t lda, 
    cuDoubleComplex       *dx, magma_int_t incx )
{
    cublasZtrsv(
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        cublas_diag_const( diag ),
        n,
        dA, lda,
        dx, incx );
}

// ========================================
// Level 3 BLAS

// --------------------
void magma_zgemm(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    cuDoubleComplex alpha, cuDoubleComplex const* dA, magma_int_t lda,
                           cuDoubleComplex const* dB, magma_int_t ldb,
    cuDoubleComplex beta,  cuDoubleComplex*       dC, magma_int_t ldc )
{
    cublasZgemm(
        cublas_trans_const( transA ),
        cublas_trans_const( transB ),
        m, n, k,
        alpha, dA, lda,
               dB, ldb,
        beta,  dC, ldc );
}

// --------------------
void magma_zhemm(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    cuDoubleComplex alpha, cuDoubleComplex const* dA, magma_int_t lda,
                           cuDoubleComplex const* dB, magma_int_t ldb,
    cuDoubleComplex beta,  cuDoubleComplex*       dC, magma_int_t ldc )
{
    cublasZhemm(
        cublas_side_const( side ),
        cublas_uplo_const( uplo ),
        m, n,
        alpha, dA, lda,
               dB, ldb,
        beta,  dC, ldc );
}

// --------------------
void magma_zherk(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    double alpha, cuDoubleComplex const* dA, magma_int_t lda,
    double beta,  cuDoubleComplex*       dC, magma_int_t ldc )
{
    cublasZherk(
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        n, k,
        alpha, dA, lda,
        beta,  dC, ldc );
}

// --------------------
void magma_zher2k(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    cuDoubleComplex alpha, cuDoubleComplex const *dA, magma_int_t lda,
                           cuDoubleComplex const *dB, magma_int_t ldb,
    double beta,           cuDoubleComplex       *dC, magma_int_t ldc )
{
    cublasZher2k(
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        n, k,
        alpha, dA, lda,
               dB, ldb,
        beta,  dC, ldc );
}

// --------------------
void magma_ztrmm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    cuDoubleComplex alpha, cuDoubleComplex const *dA, magma_int_t lda,
                           cuDoubleComplex       *dB, magma_int_t ldb )
{
    cublasZtrmm(
        cublas_side_const( side ),
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        cublas_diag_const( diag ),
        m, n,
        alpha, dA, lda,
               dB, ldb );
}

// --------------------
void magma_ztrsm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    cuDoubleComplex alpha, cuDoubleComplex const* dA, magma_int_t lda,
                           cuDoubleComplex*       dB, magma_int_t ldb )
{
    cublasZtrsm(
        cublas_side_const( side ),
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        cublas_diag_const( diag ),
        m, n,
        alpha, dA, lda,
               dB, ldb );
}

#endif // HAVE_CUBLAS
