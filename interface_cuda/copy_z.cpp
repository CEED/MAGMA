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

#endif // HAVE_CUBLAS
