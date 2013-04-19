/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011
 
       @author Mark Gates
       @precisions normal z -> s d c
*/

#include <stdlib.h>
#include <stdio.h>

#include "magma.h"
#include "error.h"

#ifdef HAVE_CUBLAS

// ========================================
// copying vectors
extern "C"
void magma_zsetvector_internal(
    magma_int_t n,
    magmaDoubleComplex const* hx_src, magma_int_t incx,
    magmaDoubleComplex*       dy_dst, magma_int_t incy,
    const char* func, const char* file, int line )
{
    cublasStatus_t status;
    status = cublasSetVector(
        n, sizeof(magmaDoubleComplex),
        hx_src, incx,
        dy_dst, incy );
    check_xerror( status, func, file, line );
}

// --------------------
extern "C"
void magma_zgetvector_internal(
    magma_int_t n,
    magmaDoubleComplex const* dx_src, magma_int_t incx,
    magmaDoubleComplex*       hy_dst, magma_int_t incy,
    const char* func, const char* file, int line )
{
    cublasStatus_t status;
    status = cublasGetVector(
        n, sizeof(magmaDoubleComplex),
        dx_src, incx,
        hy_dst, incy );
    check_xerror( status, func, file, line );
}

// --------------------
extern "C"
void magma_zsetvector_async_internal(
    magma_int_t n,
    magmaDoubleComplex const* hx_src, magma_int_t incx,
    magmaDoubleComplex*       dy_dst, magma_int_t incy,
    cudaStream_t stream,
    const char* func, const char* file, int line )
{
    cublasStatus_t status;
    status = cublasSetVectorAsync(
        n, sizeof(magmaDoubleComplex),
        hx_src, incx,
        dy_dst, incy, stream );
    check_xerror( status, func, file, line );
}

// --------------------
extern "C"
void magma_zgetvector_async_internal(
    magma_int_t n,
    magmaDoubleComplex const* dx_src, magma_int_t incx,
    magmaDoubleComplex*       hy_dst, magma_int_t incy,
    cudaStream_t stream,
    const char* func, const char* file, int line )
{
    cublasStatus_t status;
    status = cublasGetVectorAsync(
        n, sizeof(magmaDoubleComplex),
        dx_src, incx,
        hy_dst, incy, stream );
    check_xerror( status, func, file, line );
}


// ========================================
// copying sub-matrices (contiguous columns)
extern "C"
void magma_zsetmatrix_internal(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex const* hA_src, magma_int_t lda,
    magmaDoubleComplex*       dB_dst, magma_int_t ldb,
    const char* func, const char* file, int line )
{
    cublasStatus_t status;
    status = cublasSetMatrix(
        m, n, sizeof(magmaDoubleComplex),
        hA_src, lda,
        dB_dst, ldb );
    check_xerror( status, func, file, line );
}

// --------------------
extern "C"
void magma_zgetmatrix_internal(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex const* dA_src, magma_int_t lda,
    magmaDoubleComplex*       hB_dst, magma_int_t ldb,
    const char* func, const char* file, int line )
{
    cublasStatus_t status;
    status = cublasGetMatrix(
        m, n, sizeof(magmaDoubleComplex),
        dA_src, lda,
        hB_dst, ldb );
    check_xerror( status, func, file, line );
}

// --------------------
extern "C"
void magma_zsetmatrix_async_internal(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex const* hA_src, magma_int_t lda,
    magmaDoubleComplex*       dB_dst, magma_int_t ldb,
    cudaStream_t stream,
    const char* func, const char* file, int line )
{
    cublasStatus_t status;
    status = cublasSetMatrixAsync(
        m, n, sizeof(magmaDoubleComplex),
        hA_src, lda,
        dB_dst, ldb, stream );
    check_xerror( status, func, file, line );
}

// --------------------
extern "C"
void magma_zgetmatrix_async_internal(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex const* dA_src, magma_int_t lda,
    magmaDoubleComplex*       hB_dst, magma_int_t ldb,
    cudaStream_t stream,
    const char* func, const char* file, int line )
{
    cublasStatus_t status;
    status = cublasGetMatrixAsync(
        m, n, sizeof(magmaDoubleComplex),
        dA_src, lda,
        hB_dst, ldb, stream );
    check_xerror( status, func, file, line );
}

// --------------------
extern "C"
void magma_zcopymatrix_internal(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex const* dA_src, magma_int_t lda,
    magmaDoubleComplex*       dB_dst, magma_int_t ldb,
    const char* func, const char* file, int line )
{
    cudaError_t status;
    status = cudaMemcpy2D(
        dB_dst, ldb*sizeof(magmaDoubleComplex),
        dA_src, lda*sizeof(magmaDoubleComplex),
        m*sizeof(magmaDoubleComplex), n, cudaMemcpyDeviceToDevice );
    check_xerror( status, func, file, line );
}

// --------------------
extern "C"
void magma_zcopymatrix_async_internal(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex const* dA_src, magma_int_t lda,
    magmaDoubleComplex*       dB_dst, magma_int_t ldb,
    cudaStream_t stream,
    const char* func, const char* file, int line )
{
    cudaError_t status;
    status = cudaMemcpy2DAsync(
        dB_dst, ldb*sizeof(magmaDoubleComplex),
        dA_src, lda*sizeof(magmaDoubleComplex),
        m*sizeof(magmaDoubleComplex), n, cudaMemcpyDeviceToDevice, stream );
    check_xerror( status, func, file, line );
}

#endif // HAVE_CUBLAS
