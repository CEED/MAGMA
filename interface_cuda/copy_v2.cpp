/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
 
       @author Mark Gates
*/
#include <cuda_runtime.h>

#include "magma_internal.h"
#include "error.h"

#ifdef HAVE_CUBLAS

// generic, type-independent routines to copy data.
// type-safe versions which avoid the user needing sizeof(...) are in headers

// ========================================
// copying vectors
extern "C" void
magma_setvector_q_internal(
    magma_int_t n, magma_int_t elemSize,
    void const* hx_src, magma_int_t incx,
    magma_ptr   dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    assert( queue != NULL );
    cublasStatus_t status;
    status = cublasSetVectorAsync(
        int(n), int(elemSize),
        hx_src, int(incx),
        dy_dst, int(incy), queue->cuda_stream() );
    cudaStreamSynchronize( queue->cuda_stream() );
    check_xerror( status, func, file, line );
}

// --------------------
// for backwards compatability, accepts NULL queue to mean NULL stream.
extern "C" void
magma_setvector_async_internal(
    magma_int_t n, magma_int_t elemSize,
    void const* hx_src, magma_int_t incx,
    magma_ptr   dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    cudaStream_t stream = NULL;
    if ( queue != NULL ) {
        stream = queue->cuda_stream();
    }
    else {
        fprintf( stderr, "Warning: %s got NULL queue\n", __func__ );
    }
    cublasStatus_t status;
    status = cublasSetVectorAsync(
        int(n), int(elemSize),
        hx_src, int(incx),
        dy_dst, int(incy), stream );
    check_xerror( status, func, file, line );
}

// --------------------
extern "C" void
magma_getvector_q_internal(
    magma_int_t n, magma_int_t elemSize,
    magma_const_ptr dx_src, magma_int_t incx,
    void*           hy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    cublasStatus_t status;
    status = cublasGetVectorAsync(
        int(n), int(elemSize),
        dx_src, int(incx),
        hy_dst, int(incy), queue->cuda_stream() );
    cudaStreamSynchronize( queue->cuda_stream() );
    check_xerror( status, func, file, line );
}

// --------------------
// for backwards compatability, accepts NULL queue to mean NULL stream.
extern "C" void
magma_getvector_async_internal(
    magma_int_t n, magma_int_t elemSize,
    magma_const_ptr dx_src, magma_int_t incx,
    void*           hy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    cudaStream_t stream = NULL;
    if ( queue != NULL ) {
        stream = queue->cuda_stream();
    }
    else {
        fprintf( stderr, "Warning: %s got NULL queue\n", __func__ );
    }
    cublasStatus_t status;
    status = cublasGetVectorAsync(
        int(n), int(elemSize),
        dx_src, int(incx),
        hy_dst, int(incy), stream );
    check_xerror( status, func, file, line );
}

// --------------------
// TODO compare performance with cublasZcopy BLAS function.
// But this implementation can handle any element size, not just [sdcz] precisions.
extern "C" void
magma_copyvector_q_internal(
    magma_int_t n, magma_int_t elemSize,
    magma_const_ptr dx_src, magma_int_t incx,
    magma_ptr       dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    assert( queue != NULL );
    if ( incx == 1 && incy == 1 ) {
        cudaError_t status;
        status = cudaMemcpyAsync(
            dy_dst,
            dx_src,
            int(n*elemSize), cudaMemcpyDeviceToDevice, queue->cuda_stream() );
        cudaStreamSynchronize( queue->cuda_stream() );
        check_xerror( status, func, file, line );
    }
    else {
        magma_copymatrix_q_internal(
            1, n, elemSize, dx_src, incx, dy_dst, incy, queue, func, file, line );
    }
}

// --------------------
// for backwards compatability, accepts NULL queue to mean NULL stream.
extern "C" void
magma_copyvector_async_internal(
    magma_int_t n, magma_int_t elemSize,
    magma_const_ptr dx_src, magma_int_t incx,
    magma_ptr       dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    cudaStream_t stream = NULL;
    if ( queue != NULL ) {
        stream = queue->cuda_stream();
    }
    else {
        fprintf( stderr, "Warning: %s got NULL queue\n", __func__ );
    }
    if ( incx == 1 && incy == 1 ) {
        cudaError_t status;
        status = cudaMemcpyAsync(
            dy_dst,
            dx_src,
            int(n*elemSize), cudaMemcpyDeviceToDevice, stream );
        check_xerror( status, func, file, line );
    }
    else {
        magma_copymatrix_async_internal(
            1, n, elemSize, dx_src, incx, dy_dst, incy, queue, func, file, line );
    }
}


// ========================================
// copying sub-matrices (contiguous columns)
extern "C" void
magma_setmatrix_q_internal(
    magma_int_t m, magma_int_t n, magma_int_t elemSize,
    void const* hA_src, magma_int_t ldha,
    magma_ptr   dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    assert( queue != NULL );
    cublasStatus_t status;
    status = cublasSetMatrixAsync(
        int(m), int(n), int(elemSize),
        hA_src, int(ldha),
        dB_dst, int(lddb), queue->cuda_stream() );
    cudaStreamSynchronize( queue->cuda_stream() );
    check_xerror( status, func, file, line );
}

// --------------------
// for backwards compatability, accepts NULL queue to mean NULL stream.
extern "C" void
magma_setmatrix_async_internal(
    magma_int_t m, magma_int_t n, magma_int_t elemSize,
    void const* hA_src, magma_int_t ldha,
    magma_ptr   dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    cudaStream_t stream = NULL;
    if ( queue != NULL ) {
        stream = queue->cuda_stream();
    }
    else {
        fprintf( stderr, "Warning: %s got NULL queue\n", __func__ );
    }
    cublasStatus_t status;
    status = cublasSetMatrixAsync(
        int(m), int(n), int(elemSize),
        hA_src, int(ldha),
        dB_dst, int(lddb), stream );
    check_xerror( status, func, file, line );
}

// --------------------
extern "C" void
magma_getmatrix_q_internal(
    magma_int_t m, magma_int_t n, magma_int_t elemSize,
    magma_const_ptr dA_src, magma_int_t ldda,
    void*           hB_dst, magma_int_t ldhb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    assert( queue != NULL );
    cublasStatus_t status;
    status = cublasGetMatrixAsync(
        int(m), int(n), int(elemSize),
        dA_src, int(ldda),
        hB_dst, int(ldhb), queue->cuda_stream() );
    cudaStreamSynchronize( queue->cuda_stream() );
    check_xerror( status, func, file, line );
}

// --------------------
extern "C" void
magma_getmatrix_async_internal(
    magma_int_t m, magma_int_t n, magma_int_t elemSize,
    magma_const_ptr dA_src, magma_int_t ldda,
    void*           hB_dst, magma_int_t ldhb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    cudaStream_t stream = NULL;
    if ( queue != NULL ) {
        stream = queue->cuda_stream();
    }
    else {
        fprintf( stderr, "Warning: %s got NULL queue\n", __func__ );
    }
    cublasStatus_t status;
    status = cublasGetMatrixAsync(
        int(m), int(n), int(elemSize),
        dA_src, int(ldda),
        hB_dst, int(ldhb), stream );
    check_xerror( status, func, file, line );
}

// --------------------
extern "C" void
magma_copymatrix_q_internal(
    magma_int_t m, magma_int_t n, magma_int_t elemSize,
    magma_const_ptr dA_src, magma_int_t ldda,
    magma_ptr       dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    assert( queue != NULL );
    cudaError_t status;
    status = cudaMemcpy2DAsync(
        dB_dst, int(lddb*elemSize),
        dA_src, int(ldda*elemSize),
        int(m*elemSize), int(n), cudaMemcpyDeviceToDevice, queue->cuda_stream() );
    cudaStreamSynchronize( queue->cuda_stream() );
    check_xerror( status, func, file, line );
}

// --------------------
// for backwards compatability, accepts NULL queue to mean NULL stream.
extern "C" void
magma_copymatrix_async_internal(
    magma_int_t m, magma_int_t n, magma_int_t elemSize,
    magma_const_ptr dA_src, magma_int_t ldda,
    magma_ptr       dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    cudaStream_t stream = NULL;
    if ( queue != NULL ) {
        stream = queue->cuda_stream();
    }
    else {
        fprintf( stderr, "Warning: %s got NULL queue\n", __func__ );
    }
    cudaError_t status;
    status = cudaMemcpy2DAsync(
        dB_dst, int(lddb*elemSize),
        dA_src, int(ldda*elemSize),
        int(m*elemSize), int(n), cudaMemcpyDeviceToDevice, stream );
    check_xerror( status, func, file, line );
}

#endif // HAVE_CUBLAS
