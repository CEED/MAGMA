/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011
*/

#ifndef MAGMABLAS_H
#define MAGMABLAS_H

#include <cublas.h>
#include <stdint.h>

/* To use int64_t, link with mkl_intel_ilp64 or similar (instead of mkl_intel_lp64). */
#ifdef USE_INT64
typedef int64_t magma_int_t;
#else
typedef int magma_int_t;
#endif

typedef cuDoubleComplex magmaDoubleComplex;
typedef cuFloatComplex  magmaFloatComplex;

typedef int   magma_err_t;
typedef void* magma_devptr;

// For now, make these compatible with old cublas v1 prototypes.
// In the future, we will redefine these data types and
// add queues (opencl queues, cublas handles).
typedef char magma_trans_t;
typedef char magma_side_t ;
typedef char magma_uplo_t ;
typedef char magma_diag_t ;

typedef cudaStream_t magma_stream_t;
typedef cudaStream_t magma_queue_t;
typedef cudaEvent_t  magma_event_t;
typedef int          magma_device_t;

// needed by magmablas*.h, but should eventually go in magma_types.h (see clMAGMA)
#define MagmaMaxGPUs       8

#include "magmablas_z.h"
#include "magmablas_c.h"
#include "magmablas_d.h"
#include "magmablas_s.h"
#include "magmablas_zc.h"
#include "magmablas_ds.h"

#if (GPUSHMEM < 200)
#define magmablas_zgemm cublasZgemm
#endif
#define magmablas_cgemm cublasCgemm

#ifdef __cplusplus
extern "C" {
#endif

// ========================================
// Define magma streams
extern cudaStream_t magma_stream;

cublasStatus_t magmablasSetKernelStream( cudaStream_t stream );
cublasStatus_t magmablasGetKernelStream( cudaStream_t *stream );


// ========================================
// copying vectors
// set copies host to device
// get copies device to host
// Add the function, file, and line for error-reporting purposes.

#define magma_setvector(           n, elemSize, hx_src, incx, dy_dst, incy ) \
        magma_setvector_internal(  n, elemSize, hx_src, incx, dy_dst, incy, __func__, __FILE__, __LINE__ )

#define magma_getvector(           n, elemSize, dx_src, incx, hy_dst, incy ) \
        magma_getvector_internal(  n, elemSize, dx_src, incx, hy_dst, incy, __func__, __FILE__, __LINE__ )

#define magma_setvector_async(           n, elemSize, hx_src, incx, dy_dst, incy, stream ) \
        magma_setvector_async_internal(  n, elemSize, hx_src, incx, dy_dst, incy, stream, __func__, __FILE__, __LINE__ )

#define magma_getvector_async(           n, elemSize, dx_src, incx, hy_dst, incy, stream ) \
        magma_getvector_async_internal(  n, elemSize, dx_src, incx, hy_dst, incy, stream, __func__, __FILE__, __LINE__ )

void magma_setvector_internal(
    magma_int_t n, size_t elemSize,
    const void *hx_src, magma_int_t incx,
    void       *dy_dst, magma_int_t incy,
    const char* func, const char* file, int line );

void magma_getvector_internal(
    magma_int_t n, size_t elemSize,
    const void *dx_src, magma_int_t incx,
    void       *hy_dst, magma_int_t incy,
    const char* func, const char* file, int line );

void magma_setvector_async_internal(
    magma_int_t n, size_t elemSize,
    const void *hx_src, magma_int_t incx,
    void       *dy_dst, magma_int_t incy,
    magma_stream_t stream,
    const char* func, const char* file, int line );

void magma_getvector_async_internal(
    magma_int_t n, size_t elemSize,
    const void *dx_src, magma_int_t incx,
    void       *hy_dst, magma_int_t incy,
    magma_stream_t stream,
    const char* func, const char* file, int line );


// ========================================
// copying sub-matrices (contiguous columns )
// set  copies host to device
// get  copies device to host
// copy copies device to device
// (with CUDA unified addressing, copy can be between same or different devices )
// Add the function, file, and line for error-reporting purposes.

#define magma_setmatrix(           m, n, elemSize, hA_src, lda, dB_dst, lddb ) \
        magma_setmatrix_internal(  m, n, elemSize, hA_src, lda, dB_dst, lddb, __func__, __FILE__, __LINE__ )

#define magma_getmatrix(           m, n, elemSize, dA_src, ldda, hB_dst, ldb ) \
        magma_getmatrix_internal(  m, n, elemSize, dA_src, ldda, hB_dst, ldb, __func__, __FILE__, __LINE__ )

#define magma_copymatrix(          m, n, elemSize, dA_src, ldda, dB_dst, lddb ) \
        magma_copymatrix_internal( m, n, elemSize, dA_src, ldda, dB_dst, lddb, __func__, __FILE__, __LINE__ )

#define magma_setmatrix_async(           m, n, elemSize, hA_src, lda, dB_dst, lddb, stream ) \
        magma_setmatrix_async_internal(  m, n, elemSize, hA_src, lda, dB_dst, lddb, stream, __func__, __FILE__, __LINE__ )

#define magma_getmatrix_async(           m, n, elemSize, dA_src, ldda, hB_dst, ldb, stream ) \
        magma_getmatrix_async_internal(  m, n, elemSize, dA_src, ldda, hB_dst, ldb, stream, __func__, __FILE__, __LINE__ )

#define magma_copymatrix_async(          m, n, elemSize, dA_src, ldda, dB_dst, lddb, stream ) \
        magma_copymatrix_async_internal( m, n, elemSize, dA_src, ldda, dB_dst, lddb, stream, __func__, __FILE__, __LINE__ )

void magma_setmatrix_internal(
    magma_int_t m, magma_int_t n, size_t elemSize,
    const void *hA_src, magma_int_t lda,
    void       *dB_dst, magma_int_t lddb,
    const char* func, const char* file, int line );

void magma_getmatrix_internal(
    magma_int_t m, magma_int_t n, size_t elemSize,
    const void *dA_src, magma_int_t ldda,
    void       *hB_dst, magma_int_t ldb,
    const char* func, const char* file, int line );

void magma_copymatrix_internal(
    magma_int_t m, magma_int_t n, size_t elemSize,
    const void *dA_src, magma_int_t ldda,
    void       *dB_dst, magma_int_t lddb,
    const char* func, const char* file, int line );

void magma_setmatrix_async_internal(
    magma_int_t m, magma_int_t n, size_t elemSize,
    const void *hA_src, magma_int_t lda,
    void       *dB_dst, magma_int_t lddb,
    magma_stream_t stream,
    const char* func, const char* file, int line );

void magma_getmatrix_async_internal(
    magma_int_t m, magma_int_t n, size_t elemSize,
    const void *dA_src, magma_int_t ldda,
    void       *hB_dst, magma_int_t ldb,
    magma_stream_t stream,
    const char* func, const char* file, int line );

void magma_copymatrix_async_internal(
    magma_int_t m, magma_int_t n, size_t elemSize,
    const void *dA_src, magma_int_t ldda,
    void       *dB_dst, magma_int_t lddb,
    magma_stream_t stream,
    const char* func, const char* file, int line );

#ifdef __cplusplus
}
#endif

#endif /* MAGMABLAS_H */
