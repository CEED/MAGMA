/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
*/

#ifndef MAGMABLAS_H
#define MAGMABLAS_H

#include "magmablas_z.h"
#include "magmablas_c.h"
#include "magmablas_d.h"
#include "magmablas_s.h"
#include "magmablas_zc.h"
#include "magmablas_ds.h"

#ifdef __cplusplus
extern "C" {
#endif

// ========================================
// Define magma streams
extern magma_queue_t magma_stream;

cublasStatus_t magmablasSetKernelStream( magma_queue_t stream );
cublasStatus_t magmablasGetKernelStream( magma_queue_t *stream );


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
    magma_int_t n, magma_int_t elemSize,
    const void *hx_src, magma_int_t incx,
    void       *dy_dst, magma_int_t incy,
    const char* func, const char* file, int line );

void magma_getvector_internal(
    magma_int_t n, magma_int_t elemSize,
    const void *dx_src, magma_int_t incx,
    void       *hy_dst, magma_int_t incy,
    const char* func, const char* file, int line );

void magma_setvector_async_internal(
    magma_int_t n, magma_int_t elemSize,
    const void *hx_src, magma_int_t incx,
    void       *dy_dst, magma_int_t incy,
    magma_queue_t stream,
    const char* func, const char* file, int line );

void magma_getvector_async_internal(
    magma_int_t n, magma_int_t elemSize,
    const void *dx_src, magma_int_t incx,
    void       *hy_dst, magma_int_t incy,
    magma_queue_t stream,
    const char* func, const char* file, int line );


// ========================================
// copying sub-matrices (contiguous columns )
// set  copies host to device
// get  copies device to host
// copy copies device to device
// (with CUDA unified addressing, copy can be between same or different devices)
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
    magma_int_t m, magma_int_t n, magma_int_t elemSize,
    const void *hA_src, magma_int_t lda,
    void       *dB_dst, magma_int_t lddb,
    const char* func, const char* file, int line );

void magma_getmatrix_internal(
    magma_int_t m, magma_int_t n, magma_int_t elemSize,
    const void *dA_src, magma_int_t ldda,
    void       *hB_dst, magma_int_t ldb,
    const char* func, const char* file, int line );

void magma_copymatrix_internal(
    magma_int_t m, magma_int_t n, magma_int_t elemSize,
    const void *dA_src, magma_int_t ldda,
    void       *dB_dst, magma_int_t lddb,
    const char* func, const char* file, int line );

void magma_setmatrix_async_internal(
    magma_int_t m, magma_int_t n, magma_int_t elemSize,
    const void *hA_src, magma_int_t lda,
    void       *dB_dst, magma_int_t lddb,
    magma_queue_t stream,
    const char* func, const char* file, int line );

void magma_getmatrix_async_internal(
    magma_int_t m, magma_int_t n, magma_int_t elemSize,
    const void *dA_src, magma_int_t ldda,
    void       *hB_dst, magma_int_t ldb,
    magma_queue_t stream,
    const char* func, const char* file, int line );

void magma_copymatrix_async_internal(
    magma_int_t m, magma_int_t n, magma_int_t elemSize,
    const void *dA_src, magma_int_t ldda,
    void       *dB_dst, magma_int_t lddb,
    magma_queue_t stream,
    const char* func, const char* file, int line );


// ========================================
// copying vectors - version for magma_int_t
// TODO to make these truly type-safe, would need intermediate inline
//      magma_i* functions that call the generic magma_* functions.
//      Could do the same with magma_[sdcz]* set/get functions.

#define magma_isetvector(          n,                      hx_src, incx, dy_dst, incy ) \
        magma_setvector_internal(  n, sizeof(magma_int_t), hx_src, incx, dy_dst, incy, __func__, __FILE__, __LINE__ )

#define magma_igetvector(          n,                      dx_src, incx, hy_dst, incy ) \
        magma_getvector_internal(  n, sizeof(magma_int_t), dx_src, incx, hy_dst, incy, __func__, __FILE__, __LINE__ )

#define magma_icopyvector(         n,                      dx_src, incx, dy_dst, incy ) \
        magma_copyvector_internal( n, sizeof(magma_int_t), dx_src, incx, dy_dst, incy, __func__, __FILE__, __LINE__ )

#define magma_isetvector_async(          n,                      hx_src, incx, dy_dst, incy, queue ) \
        magma_setvector_async_internal(  n, sizeof(magma_int_t), hx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

#define magma_igetvector_async(          n,                      dx_src, incx, hy_dst, incy, queue ) \
        magma_getvector_async_internal(  n, sizeof(magma_int_t), dx_src, incx, hy_dst, incy, queue, __func__, __FILE__, __LINE__ )

#define magma_icopyvector_async(         n,                      dx_src, incx, dy_dst, incy, queue ) \
        magma_copyvector_async_internal( n, sizeof(magma_int_t), dx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )


// ========================================
// copying sub-matrices - version for magma_int_t

#define magma_isetmatrix(          m, n,                      hA_src, lda, dB_dst, lddb ) \
        magma_setmatrix_internal(  m, n, sizeof(magma_int_t), hA_src, lda, dB_dst, lddb, __func__, __FILE__, __LINE__ )

#define magma_igetmatrix(          m, n,                      dA_src, ldda, hB_dst, ldb ) \
        magma_getmatrix_internal(  m, n, sizeof(magma_int_t), dA_src, ldda, hB_dst, ldb, __func__, __FILE__, __LINE__ )

#define magma_icopymatrix(         m, n,                      dA_src, ldda, dB_dst, lddb ) \
        magma_copymatrix_internal( m, n, sizeof(magma_int_t), dA_src, ldda, dB_dst, lddb, __func__, __FILE__, __LINE__ )

#define magma_isetmatrix_async(          m, n,                      hA_src, lda, dB_dst, lddb, queue ) \
        magma_setmatrix_async_internal(  m, n, sizeof(magma_int_t), hA_src, lda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

#define magma_igetmatrix_async(          m, n,                      dA_src, ldda, hB_dst, ldb, queue ) \
        magma_getmatrix_async_internal(  m, n, sizeof(magma_int_t), dA_src, ldda, hB_dst, ldb, queue, __func__, __FILE__, __LINE__ )

#define magma_icopymatrix_async(         m, n,                      dA_src, ldda, dB_dst, lddb, queue ) \
        magma_copymatrix_async_internal( m, n, sizeof(magma_int_t), dA_src, ldda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )


// ========================================
// copying vectors - version for magma_index_t

#define magma_index_setvector(     n,                        hx_src, incx, dy_dst, incy ) \
        magma_setvector_internal(  n, sizeof(magma_index_t), hx_src, incx, dy_dst, incy, __func__, __FILE__, __LINE__ )

#define magma_index_getvector(     n,                        dx_src, incx, hy_dst, incy ) \
        magma_getvector_internal(  n, sizeof(magma_index_t), dx_src, incx, hy_dst, incy, __func__, __FILE__, __LINE__ )

#define magma_index_copyvector(    n,                        dx_src, incx, dy_dst, incy ) \
        magma_copyvector_internal( n, sizeof(magma_index_t), dx_src, incx, dy_dst, incy, __func__, __FILE__, __LINE__ )

#define magma_index_setvector_async(     n,                        hx_src, incx, dy_dst, incy, queue ) \
        magma_setvector_async_internal(  n, sizeof(magma_index_t), hx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

#define magma_index_getvector_async(     n,                        dx_src, incx, hy_dst, incy, queue ) \
        magma_getvector_async_internal(  n, sizeof(magma_index_t), dx_src, incx, hy_dst, incy, queue, __func__, __FILE__, __LINE__ )

#define magma_index_copyvector_async(    n,                        dx_src, incx, dy_dst, incy, queue ) \
        magma_copyvector_async_internal( n, sizeof(magma_index_t), dx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )


// ========================================
// copying sub-matrices - version for magma_index_t

#define magma_index_setmatrix(     m, n,                        hA_src, lda, dB_dst, lddb ) \
        magma_setmatrix_internal(  m, n, sizeof(magma_index_t), hA_src, lda, dB_dst, lddb, __func__, __FILE__, __LINE__ )

#define magma_index_getmatrix(     m, n,                        dA_src, ldda, hB_dst, ldb ) \
        magma_getmatrix_internal(  m, n, sizeof(magma_index_t), dA_src, ldda, hB_dst, ldb, __func__, __FILE__, __LINE__ )

#define magma_index_copymatrix(    m, n,                        dA_src, ldda, dB_dst, lddb ) \
        magma_copymatrix_internal( m, n, sizeof(magma_index_t), dA_src, ldda, dB_dst, lddb, __func__, __FILE__, __LINE__ )

#define magma_index_setmatrix_async(     m, n,                        hA_src, lda, dB_dst, lddb, queue ) \
        magma_setmatrix_async_internal(  m, n, sizeof(magma_index_t), hA_src, lda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

#define magma_index_getmatrix_async(     m, n,                        dA_src, ldda, hB_dst, ldb, queue ) \
        magma_getmatrix_async_internal(  m, n, sizeof(magma_index_t), dA_src, ldda, hB_dst, ldb, queue, __func__, __FILE__, __LINE__ )

#define magma_index_copymatrix_async(    m, n,                        dA_src, ldda, dB_dst, lddb, queue ) \
        magma_copymatrix_async_internal( m, n, sizeof(magma_index_t), dA_src, ldda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

#ifdef __cplusplus
}
#endif

#endif /* MAGMABLAS_H */
