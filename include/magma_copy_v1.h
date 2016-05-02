#ifndef MAGMA_COPY_V1_H
#define MAGMA_COPY_V1_H

#ifdef MAGMA_NO_V1
#error "Since MAGMA_NO_V1 is defined, magma.h is invalid; use magma_v2.h"
#endif

#include "magma_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// ========================================
// copying vectors
// set  copies host   to device
// get  copies device to host
// copy copies device to device
// (with CUDA unified addressing, copy can be between same or different devices)
// Add the function, file, and line for error-reporting purposes.
// async versions are same for v1 and v2; see magmablas_q.h

#define magma_setvector(           n, elemSize, hx_src, incx, dy_dst, incy ) \
        magma_setvector_internal(  n, elemSize, hx_src, incx, dy_dst, incy, __func__, __FILE__, __LINE__ )

#define magma_getvector(           n, elemSize, dx_src, incx, hy_dst, incy ) \
        magma_getvector_internal(  n, elemSize, dx_src, incx, hy_dst, incy, __func__, __FILE__, __LINE__ )

#define magma_copyvector(          n, elemSize, dx_src, incx, dy_dst, incy ) \
        magma_copyvector_internal( n, elemSize, dx_src, incx, dy_dst, incy, __func__, __FILE__, __LINE__ )

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

void magma_copyvector_internal(
    magma_int_t n, magma_int_t elemSize,
    const void *dx_src, magma_int_t incx,
    void       *dy_dst, magma_int_t incy,
    const char* func, const char* file, int line );


// ========================================
// copying sub-matrices (contiguous columns)

#define magma_setmatrix(           m, n, elemSize, hA_src, lda, dB_dst, lddb ) \
        magma_setmatrix_internal(  m, n, elemSize, hA_src, lda, dB_dst, lddb, __func__, __FILE__, __LINE__ )

#define magma_getmatrix(           m, n, elemSize, dA_src, ldda, hB_dst, ldb ) \
        magma_getmatrix_internal(  m, n, elemSize, dA_src, ldda, hB_dst, ldb, __func__, __FILE__, __LINE__ )

#define magma_copymatrix(          m, n, elemSize, dA_src, ldda, dB_dst, lddb ) \
        magma_copymatrix_internal( m, n, elemSize, dA_src, ldda, dB_dst, lddb, __func__, __FILE__, __LINE__ )

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


// ========================================
// copying vectors - version for magma_int_t

#define magma_isetvector(           n, hx_src, incx, dy_dst, incy ) \
        magma_isetvector_internal(  n, hx_src, incx, dy_dst, incy, __func__, __FILE__, __LINE__ )

#define magma_igetvector(           n, dx_src, incx, hy_dst, incy ) \
        magma_igetvector_internal(  n, dx_src, incx, hy_dst, incy, __func__, __FILE__, __LINE__ )

#define magma_icopyvector(          n, dx_src, incx, dy_dst, incy ) \
        magma_icopyvector_internal( n, dx_src, incx, dy_dst, incy, __func__, __FILE__, __LINE__ )

static inline void magma_isetvector_internal(
    magma_int_t n,
    const magma_int_t *hx_src, magma_int_t incx,
    magma_int_t       *dy_dst, magma_int_t incy,
    const char* func, const char* file, int line )
{ magma_setvector_internal( n, sizeof(magma_int_t), hx_src, incx, dy_dst, incy, func, file, line ); }

static inline void magma_igetvector_internal(
    magma_int_t n,
    const magma_int_t *dx_src, magma_int_t incx,
    magma_int_t       *hy_dst, magma_int_t incy,
    const char* func, const char* file, int line )
{ magma_getvector_internal( n, sizeof(magma_int_t), dx_src, incx, hy_dst, incy, func, file, line ); }

static inline void magma_icopyvector_internal(
    magma_int_t n,
    const magma_int_t *dx_src, magma_int_t incx,
    magma_int_t       *dy_dst, magma_int_t incy,
    const char* func, const char* file, int line )
{ magma_copyvector_internal( n, sizeof(magma_int_t), dx_src, incx, dy_dst, incy, func, file, line ); }


// ========================================
// copying sub-matrices - version for magma_int_t

#define magma_isetmatrix(           m, n, hA_src, lda, dB_dst, lddb ) \
        magma_isetmatrix_internal(  m, n, hA_src, lda, dB_dst, lddb, __func__, __FILE__, __LINE__ )

#define magma_igetmatrix(           m, n, dA_src, ldda, hB_dst, ldb ) \
        magma_igetmatrix_internal(  m, n, dA_src, ldda, hB_dst, ldb, __func__, __FILE__, __LINE__ )

#define magma_icopymatrix(          m, n, dA_src, ldda, dB_dst, lddb ) \
        magma_icopymatrix_internal( m, n, dA_src, ldda, dB_dst, lddb, __func__, __FILE__, __LINE__ )

static inline void magma_isetmatrix_internal(
    magma_int_t m, magma_int_t n,
    const magma_int_t *hA_src, magma_int_t lda,
    magma_int_t       *dB_dst, magma_int_t lddb,
    const char* func, const char* file, int line )
{ magma_setmatrix_internal( m, n, sizeof(magma_int_t), hA_src, lda, dB_dst, lddb, func, file, line ); }

static inline void magma_igetmatrix_internal(
    magma_int_t m, magma_int_t n,
    const magma_int_t *dA_src, magma_int_t ldda,
    magma_int_t       *hB_dst, magma_int_t ldb,
    const char* func, const char* file, int line )
{ magma_getmatrix_internal( m, n, sizeof(magma_int_t), dA_src, ldda, hB_dst, ldb, func, file, line ); }

static inline void magma_icopymatrix_internal(
    magma_int_t m, magma_int_t n,
    const magma_int_t *dA_src, magma_int_t ldda,
    magma_int_t       *dB_dst, magma_int_t lddb,
    const char* func, const char* file, int line )
{ magma_copymatrix_internal( m, n, sizeof(magma_int_t), dA_src, ldda, dB_dst, lddb, func, file, line ); }


// ========================================
// copying vectors - version for magma_index_t

#define magma_index_setvector(           n, hx_src, incx, dy_dst, incy ) \
        magma_index_setvector_internal(  n, hx_src, incx, dy_dst, incy, __func__, __FILE__, __LINE__ )

#define magma_index_getvector(           n, dx_src, incx, hy_dst, incy ) \
        magma_index_getvector_internal(  n, dx_src, incx, hy_dst, incy, __func__, __FILE__, __LINE__ )

#define magma_index_copyvector(          n, dx_src, incx, dy_dst, incy ) \
        magma_index_copyvector_internal( n, dx_src, incx, dy_dst, incy, __func__, __FILE__, __LINE__ )

static inline void magma_index_setvector_internal(
    magma_int_t n,
    const magma_index_t *hx_src, magma_int_t incx,
    magma_index_t       *dy_dst, magma_int_t incy,
    const char* func, const char* file, int line )
{ magma_setvector_internal( n, sizeof(magma_index_t), hx_src, incx, dy_dst, incy, func, file, line ); }

static inline void magma_index_getvector_internal(
    magma_int_t n,
    const magma_index_t *dx_src, magma_int_t incx,
    magma_index_t       *hy_dst, magma_int_t incy,
    const char* func, const char* file, int line )
{ magma_getvector_internal( n, sizeof(magma_index_t), dx_src, incx, hy_dst, incy, func, file, line ); }

static inline void magma_index_copyvector_internal(
    magma_int_t n,
    const magma_index_t *dx_src, magma_int_t incx,
    magma_index_t       *dy_dst, magma_int_t incy,
    const char* func, const char* file, int line )
{ magma_copyvector_internal( n, sizeof(magma_index_t), dx_src, incx, dy_dst, incy, func, file, line ); }


// ========================================
// copying sub-matrices - version for magma_index_t

#define magma_index_setmatrix(           m, n, hA_src, lda, dB_dst, lddb ) \
        magma_index_setmatrix_internal(  m, n, hA_src, lda, dB_dst, lddb, __func__, __FILE__, __LINE__ )

#define magma_index_getmatrix(           m, n, dA_src, ldda, hB_dst, ldb ) \
        magma_index_getmatrix_internal(  m, n, dA_src, ldda, hB_dst, ldb, __func__, __FILE__, __LINE__ )

#define magma_index_copymatrix(          m, n, dA_src, ldda, dB_dst, lddb ) \
        magma_index_copymatrix_internal( m, n, dA_src, ldda, dB_dst, lddb, __func__, __FILE__, __LINE__ )

static inline void magma_index_setmatrix_internal(
    magma_int_t m, magma_int_t n,
    const magma_index_t *hA_src, magma_int_t lda,
    magma_index_t       *dB_dst, magma_int_t lddb,
    const char* func, const char* file, int line )
{ magma_setmatrix_internal( m, n, sizeof(magma_index_t), hA_src, lda, dB_dst, lddb, func, file, line ); }

static inline void magma_index_getmatrix_internal(
    magma_int_t m, magma_int_t n,
    const magma_index_t *dA_src, magma_int_t ldda,
    magma_index_t       *hB_dst, magma_int_t ldb,
    const char* func, const char* file, int line )
{ magma_getmatrix_internal( m, n, sizeof(magma_index_t), dA_src, ldda, hB_dst, ldb, func, file, line ); }

static inline void magma_index_copymatrix_internal(
    magma_int_t m, magma_int_t n,
    const magma_index_t *dA_src, magma_int_t ldda,
    magma_index_t       *dB_dst, magma_int_t lddb,
    const char* func, const char* file, int line )
{ magma_copymatrix_internal( m, n, sizeof(magma_index_t), dA_src, ldda, dB_dst, lddb, func, file, line ); }

#ifdef __cplusplus
}
#endif

#endif        //  #ifndef MAGMA_COPY_V1_H
