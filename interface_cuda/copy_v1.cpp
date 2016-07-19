/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
 
       @author Mark Gates
*/
#ifndef MAGMA_NO_V1

#include "common_magma.h"
#include "error.h"

#ifdef HAVE_CUBLAS

// generic, type-independent routines to copy data.
// type-safe versions which avoid the user needing sizeof(...) are in [sdcz]set_get.cpp

// ========================================
// copying vectors
extern "C" void
magma_setvector_internal(
    magma_int_t n, magma_int_t elemSize,
    void const* hx_src, magma_int_t incx,
    magma_ptr   dy_dst, magma_int_t incy,
    const char* func, const char* file, int line )
{
    magma_setvector_q_internal(
        n, elemSize,
        hx_src, incx,
        dy_dst, incy,
        magmablasGetQueue(),
        func, file, line );
}

// --------------------
extern "C" void
magma_getvector_internal(
    magma_int_t n, magma_int_t elemSize,
    magma_const_ptr dx_src, magma_int_t incx,
    void*           hy_dst, magma_int_t incy,
    const char* func, const char* file, int line )
{
    magma_getvector_q_internal(
        n, elemSize,
        dx_src, incx,
        hy_dst, incy,
        magmablasGetQueue(),
        func, file, line );
}

// --------------------
extern "C" void
magma_copyvector_internal(
    magma_int_t n, magma_int_t elemSize,
    magma_const_ptr dx_src, magma_int_t incx,
    magma_ptr       dy_dst, magma_int_t incy,
    const char* func, const char* file, int line )
{
    magma_copyvector_q_internal(
        n, elemSize,
        dx_src, incx,
        dy_dst, incy,
        magmablasGetQueue(),
        func, file, line );
}


// ========================================
// copying sub-matrices (contiguous columns)
extern "C" void
magma_setmatrix_internal(
    magma_int_t m, magma_int_t n, magma_int_t elemSize,
    void const* hA_src, magma_int_t ldha,
    magma_ptr   dB_dst, magma_int_t lddb,
    const char* func, const char* file, int line )
{
    cublasStatus_t status;
    status = cublasSetMatrix(
        int(m), int(n), int(elemSize),
        hA_src, int(ldha),
        dB_dst, int(lddb) );
    check_xerror( status, func, file, line );
}

// --------------------
extern "C" void
magma_getmatrix_internal(
    magma_int_t m, magma_int_t n, magma_int_t elemSize,
    magma_const_ptr dA_src, magma_int_t ldda,
    void*           hB_dst, magma_int_t ldhb,
    const char* func, const char* file, int line )
{
    cublasStatus_t status;
    status = cublasGetMatrix(
        int(m), int(n), int(elemSize),
        dA_src, int(ldda),
        hB_dst, int(ldhb) );
    check_xerror( status, func, file, line );
}

// --------------------
extern "C" void
magma_copymatrix_internal(
    magma_int_t m, magma_int_t n, magma_int_t elemSize,
    magma_const_ptr dA_src, magma_int_t ldda,
    magma_ptr       dB_dst, magma_int_t lddb,
    const char* func, const char* file, int line )
{
    cudaError_t status;
    status = cudaMemcpy2D(
        dB_dst, int(lddb*elemSize),
        dA_src, int(ldda*elemSize),
        int(m*elemSize), int(n), cudaMemcpyDeviceToDevice );
    check_xerror( status, func, file, line );
}

#endif // HAVE_CUBLAS

#endif // MAGMA_NO_V1
