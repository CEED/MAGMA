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

#define COMPLEX

#ifdef HAVE_CUBLAS

// ----------------------------------------
// Convert MAGMA constants to CUBLAS v1 constants, which are the same as lapack.
// These must be static to avoid conflict with CUBLAS v2 translators.
extern const char *magma2lapack_constants[];

#ifdef __cplusplus
extern "C" {
#endif

static char cublas_trans_const ( magma_trans_t magma_const )
{
    assert( magma_const >= MagmaNoTrans   );
    assert( magma_const <= MagmaConjTrans );
    return *magma2lapack_constants[ magma_const ];
}

static char cublas_side_const  ( magma_side_t magma_const )
{
    assert( magma_const >= MagmaLeft  );
    assert( magma_const <= MagmaRight );
    return *magma2lapack_constants[ magma_const ];
}

static char cublas_diag_const  ( magma_diag_t magma_const )
{
    assert( magma_const >= MagmaNonUnit );
    assert( magma_const <= MagmaUnit    );
    return *magma2lapack_constants[ magma_const ];
}

static char cublas_uplo_const  ( magma_uplo_t magma_const )
{
    assert( magma_const >= MagmaUpper );
    assert( magma_const <= MagmaLower );
    return *magma2lapack_constants[ magma_const ];
}

#ifdef __cplusplus
} // extern "C"
#endif


// ========================================
// Level 1 BLAS

// --------------------
extern "C"
magma_int_t magma_izamax(
    magma_int_t n,
    const magmaDoubleComplex *dx, magma_int_t incx )
{
    return cublasIzamax( n, dx, incx );
}

// --------------------
extern "C"
magma_int_t magma_izamin(
    magma_int_t n,
    const magmaDoubleComplex *dx, magma_int_t incx )
{
    return cublasIzamin( n, dx, incx );
}

// --------------------
extern "C"
double magma_dzasum(
    magma_int_t n,
    const magmaDoubleComplex *dx, magma_int_t incx )
{
    return cublasDzasum( n, dx, incx );
}

// --------------------
extern "C"
void magma_zaxpy(
    magma_int_t n,
    magmaDoubleComplex alpha,
    const magmaDoubleComplex *dx, magma_int_t incx,
    magmaDoubleComplex       *dy, magma_int_t incy )
{
    cublasZaxpy( n, alpha, dx, incx, dy, incy );
}

// --------------------
extern "C"
void magma_zcopy(
    magma_int_t n,
    const magmaDoubleComplex *dx, magma_int_t incx,
    magmaDoubleComplex       *dy, magma_int_t incy )
{
    cublasZcopy( n, dx, incx, dy, incy );
}

// --------------------
extern "C"
magmaDoubleComplex magma_zdotc(
    magma_int_t n,
    const magmaDoubleComplex *dx, magma_int_t incx,
    const magmaDoubleComplex *dy, magma_int_t incy )
{
    return cublasZdotc( n, dx, incx, dy, incy );
}

#ifdef COMPLEX
// --------------------
extern "C"
magmaDoubleComplex magma_zdotu(
    magma_int_t n,
    const magmaDoubleComplex *dx, magma_int_t incx,
    const magmaDoubleComplex *dy, magma_int_t incy )
{
    return cublasZdotu( n, dx, incx, dy, incy );
}
#endif

// --------------------
extern "C"
double magma_dznrm2(
    magma_int_t n,
    const magmaDoubleComplex *dx, magma_int_t incx )
{
    return cublasDznrm2( n, dx, incx );
}

// --------------------
extern "C"
void magma_zrot(
    magma_int_t n,
    magmaDoubleComplex *dx, magma_int_t incx,
    magmaDoubleComplex *dy, magma_int_t incy,
    double dc, magmaDoubleComplex ds )
{
    cublasZrot( n, dx, incx, dy, incy, dc, ds );
}

#ifdef COMPLEX
// --------------------
extern "C"
void magma_zdrot(
    magma_int_t n,
    magmaDoubleComplex *dx, magma_int_t incx,
    magmaDoubleComplex *dy, magma_int_t incy,
    double dc, double ds )
{
    cublasZdrot( n, dx, incx, dy, incy, dc, ds );
}
#endif

#ifdef REAL
// --------------------
extern "C"
void magma_zrotm(
    magma_int_t n,
    double *dx, magma_int_t incx,
    double *dy, magma_int_t incy,
    const double *param )
{
    cublasZrotm( n, dx, incx, dy, incy, param );
}

// --------------------
extern "C"
void magma_zrotmg(
    double *d1, double       *d2,
    double *x1, const double *y1,
    double *param )
{
    cublasZrotmg( d1, d2, x1, y1, param );
}
#endif

// --------------------
extern "C"
void magma_zscal(
    magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex *dx, magma_int_t incx )
{
    cublasZscal( n, alpha, dx, incx );
}

#ifdef COMPLEX
// --------------------
extern "C"
void magma_zdscal(
    magma_int_t n,
    double alpha,
    magmaDoubleComplex *dx, magma_int_t incx )
{
    cublasZdscal( n, alpha, dx, incx );
}
#endif

// --------------------
extern "C"
void magma_zswap(
    magma_int_t n,
    magmaDoubleComplex *dx, magma_int_t incx,
    magmaDoubleComplex *dy, magma_int_t incy )
{
    cublasZswap( n, dx, incx, dy, incy );
}


// ========================================
// Level 2 BLAS

// --------------------
extern "C"
void magma_zgemv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    const magmaDoubleComplex *dA, magma_int_t ldda,
    const magmaDoubleComplex *dx, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex       *dy, magma_int_t incy )
{
    cublasZgemv(
        cublas_trans_const( transA ),
        m, n,
        alpha, dA, ldda,
               dx, incx,
        beta,  dy, incy );
}

// --------------------
extern "C"
void magma_zgerc(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    const magmaDoubleComplex *dx, magma_int_t incx,
    const magmaDoubleComplex *dy, magma_int_t incy,
    magmaDoubleComplex       *dA, magma_int_t ldda )
{
    cublasZgerc(
        m, n,
        alpha, dx, incx,
               dy, incy,
               dA, ldda );
}

#ifdef COMPLEX
// --------------------
extern "C"
void magma_zgeru(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    const magmaDoubleComplex *dx, magma_int_t incx,
    const magmaDoubleComplex *dy, magma_int_t incy,
    magmaDoubleComplex       *dA, magma_int_t ldda )
{
    cublasZgeru(
        m, n,
        alpha, dx, incx,
               dy, incy,
               dA, ldda );
}
#endif

// --------------------
extern "C"
void magma_zhemv(
    magma_uplo_t uplo,
    magma_int_t n,
    magmaDoubleComplex alpha,
    const magmaDoubleComplex *dA, magma_int_t ldda,
    const magmaDoubleComplex *dx, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex       *dy, magma_int_t incy )
{
    cublasZhemv(
        cublas_uplo_const( uplo ),
        n,
        alpha, dA, ldda,
               dx, incx,
        beta,  dy, incy );
}

// --------------------
extern "C"
void magma_zher(
    magma_uplo_t uplo,
    magma_int_t n,
    double alpha,
    const magmaDoubleComplex *dx, magma_int_t incx,
    magmaDoubleComplex       *dA, magma_int_t ldda )
{
    cublasZher(
        cublas_uplo_const( uplo ),
        n,
        alpha, dx, incx,
               dA, ldda );
}

// --------------------
extern "C"
void magma_zher2(
    magma_uplo_t uplo,
    magma_int_t n,
    magmaDoubleComplex alpha,
    const magmaDoubleComplex *dx, magma_int_t incx,
    const magmaDoubleComplex *dy, magma_int_t incy,
    magmaDoubleComplex       *dA, magma_int_t ldda )
{
    cublasZher2(
        cublas_uplo_const( uplo ),
        n,
        alpha, dx, incx,
               dy, incy,
               dA, ldda );
}

// --------------------
extern "C"
void magma_ztrmv(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    const magmaDoubleComplex *dA, magma_int_t ldda,
    magmaDoubleComplex       *dx, magma_int_t incx )
{
    cublasZtrmv(
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        cublas_diag_const( diag ),
        n,
        dA, ldda,
        dx, incx );
}

// --------------------
extern "C"
void magma_ztrsv(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    const magmaDoubleComplex *dA, magma_int_t ldda,
    magmaDoubleComplex       *dx, magma_int_t incx )
{
    cublasZtrsv(
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        cublas_diag_const( diag ),
        n,
        dA, ldda,
        dx, incx );
}

// ========================================
// Level 3 BLAS

// --------------------
extern "C"
void magma_zgemm(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    const magmaDoubleComplex *dA, magma_int_t ldda,
    const magmaDoubleComplex *dB, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex       *dC, magma_int_t lddc )
{
    cublasZgemm(
        cublas_trans_const( transA ),
        cublas_trans_const( transB ),
        m, n, k,
        alpha, dA, ldda,
               dB, lddb,
        beta,  dC, lddc );
}

// --------------------
extern "C"
void magma_zsymm(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    const magmaDoubleComplex *dA, magma_int_t ldda,
    const magmaDoubleComplex *dB, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex       *dC, magma_int_t lddc )
{
    cublasZsymm(
        cublas_side_const( side ),
        cublas_uplo_const( uplo ),
        m, n,
        alpha, dA, ldda,
               dB, lddb,
        beta,  dC, lddc );
}

// --------------------
extern "C"
void magma_zsyrk(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    const magmaDoubleComplex *dA, magma_int_t ldda,
    magmaDoubleComplex beta,
    magmaDoubleComplex       *dC, magma_int_t lddc )
{
    cublasZsyrk(
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        n, k,
        alpha, dA, ldda,
        beta,  dC, lddc );
}

// --------------------
extern "C"
void magma_zsyr2k(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    const magmaDoubleComplex *dA, magma_int_t ldda,
    const magmaDoubleComplex *dB, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex       *dC, magma_int_t lddc )
{
    cublasZsyr2k(
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        n, k,
        alpha, dA, ldda,
               dB, lddb,
        beta,  dC, lddc );
}

#ifdef COMPLEX
// --------------------
extern "C"
void magma_zhemm(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    const magmaDoubleComplex *dA, magma_int_t ldda,
    const magmaDoubleComplex *dB, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex       *dC, magma_int_t lddc )
{
    cublasZhemm(
        cublas_side_const( side ),
        cublas_uplo_const( uplo ),
        m, n,
        alpha, dA, ldda,
               dB, lddb,
        beta,  dC, lddc );
}

// --------------------
extern "C"
void magma_zherk(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    double alpha,
    const magmaDoubleComplex *dA, magma_int_t ldda,
    double beta,
    magmaDoubleComplex       *dC, magma_int_t lddc )
{
    cublasZherk(
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        n, k,
        alpha, dA, ldda,
        beta,  dC, lddc );
}

// --------------------
extern "C"
void magma_zher2k(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    const magmaDoubleComplex *dA, magma_int_t ldda,
    const magmaDoubleComplex *dB, magma_int_t lddb,
    double beta,
    magmaDoubleComplex       *dC, magma_int_t lddc )
{
    cublasZher2k(
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        n, k,
        alpha, dA, ldda,
               dB, lddb,
        beta,  dC, lddc );
}
#endif // COMPLEX

// --------------------
extern "C"
void magma_ztrmm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    const magmaDoubleComplex *dA, magma_int_t ldda,
    magmaDoubleComplex       *dB, magma_int_t lddb )
{
    cublasZtrmm(
        cublas_side_const( side ),
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        cublas_diag_const( diag ),
        m, n,
        alpha, dA, ldda,
               dB, lddb );
}

// --------------------
extern "C"
void magma_ztrsm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    const magmaDoubleComplex *dA, magma_int_t ldda,
    magmaDoubleComplex       *dB, magma_int_t lddb )
{
    cublasZtrsm(
        cublas_side_const( side ),
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        cublas_diag_const( diag ),
        m, n,
        alpha, dA, ldda,
               dB, lddb );
}

#endif // HAVE_CUBLAS

#undef COMPLEX
