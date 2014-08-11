/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
 
       @precisions normal z -> s d c
*/

#ifndef MAGMA_LAPACK_H
#define MAGMA_LAPACK_H

#include "magma_mangling.h"

#include "magma_zlapack.h"
#include "magma_clapack.h"
#include "magma_dlapack.h"
#include "magma_slapack.h"

#ifdef __cplusplus
extern "C" {
#endif

#define lapackf77_ieeeck FORTRAN_NAME( ieeeck, IEEECK )
#define lapackf77_lsame  FORTRAN_NAME( lsame,  LSAME  )
#define lapackf77_xerbla FORTRAN_NAME( xerbla, XERBLA )

#define lapackf77_slamch FORTRAN_NAME( slamch, SLAMCH )
#define lapackf77_dlamch FORTRAN_NAME( dlamch, DLAMCH )
#define lapackf77_slabad FORTRAN_NAME( slabad, SLABAD )
#define lapackf77_dlabad FORTRAN_NAME( dlabad, DLABAD )
#define lapackf77_zcgesv FORTRAN_NAME( zcgesv, ZCGESV )
#define lapackf77_dsgesv FORTRAN_NAME( dsgesv, DSGESV )

#define lapackf77_dsterf FORTRAN_NAME( dsterf, DSTERF )
#define lapackf77_ssterf FORTRAN_NAME( ssterf, SSTERF )

#define lapackf77_zlag2c FORTRAN_NAME( zlag2c, ZLAG2C )
#define lapackf77_clag2z FORTRAN_NAME( clag2z, CLAG2Z )
#define lapackf77_dlag2s FORTRAN_NAME( dlag2s, DLAG2S )
#define lapackf77_slag2d FORTRAN_NAME( slag2d, SLAG2D )

#define lapackf77_zlat2c FORTRAN_NAME( zlat2c, ZLAT2C )
#define lapackf77_dlat2s FORTRAN_NAME( dlat2s, DLAT2S )
//#define lapackf77_clat2z FORTRAN_NAME( clat2z, CLAT2Z )
//#define lapackf77_slat2d FORTRAN_NAME( slat2d, SLAT2D )

#define lapackf77_dlapy2 FORTRAN_NAME( dlapy2, DLAPY2 )
#define lapackf77_slapy2 FORTRAN_NAME( slapy2, SLAPY2 )

// Can't call lapack ilaenv directly because passing strings, i.e.,
// character*(*), to Fortran is not portable.
// This wrapper passes a character(*) array and separate length,
// which should be portable.
#define magmaf77_ilaenv  FORTRAN_NAME( magmaf77_ilaenv, MAGMAF77_ILAENV )

magma_int_t magmaf77_ilaenv( const magma_int_t* ispec,
                             const char* name, const magma_int_t* namelen,
                             const char* opts, const magma_int_t* optslen,
                             const magma_int_t* n1, const magma_int_t* n2,
                             const magma_int_t* n3, const magma_int_t* n4 );

// Convenience C wrapper that calls magmaf77_ilaenv.
magma_int_t magma_ilaenv( magma_int_t ispec, const char* name, const char* opts,
                          magma_int_t n1, magma_int_t n2,
                          magma_int_t n3, magma_int_t n4 );

magma_int_t lapackf77_ieeeck( magma_int_t* ispec, float* zero, float* one );

long   lapackf77_lsame(  const char *ca, const char *cb );

// NOT portable! Assumes particular Fortran string convention (Intel/gfortran).
void   lapackf77_xerbla( const char *name, magma_int_t *info, magma_int_t name_len );

float  lapackf77_slamch( const char *cmach );
double lapackf77_dlamch( const char *cmach );

// "small" (lowercase) defined as char on Windows (reported by MathWorks)
void   lapackf77_slabad( float  *Small, float  *large );
void   lapackf77_dlabad( double *Small, double *large );

void   lapackf77_zcgesv( const magma_int_t *n, const magma_int_t *nrhs,
                         magmaDoubleComplex *A, const magma_int_t *lda,
                         magma_int_t *ipiv,
                         const magmaDoubleComplex *B, const magma_int_t *ldb,
                               magmaDoubleComplex *X, const magma_int_t *ldx,
                         magmaDoubleComplex *work, magmaFloatComplex *swork, double *rwork,
                         magma_int_t *iter,
                         magma_int_t *info );

void   lapackf77_dsgesv( const magma_int_t *n, const magma_int_t *nrhs,
                         double *A, const magma_int_t *lda,
                         magma_int_t *ipiv,
                         const double *B, const magma_int_t *ldb,
                               double *X, const magma_int_t *ldx,
                         double *work, float *swork,
                         magma_int_t *iter,
                         magma_int_t *info );

void   lapackf77_dsterf( const magma_int_t *n,
                         double *d, double *e,
                         magma_int_t *info );

void   lapackf77_ssterf( const magma_int_t *n,
                         float *d, float *e,
                         magma_int_t *info );

// precision conversion, general matrix
void   lapackf77_zlag2c( magma_int_t *m, magma_int_t *n,
                         const magmaDoubleComplex *A,  magma_int_t *lda,
                               magmaFloatComplex  *SA, magma_int_t *ldsa,
                         magma_int_t *info );

void   lapackf77_clag2z( magma_int_t *m, magma_int_t *n,
                         const magmaFloatComplex  *SA, magma_int_t *ldsa,
                               magmaDoubleComplex *A,  magma_int_t *lda,
                         magma_int_t *info );

void   lapackf77_dlag2s( magma_int_t *m, magma_int_t *n,
                         const double *A,  magma_int_t *lda,
                               float  *SA, magma_int_t *ldsa,
                         magma_int_t *info );

void   lapackf77_slag2d( magma_int_t *m, magma_int_t *n,
                         const float  *SA, magma_int_t *ldsa,
                               double *A,  magma_int_t *lda,
                         magma_int_t *info );

// precision conversion, triangular (or symmetric) matrix
void   lapackf77_zlat2c( const char* uplo, magma_int_t *n,
                         const magmaDoubleComplex *A,  magma_int_t *lda,
                               magmaFloatComplex  *SA, magma_int_t *ldsa,
                         magma_int_t *info );

void   lapackf77_dlat2s( const char* uplo, magma_int_t *n,
                         const double *A,  magma_int_t *lda,
                               float  *SA, magma_int_t *ldsa,
                         magma_int_t *info );

// not implemented in LAPACK
//void lapackf77_clat2z(const char* uplo, magma_int_t *n,
//                       const magmaFloatComplex  *SA, magma_int_t *ldsa,
//                             magmaDoubleComplex *A,  magma_int_t *lda,
//                       magma_int_t *info );
//
//void lapackf77_slat2d( const char* uplo, magma_int_t *n,
//                       const float  *SA, magma_int_t *ldsa,
//                             double *A,  magma_int_t *lda,
//                       magma_int_t *info );

double lapackf77_dlapy2( const double *x, const double *y );
float  lapackf77_slapy2( const float  *x, const float  *y );

#ifdef __cplusplus
}
#endif

#undef return_float

#endif /* MAGMA_LAPACK_H */
