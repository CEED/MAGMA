#ifndef MAGMA_LAPACK_H
#define MAGMA_LAPACK_H

#include "magma_zlapack.h"
#include "magma_clapack.h"
#include "magma_dlapack.h"
#include "magma_slapack.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(ADD_)

#    define lapackf77_lsame  lsame_
#    define lapackf77_slamch slamch_
#    define lapackf77_dlamch dlamch_
#    define lapackf77_slabad slabad_
#    define lapackf77_dlabad dlabad_
#    define lapackf77_zcgesv zcgesv_
#    define lapackf77_dsgesv dsgesv_

#    define lapackf77_zlag2c zlag2c_
#    define lapackf77_clag2z clag2z_
#    define lapackf77_dlag2s dlag2s_
#    define lapackf77_slag2d slag2d_

#elif defined(NOCHANGE)

#    define lapackf77_lsame  lsame
#    define lapackf77_slamch slamch
#    define lapackf77_dlamch dlamch
#    define lapackf77_slabad slabad
#    define lapackf77_dlabad dlabad
#    define lapackf77_zcgesv zcgesv
#    define lapackf77_dsgesv dsgesv

#    define lapackf77_zlag2c zlag2c
#    define lapackf77_clag2z clag2z
#    define lapackf77_dlag2s dlag2s
#    define lapackf77_slag2d slag2d

#endif

long int lapackf77_lsame( const char *ca, const char *cb);
float    lapackf77_slamch(const char *cmach);
double   lapackf77_dlamch(const char *cmach);
void     lapackf77_slabad(float  *small, float  *large);
void     lapackf77_dlabad(double *small, double *large);
void     lapackf77_zcgesv(int *n, int *nrhs, cuDoubleComplex *A, int *lda, int *IPIV, cuDoubleComplex *B, int *ldb, cuDoubleComplex *X, int *ldx, cuDoubleComplex *work, float2 *swork, double *rwork, int *iter, int *info);
void     lapackf77_dsgesv(int *n, int *nrhs, double  *A, int *lda, int *IPIV, double  *B, int *ldb, double  *X, int *ldx, double  *work, float  *swork,                int *iter, int *info);

void     lapackf77_zlag2c( int *m, int *n, cuDoubleComplex *a,  int *lda,  float2  *sa, int *ldsa, int *info );
void     lapackf77_clag2z( int *m, int *n, float2  *sa, int *ldsa, cuDoubleComplex *a,  int *lda,  int *info );
void     lapackf77_dlag2s( int *m, int *n, double  *a,  int *lda,  float   *sa, int *ldsa, int *info );
void     lapackf77_slag2d( int *m, int *n, float   *sa, int *ldsa, double  *a,  int *lda,  int *info );

#ifdef __cplusplus
}
#endif

#endif /* MAGMA LAPACK */
