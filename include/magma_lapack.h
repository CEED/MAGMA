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

#elif defined(NOCHANGE)

#    define lapackf77_lsame  lsame
#    define lapackf77_slamch slamch
#    define lapackf77_dlamch dlamch
#    define lapackf77_slabad slabad
#    define lapackf77_dlabad dlabad
#    define lapackf77_zcgesv zcgesv
#    define lapackf77_dsgesv dsgesv

#endif

long int lapackf77_lsame( const char *ca, const char *cb);
float    lapackf77_slamch(const char *cmach);
double   lapackf77_dlamch(const char *cmach);
void     lapackf77_slabad(float  *small, float  *large);
void     lapackf77_dlabad(double *small, double *large);
void     lapackf77_zcgesv(int *n, int *nrhs, double2 *A, int *lda, int *IPIV, double2 *B, int *ldb, double2 *X, int *ldx, double2 *work, float2 *swork, double *rwork, int *iter, int *info);
void     lapackf77_dsgesv(int *n, int *nrhs, double  *A, int *lda, int *IPIV, double  *B, int *ldb, double  *X, int *ldx, double  *work, float  *swork,                int *iter, int *info);

#ifdef __cplusplus
}
#endif

#endif /* MAGMA LAPACK */
