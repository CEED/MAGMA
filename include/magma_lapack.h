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
#    define lapackf77_zcgesv zcgesv_
#    define lapackf77_dsgesv dsgesv_

#elif defined(NOCHANGE)

#    define lapackf77_lsame  lsame
#    define lapackf77_slamch slamch
#    define lapackf77_dlamch dlamch
#    define lapackf77_zcgesv zcgesv
#    define lapackf77_dsgesv dsgesv

#endif

long int lapackf77_lsame(char *, char *);
float    lapackf77_slamch(char *);
double   lapackf77_dlamch(char *);
int      lapackf77_zcgesv(int *, int *, double2 *, int *, int *, double2 *, int *, double2 *, int *, double2 *, float2 *, int *, int *);
int      lapackf77_dsgesv(int *, int *, double *, int *, int *, double *, int *, double *, int *, double *, float *, int *, int *);

#ifdef __cplusplus
}
#endif

#endif /* MAGMA LAPACK */
