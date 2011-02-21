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

#    define lapackf77_xerbla xerbla_
#    define lapackf77_dsterf dsterf_
#    define lapackf77_ssterf ssterf_
#    define lapackf77_zstedc zstedc_
#    define lapackf77_cstedc cstedc_
#    define lapackf77_dstedc dstedc_
#    define lapackf77_sstedc sstedc_
#    define lapackf77_zunmtr zunmtr_
#    define lapackf77_cunmtr cunmtr_
#    define lapackf77_dormtr dormtr_
#    define lapackf77_sormtr sormtr_

#    define lapackf77_zlag2c zlag2c_
#    define lapackf77_clag2z clag2z_
#    define lapackf77_dlag2s dlag2s_
#    define lapackf77_slag2d slag2d_

#    define lapackf77_dlapy2 dlapy2_
#    define lapackf77_slapy2 slapy2_

#    define blasf77_ddot     ddot_
#    define blasf77_sdot     sdot_ 

#elif defined(NOCHANGE)

#    define lapackf77_lsame  lsame
#    define lapackf77_slamch slamch
#    define lapackf77_dlamch dlamch
#    define lapackf77_slabad slabad
#    define lapackf77_dlabad dlabad
#    define lapackf77_zcgesv zcgesv
#    define lapackf77_dsgesv dsgesv

#    define lapackf77_xerbla xerbla
#    define lapackf77_dsterf dsterf
#    define lapackf77_ssterf ssterf
#    define lapackf77_zstedc zstedc
#    define lapackf77_cstedc cstedc
#    define lapackf77_dstedc dstedc
#    define lapackf77_sstedc sstedc
#    define lapackf77_zunmtr zunmtr
#    define lapackf77_cunmtr cunmtr
#    define lapackf77_dormtr dormtr
#    define lapackf77_sormtr sormtr

#    define lapackf77_zlag2c zlag2c
#    define lapackf77_clag2z clag2z
#    define lapackf77_dlag2s dlag2s
#    define lapackf77_slag2d slag2d

#    define lapackf77_dlapy2 dlapy2
#    define lapackf77_slapy2 slapy2

#    define blasf77_ddot     ddot
#    define blasf77_sdot     sdot

#endif

long int lapackf77_lsame( const char *ca, const char *cb);
float    lapackf77_slamch(const char *cmach);
double   lapackf77_dlamch(const char *cmach);
void     lapackf77_slabad(float  *small, float  *large);
void     lapackf77_dlabad(double *small, double *large);
void     lapackf77_zcgesv(magma_int_t *n, magma_int_t *nrhs, cuDoubleComplex *A, magma_int_t *lda, magma_int_t *IPIV, cuDoubleComplex *B, magma_int_t *ldb, 
                          cuDoubleComplex *X, magma_int_t *ldx, cuDoubleComplex *work, cuFloatComplex *swork, double *rwork, magma_int_t *iter, magma_int_t *info);
void     lapackf77_dsgesv(magma_int_t *n, magma_int_t *nrhs, double          *A, magma_int_t *lda, magma_int_t *IPIV, double          *B, magma_int_t *ldb, 
                          double          *X, magma_int_t *ldx, double          *work, float          *swork,                magma_int_t *iter, magma_int_t *info);

int lapackf77_xerbla(char *, magma_int_t *);
int lapackf77_dsterf(magma_int_t *, double *, double *, magma_int_t *);
int lapackf77_ssterf(magma_int_t *, float *, float *, magma_int_t *);
int lapackf77_zstedc(char *, magma_int_t *, double *, double *, cuDoubleComplex *,
                     magma_int_t *, cuDoubleComplex *, magma_int_t *, double *, magma_int_t *,
                     magma_int_t *, magma_int_t *, magma_int_t *);
int lapackf77_cstedc(char *, magma_int_t *, float *, float *, cuFloatComplex *,
                     magma_int_t *, cuFloatComplex *, magma_int_t *, float *, magma_int_t *,
                     magma_int_t *, magma_int_t *, magma_int_t *);
int lapackf77_dstedc(char *, magma_int_t *, double *, double *, double *,
                     magma_int_t *, double *, magma_int_t *,
                      magma_int_t *, magma_int_t *, magma_int_t *);
int lapackf77_sstedc(char *, magma_int_t *, float *, float *, float *,
                     magma_int_t *, float *, magma_int_t *,
                     magma_int_t *, magma_int_t *, magma_int_t *);
int lapackf77_zunmtr(char *, char *, char *, magma_int_t *, magma_int_t *,
                     cuDoubleComplex *, magma_int_t *, cuDoubleComplex *,
                     cuDoubleComplex *, magma_int_t *,
                     cuDoubleComplex *, magma_int_t *, magma_int_t *);
int lapackf77_cunmtr(char *, char *, char *, magma_int_t *, magma_int_t *,
                     cuFloatComplex *, magma_int_t *, cuFloatComplex *,
                     cuFloatComplex *, magma_int_t *,
                     cuFloatComplex *, magma_int_t *, magma_int_t *);
int lapackf77_dormtr(char *, char *, char *, magma_int_t *, magma_int_t *,
            double *, magma_int_t *, double *,
            double *, magma_int_t *, 
            double *, magma_int_t *, magma_int_t *);
int lapackf77_sormtr(char *, char *, char *, magma_int_t *, magma_int_t *,
            float *, magma_int_t *, float *,
            float *, magma_int_t *, 
            float *, magma_int_t *, magma_int_t *);

void     lapackf77_zlag2c( magma_int_t *m, magma_int_t *n, cuDoubleComplex *a,  magma_int_t *lda,  cuFloatComplex  *sa, magma_int_t *ldsa, magma_int_t *info );
void     lapackf77_clag2z( magma_int_t *m, magma_int_t *n, cuFloatComplex  *sa, magma_int_t *ldsa, cuDoubleComplex *a,  magma_int_t *lda,  magma_int_t *info );
void     lapackf77_dlag2s( magma_int_t *m, magma_int_t *n, double          *a,  magma_int_t *lda,  float           *sa, magma_int_t *ldsa, magma_int_t *info );
void     lapackf77_slag2d( magma_int_t *m, magma_int_t *n, float           *sa, magma_int_t *ldsa, double          *a,  magma_int_t *lda,  magma_int_t *info );

double   lapackf77_dlapy2( double *x, double *y  );
float    lapackf77_slapy2( float  *x, float  *y  );

double   blasf77_ddot( magma_int_t *, double *, magma_int_t *, double *, magma_int_t *);
float    blasf77_sdot( magma_int_t *, float *,  magma_int_t *, float *,  magma_int_t *);

#ifdef __cplusplus
}
#endif

#endif /* MAGMA LAPACK */
