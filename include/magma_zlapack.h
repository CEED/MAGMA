/*
 *   -- MAGMA (version 1.0) --
 *      Univ. of Tennessee, Knoxville
 *      Univ. of California, Berkeley
 *      Univ. of Colorado, Denver
 *      November 2010
 *
 * @precisions normal z -> s d c
 */

#ifndef MAGMA_ZLAPACK_H
#define MAGMA_ZLAPACK_H

#ifdef __cplusplus
extern "C" {
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- LAPACK Externs used in MAGMA
*/
#if defined(ADD_)
#    define blasf77_zdot zdot_
#    define blasf77_ztrmm ztrmm_
#    define blasf77_ztrsm ztrsm_
#    define blasf77_zgemv zgemv_
#    define blasf77_zsymv zsymv_
#    define blasf77_zaxpy zaxpy_
#    define blasf77_zcopy zcopy_
#    define blasf77_ztrmv ztrmv_
#    define blasf77_zherk zherk_
#    define blasf77_zscal zscal_
     
#    define lapackf77_zpotrf zpotrf_
#    define lapackf77_zgeqrf zgeqrf_
#    define lapackf77_zgeqlf zgeqlf_
#    define lapackf77_zgelqf zgelqf_
#    define lapackf77_zgehd2 zgehd2_
#    define lapackf77_zsytrd zsytrd_
#    define lapackf77_zlarft zlarft_
#    define lapackf77_zlarfb zlarfb_
#    define lapackf77_zgetrf zgetrf_
#    define lapackf77_zlaswp zlaswp_
#    define lapackf77_zlange zlange_
#    define lapackf77_ztrtri ztrtri_
#    define lapackf77_zlarfg zlarfg_
#    define lapackf77_zunmqr zunmqr_

#elif defined(NOCHANGE)

#    define blasf77_zdot zdot
#    define blas77_zaxpy zaxpy
#    define blas77_zcopy zcopy
#    define blas77_zscal zscal
#    define blas77_ztrmm ztrmm
#    define blas77_ztrsm ztrsm
#    define blas77_zgemv zgemv
#    define blas77_zsymv zsymv
#    define blas77_ztrmv ztrmv
#    define blas77_zherk zherk
     
#    define lapackf77_zpotrf zpotrf
#    define lapackf77_zgeqrf zgeqrf
#    define lapackf77_zgeqlf zgeqlf
#    define lapackf77_zgelqf zgelqf
#    define lapackf77_zgehd2 zgehd2
#    define lapackf77_zsytrd zsytrd
#    define lapackf77_zlarft zlarft
#    define lapackf77_zlarfb zlarfb
#    define lapackf77_zgetrf zgetrf
#    define lapackf77_zlaswp zlaswp
#    define lapackf77_zlange zlange
#    define lapackf77_ztrtri ztrtri
#    define lapackf77_zlarfg zlarfg
#    define lapackf77_zunmqr zunmqr
#endif

double2 blasf77_zdot(int *, double2 *, int *, double2 *, int *);
void    blasf77_zaxpy(int *, double2 *, double2 *, int *, double2 *, int *);
void    blasf77_zcopy(int *, double2 *, int *, double2 *, int *);
void    blasf77_zscal(int *, double2 *, double2 *, int *);
int     blasf77_ztrmm(char *, char *, char *, char *, int *, int *, double2 *, double2 *, int *, double2 *,int *);
void    blasf77_ztrsm(char *, char *, char *, char *, int *, int *, double2 *, double2 *, int *, double2 *,int*);
int     blasf77_zgemv(char *, int *, int *, double2 *, double2 *, int *, double2 *, int *, double2 *, double2 *, int *);
int     blasf77_zsymv(char *, int *, double2 *, double2 *, int *, double2 *, int *, double2 *, double2 *, int *);
void    blasf77_ztrmv(char*,char*,char*,int *,double2*,int*,double2*,int*);
void    blasf77_zherk(char *, char *, int *, int *, double *, double2 *, int *, double *, double2 *, int *);

int    lapackf77_zpotrf(char *uplo, int *n, double2 *a, int *lda, int *info);
int    lapackf77_zgeqrf(int*, int*, double2 *, int*, double2 *, double2 *, int *, int *);
int    lapackf77_zgeqlf(int*,int*,double2 *,int*,double2 *,double2 *,int *,int *);
int    lapackf77_zgelqf(int*,int*,double2 *,int*,double2 *,double2 *,int *,int *);
int    lapackf77_zgehd2(int*,int*,int*,double2*,int*,double2*,double2*,int*);
int    lapackf77_zsytrd(char *, int *, double2 *, int *, double2 *, double2 *, double2 *, double2 *, int *, int *);
int    lapackf77_zlarft(char *, char *, int *, int *, double2 *, int *, double2 *, double2 *, int *);
int    lapackf77_zlarfb(char *, char *, char *, char *, int *, int *, int *, double2 *, int *, double2 *, int *, double2 *, int *, double2 *, int *);
int    lapackf77_zgetrf(int *, int *, double2 *, int *, int *, int *);
int    lapackf77_zlaswp(int *, double2 *, int *, int *, int *, int *, int *);
double lapackf77_zlange(char *, int *, int *, double2 *, int *, double2 *);
void   lapackf77_ztrtri(char *, char *, int *, double2 *, int *, int *);
void   lapackf77_zlarfg(int *, double2 *, double2 *x, int *, double2 *);
int    lapackf77_zunmqr(char *, char *, int *, int *, int *, double2 *, int *, double2 *, double2 *, int *, double2 *, int *, int *);

#ifdef __cplusplus
}
#endif

#endif /* MAGMA ZLAPACK */
