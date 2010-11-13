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
#    define blasf77_zdot   zdot_
#    define blasf77_zaxpy  zaxpy_
#    define blasf77_zcopy  zcopy_
#    define blasf77_zscal  zscal_
#    define blasf77_ztrmm  ztrmm_
#    define blasf77_ztrsm  ztrsm_
#    define blasf77_zgemv  zgemv_
#    define blasf77_zhemv  zhemv_
#    define blasf77_ztrmv  ztrmv_
#    define blasf77_zherk  zherk_
#    define blasf77_zsyr2k zsyr2k_
#    define blasf77_zher2k zher2k_
#    define blasf77_zgemm  zgemm_
     
#    define lapackf77_zlacpy zlacpy_
#    define lapackf77_zpotrf zpotrf_
#    define lapackf77_zgeqrf zgeqrf_
#    define lapackf77_zgeqlf zgeqlf_
#    define lapackf77_zgelqf zgelqf_
#    define lapackf77_zgehrd zgehrd_
#    define lapackf77_zgehd2 zgehd2_
#    define lapackf77_zgebrd zgebrd_
#    define lapackf77_zgebd2 zgebd2_
#    define lapackf77_zhetrd zhetrd_
#    define lapackf77_zlarft zlarft_
#    define lapackf77_zlarfb zlarfb_
#    define lapackf77_zgetrf zgetrf_
#    define lapackf77_zlaswp zlaswp_
#    define lapackf77_zlange zlange_
#    define lapackf77_ztrtri ztrtri_
#    define lapackf77_zlarfg zlarfg_
#    define lapackf77_zunmqr zunmqr_
#    define lapackf77_zunmlq zunmlq_
#    define lapackf77_zunmql zunmql_
#    define lapackf77_zunmr2 zunmr2_
#    define lapackf77_zung2r zung2r_
#    define lapackf77_zhetd2 zhetd2_

#elif defined(NOCHANGE)

#    define blasf77_zdot zdot
#    define blasf77_zaxpy zaxpy
#    define blasf77_zcopy zcopy
#    define blasf77_zscal zscal
#    define blasf77_ztrmm ztrmm
#    define blasf77_ztrsm ztrsm
#    define blasf77_zgemv zgemv
#    define blasf77_zhemv zhemv
#    define blasf77_ztrmv ztrmv
#    define blasf77_zherk zherk
#    define blasf77_zsyr2k zsyr2k
#    define blasf77_zher2k zher2k
#    define blasf77_zgemm zgemm
     
#    define lapackf77_zlacpy zlacpy
#    define lapackf77_zpotrf zpotrf
#    define lapackf77_zgeqrf zgeqrf
#    define lapackf77_zgeqlf zgeqlf
#    define lapackf77_zgelqf zgelqf
#    define lapackf77_zgehrd zgehrd
#    define lapackf77_zgehd2 zgehd2
#    define lapackf77_zgebrd zgebrd
#    define lapackf77_zgebd2 zgebd2
#    define lapackf77_zhetrd zhetrd
#    define lapackf77_zlarft zlarft
#    define lapackf77_zlarfb zlarfb
#    define lapackf77_zgetrf zgetrf
#    define lapackf77_zlaswp zlaswp
#    define lapackf77_zlange zlange
#    define lapackf77_ztrtri ztrtri
#    define lapackf77_zlarfg zlarfg
#    define lapackf77_zunmqr zunmqr
#    define lapackf77_zunmlq zunmlq
#    define lapackf77_zunmql zunmql
#    define lapackf77_zunmr2 zunmr2
#    define lapackf77_zung2r zung2r
#    define lapackf77_zhetd2 zhetd2

#endif

double2 blasf77_zdot(  const int *, double2 *, const int *, double2 *, const int *);
void    blasf77_zaxpy( const int *, double2 *, double2 *, const int *, double2 *, const int *);
void    blasf77_zcopy( const int *, double2 *, const int *, double2 *, const int *);
void    blasf77_zscal( const int *, double2 *, double2 *, const int *);
void    blasf77_ztrmm( const char *, const char *, const char *, const char *, const int *, const int *, double2 *, double2 *, const int *, double2 *,const int *);
void    blasf77_ztrsm( const char *, const char *, const char *, const char *, const int *, const int *, double2 *, double2 *, const int *, double2 *,const int*);
void    blasf77_zgemv( const char *, const int  *, const int *, double2 *, double2 *, const int *, double2 *, const int *, double2 *, double2 *, const int *);
void    blasf77_zhemv( const char *, const int  *, double2 *, double2 *, const int *, double2 *, const int *, double2 *, double2 *, const int *);
void    blasf77_ztrmv( const char *, const char *, const char*, const int *, double2*,  const int *, double2 *, const int*);
void    blasf77_zherk( const char *, const char *, const int *, const int *, double  *, double2 *, const int *, double *, double2 *, const int *);
void    blasf77_zsyr2k(const char *, const char *, const int *, const int *, double2 *, double2 *, const int *, double2 *, const int *, double2 *, double2 *, const int *);
void    blasf77_zher2k(const char *, const char *, const int *, const int *, double2 *, double2 *, const int *, double2 *, const int *, double *,  double2 *, const int *);
void    blasf77_zgemm( const char *, const char *, const int *, const int *, const int *, double2 *, double2 *, const int *, double2 *, const int *, double2 *,double2 *, const int *);

int    lapackf77_zlacpy(const char *, int *, int *, double2 *, int *, double2 *, int *);
int    lapackf77_zpotrf(const char *uplo, int *n, double2 *a, int *lda, int *info);
int    lapackf77_zgeqrf(int*, int*, double2 *, int*, double2 *, double2 *, int *, int *);
int    lapackf77_zgeqlf(int*,int*,double2 *,int*,double2 *,double2 *,int *,int *);
int    lapackf77_zgelqf(int*,int*,double2 *,int*,double2 *,double2 *,int *,int *);
int    lapackf77_zgehrd(int *, int *, int *, double2 *, int *, double2 *, double2 *, int *, int *);
int    lapackf77_zgehd2(int*,int*,int*,double2*,int*,double2*,double2*,int*);
int    lapackf77_zgebrd(int *, int *, double2 *, int *, double *, double *, double2 *, double2 *, double2 *, int *, int *);
int    lapackf77_zgebd2(int *, int *, double2 *, int *, double2 *, double2 *, double2 *, double2 *, double2 *, int *);
int    lapackf77_zhetrd(const char *, int *, double2 *, int *, double2 *, double2 *, double2 *, double2 *, int *, int *);
int    lapackf77_zlarft(const char *, const char *, int *, int *, double2 *, int *, double2 *, double2 *, int *);
int    lapackf77_zlarfb(const char *, const char *, const char *, const char *, int *, int *, int *, double2 *, int *, double2 *, int *, double2 *, int *, double2 *, int *);
int    lapackf77_zgetrf(int *, int *, double2 *, int *, int *, int *);
int    lapackf77_zlaswp(int *, double2 *, int *, int *, int *, int *, int *);
double lapackf77_zlange(const char *, int *, int *, double2 *, int *, double2 *);
void   lapackf77_ztrtri(const char *, const char *, int *, double2 *, int *, int *);
void   lapackf77_zlarfg(int *, double2 *, double2 *x, int *, double2 *);
int    lapackf77_zunmqr(const char *, const char *, int *, int *, int *, double2 *, int *, double2 *, double2 *, int *, double2 *, int *, int *);
int    lapackf77_zunmlq(const char *, const char *, int *, int *, int *, double2 *, int *, double2 *, double2 *, int *, double2 *, int *, int *);
int    lapackf77_zunmql(const char *, const char *, int *, int *, int *, double2 *, int *, double2 *, double2 *, int *, double2 *, int *, int *);
int    lapackf77_zunm2r(const char *, const char *, int *, int *, int *, double2 *, int *, double2 *, double2 *, int *, double2 *, int *);
int    lapackf77_zung2r(int*, int*, int*, double2*, int*, double2*, double2*, int*);
int    lapackf77_zhetd2(const char *, int *, double2 *, int *, double2 *, double2 *, double2 *, int *);

#ifdef __cplusplus
}
#endif

#endif /* MAGMA ZLAPACK */
