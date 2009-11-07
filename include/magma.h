/*
    -- MAGMA (version 0.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       June 2009
*/

#ifndef _MAGMA_
#define _MAGMA_

#include "auxiliary.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA function definitions
*/
int magma_spotrf(char *, int *, float *, int *, float *, int *);
int magma_spotrf2(char *, int *, float *, int *, float *, int *);
int magma_spotrf_gpu(char *, int *, float *, int *, float *, int *);
int magma_slarfb(char, char, int, int, int *, float *, int *, float *,
                 int *, float *, int *, float *, int *);
int magma_sgeqrf(int *, int *, float *, int  *,  float  *,
		 float *, int *, float *, int *);
int magma_sgeqrf_gpu(int *, int *, float *, int  *,  float  *,
		     float *, int *, float *, int *);
int magma_sgeqrf_gpu2(int *, int *, float *, int  *,  float  *,
		      float *, int *, float *, int *);
int magma_sgeqlf(int *, int *, float *, int  *,  float  *,
                 float *, int *, float *, int *);
int magma_sgelqf(int *, int *, float *, int  *,  float  *,
                 float *, int *, float *, int *);
int magma_sgelqf2(int *, int *, float *, int  *,  float  *,
		  float *, int *, float *, int *);
int magma_sgetrf(int *, int *, float *, int *, int *, float *, float *, int *);
int magma_sgetrf_gpu(int *, int *, float *, int *, int *, float *, int *);
int magma_sgetrf_gpu2(int *, int *, float *, int *, int *,int *,float*,int*);
int magma_sgehrd(int *, int *, int *, float *, int *, float *, float *,
                 int *, float *, int *);
int magma_slahr2(int *, int *, int *, float *, float *, float *, int *,
                 float *, float *, int *, float *, int *);
int magma_slahru(int, int, int,  float *, int,
                 float *, float *, float *, float *, float *);

int magma_dpotrf(char *, int *, double *, int *, double *, int *);
int magma_dpotrf_gpu(char *, int *, double *, int *, double *, int *);
int magma_dlarfb(int, int, int *, double *, int *, double *,
                 int *, double *, int *, double *, int *);
int magma_dgeqrf(int *, int *, double *, int  *,  double  *,
		 double *, int *, double *, int *);
int magma_dgeqrf_gpu(int *, int *, double *, int  *, double  *,
		     double *, int *, double *, int *);
int magma_dgetrf(int *, int *, double *, int *, int *, double*, double*, int*);
int magma_dgetrf_gpu(int *, int *, double *, int *, int *, double *, int *);
int magma_dgehrd(int *, int *, int *, double *, int *, double *, double *,
		 int *, double *, int *);
int magma_dlahr2(int *, int *, int *, double *, double *, double *, int *,
                 double *, double *, int *, double *, int *);
int magma_dlahru(int, int, int,  double *, int,
		 double *, double *, double *, double *, double *);

int magma_cpotrf(char *, int *, float2 *, int *, float2 *, int *);
int magma_cgetrf(int *, int *, float2 *, int *, int *, 
		 float2 *, float2 *, int *);
int magma_clarfb(char, char, int, int, int *, float2 *, int *, float2 *,
                 int *, float2 *, int *, float2 *, int *);
int magma_cgeqrf(int *, int *, float2 *, int  *,  float2  *,
                 float2 *, int *, float2 *, int *);

int magma_zgetrf(int *, int *, double2 *, int *, int *,
                 double2 *, double2 *, int *);

int magma_sdgetrs_gpu(int *n, int *nrhs, float *a, int *lda,
                  int *ipiv, float *x, double *b, int *ldb, int *info);
void magma_dgetrs_v2( char *TRANS , int N , int NRHS, double *A , int LDA , int *IPIV , double *B, int LDB, int *INFO, double *BB1 );
void magma_dsgesv(int N , int NRHS,double *A,int LDA ,int *IPIV,double *B,int LDB,double *X,int LDX,double *WORK, float *SWORK,int *ITER,int *INFO,float *h_work,double *h_work2,int *DIPIV );
void magma_spotrs_gpu( char *UPLO , int N , int NRHS, float *A , int LDA ,float *B, int LDB, int *INFO);
void magma_dpotrs_gpu( char *UPLO , int N , int NRHS, double *A , int LDA ,double *B, int LDB, int *INFO);
void magma_dsposv(char UPLO,int N ,int NRHS,double *A,int LDA ,double *B,int LDB,double *X,int LDX,double *WORK,float *SWORK,int *ITER,int *INFO,float *h_work,double *h_work2 );
/* ////////////////////////////////////////////////////////////////////////////
   -- LAPACK Externs used in MAGMA
*/
extern "C" void strtri_(char *, char *, int *, float *, int *, int *);
extern "C" void strsm_(char *, char *, char *, char *,
		       int *, int *, float *, float *, int *, float *, int*);
extern "C" void sgemm_(char *, char *, int *, int *, int *, float *,
		       float *, int *, float *, int *, float *,
		       float *, int *);
extern "C" int sgemv_(char *, int *, int *, float *, float *, int *,
                      float *, int *, float *, float *, int *);
extern "C" void saxpy_(int *, float *, float *, int *, float *, int *);
extern "C" int scopy_(int *, float *, int *, float *, int *);
extern "C" int strmv_(char*,char*,char*,int *,float *, int *, float *, int *);
extern "C" int slarfg_(int *, float *, float *x, int *, float *);
extern "C" int sscal_(int *, float *, float *, int *);
extern "C" void ssyrk_(char *, char *, int *, int *, float *, float *,
		       int *, float *, float *, int *);
extern "C" int strmm_(char *, char *, char *, char *,
                      int *, int *, float *, float *, int *, float *, int *);
extern "C" int slaswp_(int *, float *, int *, int *, int *, int *, int *);

extern "C" float snrm2_(const int, const float *, const int);
extern "C" float slange_(char *norm, int *, int *, float *, int *, float *);

extern "C" int sgehd2_(int*, int*, int*, float*, int*, float*, float*, int*);
extern "C" int spotrf_(char *uplo, int *n, float *a, int *lda, int *info);
extern "C" int spotf2_(char *, int *, float *, int *, int *);
extern "C" int sgeqrf_(int*,int*,float *,int*,float *,float *,int *,int *);
extern "C" int sgeqlf_(int*,int*,float *,int*,float *,float *,int *,int *);
extern "C" int sgelqf_(int*,int*,float *,int*,float *,float *,int *,int *);
extern "C" int sgelq2_(int*,int*,float *,int*,float *,float *,int *);
extern "C" int sgeql2_(int*,int*,float *,int*,float *,float *,int *);
extern "C" int sgehrd_(int *, int *, int *, float *, int *,
                       float *, float *, int *, int *);
extern "C" int slarft_(char *, char *, int *, int *, float *, int *, float *,
		       float *, int *);
extern "C" int slarfb_(char *, char *, char *, char *, int *, int *, int *, 
		       float *, int *, float *, int *, float *, int *, 
		       float *, int *);
extern "C" int sgetrf_(int *, int *, float *, int *, int *, int *);

extern "C" int slaset_(char *,int *,int *,float *,float *,float *a,int *);
extern "C" float slamch_(char *);
extern "C" float slansy_(char *, char *, int *, float *, int *, float *);
extern "C" int slacpy_(char *, int *, int *, float *, int *, float *, int *);
extern "C" int sorgqr_(int *, int *, int *, float *, int *, float *, 
		       float *, int *, int *);

extern "C" void ctrsm_(char *, char *, char *, char *,
                       int *, int *, float2 *, float2 *, int *, float2 *,int*);
extern "C" int ctrmm_(char *, char *, char *, char *,
                      int *, int *, float2 *, float2 *, int *, float2 *,int *);
extern "C" void caxpy_(int *, float2 *, float2 *, int *, float2 *, int *);
extern "C" void csyrk_(char *, char *, int *, int *, float2 *,
		       float2 *, int *, float2 *, float2 *, int *);
extern "C" void cherk_(char *, char *, int *, int *, float *,
		       float2 *, int *, float *, float2 *, int *);
extern "C" int cpotrf_(char *uplo, int *n, float2 *a, int *lda, int *info);
extern "C" int cgeqrf_(int*,int*,float2 *,int*,float2 *,float2 *,int *,int *);
extern "C" int clarft_(char *, char *, int *, int *, float2 *, int *, float2 *,
                       float2 *, int *);
extern "C" int cgetrf_(int *, int *, float2 *, int *, int *, int *);
extern "C" int claswp_(int *, float2 *, int *, int *, int *, int *, int *);
extern "C" float clange_(char *norm, int *, int *, float2 *, int *, float *);


extern "C" void dtrsm_(char *, char *, char *, char *,
		       int *, int *, double *, double *, int *, double *,int*);
extern "C" void dgemm_(char *, char *, int *, int *, int *, double *,
		       double *, int *, double *, int *, double *,
		       double *, int *);
extern "C" int dgemv_(char *, int *, int *, double *, double *, int *,
                      double *, int *, double *, double *, int *);
extern "C" void daxpy_(int *, double *, double *, int *, double *, int *);
extern "C" int dcopy_(int *, double *, int *, double *, int *);
extern "C" int dtrmv_(char*,char*,char*,int *,double*,int*,double*,int*);
extern "C" int dlarfg_(int *, double *, double *x, int *, double *);
extern "C" int dscal_(int *, double *, double *, int *);
extern "C" void dsyrk_(char *, char *, int *, int *, double *, double *,
		       int *, double *, double *, int *);
extern "C" int dtrmm_(char *, char *, char *, char *, int *, int *, 
		      double *, double *, int *, double *, int *);
extern "C" int dlaswp_(int *, double *, int *, int *, int *, int *, int *);

extern "C" double dnrm2_(int *, double *, int *);
extern "C" double dlange_(char *norm, int *, int *, double *, int *, double *);

extern "C" int dgehd2_(int*,int*,int*,double*,int*,double*,double*,int*);
extern "C" int dpotrf_(char *uplo, int *n, double *a, int *lda, int *info);
extern "C" int dpotf2_(char *, int *, double *, int *, int *);
extern "C" int dgeqrf_(int*,int*,double *,int*,double *,double *,int *,int *);
extern "C" int dgetrf_(int *, int *, double *, int *, int *, int *);
extern "C" int dgehrd_(int *, int *, int *, double *, int *, 
		       double *, double *, int *, int *);
extern "C" int dlarft_(char *, char *, int *, int *, double *, int *, double *,
		       double *, int *);
extern "C" int dlaset_(char *,int *,int *,double *,double *,double *a,int *);
extern "C" double dlamch_(char *);
extern "C" double dlansy_(char *, char *, int *, double *, int *, double *);
extern "C" int dlacpy_(char *, int *, int *, double *, int *, double *, int *);
extern "C" int dorgqr_(int *, int *, int *, double *, int *, double *, 
		       double *, int *, int *);

extern "C" long int lsame_(char *, char *);

extern "C" int zgetrf_(int *, int *, double2 *, int *, int *, int *);
extern "C" int zlaswp_(int *, double2 *, int *, int *, int *, int *, int *);
extern "C" double zlange_(char *, int *, int *, double2 *, int *, double *);
extern "C" int ztrmm_(char *, char *, char *, char *, int *, int *, 
		      double2 *, double2 *, int *, double2 *,int *);
extern "C" void ztrsm_(char *, char *, char *, char *, int *, int *, 
		       double2 *, double2 *, int *, double2 *,int*);
extern "C" int dsgesv_( int *, int *, double *, int *, int *, double *, int *, double *, int *, double *, float *, int *, int *);


// Remove these stuff -- rajib 
extern "C" int dsgesv_( int *, int *, double *, int *, int *, double *, int *, double *, int *, double *, float *, int *, int *);
extern "C" int dgesv_( int *, int *, double *, int *, int *, double *, int *,  int *);
extern "C" int sgesv_( int *, int *, float *, int *, int *, float *, int *,  int *);
extern "C" int dlag2s_( int *, int *, double *, int *,  float *, int *,  int *);
extern "C" int slag2d_( int *, int *, float *, int *,  double *, int *,  int *);
extern "C" int sgetrs_(char *, int *, int *, float *, int *, int *, float * ,int * ,  int *);
extern "C" int idamax_(int *, double *, int *);
extern "C" int dgetrs_(char *, int *, int *, double *, int *, int *, double * ,int * ,  int *);
extern "C" double dsymm_     (char *,char*,int *,int *,double *,double *,int *,double *,int *,double *,double *,int *);
//001:       SUBROUTINE DSYMM(SIDE,  UPLO, M,    N,    ALPHA,     A,      LDA,     B,     LDB,   BETA,   C,        LDC)


#endif
