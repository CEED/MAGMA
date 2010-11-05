#ifndef MAGMA_LAPACK_H
#define MAGMA_LAPACK_H

#ifdef __cplusplus
extern "C" {
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- LAPACK Externs used in MAGMA
*/
void strtri_(char *, char *, int *, float *, int *, int *);
void strsm_(char *, char *, char *, char *,
	    int *, int *, float *, float *, int *, float *, int*);
void sgemm_(char *, char *, int *, int *, int *, float *,
	    float *, int *, float *, int *, float *, float *, int *);
int sgemv_(char *, int *, int *, float *, float *, int *,
	   float *, int *, float *, float *, int *);
void saxpy_(int *, float *, float *, int *, float *, int *);
int scopy_(int *, float *, int *, float *, int *);
int strmv_(char*,char*,char*,int *,float *, int *, float *, int *);
int slarfg_(int *, float *, float *x, int *, float *);
int sscal_(int *, float *, float *, int *);
void ssyrk_(char *, char *, int *, int *, float *, float *,
	    int *, float *, float *, int *);
int strmm_(char *, char *, char *, char *,
	   int *, int *, float *, float *, int *, float *, int *);
int slaswp_(int *, float *, int *, int *, int *, int *, int *);
int ssymv_(char *, int *, float *, float *, int *, 
	   float *, int *, float *, float *, int *);

float snrm2_(int *, float *, int *);
float slange_(char *norm, int *, int *, float *, int *, float *);
float sdot_(int *, float *, int *, float *, int *);

int sgehd2_(int*, int*, int*, float*, int*, float*, float*, int*);
int spotrf_(char *uplo, int *n, float *a, int *lda, int *info);
int spotf2_(char *, int *, float *, int *, int *);
int sgeqrf_(int*,int*,float *,int*,float *,float *,int *,int *);
int sgeqr2_(int*,int*,float *,int*,float *,float *,int *);
int sgeqlf_(int*,int*,float *,int*,float *,float *,int *,int *);
int sgelqf_(int*,int*,float *,int*,float *,float *,int *,int *);
int sgelq2_(int*,int*,float *,int*,float *,float *,int *);
int sgeql2_(int*,int*,float *,int*,float *,float *,int *);
int sgehrd_(int *, int *, int *, float *, int *,
	    float *, float *, int *, int *);
int ssytrd_(char *, int *, float *, int *, float *, float *, 
	    float *, float *, int *, int *);
int sgebrd_(int *, int *, float *, int *, float *, float *, float *, 
	    float *, float *, int *, int *);

int slarft_(char *, char *, int *, int *, float *, int *, float *,
	    float *, int *);
int slarfb_(char *, char *, char *, char *, int *, int *, int *, float *, 
	    int *, float *, int *, float *, int *, float *, int *);
int sgetrf_(int *, int *, float *, int *, int *, int *);

int slaset_(char *,int *,int *,float *,float *,float *a,int *);
float slamch_(char *);
float slansy_(char *, char *, int *, float *, int *, float *);
int slacpy_(char *, int *, int *, float *, int *, float *, int *);
int sorgqr_(int *, int *, int *, float *, int *, float *, 
	    float *, int *, int *);
int sormqr_(char *, char *, int *, int *, int *, float *, int *,
	    float *, float *, int *, float *, int *, int *);

void ctrtri_(char *, char *, int *, double2 *, int *, int *);
void ctrsm_(char *, char *, char *, char *,
	    int *, int *, float2 *, float2 *, int *, float2 *,int*);
int ctrmm_(char *, char *, char *, char *,
	   int *, int *, float2 *, float2 *, int *, float2 *,int *);
void caxpy_(int *, float2 *, float2 *, int *, float2 *, int *);
void csyrk_(char *, char *, int *, int *, float2 *,
	    float2 *, int *, float2 *, float2 *, int *);
void cherk_(char *, char *, int *, int *, float *,
	    float2 *, int *, float *, float2 *, int *);
int cpotrf_(char *uplo, int *n, float2 *a, int *lda, int *info);
int cgeqrf_(int*,int*,float2 *,int*,float2 *,float2 *,int *,int *);
int clarft_(char *, char *, int *, int *, float2 *, int *, float2 *,
	    float2 *, int *);
int cgetrf_(int *, int *, float2 *, int *, int *, int *);
int claswp_(int *, float2 *, int *, int *, int *, int *, int *);
float clange_(char *norm, int *, int *, float2 *, int *, float *);

void dtrtri_(char *, char *, int *, double *, int *, int *);
void dtrsm_(char *, char *, char *, char *,
	    int *, int *, double *, double *, int *, double *,int*);
void dgemm_(char *, char *, int *, int *, int *, double *,
	    double *, int *, double *, int *, double *,double *, int *);
int dgemv_(char *, int *, int *, double *, double *, int *,
	   double *, int *, double *, double *, int *);
void daxpy_(int *, double *, double *, int *, double *, int *);
int dcopy_(int *, double *, int *, double *, int *);
int dtrmv_(char*,char*,char*,int *,double*,int*,double*,int*);
int dlarfg_(int *, double *, double *x, int *, double *);
int dscal_(int *, double *, double *, int *);
void dsyrk_(char *, char *, int *, int *, double *, double *,
	    int *, double *, double *, int *);
int dtrmm_(char *, char *, char *, char *, int *, int *, 
	   double *, double *, int *, double *, int *);
int dlaswp_(int *, double *, int *, int *, int *, int *, int *);
int dsymv_(char *, int *, double *, double *, int *,
	   double *, int *, double *, double *, int *);

double dnrm2_(int *, double *, int *);
double dlange_(char *norm, int *, int *, double *, int *, double *);
double ddot_(int *, double *, int *, double *, int *);

int dgehd2_(int*,int*,int*,double*,int*,double*,double*,int*);
int dpotrf_(char *uplo, int *n, double *a, int *lda, int *info);
int dpotf2_(char *, int *, double *, int *, int *);
int dgeqrf_(int*,int*,double *,int*,double *,double *,int *,int *);
int dgeqlf_(int*,int*,double *,int*,double *,double *,int *,int *);
int dgelqf_(int*,int*,double *,int*,double *,double *,int *,int *);
int dgelq2_(int*,int*,double *,int*,double *,double *,int *);
int dgeql2_(int*,int*,double *,int*,double *,double *,int *);
int dgetrf_(int *, int *, double *, int *, int *, int *);
int dgehrd_(int *, int *, int *, double *, int *, 
	    double *, double *, int *, int *);
int dsytrd_(char *, int *, double *, int *, double *, double *,
	    double *, double *, int *, int *);
int dgebrd_(int *, int *, double *, int *, double *, double *, double *, 
	    double *, double *, int *, int *);

int dlarft_(char *, char *, int *, int *, double *, int *, double *,
	    double *, int *);
int dlarfb_(char *, char *, char *, char *, int *, int *, int *, double *, 
	    int *, double *, int *, double *, int *, double *, int *);
int dlaset_(char *,int *,int *,double *,double *,double *a,int *);
double dlamch_(char *);
double dlansy_(char *, char *, int *, double *, int *, double *);
int dlacpy_(char *, int *, int *, double *, int *, double *, int *);
int dorgqr_(int *, int *, int *, double *, int *, double *, 
	    double *, int *, int *);
int dormqr_(char *, char *, int *, int *, int *, double *, int *,
	    double *, double *, int *, double *, int *, int *);

long int lsame_(char *, char *);

int zpotrf_(char *uplo, int *n, double2 *a, int *lda, int *info);
int zgeqrf_(int*, int*, double2 *, int*, double2 *, double2 *,
	    int *, int *);
int zlarft_(char *, char *, int *, int *, double2 *, int *, 
	    double2 *, double2 *, int *);
int zgetrf_(int *, int *, double2 *, int *, int *, int *);
int zlaswp_(int *, double2 *, int *, int *, int *, int *, int *);
double zlange_(char *, int *, int *, double2 *, int *, double *);
int ztrmm_(char *, char *, char *, char *, int *, int *, 
	   double2 *, double2 *, int *, double2 *,int *);
void ztrtri_(char *, char *, int *, double2 *, int *, int *);
void ztrsm_(char *, char *, char *, char *, int *, int *, 
	    double2 *, double2 *, int *, double2 *,int*);
int dsgesv_( int *, int *, double *, int *, int *, double *, int *, 
	     double *, int *, double *, float *, int *, int *);
void zaxpy_(int *, double2 *, double2 *, int *, double2 *, int *);
void zherk_(char *, char *, int *, int *, double *,
	    double2 *, int *, double *, double2 *, int *);
double dsymm_(char *,char*,int *,int *,double *,double *,int *,
	      double *,int *,double *,double *,int *);

#ifdef __cplusplus
}
#endif

#endif /* MAGMA LAPACK */
