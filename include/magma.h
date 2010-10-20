/*
    -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010
*/

#ifndef _MAGMA_
#define _MAGMA_

#include "auxiliary.h"
#include "magmablas.h"
#include "magma_lapack.h"

typedef int magma_int_t;

#include "magma_z.h"
#include "magma_c.h"
#include "magma_d.h"
#include "magma_s.h"


#ifdef __cplusplus
extern "C" {
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA function definitions
*/


int magma_slarfb(char, char, int, int, int *, float *, int *, float *,
                 int *, float *, int *, float *, int *);
int magma_sgeqrf(int *, int *, float *, int  *,  float  *,
		 float *, int *, float *, int *);
int magma_sgeqrf2(int *, int *, float *, int  *,  float  *,
		  float *, int *, int *);
int magma_sgeqrf_gpu(int *, int *, float *, int  *,  float  *,
		     float *, int *, float *, int *);
int magma_sgeqrf_gpu2(int *, int *, float *, int  *,  float  *,
		      float *, int *, float *, int *);
int magma_sgeqrf_gpu3(int *, int *, float *, int  *,  float  *,
		      float *, int *, float *, int *);
int magma_sgeqrs_gpu(int *, int *, int *, float *, int *, float *, float *,
                     int *, float *, int *, float *, int *);
int magma_sgeqlf(int *, int *, float *, int  *,  float  *,
                 float *, int *, float *, int *);
int magma_sgelqf(int *, int *, float *, int  *,  float  *,
                 float *, int *, float *, int *);
int magma_sgelqf2(int *, int *, float *, int  *,  float  *,
		  float *, int *, float *, int *);

int magma_sgetrs_gpu(char *, int, int, float *, int,
		     int *, float *, int, int *, float *);
int magma_sgehrd(int *, int *, int *, float *, int *, float *, float *,
                 int *, float *, int *);
int magma_slahr2(int *, int *, int *, float *, float *, float *, int *,
                 float *, float *, int *, float *, int *);
int magma_slahru(int, int, int,  float *, int,
                 float *, float *, float *, float *, float *);
int magma_ssytrd(char *, int *, float *, int *, float *, float *, 
		 float *, float *, int *, float *, int *);
int magma_sgebrd(int *, int *, float *, int *, float *, float *, float *, 
		 float *, float *, int *, float *, int *);
int magma_slabrd(int *, int *, int *, float *, int *, float *, float *, 
		 float *, float *, float *, int *, float *, int *,
		 float *, int *, float *, int *, float *, int *);


int magma_dlarfb(char, char, int, int, int *, double *, int *, double *,
                 int *, double *, int *, double *, int *);
int magma_dgeqrf(int *, int *, double *, int  *,  double  *,
		 double *, int *, double *, int *);
int magma_dgeqrf_gpu(int *, int *, double *, int  *, double  *,
		     double *, int *, double *, int *);
int magma_dgeqrf_gpu2(int *, int *, double *, int  *, double  *,
                      double *, int *, double *, int *);
int magma_dgeqrs_gpu(int *, int *, int *, double *, int *, double *, double *,
                     int *, double *, int *, double *, int *);
int magma_dgeqlf(int *, int *, double *, int  *, double  *,
		 double *, int *, double *, int *);
int magma_dgelqf(int *, int *, double *, int  *, double *,
		 double *, int *, double *, int *);
int magma_dgelqf2(int *, int *, double *, int  *, double *,
		  double *, int *, double *, int *);
int magma_dgetrs_gpu(char *, int, int, double *, int,
                     int *, double *, int, int *, double *);
int magma_dgehrd(int *, int *, int *, double *, int *, double *, double *,
		 int *, double *, int *);
int magma_dlahr2(int *, int *, int *, double *, double *, double *, int *,
                 double *, double *, int *, double *, int *);
int magma_dlahru(int, int, int,  double *, int,
		 double *, double *, double *, double *, double *);
int magma_dsytrd(char *, int *, double *, int *, double *, double *,
		 double *, double *, int *, double *, int *);

int magma_clarfb(char, char, int, int, int *, float2 *, int *, float2 *,
                 int *, float2 *, int *, float2 *, int *);
int magma_cgeqrf(int *, int *, float2 *, int  *,  float2  *,
                 float2 *, int *, float2 *, int *);
int magma_cgeqrf_gpu(int *, int *, float2 *, int  *,  float2  *,
		     float2 *, int *, float2 *, int *);

int magma_sdgetrs_gpu(int *n, int *nrhs, float *a, int *lda,
                  int *ipiv, float *x, double *b, int *ldb, int *info);
int magma_dsgesv_gpu(int, int, double *, int, int *, double *, int, double *,
		     int, double *, float *, int *, int *, float *, double *,
		     int *);
int magma_spotrs_gpu(char *, int, int, float *, int, float *, int, int *);
int magma_dpotrs_gpu(char *, int, int, double *, int, double *, int, int *);
int magma_dsposv_gpu(char, int, int, double *, int, double *, int, double *,
		     int, double *, float *, int *, int *, float *, double *);

void magma_xerbla(char *name , int *info);

int magma_dsgeqrsv_gpu(int, int, int, double *, int, double *, int, double *,
		       int, double *, float *, int *, int *, float *, int, 
		       float *, float *, double *, int, double *, double *);

#ifdef __cplusplus
}
#endif

#endif

