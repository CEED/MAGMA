/*
    -- MAGMA (version 0.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       June 2009
*/

#ifndef _MAGMABLAS_
#define _MAGMABLAS_
#include "cublas.h"
#include "cuda.h"

extern "C" void magmablas_sinplace_transpose(float *, int, int);
extern "C" void magmablas_spermute_long(float *, int, int *, int, int);
extern "C" void magmablas_stranspose(float *, int, float *, int, int, int);

extern "C" void magmablas_dinplace_transpose(double *, int, int);
extern "C" void magmablas_dpermute_long(double *, int, int *, int, int);
extern "C" void magmablas_dtranspose(double *, int, double *, int, int, int);

extern "C" void magmablas_cinplace_transpose(float2 *, int, int);
extern "C" void magmablas_cpermute_long(float2 *, int, int *, int, int);
extern "C" void magmablas_ctranspose(float2 *, int, float2 *, int, int, int);

extern "C" void magmablas_zinplace_transpose(double2 *, int, int);
extern "C" void magmablas_zpermute_long(double2 *, int, int *, int, int);
extern "C" void magmablas_ztranspose(double2 *, int, double2 *, int, int, int);
extern "C" void magmablas_ztrsm(char, char, char, char, int, int, double2,
                                double2 *, int, double2 *, int);

extern "C" void magmablas_strsm(char, char, char, char,
				int, int, float*, int, float*, int);
extern "C" void magmablas_csyrk(char, char, int, int, float2,
				float2 *, int, float2, float2 *, int);
extern "C" void magmablas_cherk(char, char, int, int, float,
				float2 *, int, float, float2 *, int);
extern "C" void magmablas_ctrsm(char, char, char, char, int, int, float2,
				float2 *, int, float2 *, int);

extern "C" void magmablas_sgemv(int, int, float *, int, float *, float *);
extern "C" void magmablas_dgemv(int, int, double *, int, double *, double *);

extern "C" void magmablas_sdlaswp(int, double *, int, float *, int, int *);

extern "C" void magmablas_dtrsm (char side, char uplo, char tran, char diag, int M, int N,  double* A, int lda, double* b, int ldb);
/* ////////////////////////////////////////////////////////////////////////////
   -- Auxiliary CUDA routines
*/
void dzero_32x32_block(double *, int);
void dzero_nbxnb_block(int, double *, int);

void szero_32x32_block(float *, int);
void szero_nbxnb_block(int, float *, int);

/*
All the Iterative Refinement Stuffs Here
*/

extern "C" double  magma_dlange( char NORM, int M, int N , double *A, int LDA , double * WORK );
extern "C" void magma_dlag2s(int M, int N , const double *A, int lda, float *SA , int LDSA, float RMAX );
extern "C" void magmablas_sdaxpycp(float *R, double *X, int M, int ldr,int lda, double *B, double *W);
extern "C" void magmablas_magma_dgemv_MLU(int n, int m, double *A, int lda, double *x, double *z);
extern "C" void magmablas_slag2d(int M, int N, float *SA, int LDSA , double *A , int LDA, int *INFO);
extern "C" void magma_dlacpy(int M, int N, double *SA, int LDSA , double *A , int LDA);
#endif
