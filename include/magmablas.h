/*
    -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010
*/

#ifndef _MAGMABLAS_
#define _MAGMABLAS_

#include "cublas.h"
#include "cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

void magmablas_sinplace_transpose(float *, int, int);
void magmablas_spermute_long(float *, int, int *, int, int);
void magmablas_stranspose(float *, int, float *, int, int, int);

void magmablas_dinplace_transpose(double *, int, int);
void magmablas_dpermute_long(double *, int, int *, int, int);
void magmablas_dtranspose(double *, int, double *, int, int, int);

void magmablas_cinplace_transpose(float2 *, int, int);
void magmablas_cpermute_long(float2 *, int, int *, int, int);
void magmablas_ctranspose(float2 *, int, float2 *, int, int, int);

void magmablas_zinplace_transpose(double2 *, int, int);
void magmablas_zpermute_long(double2 *, int, int *, int, int);
void magmablas_ztranspose(double2 *, int, double2 *, int, int, int);
void magmablas_zherk(char, char, int, int, double,
                                double2 *, int, double, double2 *, int);
void magmablas_ztrsm(char, char, char, char, int, int, double2,
                                double2 *, int, double2 *, int);
void magmablas_ztrmm(char, char, char, char, int, int, double2,
                                double2 *, int, double2 *, int);

void magmablas_strsm(char, char, char, char,
				int, int, float, float*, int, float*, int);
void magmablas_csyrk(char, char, int, int, float2,
				float2 *, int, float2, float2 *, int);
void magmablas_cherk(char, char, int, int, float,
				float2 *, int, float, float2 *, int);
void magmablas_ctrsm(char, char, char, char, int, int, float2,
				float2 *, int, float2 *, int);
void magmablas_ctrmm(char, char, char, char, int, int, float2,
                                float2 *, int, float2 *, int);

void magmablas_sgemv(int, int, float *, int, float *, float *);
void magmablas_sgemvt(int,int,float,float *,int,float *,float *);
void magmablas_sgemvt1(int,int,float,float *,int,float *,float *);
void magmablas_sgemvt2(int,int,float,float *,int,float *,float *);
void magmablas_sgemv32(char, int, int, float, float *, int, float *, float *);
void magmablas_dgemv(int, int, double *, int, double *, double *);
void magmablas_dgemvt(int,int,double,double *,int,double *,double *);
void magmablas_dgemv32(char, int, double, double *, int, double *, double *);

void magmablas_sdlaswp(int, double *, int, float *, int, int *);

void magmablas_dtrsm(char, char, char, char, int, int, double, 
				double *, int, double* , int);
/* ////////////////////////////////////////////////////////////////////////////
   -- Auxiliary CUDA routines
*/
void dzero_32x32_block(double *, int);
void dzero_nbxnb_block(int, double *, int);

void szero_32x32_block(float *, int);
void szero_nbxnb_block(int, float *, int);

/* ////////////////////////////////////////////////////////////////////////////
   -- Iterative Refinement Kernels 
*/
double magma_dlange(char, int, int, double *, int, double *);
void magma_dlag2s(int, int, const double *, int, float * , int, float);
void magmablas_sdaxpycp(float *, double *, int, int, int, double *, double *);
void magmablas_magma_dgemv_MLU(int, int, double *, int, double *, double *);
void magmablas_slag2d(int, int, float *, int, double *, int, int *);
void magma_dlacpy(int, int, double *, int, double *, int);
double magma_dlansy(char, char , int, double *, int,  double *);
void magma_dlat2s(char, int, double *, int, float *, int, int *);
void magmablas_dsymv(char, int, double, double *, int, double *, int, 
		 double, double *, int);
void magmablas_ssymv(char, int, float, float *, int, float *, int, 
		 float, float *, int);
void magmablas_ssymv6(char, int, float, float *, int, float *, int,
		      float, float *, int, float *, int);
void magmablas_ssyr2k(char, char, int, int, float, const float *, int, 
                      const float *, int, float, float *, int);
void magmablas_dsyr2k(char, char, int, int, double, const double *, int,
			const double *, int, double, double *, int);

/*
Gemm Kernels
*/
void magmablas_dgemm_kernel_a_0(double *, const double *, const double *, 
				int, int, int, int, int, int, double, double);
void magmablas_dgemm_kernel_ab_0(double *, const double *, const double *, 
				 int, int, int, int, int, int, double, double);
void magmablas_dgemm_kernel_N_N_64_16_16_16_4_special(double *, const double *, 
                 const double *, int, int, int, int, int, int, double, double);
void magmablas_dgemm_kernel_N_N_64_16_16_16_4(double *, const double *, 
                 const double *, int, int, int, int, int, int, double, double);
void magmablas_dgemm_kernel_N_T_64_16_4_16_4(double *, const double *, 
                 const double *, int, int, int, int, int, int, double, double);
void magmablas_dgemm_kernel_T_N_32_32_8_8_8(double *, const double *, 
                 const double *, int, int, int, int, int, int, double, double);
void magmablas_dgemm_kernel_T_T_64_16_16_16_4_v2(double *, const double *, 
                 const double *, int, int, int, int, int, int, double, double);
void magmablas_dgemm_kernel_T_T_64_16_16_16_4(double *, const double *, 
                 const double *, int, int, int, int, int, int, double, double);

void magmablas_sgemm_kernel_a_0(float *, const float *, const float *, int, 
				int, int, int, int, int, float, float);
void magmablas_sgemm_kernel_ab_0(float *, const float *, const float *, int,
				 int, int, int, int, int, float, float);
void magmablas_sgemm_kernel_N_N_64_16_16_16_4_special(float *, const float *, 
		  const float *, int, int, int, int, int, int, float, float);
void magmablas_sgemm_kernel_N_N_64_16_16_16_4(float *, const float *, const float *, 
                  int, int, int, int, int, int, float, float);
void magmablas_sgemm_kernel_N_T_64_16_4_16_4(float *, const float *, const float *, 
                  int, int, int, int, int, int, float, float);
void magmablas_sgemm_kernel_T_N_32_32_8_8_8(float *, const float *, const float *, 
                  int, int, int, int, int, int, float, float);
void magmablas_sgemm_kernel_T_T_64_16_16_16_4_v2(float *, const float *, 
                  const float *, int, int, int, int, int, int, float, float);
void magmablas_sgemm_kernel_T_T_64_16_16_16_4(float *, const float *, const float *, 
                  int, int, int, int, int, int, float, float);

void magmablas_dgemm(char, char, int, int, int, double, const double *, int, 
		   const double *, int, double, double *, int);
void magmablas_sgemm(char, char, int, int, int, float, const float *, int, 
		   const float *, int, float, float *, int);

#ifdef __cplusplus
}
#endif

#endif
