/*
    -- MAGMA (version 0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2009
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <quark.h>

// includes, project
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cublas.h"
#include "magma.h"


int EN_BEE;

int TRACE;

float get_LU_error(int M, int N, float *A, int *lda, float *LU, int *IPIV){
int min_mn = min(M,N), intONE = 1, i, j;

slaswp_( &N, A, lda, &intONE, &min_mn, IPIV, &intONE);

float *L = (float *) calloc (M*min_mn, sizeof(float));
float *U = (float *) calloc (N*min_mn, sizeof(float));
float *work = (float *) calloc (M+1, sizeof(float));

for(j=0; j<min_mn; j++)
for(i=0; i<M; i++)
L[i+j*M] = (i > j  ? LU[i+j*(*lda)] : (i == j ? 1. : 0.));

for(j=0; j<N; j++)
for(i=0; i<min_mn; i++)
U[i+j*min_mn] = (i <= j ? LU[i+j*(*lda)] :  0.);

float matnorm = slange_("f", &M, &N, A, lda, work);
float alpha = 1., beta = 0.;

sgemm_("N", "N", &M, &N, &min_mn, &alpha, L, &M, U, &min_mn,  
&beta, LU, lda);

for( j = 0; j < N; j++ )
for( i = 0; i < M; i++ )
LU[i+j*(*lda)] = LU[i+j*(*lda)] - A[i+j*(*lda)];

float residual = slange_("f", &M, &N, LU, lda, work);

free(L);
free(work);

return residual / (matnorm * N);
}


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing sgetrf
*/
int main( int argc, char** argv) 
{
    ////cuInit( 0 );
    ////cublasInit( );
    ////printout_devices( );

    float *h_A, *h_A2, *h_R, *h_work;
    float *d_A;
    int *ipiv, *dipiv;
    float gpu_perf, cpu_perf, cpu2_perf;

    TimeStruct start, end;

    /* Matrix size */
    int N=0, n2, lda, M=0;
    int size[10] = {1024,2048,3072,4032,5184,6016,7040,8064,9088,10112};
    
    ////cublasStatus status;
    int i, j, info[1];

    EN_BEE = 128;

    TRACE = 0;

    int loop = argc;

    if (argc != 1){
      for(i = 1; i<argc; i++){      
        if (strcmp("-N", argv[i])==0)
          N = atoi(argv[++i]);
        else if (strcmp("-M", argv[i])==0)
          M = atoi(argv[++i]);
        else if (strcmp("-T", argv[i])==0)
          TRACE = atoi(argv[++i]);
        else if (strcmp("-B", argv[i])==0)
          EN_BEE = atoi(argv[++i]);
      }
      if ((M>0 && N>0) || (M==0 && N==0)) {
        printf("  testing_sgetrf_gpu -M %d -N %d -B %d -T %d\n\n", M, N, EN_BEE, TRACE);
        if (M==0 && N==0) {
          M = N = size[9];
          loop = 1;
        }
      } else {
        printf("\nUsage: \n");
        printf("  testing_sgetrf_gpu -M %d -N %d -B 128 -T 1\n\n", 1024, 1024);
        exit(1);
      }
    } else {
      printf("\nUsage: \n");
      printf("  testing_sgetrf_gpu -M %d -N %d -B 128 -T 1\n\n", 1024, 1024);
      M = N = size[9];
    }


    /* Initialize CUBLAS */
    ////status = cublasInit();
    ////if (status != CUBLAS_STATUS_SUCCESS) {
        ////fprintf (stderr, "!!!! CUBLAS initialization error\n");
    ////}

    n2 = M * N;

    int min_mn = min(M, N);

    /* Allocate host memory for the matrix */
    h_A = (float*)malloc(n2 * sizeof(h_A[0]));
    if (h_A == 0) {
        fprintf (stderr, "!!!! host memory allocation error (A)\n");
    }

    h_A2 = (float*)malloc(n2 * sizeof(h_A2[0]));
    if (h_A2 == 0) {
        fprintf (stderr, "!!!! host memory allocation error (A2)\n");
    }

    ipiv = (int*)malloc(min_mn * sizeof(int));
    if (ipiv == 0) {
      fprintf (stderr, "!!!! host memory allocation error (ipiv)\n");
    }

    ////status = cublasAlloc(size[9],sizeof(int), (void**)&dipiv);
    ////if (status != CUBLAS_STATUS_SUCCESS) {
      ////fprintf (stderr, "!!!! device memory allocation error (dipiv)\n");
    ////}
  
    ////cudaMallocHost( (void**)&h_R,  n2*sizeof(float) );
    ////if (h_R == 0) {
        ////fprintf (stderr, "!!!! host memory allocation error (R)\n");
    ////}

    ////int maxnb = magma_get_sgetrf_nb(size[9]);
    ////int lwork = size[9]*maxnb;
    ////status = cublasAlloc((size[9]+32)*(size[9]+32) + 32*maxnb + 
      ////lwork+2*maxnb*maxnb,sizeof(float), (void**)&d_A);
    ////if (status != CUBLAS_STATUS_SUCCESS) {
      ////fprintf (stderr, "!!!! device memory allocation error (d_A)\n");
    ////}

    ////cudaMallocHost( (void**)&h_work, (lwork+32*maxnb)*sizeof(float) );
    ////if (h_work == 0) {
      ////fprintf (stderr, "!!!! host memory allocation error (work)\n");
    ////}

    printf("\n\n");
    printf("  M    N   magma_sgetrf_mc GFlop/s     ||PA-LU|| / (||A||*N)\n");
    printf("========================================================\n");
    for(i=0; i<10; i++){

      if (loop == 1) {
        M = N = min_mn = size[i];
        n2 = M*N;
      }

      for(j = 0; j < n2; j++){
        h_A2[j] = h_A[j] = rand() / (float)RAND_MAX;
      }       

      ////cublasSetMatrix( N, N, sizeof(float), h_A, N, d_A, lda);
      ////magma_sgetrf_gpu2(&N, &N, d_A, &lda, ipiv, dipiv, h_work, info);

      /* =====================================================================
         Performs operation using LAPACK
         =================================================================== */
      start = get_current_time();
      //sgetrf_(&M, &N, h_A, &M, ipiv, info);
      end = get_current_time();

      //float error = get_LU_error(M, N, h_A2, &M, h_A, ipiv);

      if (info[0] < 0)
        printf("Argument %d of sgetrf had an illegal value.\n", -info[0]);

      cpu_perf = 2.*N*N*N/(3.*1000000*GetTimerValue(start,end));
      // printf("CPU Processing time: %f (ms) \n", GetTimerValue(start,end));
      
      //for(j=0; j<n2; j++)
        //h_A[j] = h_R[j];

      /* =====================================================================
         Performs operation using multi-core
         =================================================================== */
      start = get_current_time();
      magma_sgetrf_mc(&M, &N, h_A2, &M, ipiv, info);
      end = get_current_time();

      if (info[0] < 0)      
        printf("Argument %d of magma_sgeqrf_mc had an illegal value.\n", -info[0]);

      cpu2_perf = 2.*N*N*N/(3.*1000000*GetTimerValue(start,end));
  
      /* ====================================================================
         Performs operation using MAGMA
      =================================================================== */
      ////cublasSetMatrix( N, N, sizeof(float), h_A, N, d_A, lda);
      ////start = get_current_time();
      ////magma_sgetrf_gpu2(&N, &N, d_A, &lda, ipiv, dipiv, h_work, info);
      ////end = get_current_time();
      ////cublasGetMatrix( N, N, sizeof(float), d_A, lda, h_R, N);

      ////gpu_perf = 2.*N*N*N/(3.*1000000*GetTimerValue(start,end));
      // printf("GPU Processing time: %f (ms) \n", GetTimerValue(start,end));
     
      /* =====================================================================
      Check the factorization
      =================================================================== */
  
      float error = get_LU_error(M, N, h_A, &M, h_A2, ipiv);

      printf("%5d %5d       %6.2f                  %e\n",
             M, N, cpu2_perf, error);

      if (loop != 1)
        break;
    }

    /* Memory clean up */
    free(h_A);
    free(h_A2);
    free(ipiv);
    //cublasFree(h_work);
    //cublasFree(h_R);
    //cublasFree(d_A);
    //cublasFree(dipiv);

    /* Shutdown */
    //status = cublasShutdown();
    //if (status != CUBLAS_STATUS_SUCCESS) {
        //fprintf (stderr, "!!!! shutdown error (A)\n");
    //}
}
