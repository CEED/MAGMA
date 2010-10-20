/*
    -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cublas.h"
#include "magma.h"


float get_LU_error(float *A, float *LU, int *IPIV, int N){
  int k1 = 1, k2 = N, intONE = 1, i, j;

  slaswp_( &N, A, &N, &k1, &k2, IPIV, &intONE);

  float *L = (float *) calloc (N*N, sizeof(float));
  float *work = (float *) calloc (N+1, sizeof(float));

  for( j = 0; j < N; j++ )
    for( i = 0; i < N; i++ )
      L[i+j*N] = (i > j ? LU[i+j*N] : (i == j ? 1. : 0.));

  for( j = 0; j < N; j++ )
    for( i = j+1; i < N; i++ )
      LU[i+j*N] = 0;

  float matnorm = slange_("f", &N, &N, A, &N, work);
  float alpha = 1.;

  strmm_("L", "L", "N", "N", &N, &N, &alpha, L, &N, LU, &N);

  for( j = 0; j < N*N; j++ )
    LU[j] = LU[j] - A[j];
  
  /*
  //int n1 = 898, n2 = 1025;
  int n1 = N, n2 = 1024;
  float residual = slange_("f", &n1, &n2, LU, &N, work);
  printf("%e\n", residual/(matnorm * N));

  n1 = 1024; n2 = N;
  residual = slange_("f", &n1, &n2, LU, &N, work);
  printf("%e\n", residual/(matnorm * N));

  n1 = 800, n2 = 800;
  residual = slange_("f", &n1, &n2, LU, &N, work);
  printf("%e\n", residual/(matnorm * N));

  n1 = 962, n2 = N;
  residual = slange_("f", &n1, &n2, LU, &N, work);
  printf("%e\n", residual/(matnorm * N));

  n1 = N, n2 = 800;
  residual = slange_("f", &n1, &n2, LU, &N, work);
  printf("%e\n", residual/(matnorm * N));


  residual = slange_("f", &N, &N, LU, &N, work);
  */
  float residual = slange_("f", &N, &N, LU, &N, work);

  free(L);
  free(work);
  
  return residual / (matnorm * N);
}


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing sgetrf
*/
int main( int argc, char** argv) 
{
    cuInit( 0 );
    cublasInit( );
    printout_devices( );

    float *h_A, *h_R, *h_work;
    float *d_A;
    int *ipiv, *dipiv;
    float gpu_perf, cpu_perf;

    TimeStruct start, end;

    /* Matrix size */
    int N=0, n2, lda;
    int size[10] = {1024,2048,3072,4032,5184,6016,7040,8064,9088,10112};
    
    cublasStatus status;
    int i, j, info[1];

    if (argc != 1){
      for(i = 1; i<argc; i++){	
	if (strcmp("-N", argv[i])==0)
	  N = atoi(argv[++i]);
      }
      if (N>0) size[0] = size[9] = N;
      else exit(1);
    }
    else {
      printf("\nUsage: \n");
      printf("  testing_sgetrf_gpu -N %d\n\n", 1024);
    }

    /* Initialize CUBLAS */
    status = cublasInit();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! CUBLAS initialization error\n");
    }

    lda = N;
    n2 = size[9] * size[9];

    /* Allocate host memory for the matrix */
    h_A = (float*)malloc(n2 * sizeof(h_A[0]));
    if (h_A == 0) {
        fprintf (stderr, "!!!! host memory allocation error (A)\n");
    }

    ipiv = (int*)malloc(size[9] * sizeof(int));
    if (ipiv == 0) {
      fprintf (stderr, "!!!! host memory allocation error (ipiv)\n");
    }

    status = cublasAlloc(size[9],sizeof(int), (void**)&dipiv);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (dipiv)\n");
    }
  
    cudaMallocHost( (void**)&h_R,  n2*sizeof(float) );
    if (h_R == 0) {
        fprintf (stderr, "!!!! host memory allocation error (R)\n");
    }

    int maxnb = magma_get_sgetrf_nb(size[9]);
    int lwork = size[9]*maxnb;
    status = cublasAlloc((size[9]+32)*(size[9]+32) + 32*maxnb + 
			 lwork+2*maxnb*maxnb,sizeof(float), (void**)&d_A);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (d_A)\n");
    }

    cudaMallocHost( (void**)&h_work, (lwork+32*maxnb)*sizeof(float) );
    if (h_work == 0) {
      fprintf (stderr, "!!!! host memory allocation error (work)\n");
    }

    printf("\n\n");
    printf("  N    CPU GFlop/s    GPU GFlop/s    ||PA-LU|| / (||A||*N)\n");
    printf("==========================================================\n");
    for(i=0; i<10; i++){
      N = lda = size[i];
      n2 = N*N;

      for(j = 0; j < n2; j++)
	h_R[j] = h_A[j] = rand() / (float)RAND_MAX;

      lda = (N/32)*32;
      if (lda<N) lda+=32;

      cublasSetMatrix( N, N, sizeof(float), h_A, N, d_A, lda);
      magma_sgetrf_gpu2(&N, &N, d_A, &lda, ipiv, dipiv, h_work, info);

      /* =====================================================================
         Performs operation using LAPACK
         =================================================================== */
      start = get_current_time();
      sgetrf_(&N, &N, h_A, &N, ipiv, info);
      end = get_current_time();
      if (info[0] < 0)
        printf("Argument %d of sgetrf had an illegal value.\n", -info[0]);

      cpu_perf = 2.*N*N*N/(3.*1000000*GetTimerValue(start,end));
      // printf("CPU Processing time: %f (ms) \n", GetTimerValue(start,end));
      
      for(j=0; j<n2; j++)
        h_A[j] = h_R[j];

      /* ====================================================================
         Performs operation using MAGMA
	 =================================================================== */
      cublasSetMatrix( N, N, sizeof(float), h_A, N, d_A, lda);
      start = get_current_time();
      magma_sgetrf_gpu2(&N, &N, d_A, &lda, ipiv, dipiv, h_work, info);
      end = get_current_time();
      cublasGetMatrix( N, N, sizeof(float), d_A, lda, h_R, N);

      gpu_perf = 2.*N*N*N/(3.*1000000*GetTimerValue(start,end));
      // printf("GPU Processing time: %f (ms) \n", GetTimerValue(start,end));
     
      /* =====================================================================
	 Check the factorization
	 =================================================================== */
      float error = get_LU_error(h_A, h_R, ipiv, N);
      
      printf("%5d    %6.2f         %6.2f         %e\n",
             size[i], cpu_perf, gpu_perf, error);

      if (argc != 1)
	break;
    }

    /* Memory clean up */
    free(h_A);
    free(ipiv);
    cublasFree(h_work);
    cublasFree(h_R);
    cublasFree(d_A);
    cublasFree(dipiv);

    /* Shutdown */
    status = cublasShutdown();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! shutdown error (A)\n");
    }
}
