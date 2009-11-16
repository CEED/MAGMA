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

// includes, project
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cublas.h"
#include "magma.h"


float get_LU_error(float2 *A, float2 *LU, int *IPIV, int N){
  int k1 = 1, k2 = N, intONE = 1, i, j;
  float2 cone = {1.0, 0.0}, czero = {0.0, 0.0};

  claswp_( &N, A, &N, &k1, &k2, IPIV, &intONE);

  float2 *L = (float2 *) calloc (N*N, sizeof(float2));
  float *work = (float *) calloc (N+1, sizeof(float));

  for( j = 0; j < N; j++ )
    for( i = 0; i < N; i++ )
      L[i+j*N] = (i > j ? LU[i+j*N] : (i == j ? cone : czero));

  for( j = 0; j < N; j++ )
    for( i = j+1; i < N; i++ )
      LU[i+j*N] = czero;

  float matnorm = clange_("f", &N, &N, A, &N, work);
  float2 alpha = {1.0, 0.0};

  ctrmm_("L", "L", "N", "N", &N, &N, &alpha, L, &N, LU, &N);

  for( j = 0; j < N*N; j++ )
    { LU[j].x = LU[j].x - A[j].x; LU[j].y = LU[j].y - A[j].y; }
  
  /*
  //int n1 = 898, n2 = 1025;
  int n1 = N, n2 = 1024;
  float residual = clange_("f", &n1, &n2, LU, &N, work);
  printf("%e\n", residual/(matnorm * N));

  n1 = 1024; n2 = N;
  residual = clange_("f", &n1, &n2, LU, &N, work);
  printf("%e\n", residual/(matnorm * N));

  n1 = 800, n2 = 800;
  residual = clange_("f", &n1, &n2, LU, &N, work);
  printf("%e\n", residual/(matnorm * N));

  n1 = 962, n2 = N;
  residual = clange_("f", &n1, &n2, LU, &N, work);
  printf("%e\n", residual/(matnorm * N));

  n1 = N, n2 = 800;
  residual = clange_("f", &n1, &n2, LU, &N, work);
  printf("%e\n", residual/(matnorm * N));


  residual = clange_("f", &N, &N, LU, &N, work);
  */
  float residual = clange_("f", &N, &N, LU, &N, work);

  free(L);
  free(work);
  
  return residual / (matnorm * N);
}


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing cgetrf
*/
int main( int argc, char** argv) 
{
    cuInit( 0 );
    cublasInit( );
    printout_devices( );

    float2 *h_A, *h_R, *h_work;
    float2 *d_A;
    int *ipiv;
    float gpu_perf_cgetrf, cpu_perf_cgetrf;

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
      printf("  testing_cgetrf_gpu -N %d\n\n", 1024);
    }

    /* Initialize CUBLAS */
    status = cublasInit();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! CUBLAS initialization error\n");
    }

    lda = N;
    n2 = size[9] * size[9];

    /* Allocate host memory for the matrix */
    h_A = (float2*)malloc(n2 * sizeof(h_A[0]));
    if (h_A == 0) {
        fprintf (stderr, "!!!! host memory allocation error (A)\n");
    }

    ipiv = (int*)malloc(size[9] * sizeof(int));
    if (ipiv == 0) {
      fprintf (stderr, "!!!! host memory allocation error (ipiv)\n");
    }
  
    cudaMallocHost( (void**)&h_R,  n2*sizeof(float2) );
    if (h_R == 0) {
        fprintf (stderr, "!!!! host memory allocation error (R)\n");
    }

    int maxnb = magma_get_cgetrf_nb(size[9]);
    int lwork = size[9]*maxnb;
    int k, n3;
    n3 = (size[9]+32)*(size[9]+32)+32*maxnb+lwork+2*maxnb*maxnb;
    status = cublasAlloc(n3,sizeof(float), (void**)&d_A);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (d_A)\n");
    }

    cudaMallocHost( (void**)&h_work, (lwork+32*maxnb)*sizeof(float2) );
    if (h_work == 0) {
      fprintf (stderr, "!!!! host memory allocation error (work)\n");
    }

    printf("\n\n");
    printf("  N    CPU GFlop/s    GPU GFlop/s    ||PA-LU|| / (||A||*N)\n");
    printf("==========================================================\n");
    for(i=0; i<10; i++){
      N = lda = size[i];
      n2 = N*N;

      for(k = 0; k < n2; k++)
	h_R[k].x = h_A[k].x = rand() / (float)RAND_MAX; h_R[k].y = h_A[k].y = rand() / (float)RAND_MAX;

      lda = (N/32)*32;
      if (lda<N) lda+=32;

      cublasSetMatrix( N, N, sizeof(float2), h_A, N, d_A, lda);
      magma_cgetrf_gpu(&N, &N, d_A, &lda, ipiv, h_work, info);


      /* =====================================================================
         Performs operation using LAPACK
         =================================================================== */
      start = get_current_time();
      cgetrf_(&N, &N, h_A, &N, ipiv, info);
      end = get_current_time();
      if (info[0] < 0)
        printf("Argument %d of cgetrf had an illegal value.\n", -info[0]);

      cpu_perf_cgetrf = 4.*2.*N*N*N/(3.*1000000*GetTimerValue(start,end));
      // printf("CPU Processing time: %f (ms) \n", GetTimerValue(start,end));
      
      for(j=0; j<n2; j++)
        h_A[j].x = h_R[j].x; h_A[j].y = h_R[j].y;

      /* ====================================================================
         Performs operation using MAGMA
	 =================================================================== */
      cublasSetMatrix( N, N, sizeof(float2), h_A, N, d_A, lda);
      start = get_current_time();
      magma_cgetrf_gpu(&N, &N, d_A, &lda, ipiv, h_work, info);
      end = get_current_time();
      cublasGetMatrix( N, N, sizeof(float2), d_A, lda, h_R, N);

      gpu_perf_cgetrf = 4.*2.*N*N*N/(3.*1000000*GetTimerValue(start,end));
      // printf("GPU Processing time: %f (ms) \n", GetTimerValue(start,end));
     
      /* =====================================================================
	 Check the factorization
	 =================================================================== */
      float error = get_LU_error(h_A, h_R, ipiv, N);
      
      printf("%5d    %6.2f         %6.2f         %e\n",
             size[i], cpu_perf_cgetrf, gpu_perf_cgetrf, error);

      if (argc != 1)
	break;
    }

    /* Memory clean up */
    free(h_A);
    free(ipiv);
    cublasFree(h_work);
    cublasFree(h_R);
    cublasFree(d_A);

    /* Shutdown */
    status = cublasShutdown();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! shutdown error (A)\n");
    }
}
