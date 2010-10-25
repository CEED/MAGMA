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

#define min(a,b)  (((a)<(b))?(a):(b))
#define max(a,b)  (((a)<(b))?(b):(a))

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
    cuInit( 0 );
    cublasInit( );
    printout_devices( );

    float *h_A, *h_R, *h_work;
    float *d_A;
    int *ipiv;
    float gpu_perf, cpu_perf;

    TimeStruct start, end;

    /* Matrix size */
    int M = 0, N=0, n2, lda;
    int size[10] = {1024,2048,3072,4032,5184,6016,7040,8064,9088,10112};
    
    cublasStatus status;
    int i, j, info[1];

    if (argc != 1){
      for(i = 1; i<argc; i++){	
	if (strcmp("-N", argv[i])==0)
	  N = atoi(argv[++i]);
	else if (strcmp("-M", argv[i])==0)
          M = atoi(argv[++i]);
      }
      if (M>0 && N>0)
        printf("  testing_sgetrf_gpu -M %d -N %d\n\n", M, N);
      else
        {
          printf("\nUsage: \n");
          printf("  testing_sgetrf_gpu -M %d -N %d\n\n", 1024, 1024);
          exit(1);
        }
    }
    else {
      printf("\nUsage: \n");
      printf("  testing_sgetrf_gpu -M %d -N %d\n\n", 1024, 1024);
      M = N = size[9];
    }

    /* Initialize CUBLAS */
    status = cublasInit();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! CUBLAS initialization error\n");
    }

    lda = ((M+31)/32)*32;
    //lda = M;
    n2 = M * N;

    int min_mn = min(M, N);

    /* Allocate host memory for the matrix */
    h_A = (float*)malloc(n2 * sizeof(h_A[0]));
    if (h_A == 0) {
        fprintf (stderr, "!!!! host memory allocation error (A)\n");
    }

    ipiv = (int*)malloc(min_mn * sizeof(int));
    if (ipiv == 0) {
      fprintf (stderr, "!!!! host memory allocation error (ipiv)\n");
    }
  
    cudaMallocHost( (void**)&h_R,  n2*sizeof(float) );
    if (h_R == 0) {
        fprintf (stderr, "!!!! host memory allocation error (R)\n");
    }

    int nb = magma_get_sgetrf_nb(min_mn);
    int lwork = (M+32) * nb;
    status = cublasAlloc(lda *N, sizeof(float), (void**)&d_A);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (d_A)\n");
    }

    cudaMallocHost( (void**)&h_work, lwork*sizeof(float) );
    if (h_work == 0) {
      fprintf (stderr, "!!!! host memory allocation error (work)\n");
    }

    printf("\n\n");
    printf("  M     N   CPU GFlop/s    GPU GFlop/s   ||PA-LU||/(||A||*N)\n");
    printf("============================================================\n");
    for(i=0; i<10; i++){
      if (argc==1){
	M = N = min_mn = size[i];
	n2 = M*N;

	lda = ((M+31)/32)*32;
	//lda= M;
      }

      for(j = 0; j < n2; j++)
	h_R[j] = h_A[j] = rand() / (float)RAND_MAX;

      cublasSetMatrix( M, N, sizeof(float), h_A, M, d_A, lda);
      magma_sgetrf_gpu( M, N, d_A, lda, ipiv, info);
      cublasSetMatrix( M, N, sizeof(float), h_A, M, d_A, lda);

      /* =====================================================================
         Performs operation using LAPACK
         =================================================================== */
      start = get_current_time();
      sgetrf_(&M, &N, h_A, &M, ipiv, info);
      end = get_current_time();
      if (info[0] < 0)
        printf("Argument %d of sgetrf had an illegal value.\n", -info[0]);

      cpu_perf = 2.*M*N*min_mn/(3.*1000000*GetTimerValue(start,end));
      // printf("CPU Processing time: %f (ms) \n", GetTimerValue(start,end));
      
      for(j=0; j<n2; j++)
        h_A[j] = h_R[j];

      /* ====================================================================
         Performs operation using MAGMA
	 =================================================================== */
      start = get_current_time();
      magma_sgetrf_gpu( M, N, d_A, lda, ipiv, info);
      end = get_current_time();
      cublasGetMatrix( M, N, sizeof(float), d_A, lda, h_R, M);

      gpu_perf = 2.*M*N*min_mn/(3.*1000000*GetTimerValue(start,end));
      // printf("GPU Processing time: %f (ms) \n", GetTimerValue(start,end));
     
      /* =====================================================================
	 Check the factorization
	 =================================================================== */
      float error = get_LU_error(M, N, h_A, &M, h_R, ipiv);
      
      printf("%5d %5d  %6.2f         %6.2f         %e\n",
             M, N, cpu_perf, gpu_perf, error);

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
