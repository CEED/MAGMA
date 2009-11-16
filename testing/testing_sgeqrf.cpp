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

extern "C" void sqrt02_(int *, int *, int *, float *, float *, float *,
			float *, int *, float *, float *,
			int *, float *, float *);


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing sgeqrf
*/
int main( int argc, char** argv) 
{
    cuInit( 0 );
    cublasInit( );
    printout_devices( );

    float *h_A, *h_R, *h_work, *tau;
    float *d_A;
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
      printf("  testing_sgeqrf -N %d\n\n", 1024);
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

    tau = (float*)malloc(size[9] * sizeof(float));
    if (tau == 0) {
      fprintf (stderr, "!!!! host memory allocation error (tau)\n");
    }
  
    cudaMallocHost( (void**)&h_R,  n2*sizeof(float) );
    if (h_R == 0) {
        fprintf (stderr, "!!!! host memory allocation error (R)\n");
    }

    int lwork = size[9]*magma_get_sgeqrf_nb(size[9]);
    status = cublasAlloc(n2+lwork, sizeof(float), (void**)&d_A);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (d_A)\n");
    }

    cudaMallocHost( (void**)&h_work, lwork*sizeof(float) );
    if (h_work == 0) {
      fprintf (stderr, "!!!! host memory allocation error (work)\n");
    }

    printf("\n\n");
    printf("  N    CPU GFlop/s    GPU GFlop/s    ||R - Q'A|| / ||A||\n");
    printf("========================================================\n");
    for(i=0; i<10; i++){
      N = lda = size[i];
      n2 = N*N;

      for(j = 0; j < n2; j++)
	h_A[j] = rand() / (float)RAND_MAX;

      magma_sgeqrf(&N, &N, h_R, &N, tau, h_work, &lwork, d_A, info);

      for(j=0; j<n2; j++)
        h_R[j] = h_A[j];    
  
      /* ====================================================================
         Performs operation using MAGMA
	 =================================================================== */
      start = get_current_time();
      magma_sgeqrf(&N, &N, h_R, &N, tau, h_work, &lwork, d_A, info);
      end = get_current_time();
    
      gpu_perf = 4.*N*N*N/(3.*1000000*GetTimerValue(start,end));
      // printf("GPU Processing time: %f (ms) \n", GetTimerValue(start,end));

      /* =====================================================================
         Check the factorization
         =================================================================== */
      float result[2];
      float *hwork_Q = (float*)malloc( N * N * sizeof(float));
      float *hwork_R = (float*)malloc( N * N * sizeof(float));
      float *rwork   = (float*)malloc( N * sizeof(float));

      sqrt02_(&N, &N, &N, h_A, h_R, hwork_Q, hwork_R, &N, tau,
	      h_work, &lwork, rwork, result);

      //printf("norm( R - Q'*A ) / ( M * norm(A) * EPS ) = %f\n", result[0]);
      //printf("norm( I - Q'*Q ) / ( M * EPS )           = %f\n", result[1]);
      free(hwork_Q);
      free(hwork_R);
      free(rwork);

      /* =====================================================================
         Performs operation using LAPACK 
	 =================================================================== */
      start = get_current_time();
      sgeqrf_(&N, &N, h_A, &lda, tau, h_work, &lwork, info);
      end = get_current_time();
      if (info[0] < 0)  
	printf("Argument %d of sgeqrf had an illegal value.\n", -info[0]);     
  
      cpu_perf = 4.*N*N*N/(3.*1000000*GetTimerValue(start,end));
      // printf("CPU Processing time: %f (ms) \n", GetTimerValue(start,end));
      
      /* =====================================================================
         Print performance and error.
         =================================================================== */
      printf("%5d    %6.2f         %6.2f        %e\n", 
	     size[i], cpu_perf, gpu_perf,
	     N*result[0]*5.96e-08);
      
      if (argc != 1)
	break;
    }

    /* Memory clean up */
    free(h_A);
    free(tau);
    cublasFree(h_work);
    cublasFree(h_R);
    cublasFree(d_A);

    /* Shutdown */
    status = cublasShutdown();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! shutdown error (A)\n");
    }
}
