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

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing cpotrf
*/
int main( int argc, char** argv) 
{
    cuInit( 0 );
    cublasInit( );
    printout_devices( );

    float2 *h_A, *h_R;
    float2 *d_A;
    float gpu_perf, cpu_perf;

    TimeStruct start, end;

    /* Matrix size */
    int N=0, n2, lda;
    int size[10] = {1024,2048,3072,4032,5184,6048,7200,8064,8928,10080};
    
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
      printf("  testing_cpotrf -N %d\n\n", 1024);
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
  
    cudaMallocHost( (void**)&h_R,  n2*sizeof(float2) );
    if (h_R == 0) {
        fprintf (stderr, "!!!! host memory allocation error (R)\n");
    }

    status = cublasAlloc(n2, sizeof(float2), (void**)&d_A);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (d_A)\n");
    }

    printf("\n\n");
    printf("  N    CPU GFlop/s    GPU GFlop/s    ||R||_F / ||A||_F\n");
    printf("========================================================\n");
    for(i=0; i<10; i++){
      N = lda = size[i];
      n2 = N*N;

      for(j = 0; j < n2; j++){
	h_A[j].x = rand() / (float)RAND_MAX;
	h_A[j].y = rand() / (float)RAND_MAX;
      }
      for(j=0; j<n2; j+=(lda+1)){
      	h_R[j].x = (h_A[j].x += N);
	h_R[j].y = h_A[j].y = 0.;
      }
      magma_cpotrf("L", &N, h_R, &lda, d_A, info);
      //magma_cpotrf("U", &N, h_R, &lda, d_A, info);

      for(j=0; j<n2; j++)
	h_R[j] = h_A[j];
     
      /* ====================================================================
         Performs operation using MAGMA 
	 =================================================================== */
      start = get_current_time();
      magma_cpotrf("L", &N, h_R, &lda, d_A, info);
      //magma_cpotrf("U", &N, h_R, &lda, d_A, info);
      end = get_current_time();
    
      gpu_perf = 4.*N*N*N/(3.*1000000*GetTimerValue(start,end));
      // printf("GPU Processing time: %f (ms) \n", GetTimerValue(start,end));
      // printf("Speed: %f GFlops \n", gpu_perf);

      /* =====================================================================
         Performs operation using LAPACK 
	 =================================================================== */
      start = get_current_time();
      cpotrf_("L", &N, h_A, &lda, info);
      //cpotrf_("U", &N, h_A, &lda, info);
      end = get_current_time();

      if (info[0] < 0)  
	printf("Argument %d of cpotrf had an illegal value.\n", -info[0]);     
  
      cpu_perf = 4.*N*N*N/(3.*1000000*GetTimerValue(start,end));
      // printf("CPU Processing time: %f (ms) \n", GetTimerValue(start,end));
      // printf("Speed: %f GFlops \n", cpu_perf);
      
      /* =====================================================================
         Check the result compared to LAPACK
         =================================================================== */
      float work[1], matnorm;
      float2 mone = {-1., 0.};
      int one = 1;
      matnorm = clange_("f", &N, &N, h_A, &N, work);
      caxpy_(&n2, &mone, h_A, &one, h_R, &one);
      printf("%5d    %6.2f         %6.2f        %e\n", 
	     size[i], cpu_perf, gpu_perf,
	     clange_("f", &N, &N, h_R, &N, work) / matnorm);

      if (argc != 1)
	break;
    }

    /* Memory clean up */
    free(h_A);
    cublasFree(h_R);
    cublasFree(d_A);

    /* Shutdown */
    status = cublasShutdown();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! shutdown error (A)\n");
    }
}
