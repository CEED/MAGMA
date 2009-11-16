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
   -- Testing zpotrf
*/
int main( int argc, char** argv) 
{
    cuInit( 0 );
    cublasInit( );
    printout_devices( );

    double2 *h_A, *h_R, *h_work;
    double2 *d_A;
    double gpu_perf_zpotrf, cpu_perf_zpotrf;

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
      printf("  testing_zpotrf_gpu -N %d\n\n", 1024);
    }

    /* Initialize CUBLAS */
    status = cublasInit();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! CUBLAS initialization error\n");
    }

    lda = N;
    n2 = size[9] * size[9];

    int maxNB = magma_get_zpotrf_nb(size[9]);
    cudaMallocHost( (void**)&h_work,  maxNB*maxNB*sizeof(double2) );

    /* Allocate host memory for the matrix */
    h_A = (double2 *)malloc(n2 * sizeof(h_A[0]));
    if (h_A == 0) {
        fprintf (stderr, "!!!! host memory allocation error (A)\n");
    }
    //h_R = (double2 *)malloc(n2 * sizeof(h_R[0]));
    cudaMallocHost( (void**)&h_R,  n2*sizeof(double2) );
    if (h_R == 0) {
        fprintf (stderr, "!!!! host memory allocation error (R)\n");
    }

    status = cublasAlloc(n2, sizeof(double2), (void**)&d_A);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (d_A)\n");
    }

    printf("\n\n");
    printf("  N    CPU GFlop/s    GPU GFlop/s    ||R||_F / ||A||_F\n");
    printf("========================================================\n");
    for(i=0; i<10; i++){
      N = lda = size[i];
      n2 = N*N;

      for(j = 0; j < n2; j++)
	h_A[j].x = rand() / (double)RAND_MAX; h_A[j].y = rand() / (double)RAND_MAX;
      
      for(j=0; j<n2; j+=(lda+1)){
	h_R[j].x = (h_A[j].x += 2000); h_R[j].y = h_A[j].y = 0.;
      }

      cublasSetVector(n2, sizeof(double2), h_A, 1, d_A, 1);
      magma_zpotrf_gpu("U", &N, d_A, &lda, h_work, info);
      cublasSetVector(n2, sizeof(double2), h_A, 1, d_A, 1);
      
      /* ====================================================================
         Performs operation using MAGMA 
	 =================================================================== */
      start = get_current_time();
      magma_zpotrf_gpu("L", &N, d_A, &lda, h_work, info);
      //magma_zpotrf_gpu("U", &N, d_A, &lda, h_work, info);
      end = get_current_time();
    
      gpu_perf_zpotrf = 4.*N*N*N/(3.*1000000*GetTimerValue(start,end));
      // printf("GPU Processing time: %f (ms) \n", GetTimerValue(start,end));
      // printf("Speed: %f GFlops \n", gpu_perf);

      /* =====================================================================
         Performs operation using LAPACK 
	 =================================================================== */
      start = get_current_time();
      zpotrf_("L", &N, h_A, &lda, info);
      //zpotrf_("U", &N, h_A, &lda, info);
      end = get_current_time();
      if (info[0] < 0)  
	printf("Argument %d of zpotrf had an illegal value.\n", -info[0]);     
  
      cpu_perf_zpotrf = 4.*N*N*N/(3.*1000000*GetTimerValue(start,end));
      // printf("CPU Processing time: %f (ms) \n", GetTimerValue(start,end));
      // printf("Speed: %f GFlops \n", cpu_perf);
      
      /* =====================================================================
         Check the result compared to LAPACK
         =================================================================== */
      cublasGetVector(n2, sizeof(double2), d_A, 1, h_R, 1);
      double work[1], matnorm;
      double2 mone = {-1., 0.};
      int one = 1;
      matnorm = zlange_("f", &N, &N, h_A, &N, work);
      zaxpy_(&n2, &mone, h_A, &one, h_R, &one);
      printf("%5d    %6.2f         %6.2f        %e\n", 
	     size[i], cpu_perf_zpotrf, gpu_perf_zpotrf,
	     zlange_("f", &N, &N, h_R, &N, work) / matnorm);

      if (argc != 1)
	break;
    }

    /* Memory clean up */
    free(h_A);
    cublasFree(h_work);
    cublasFree(h_R);
    cublasFree(d_A);

    /* Shutdown */
    status = cublasShutdown();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! shutdown error (A)\n");
    }
}
