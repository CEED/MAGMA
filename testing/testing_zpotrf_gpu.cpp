/*
 *  -- MAGMA (version 1.0) --
 *     Univ. of Tennessee, Knoxville
 *     Univ. of California, Berkeley
 *     Univ. of Colorado, Denver
 *     November 2010
 *
 * @precisions normal z -> c d s
 *
 **/
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
#include "magmablas.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zpotrf
*/
int main( int argc, char** argv) 
{
    cuInit( 0 );
    cublasInit( );
    printout_devices( );

    double2 *h_A, *h_R;
    double2 *d_A;
    double2 gpu_perf, cpu_perf;

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

    /* Allocate host memory for the matrix */
    h_A = (double2*)malloc(n2 * sizeof(h_A[0]));
    if (h_A == 0) {
        fprintf (stderr, "!!!! host memory allocation error (A)\n");
    }
    //h_R = (double2*)malloc(n2 * sizeof(h_R[0]));
    cudaMallocHost( (void**)&h_R,  n2*sizeof(double2) );
    if (h_R == 0) {
        fprintf (stderr, "!!!! host memory allocation error (R)\n");
    }

    status = cublasAlloc(n2+32*size[9], sizeof(double2), (void**)&d_A);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (d_A)\n");
    }

    printf("\n\n");
    printf("  N    CPU GFlop/s    GPU GFlop/s    ||R||_F / ||A||_F\n");
    printf("========================================================\n");
    for(i=0; i<10; i++){
      //N = lda = size[i];
      N = size[i]; lda = size[9]; 
      n2 = N*N;

      lda = (N/32)*32;
      if (lda<N) lda+=32;

      for(j = 0; j < n2; j++)
	h_A[j] = rand() / (double2)RAND_MAX;
      for(j=0; j<n2; j+=(N+1))
	h_R[j] = (h_A[j]+=2000);

      cublasSetMatrix( N, N, sizeof(double2), h_A, N, d_A, lda);
      magma_zpotrf_gpu('U', N, d_A, lda, info);
      cublasSetMatrix( N, N, sizeof(double2), h_A, N, d_A, lda);
      
      /* ====================================================================
         Performs operation using MAGMA 
	 =================================================================== */
      start = get_current_time();
      magma_zpotrf_gpu('L', N, d_A, lda, info);
      //magma_zpotrf_gpu('U', N, d_A, lda, info);
      end = get_current_time();
    
      gpu_perf = 1.*N*N*N/(3.*1000000*GetTimerValue(start,end));
      // printf("GPU Processing time: %f (ms) \n", GetTimerValue(start,end));
      // printf("Speed: %f GFlops \n", gpu_perf);

      /* =====================================================================
         Performs operation using LAPACK 
	 =================================================================== */
      start = get_current_time();
      lapackf77_zpotrf("L", &N, h_A, &N, info);
      //lapackf77_zpotrf("U", &N, h_A, &N, info);
      end = get_current_time();
      if (info[0] < 0)  
	printf("Argument %d of zpotrf had an illegal value.\n", -info[0]);     
  
      cpu_perf = 1.*N*N*N/(3.*1000000*GetTimerValue(start,end));
      // printf("CPU Processing time: %f (ms) \n", GetTimerValue(start,end));
      // printf("Speed: %f GFlops \n", cpu_perf);
      
      /* =====================================================================
         Check the result compared to LAPACK
         =================================================================== */
      cublasGetMatrix( N, N, sizeof(double2), d_A, lda, h_R, N);
      double2 work[1], matnorm, mone = -1.0;
      int one = 1;
      matnorm = lapackf77_zlange("f", &N, &N, h_A, &N, work);
      blasf77_zaxpy(&n2, &mone, h_A, &one, h_R, &one);
      printf("%5d    %6.2f         %6.2f        %e\n", 
	     size[i], cpu_perf, gpu_perf,
	     lapackf77_zlange("f", &N, &N, h_R, &N, work) / matnorm);

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
