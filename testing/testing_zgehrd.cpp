/*
    -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

       @precisions normal z -> s d c

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

extern "C" void shst01_(int *, int *, int *, double2 *, int *, double2 *, int *, 
			double2 *, int *, double2 *, int *, double2 *);
extern "C" void sorghr_(int *, int *, int *, double2 *, int *, double2 *, 
			double2 *, int *, int *);


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgehrd
*/
int main( int argc, char** argv) 
{
    cuInit( 0 );
    cublasInit( );
    printout_devices( );

    double2 *h_A, *h_R, *h_work, *tau;
    double2 *d_A;
    double2 gpu_perf, cpu_perf;

    TimeStruct start, end;

    /* Matrix size */
    int N=0, n2, lda, ione = 1;
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
      printf("  testing_zgehrd -N %d\n\n", 1024);
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

    tau = (double2*)malloc(size[9] * sizeof(double2));
    if (tau == 0) {
      fprintf (stderr, "!!!! host memory allocation error (tau)\n");
    }
  
    cudaMallocHost( (void**)&h_R,  n2*sizeof(double2) );
    if (h_R == 0) {
        fprintf (stderr, "!!!! host memory allocation error (R)\n");
    }

    int nb = magma_get_zgehrd_nb(size[9]);
    int lwork = size[9]*nb;
    status = cublasAlloc(n2+2*lwork+nb*nb, sizeof(double2), (void**)&d_A);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (d_A)\n");
    }

    cudaMallocHost( (void**)&h_work, lwork*sizeof(double2) );
    if (h_work == 0) {
      fprintf (stderr, "!!!! host memory allocation error (work)\n");
    }

    printf("\n\n");
    printf("  N    CPU GFlop/s    GPU GFlop/s   |A-QHQ'|/N|A|  |I-QQ'|/N \n");
    printf("=============================================================\n");
    for(i=0; i<10; i++){
      N = lda = size[i];
      n2 = N*N;

      for(j = 0; j < n2; j++)
	h_A[j] = rand() / (double2)RAND_MAX;

      //magma_zgehrd(&N, &ione, &N, h_R, &N, tau, h_work, &lwork, d_A, info);

      for(j=0; j<n2; j++)
        h_R[j] = h_A[j];    
  
      /* ====================================================================
         Performs operation using MAGMA
	 =================================================================== */
      start = get_current_time();
      magma_zgehrd( N, ione, N, h_R, N, tau, h_work, &lwork, d_A, info);
      end = get_current_time();
    
      gpu_perf = 10.*N*N*N/(3.*1000000*GetTimerValue(start,end));
      // printf("GPU Processing time: %f (ms) \n", GetTimerValue(start,end));

      /* =====================================================================
         Check the factorization
         =================================================================== */
      
      double2 result[2];
      double2 *hwork_Q = (double2*)malloc( N * N * sizeof(double2));
      double2 *twork    = (double2*)malloc( 2* N * N * sizeof(double2));
      int ltwork = 2*N*N;
      
      for(j=0; j<n2; j++)
        hwork_Q[j] = h_R[j];
      
      for(j=0; j<N-1; j++)
        for(int i=j+2; i<N; i++)
          h_R[i+j*N] = 0.;

      sorghr_(&N, &ione, &N, hwork_Q, &N, tau, h_work, &lwork, info);
      shst01_(&N, &ione, &N, h_A, &N, h_R, &N, hwork_Q, &N,
              twork, &ltwork, result);
      
      //printf("N = %d\n", N);
      //printf("norm( A - Q H Q') / ( M * norm(A) * EPS ) = %f\n", result[0]);
      //printf("norm( I - Q'  Q ) / ( M * EPS )           = %f\n", result[1]);
      //printf("\n");

      free(hwork_Q);
      free(twork);
      /* =====================================================================
         Performs operation using LAPACK 
	 =================================================================== */
      start = get_current_time();
      zgehrd_(&N, &ione, &N, h_R, &lda, tau, h_work, &lwork, info);
      end = get_current_time();
      if (info[0] < 0)  
	printf("Argument %d of zgehrd had an illegal value.\n", -info[0]);
  
      cpu_perf = 10.*N*N*N/(3.*1000000*GetTimerValue(start,end));
      // printf("CPU Processing time: %f (ms) \n", GetTimerValue(start,end));
      
      /* =====================================================================
         Print performance and error.
         =================================================================== */
      printf("%5d    %6.2f         %6.2f      %e %e\n", 
	     size[i], cpu_perf, gpu_perf,
	     result[0]*5.96e-08, result[1]*5.96e-08);
      
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
