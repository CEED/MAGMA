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

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zhetrd
*/
int main( int argc, char** argv) 
{
    cuInit( 0 );
    cublasInit( );
    printout_devices( );

    double2 *h_A, *h_R, *h_work;
    double2 *tau, *diag, *offdiag, *tau2, *diag2, *offdiag2;
    double2 *d_A;
    double2 gpu_perf, cpu_perf;

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
      printf("  testing_zsytrd -N %d\n\n", 1024);
      N = size[9];
    }

    /* Initialize CUBLAS */
    status = cublasInit();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! CUBLAS initialization error\n");
    }

    lda = N;
    if (N%32!=0)
      lda = (N/32)*32 + 32;
    n2 = size[9] * lda;

    /* Allocate host memory for the matrix */
    h_A = (double2*)malloc(n2 * sizeof(h_A[0]));
    if (h_A == 0) {
        fprintf (stderr, "!!!! host memory allocation error (A)\n");
    }

    tau = (double2*)malloc(size[9] * sizeof(double2));
    tau2= (double2*)malloc(size[9] * sizeof(double2));
    if (tau == 0) {
      fprintf (stderr, "!!!! host memory allocation error (tau)\n");
    }
    

    diag = (double2*)malloc(size[9] * sizeof(double2));
    diag2= (double2*)malloc(size[9] * sizeof(double2));
    if (diag == 0) {
      fprintf (stderr, "!!!! host memory allocation error (diag)\n");
    }

    offdiag = (double2*)malloc(size[9] * sizeof(double2));
    offdiag2= (double2*)malloc(size[9] * sizeof(double2));
    if (offdiag == 0) {
      fprintf (stderr, "!!!! host memory allocation error (offdiag)\n");
    }

    cudaMallocHost( (void**)&h_R,  n2*sizeof(double2) );
    if (h_R == 0) {
        fprintf (stderr, "!!!! host memory allocation error (R)\n");
    }

    int nb = magma_get_zhetrd_nb(size[9]);
    //int lwork = 2*size[9]*nb;
    int lwork = 2*size[9]*lda/nb;
    status = cublasAlloc(n2+lwork, sizeof(double2), (void**)&d_A);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (d_A)\n");
    }

    cudaMallocHost( (void**)&h_work, (lwork)*sizeof(double2) );
    //h_work = (double2*)malloc( lwork * sizeof(double2) );
    if (h_work == 0) {
      fprintf (stderr, "!!!! host memory allocation error (work)\n");
    }

    printf("\n\n");
    printf("  N    CPU GFlop/s    GPU GFlop/s   |A-QHQ'|/N|A|  |I-QQ'|/N \n");
    printf("=============================================================\n");
    for(i=0; i<10; i++){
      N = size[i];

      if (N%32==0)
	lda = N;
      else
	lda = (N/32)*32+32;

      n2 = N*lda;

      for(j = 0; j < n2; j++)
	h_A[j] = rand() / (double2)RAND_MAX;
      /*
      magma_zhetrd("L", &N, h_R, &lda, diag, offdiag,
		   tau, h_work, &lwork, d_A, info);
      */
      for(j=0; j<n2; j++)
        h_R[j] = h_A[j];    
  
      /* ====================================================================
         Performs operation using MAGMA
	 =================================================================== */
      start = get_current_time();      
      magma_zsytrd('L', N, h_R, lda, diag, offdiag, 
		   tau, h_work, &lwork, d_A, info);
      end = get_current_time();
    
      gpu_perf = 4.*N*N*N/(3.*1000000*GetTimerValue(start,end));
      // printf("GPU Processing time: %f (ms) \n", GetTimerValue(start,end));

      /* =====================================================================
         Check the factorization
         =================================================================== */
      double2 *hwork_Q = (double2*)malloc( N * N * sizeof(double2));
      double2 *work    = (double2*)malloc( 2 * N * N * sizeof(double2));
      
      double2 result[2] = {0., 0.};

      int test, one = 1;

      lapackf77_zlacpy("L", &N, &N, h_R, &lda, hwork_Q, &N);
      lapackf77_zungtr("L", &N, hwork_Q, &N, tau, h_work, &lwork, info);

      test = 2;
      lapackf77_zhet21(&test, "L", &N, &one, h_A, &lda, diag, offdiag, 
	      hwork_Q, &N, h_R, &lda, tau, work, &result[0]);

      test = 3;
      lapackf77_zhet21(&test, "L", &N, &one, h_A, &lda, diag, offdiag,
	      hwork_Q, &N, h_R, &lda, tau, work, &result[1]);
      
      //printf("N = %d\n", N);
      //printf("norm( A - Q H Q') / ( N * norm(A) * EPS ) = %f\n", result[0]);
      //printf("norm( I - Q'  Q ) / ( N * EPS )           = %f\n", result[1]);
      //printf("\n");

      free(hwork_Q);
      free(work);
      /* =====================================================================
         Performs operation using LAPACK 
	 =================================================================== */
      start = get_current_time();
      lapackf77_zhetrd("L", &N, h_A, &lda, diag2, offdiag2, tau2, h_work, &lwork, info);
      end = get_current_time();
     
      if (info[0] < 0)  
	printf("Argument %d of zhetrd had an illegal value.\n", -info[0]);
  
      cpu_perf = 4.*N*N*N/(3.*1000000*GetTimerValue(start,end));
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
    free(tau);     free(tau2);
    free(diag);    free(diag2);
    free(offdiag); free(offdiag2);
    cublasFree(h_work);
    //free(h_work);
    cublasFree(h_R);
    cublasFree(d_A);

    /* Shutdown */
    status = cublasShutdown();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! shutdown error (A)\n");
    }
}
