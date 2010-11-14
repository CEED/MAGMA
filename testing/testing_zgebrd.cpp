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
   -- Testing zgebrd
*/
int main( int argc, char** argv) 
{
    cuInit( 0 );
    cublasInit( );
    printout_devices( );

    double2 *h_A, *h_R, *h_work;
    double2 *taup, *tauq, *diag, *offdiag, *diag2, *offdiag2;
    double2 *d_A;
    double2 gpu_perf, cpu_perf;

    TimeStruct start, end;

    /* Matrix size */
    int M, N=0, n2, lda, ione = 1;
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
      printf("  testing_zgebrd -N %d\n\n", 1024);
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

    taup = (double2*)malloc(size[9] * sizeof(double2));
    tauq = (double2*)malloc(size[9] * sizeof(double2));
    if (taup == 0) {
      fprintf (stderr, "!!!! host memory allocation error (taup)\n");
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

    int nb = magma_get_zgebrd_nb(size[9]);
    int lwork = 2*size[9]*nb;
    status = cublasAlloc(n2+lwork, sizeof(double2), (void**)&d_A);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (d_A)\n");
    }

    cudaMallocHost( (void**)&h_work, (lwork)*sizeof(double2) );
    //h_work = (double2*)malloc( nb *lwork * sizeof(double2) );
    if (h_work == 0) {
      fprintf (stderr, "!!!! host memory allocation error (work)\n");
    }

    printf("\n\n");
    printf("  N    CPU GFlop/s    GPU GFlop/s   |A-QHQ'|/N|A|  |I-QQ'|/N \n");
    printf("=============================================================\n");
    for(i=0; i<10; i++){
      M = N = lda = size[i];
      n2 = M*N;

      for(j = 0; j < n2; j++)
	h_A[j] = rand() / (double2)RAND_MAX;
      /*
      magma_zgebrd(&M, &N, h_R, &N, diag, offdiag,
		   taup, tauq, h_work, &lwork, d_A, info);
      */
      for(j=0; j<n2; j++)
        h_R[j] = h_A[j];    
  
      /* ====================================================================
         Performs operation using MAGMA
	 =================================================================== */
      start = get_current_time();
      magma_zgebrd( M, N, h_R, N, diag, offdiag, 
		   tauq, taup, h_work, &lwork, d_A, info);
      end = get_current_time();
    
      gpu_perf =(4.*M*N*N-4.*N*N*N/3.)/(1000000.*GetTimerValue(start,end));
      // printf("GPU Processing time: %f (ms) \n", GetTimerValue(start,end));
      
      /* =====================================================================
         Check the factorization
         =================================================================== */
      int lwork = nb * N * N;
      double2 *PT      = (double2*)malloc( N * N * sizeof(double2));
      double2 *work    = (double2*)malloc( lwork * sizeof(double2));

      double2 result[3] = {0., 0., 0.};
      int test, one = 1;
      
      lapackf77_zlacpy(" ", &N, &N, h_R, &N, PT, &N);

      // generate Q & P'
      lapackf77_zungbr("Q", &M, &M, &M, h_R, &N, tauq, work, &lwork, info);

      lapackf77_zungbr("P", &M, &M, &M,  PT, &N, taup, work, &lwork, info);

      // Test 1:  Check the decomposition A := Q * B * PT
      //      2:  Check the orthogonality of Q
      //      3:  Check the orthogonality of PT
      lapackf77_zbdt01(&M, &N, &one, h_A, &M, h_R, &M, diag, offdiag, PT, &M,
	       work, &result[0]);
      lapackf77_zunt01("Columns", &M, &M, h_R, &M, work, &lwork, &result[1]);
      lapackf77_zunt01("Rows", &M, &N, PT, &M, work, &lwork, &result[2]);
     
      //printf("N = %d\n", N);
      //printf("norm(A -  Q  B  PT) / ( N * norm(A) * EPS ) = %f\n", result[0]);
      //printf("norm(I -  Q' *  Q ) / ( N * EPS )           = %f\n", result[1]);
      //printf("norm(I - PT' * PT ) / ( N * EPS )           = %f\n", result[2]);
      //printf("\n");
      
      free(PT);
      free(work);
      /* =====================================================================
         Performs operation using LAPACK 
	 =================================================================== */
      start = get_current_time();
      lapackf77_zgebrd(&M, &N, h_A, &N, diag2, offdiag2, tauq, taup,
      	      h_work, &lwork, info);
      end = get_current_time();
     
      if (info[0] < 0)  
	printf("Argument %d of zgebrd had an illegal value.\n", -info[0]);
  
      cpu_perf = (4.*M*N*N-4.*N*N*N/3.)/(1000000.*GetTimerValue(start,end));
      // printf("CPU Processing time: %f (ms) \n", GetTimerValue(start,end));
      
      /* =====================================================================
         Print performance and error.
         =================================================================== */
      printf("%5d   %6.2f        %6.2f       %4.2e %4.2e %4.2e\n",
             size[i], cpu_perf, gpu_perf, 
	     result[0]*5.96e-08, result[1]*5.96e-08, result[2]*5.96e-08);

      if (argc != 1)
	break;
    }

    /* Memory clean up */
    free(h_A);
    free(taup);
    free(tauq);
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
