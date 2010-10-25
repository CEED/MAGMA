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

extern "C" int sorgbr_(char *, int *, int *, int *, float *a, int *,
		       float *, float *, int *, int *); 
extern "C" int sbdt01_(int *, int *, int *, float *, int *, float *, int *, 
		       float *, float *, float *, int *, float *, float *);
extern "C" int sort01_(char *, int *, int *, float *, int *, 
		       float *, int *, float *);

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing sgebrd
*/
int main( int argc, char** argv) 
{
    cuInit( 0 );
    cublasInit( );
    printout_devices( );

    float *h_A, *h_R, *h_work;
    float *taup, *tauq, *diag, *offdiag, *diag2, *offdiag2;
    float *d_A;
    float gpu_perf, cpu_perf;

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
      printf("  testing_sgebrd -N %d\n\n", 1024);
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

    taup = (float*)malloc(size[9] * sizeof(float));
    tauq = (float*)malloc(size[9] * sizeof(float));
    if (taup == 0) {
      fprintf (stderr, "!!!! host memory allocation error (taup)\n");
    }
    

    diag = (float*)malloc(size[9] * sizeof(float));
    diag2= (float*)malloc(size[9] * sizeof(float));
    if (diag == 0) {
      fprintf (stderr, "!!!! host memory allocation error (diag)\n");
    }

    offdiag = (float*)malloc(size[9] * sizeof(float));
    offdiag2= (float*)malloc(size[9] * sizeof(float));
    if (offdiag == 0) {
      fprintf (stderr, "!!!! host memory allocation error (offdiag)\n");
    }

    cudaMallocHost( (void**)&h_R,  n2*sizeof(float) );
    if (h_R == 0) {
        fprintf (stderr, "!!!! host memory allocation error (R)\n");
    }

    int nb = magma_get_sgebrd_nb(size[9]);
    int lwork = 2*size[9]*nb;
    status = cublasAlloc(n2+lwork, sizeof(float), (void**)&d_A);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (d_A)\n");
    }

    cudaMallocHost( (void**)&h_work, (lwork)*sizeof(float) );
    //h_work = (float*)malloc( nb *lwork * sizeof(float) );
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
	h_A[j] = rand() / (float)RAND_MAX;
      /*
      magma_sgebrd(&M, &N, h_R, &N, diag, offdiag,
		   taup, tauq, h_work, &lwork, d_A, info);
      */
      for(j=0; j<n2; j++)
        h_R[j] = h_A[j];    
  
      /* ====================================================================
         Performs operation using MAGMA
	 =================================================================== */
      start = get_current_time();
      magma_sgebrd( M, N, h_R, N, diag, offdiag, 
		   tauq, taup, h_work, &lwork, d_A, info);
      end = get_current_time();
    
      gpu_perf =(4.*M*N*N-4.*N*N*N/3.)/(1000000.*GetTimerValue(start,end));
      // printf("GPU Processing time: %f (ms) \n", GetTimerValue(start,end));
      
      /* =====================================================================
         Check the factorization
         =================================================================== */
      int lwork = nb * N * N;
      float *PT      = (float*)malloc( N * N * sizeof(float));
      float *work    = (float*)malloc( lwork * sizeof(float));

      float result[3] = {0., 0., 0.};
      int test, one = 1;
      
      slacpy_(" ", &N, &N, h_R, &N, PT, &N);

      // generate Q & P'
      sorgbr_("Q", &M, &M, &M, h_R, &N, tauq, work, &lwork, info);

      sorgbr_("P", &M, &M, &M,  PT, &N, taup, work, &lwork, info);

      // Test 1:  Check the decomposition A := Q * B * PT
      //      2:  Check the orthogonality of Q
      //      3:  Check the orthogonality of PT
      sbdt01_(&M, &N, &one, h_A, &M, h_R, &M, diag, offdiag, PT, &M,
	       work, &result[0]);
      sort01_("Columns", &M, &M, h_R, &M, work, &lwork, &result[1]);
      sort01_("Rows", &M, &N, PT, &M, work, &lwork, &result[2]);
     
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
      sgebrd_(&M, &N, h_A, &N, diag2, offdiag2, tauq, taup,
      	      h_work, &lwork, info);
      end = get_current_time();
     
      if (info[0] < 0)  
	printf("Argument %d of sgebrd had an illegal value.\n", -info[0]);
  
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
