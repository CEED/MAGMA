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
   -- Testing dgeqrs
*/
int main( int argc, char** argv) 
{
    cuInit( 0 );
    cublasInit( );
    printout_devices( );

    double *h_A, *h_R, *h_work, *tau;
    double *d_A, *d_work, *d_x;
    double gpu_perf, cpu_perf;

    double *x, *b, *r;
    double *d_b;

    TimeStruct start, end;

    /* Matrix size */
    int M, N=0, n2, lda;
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
      printf("  testing_dgeqrs_gpu -N %d\n\n", 1024);
    }

    /* Initialize CUBLAS */
    status = cublasInit();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! CUBLAS initialization error\n");
    }

    lda = N;
    n2 = size[9] * size[9];

    /* Allocate host memory for the matrix */
    h_A = (double*)malloc(n2 * sizeof(h_A[0]));
    if (h_A == 0) {
        fprintf (stderr, "!!!! host memory allocation error (A)\n");
    }

    tau = (double*)malloc(size[9] * sizeof(double));
    if (tau == 0) {
      fprintf (stderr, "!!!! host memory allocation error (tau)\n");
    }
  
    x = (double*)malloc(size[9] * sizeof(double));
    b = (double*)malloc(size[9] * sizeof(double));
    r = (double*)malloc(size[9] * sizeof(double));

    cudaMallocHost( (void**)&h_R,  n2*sizeof(double) );
    if (h_R == 0) {
        fprintf (stderr, "!!!! host memory allocation error (R)\n");
    }

    int nb = magma_get_dgeqrf_nb(size[9]);
    int lwork = (3*size[9]+nb)*nb;
    status = cublasAlloc(n2, sizeof(double), (void**)&d_A);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (d_A)\n");
    }

    status = cublasAlloc(size[9], sizeof(double), (void**)&d_b);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (d_b)\n");
    }

    status = cublasAlloc(lwork, sizeof(double), (void**)&d_work);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (d_work)\n");
    }

    //status = cublasAlloc(nb, sizeof(double), (void**)&d_x);
    status = cublasAlloc(size[9], sizeof(double), (void**)&d_x);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (d_x)\n");
    }

    cudaMallocHost( (void**)&h_work, lwork*sizeof(double) );
    if (h_work == 0) {
      fprintf (stderr, "!!!! host memory allocation error (work)\n");
    }

    printf("\n\n");
    printf("  N    CPU GFlop/s    GPU GFlop/s    || b-Ax || / ||A||\n");
    printf("========================================================\n");
    for(i=0; i<10; i++){
      M = N = lda = size[i];
      n2 = N*N;

      for(j = 0; j < n2; j++)
	h_A[j] = rand() / (double)RAND_MAX;

      for(j=0; j<N; j++)
	r[j] = b[j] = rand() / (double)RAND_MAX;

      cublasSetVector(n2, sizeof(double), h_A, 1, d_A, 1);
      magma_dgeqrf_gpu(&N, &N, d_A, &N, tau, h_work, &lwork, d_work, info);
      cublasSetVector(n2, sizeof(double), h_A, 1, d_A, 1);
      cublasSetVector(N, sizeof(double), b, 1, d_b, 1);

      /* ====================================================================
         Performs operation using MAGMA
	 =================================================================== */
      start = get_current_time();
      magma_dgeqrf_gpu2(&M, &N, d_A, &N, tau, h_work, &lwork, d_work, info);

      // Solve the least-squares problem min || A * X - B ||
      int nrhs = 1; 
      magma_dgeqrs_gpu(&M, &N, &nrhs, d_A, &N, tau, 
		       d_b, &M, h_work, &lwork, d_work, info);
      end = get_current_time();

      gpu_perf = (4.*N*N*N/3.+2.*N*N)/(1000000.*GetTimerValue(start,end));

      double work[1], fone = 1.0, mone = -1., matnorm;
      int one = 1;
      
      // get the solution in x
      cublasGetVector(N, sizeof(double), d_b, 1, x, 1);

      dgemv_("n", &N, &N, &mone, h_A, &N, x, &one, &fone, r, &one);
      matnorm = dlange_("f", &N, &N, h_A, &N, work);

      /* =====================================================================
         Performs operation using LAPACK 
	 =================================================================== */
      start = get_current_time();
      dgeqrf_(&M, &N, h_A, &lda, tau, h_work, &lwork, info);
      if (info[0] < 0)  
	printf("Argument %d of sgeqrf had an illegal value.\n", -info[0]);

      // Solve the least-squares problem: min || A * X - B ||
      dormqr_("l", "t", &M, &nrhs, &M, h_A, &lda,
	      tau, b, &M, h_work, &lwork, info);

      // B(1:N,1:NRHS) := inv(R) * B(1:N,1:NRHS)
      dtrsm_("l", "u", "n", "n", &M, &nrhs, &fone, h_A, &lda, b, &M);

      end = get_current_time();
      cpu_perf = (4.*N*N*N/3.+2.*N*N)/(1000000.*GetTimerValue(start,end));

      printf("%5d    %6.2f         %6.2f        %e\n",
             size[i], cpu_perf, gpu_perf,
             dlange_("f", &N, &nrhs, r, &N, work)/matnorm );

      if (argc != 1)
	break;
    }

    /* Memory clean up */
    free(h_A);
    free(tau);
    free(x);
    free(b);
    free(r);
    cublasFree(h_work);
    cublasFree(d_work);
    cublasFree(d_x);
    cublasFree(h_R);
    cublasFree(d_A);
    cublasFree(d_b);

    /* Shutdown */
    status = cublasShutdown();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! shutdown error (A)\n");
    }
}
