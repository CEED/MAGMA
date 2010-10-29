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

#define min(a,b)  (((a)<(b))?(a):(b))

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgeqrs
*/
int main( int argc, char** argv) 
{
    cuInit( 0 );
    cublasInit( );
    printout_devices( );

    double2 *h_A, *h_R, *h_work, *tau;
    double2 *d_A, *d_work, *d_x;
    double2 gpu_perf, cpu_perf;

    double2 *x, *b, *r;
    double2 *d_b;

    int nrhs = 3;

    TimeStruct start, end;

    /* Matrix size */
    int M=0, N=0, n2, lda;
    int size[10] = {1024,2048,3072,4032,5184,6016,7040,8064,9088,10112};
    
    cublasStatus status;
    int i, j, info[1];

    if (argc != 1){
      for(i = 1; i<argc; i++){	
	if (strcmp("-N", argv[i])==0)
	  N = atoi(argv[++i]);
	else if (strcmp("-M", argv[i])==0)
          M = atoi(argv[++i]);
        else if (strcmp("-nrhs", argv[i])==0)
	  nrhs = atoi(argv[++i]);
      }
      if (N>0 && M>0 && M >= N)
	printf("  testing_zgeqrs_gpu -nrhs %d -M %d -N %d\n\n", nrhs, M, N);
      else
        {
          printf("\nUsage: \n");
          printf("  testing_zgeqrs_gpu -nrhs %d  -M %d  -N %d\n\n", nrhs, M, N);
	  printf("  M has to be >= N, exit.\n");
          exit(1);
        }
    }
    else {
      printf("\nUsage: \n");
      printf("  testing_zgeqrs_gpu -nrhs %d  -M %d  -N %d\n\n", nrhs, 1024, 1024);
      M = N = size[9];
    }

    /* Initialize CUBLAS */
    status = cublasInit();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! CUBLAS initialization error\n");
    }

    lda = (M/32)*32;
    if (lda<M) lda+=32;
    n2  = M * N;

    int min_mn = min(M, N);

    /* Allocate host memory for the matrix */
    h_A = (double2*)malloc(n2 * sizeof(h_A[0]));
    if (h_A == 0) {
        fprintf (stderr, "!!!! host memory allocation error (A)\n");
    }

    tau = (double2*)malloc(min_mn * sizeof(double2));
    if (tau == 0) {
      fprintf (stderr, "!!!! host memory allocation error (tau)\n");
    }
  
    x = (double2*)malloc(nrhs* M * sizeof(double2));
    b = (double2*)malloc(nrhs* M * sizeof(double2));
    r = (double2*)malloc(nrhs* M * sizeof(double2));

    cudaMallocHost( (void**)&h_R,  n2*sizeof(double2) );
    if (h_R == 0) {
        fprintf (stderr, "!!!! host memory allocation error (R)\n");
    }

    int nb = magma_get_zgeqrf_nb(M);
    // int lwork = (3*size[9]+nb)*nb;
    int lwork = (M+2*N+nb)*nb;

    if (nrhs > nb)
      lwork = (M+2*N+nb)*nrhs;

    status = cublasAlloc(lda*N, sizeof(double2), (void**)&d_A);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (d_A)\n");
    }

    status = cublasAlloc(nrhs * M, sizeof(double2), (void**)&d_b);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (d_b)\n");
    }

    status = cublasAlloc(lwork, sizeof(double2), (void**)&d_work);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (d_work)\n");
    }

    status = cublasAlloc(nrhs * N, sizeof(double2), (void**)&d_x);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (d_x)\n");
    }

    cudaMallocHost( (void**)&h_work, lwork*sizeof(double2) );
    if (h_work == 0) {
      fprintf (stderr, "!!!! host memory allocation error (work)\n");
    }

    printf("\n");
    printf("                                         ||b-Ax|| / (N||A||)\n");
    printf("  M     N    CPU GFlop/s   GPU GFlop/s      GPU      CPU    \n");
    printf("============================================================\n");
    for(i=0; i<10; i++){
      if (argc == 1){
	M = N = min_mn = size[i];
        n2 = M*N;

        lda = (M/32)*32;
	if (lda<M) lda+=32;
      }

      for(j = 0; j < n2; j++)
	h_A[j] = h_R[j] = rand() / (double2)RAND_MAX;

      for(int k=0; k<nrhs; k++)
	for(j=0; j<M; j++)
	  r[j+k*M] = b[j+k*M] = rand() / (double2)RAND_MAX;

      cublasSetMatrix( M, N, sizeof(double2), h_A, M, d_A, lda);
      magma_zgeqrf_gpu( M, N, d_A, lda, tau, info);
      cublasSetMatrix( M, N, sizeof(double2), h_A, M, d_A, lda);
      cublasSetMatrix( M, nrhs, sizeof(double2), b, M, d_b, M);

      /* ====================================================================
         Performs operation using MAGMA
	 =================================================================== */
      start = get_current_time();
      magma_zgeqrf_gpu2( M, N, d_A, lda, tau, d_work, info);
      
      // Solve the least-squares problem min || A * X - B || 
      magma_zgeqrs_gpu( M, N, nrhs, d_A, lda, tau, 
		       d_b, M, h_work, &lwork, d_work, info);
      end = get_current_time();

      gpu_perf=(4.*M*N*min_mn/3. + 3.*nrhs*N*N)/(1000000.*
						 GetTimerValue(start,end));

      double2 work[1], fone = 1.0, mone = -1., matnorm;
      int one = 1;
      
      // get the solution in x
      cublasGetMatrix(N, nrhs, sizeof(double2), d_b, M, x, N);

      // compute the residual
      if (nrhs == 1)
	sgemv_("n", &M, &N, &mone, h_A, &M, x, &one, &fone, r, &one);
      else
	zgemm_("n","n", &M, &nrhs, &N, &mone, h_A, &M, x, &N, &fone, r, &M);
      matnorm = zlange_("f", &M, &N, h_A, &M, work);

      /* =====================================================================
         Performs operation using LAPACK 
	 =================================================================== */
      for(int k=0; k<nrhs; k++)
	for(j=0; j<M; j++)
          x[j+k*M] = b[j+k*M];
      
      start = get_current_time();
      zgeqrf_(&M, &N, h_R, &M, tau, h_work, &lwork, info);

      if (info[0] < 0)  
	printf("Argument %d of zgeqrf had an illegal value.\n", -info[0]);

      // Solve the least-squares problem: min || A * X - B ||
      // 1. B(1:M,1:NRHS) = Q^T B(1:M,1:NRHS)
      zunmqr_("l", "t", &M, &nrhs, &min_mn, h_R, &M,
	      tau, x, &M, h_work, &lwork, info);

      // 2. B(1:N,1:NRHS) := inv(R) * B(1:M,1:NRHS)
      ztrsm_("l", "u", "n", "n", &N, &nrhs, &fone, h_R, &M, x, &M);

      end = get_current_time();
      cpu_perf = (4.*M*N*min_mn/3.+3.*nrhs*N*N)/(1000000.*
						 GetTimerValue(start,end));

      if (nrhs == 1)
        sgemv_("n", &M, &N, &mone, h_A, &M, x, &one, &fone, b, &one);
      else
        zgemm_("n","n", &M, &nrhs, &N, &mone, h_A, &M, x, &M, &fone, b, &M);

      printf("%5d %5d   %6.1f       %6.1f       %7.2e   %7.2e\n",
             M, N, cpu_perf, gpu_perf,
             zlange_("f", &M, &nrhs, r, &M, work)/(min_mn*matnorm),
	     zlange_("f", &M, &nrhs, b, &M, work)/(min_mn*matnorm) );

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
