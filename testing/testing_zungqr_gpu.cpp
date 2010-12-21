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
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas.h>

// includes, project
#include "magma.h"

#define min(a,b)  (((a)<(b))?(a):(b))

// Flops formula
#define PRECISION_z
#if defined(PRECISION_z) || defined(PRECISION_c)
#define FLOPS(m, n, k) ( 4.* 4.* m * min(m, n) * k /3. )
#else
#define FLOPS(m, n, k) (     4.* m * min(m, n) * k /3. )
#endif


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing sorgqr_gpu
*/
int main( int argc, char** argv) 
{
    cuInit( 0 );
    cublasInit( );
    printout_devices( );

    cuDoubleComplex *h_R, *h_A, *h_work, *tau;
    cuDoubleComplex *d_A, *d_work;
    double gpu_perf, cpu_perf;

    TimeStruct start, end;

    /* Matrix size */
    magma_int_t M=0, N=0, K, n2, lda;
    magma_int_t size[10] = {1024,2048,3072,4032,5184,6016,7040,8064,9088,9984};
    
    cublasStatus status;
    magma_int_t i, info[1];
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};

    if (argc != 1){
      for(i = 1; i<argc; i++){	
	if (strcmp("-N", argv[i])==0)
	  N = atoi(argv[++i]);
	else if (strcmp("-M", argv[i])==0)
          M = atoi(argv[++i]);
        else if (strcmp("-K", argv[i])==0)
	  K = atoi(argv[++i]);
      }
      if (N>0 && M>0 && M >= N && K >0 && K <= N)
	printf("  testing_sorgqr_gpu -M %d -N %d -K %d\n\n", M, N, K);
      else
        {
          printf("\nUsage: \n");
          printf("  testing_sorgqr_gpu  -M %d  -N %d  -K %d\n\n", M, N, K);
	  printf("  M, N, and K have to to be K <= N <= M, exit.\n");
          exit(1);
        }
    }
    else {
      printf("\nUsage: \n");
      printf("  testing_sorgqr_gpu -M %d  -N %d  -K %d\n\n", 1024, 1024, 1024);
      M = N = K = size[9];
    }
    
    /* Initialize CUBLAS */
    status = cublasInit();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! CUBLAS initialization error\n");
    }

    lda = (M/32)*32;
    if (lda<M) lda+=32;
    n2  = M * N;

    magma_int_t min_mn = min(M, N);

    /* Allocate host memory for the matrix */
    h_R = (cuDoubleComplex*)malloc(n2 * sizeof(h_R[0]));
    if (h_R == 0) {
        fprintf (stderr, "!!!! host memory allocation error (h_R)\n");
    }

    tau = (cuDoubleComplex*)malloc(min_mn * sizeof(cuDoubleComplex));
    if (tau == 0) {
      fprintf (stderr, "!!!! host memory allocation error (tau)\n");
    }
  
    cudaMallocHost( (void**)&h_A,  n2*sizeof(cuDoubleComplex) );
    if (h_A == 0) {
        fprintf (stderr, "!!!! host memory allocation error (h_A)\n");
    }

    magma_int_t nb = magma_get_zgeqrf_nb(M);
    magma_int_t lwork = (M+2*N+nb)*nb;

    lwork = (M+2*N+nb)*nb;

    status = cublasAlloc(lda*N, sizeof(cuDoubleComplex), (void**)&d_A);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (d_A)\n");
    }

    status = cublasAlloc(lwork, sizeof(cuDoubleComplex), (void**)&d_work);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (d_work)\n");
    }

    cudaMallocHost( (void**)&h_work, lwork*sizeof(cuDoubleComplex) );
    if (h_work == 0) {
      fprintf (stderr, "!!!! host memory allocation error (work)\n");
    }

    printf("\n");
    printf("  M     N    CPU GFlop/s   GPU GFlop/s   ||R|| / ||A||\n");
    printf("=======================================================\n");
    for(i=0; i<10; i++){
      if (argc == 1){
	M = N = K = min_mn = size[i];
        n2 = M*N;

        lda = (M/32)*32;
	if (lda<M) lda+=32;
      }


      lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
      lapackf77_zlacpy( MagmaUpperLowerStr, &M, &N, h_A, &M, h_R, &M );

      cublasSetMatrix( M, N, sizeof(cuDoubleComplex), h_A, M, d_A, lda);
      magma_zgeqrf2_gpu(M, N, d_A, lda, tau, info);
      cublasSetMatrix( M, N, sizeof(cuDoubleComplex), h_A, M, d_A, lda);

      /* ====================================================================
         Performs operation using MAGMA
	 =================================================================== */
      magma_zgeqrf_gpu(M, N, d_A, lda, tau, d_work, info);

      start = get_current_time();
      nb = magma_get_zgeqrf_nb(M);
      magma_zungqr_gpu(M, N, K, d_A, lda, tau, d_work, nb, info);
      end = get_current_time();

      // Get d_A back to the CPU to compare with the CPU result.
      cublasGetMatrix(M, N, sizeof(cuDoubleComplex), d_A, lda, h_R, M);

      gpu_perf= FLOPS(M, N, K)/(1000000. * GetTimerValue(start,end));
      cuDoubleComplex mone = MAGMA_Z_NEG_ONE;
      double work[1];
      double matnorm = lapackf77_zlange("f", &M, &N, h_A, &M, work);

      /* =====================================================================
         Performs operation using LAPACK 
	 =================================================================== */
      lapackf77_zgeqrf(&M, &N, h_A, &M, tau, h_work, &lwork, info);
      if (info[0] < 0)  
	printf("Argument %d of sgeqrf had an illegal value.\n", -info[0]);

      start = get_current_time();
      //lapackf77_zungqr(&M, &N, &K, h_A, &M, tau, h_work, &lwork, info);
      magma_zungqr(M, N, K, h_A, M, tau, d_work, nb, info);
      end = get_current_time();

      cpu_perf = FLOPS(M, N, K)/(1000000.* GetTimerValue(start,end));      
      blasf77_zaxpy(&n2, &mone, h_A, &ione, h_R, &ione);
      
      printf("%5d %5d   %6.1f       %6.1f         %7.2e \n",
             M, N, cpu_perf, gpu_perf,
             lapackf77_zlange("f", &M, &N, h_R, &M, work) / matnorm );

      if (argc != 1)
	break;
    }

    /* Memory clean up */
    free(h_R);
    free(tau);
    cudaFreeHost(h_work);
    cublasFree(d_work);
    cudaFreeHost(h_A);
    cublasFree(d_A);

    /* Shutdown */
    status = cublasShutdown();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! shutdown error (A)\n");
    }
}
