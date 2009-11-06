/*
    -- MAGMA (version 0.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       June 2009
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


double get_LU_error(double2 *A, double2 *LU, int *IPIV, int N){
  int k1 = 1, k2 = N, intONE = 1, i, j;
  double2 cone = {1.0, 0.0}, czero = {0.0, 0.0};

  zlaswp_( &N, A, &N, &k1, &k2, IPIV, &intONE);

  double2 *L = (double2 *) calloc (N*N, sizeof(double2));
  double *work = (double *) calloc (N+1, sizeof(double));

  for( j = 0; j < N; j++ )
    for( i = 0; i < N; i++ )
      L[i+j*N] = (i > j ? LU[i+j*N] : (i == j ? cone : czero));

  for( j = 0; j < N; j++ )
    for( i = j+1; i < N; i++ )
      LU[i+j*N] = czero;

  double matnorm = zlange_("f", &N, &N, A, &N, work);
  double2 alpha = {1.0, 0.0};

  ztrmm_("L", "L", "N", "N", &N, &N, &alpha, L, &N, LU, &N);

  for( j = 0; j < N*N; j++ ){
    LU[j].x = LU[j].x - A[j].x;
    LU[j].y = LU[j].y - A[j].y;
  }
  
  double residual = zlange_("f", &N, &N, LU, &N, work);

  free(L);
  free(work);
  
  return residual / (matnorm * N);
}


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgetrf
*/
int main( int argc, char** argv) 
{
  //int device = 0;
  //cudaSetDevice(device);

    cuInit( 0 );
    cublasInit( );
    printout_devices( );

    double2 *h_A, *h_R, *h_work;
    double2 *d_A;
    int *ipiv;
    double gpu_perf, cpu_perf;

    TimeStruct start, end;

    /* Matrix size */
    int N=0, n2, lda; 
    int size[7] = {1024,2048,3072,4032,5184,6016,7040};
    
    cublasStatus status;
    int i, j, info[1];

    if (argc != 1){
      for(i = 1; i<argc; i++){	
	if (strcmp("-N", argv[i])==0)
	  N = atoi(argv[++i]);
      }
      if (N>0) size[0] = size[6] = N;
      else exit(1);
    }
    else {
      printf("\nUsage: \n");
      printf("  testing_zgetrf -N %d\n\n", 1024);
    }

    /* Initialize CUBLAS */
    status = cublasInit();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! CUBLAS initialization error\n");
    }

    lda = N;
    n2 = size[6] * size[6];

    /* Allocate host memory for the matrix */
    h_A = (double2*)malloc(n2 * sizeof(h_A[0]));
    if (h_A == 0) {
        fprintf (stderr, "!!!! host memory allocation error (A)\n");
    }

    ipiv = (int*)malloc(size[6] * sizeof(int));
    if (ipiv == 0) {
      fprintf (stderr, "!!!! host memory allocation error (ipiv)\n");
    }
  
    cudaMallocHost( (void**)&h_R,  n2*sizeof(double2) );
    if (h_R == 0) {
        fprintf (stderr, "!!!! host memory allocation error (R)\n");
    }

    int maxnb = magma_get_zgetrf_nb(size[6]);
    int lwork = size[6]*maxnb;
    //status=cublasAlloc(n2+lwork+2*maxnb*maxnb,sizeof(double2), (void**)&d_A);
    status = cublasAlloc((size[6]+32)*(size[6]+32) + 32*maxnb + 
			 lwork+2*maxnb*maxnb,sizeof(double2), (void**)&d_A);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (d_A)\n");
    }

    cudaMallocHost( (void**)&h_work, lwork*sizeof(double2) );
    if (h_work == 0) {
      fprintf (stderr, "!!!! host memory allocation error (work)\n");
    }

    printf("\n\n");
    printf("  N    CPU GFlop/s    GPU GFlop/s    ||PA-LU|| / (||A||*N)\n");
    printf("==========================================================\n");
    for(i=0; i<7; i++){
      N = lda = size[i];
      n2 = N*N;

      for(j = 0; j < n2; j++){
	h_R[j].x = h_A[j].x = rand() / (double)RAND_MAX;
	h_R[j].y = h_A[j].y = rand() / (double)RAND_MAX;
      }
      magma_zgetrf(&N, &N, h_R, &N, ipiv, h_work, d_A, info);

      for(j=0; j<n2; j++){
        h_R[j].x = h_A[j].x;  
	h_R[j].y = h_A[j].y;
      }

      /* =====================================================================
         Performs operation using LAPACK
         =================================================================== */
      start = get_current_time();
      zgetrf_(&N, &N, h_A, &lda, ipiv, info);
      end = get_current_time();
      if (info[0] < 0)
        printf("Argument %d of zgetrf had an illegal value.\n", -info[0]);

      cpu_perf = 4.*2.*N*N*N/(3.*1000000*GetTimerValue(start,end));
      // printf("CPU Processing time: %f (ms) \n", GetTimerValue(start,end));
      
      for(j=0; j<n2; j++){
        h_A[j].x = h_R[j].x;
	h_A[j].y = h_R[j].y;
      }
      /* ====================================================================
         Performs operation using MAGMA
	 =================================================================== */
      start = get_current_time();
      magma_zgetrf(&N, &N, h_R, &N, ipiv, h_work, d_A, info);
      end = get_current_time();
    
      gpu_perf = 4.*2.*N*N*N/(3.*1000000*GetTimerValue(start,end));
      //printf("GPU Processing time: %f (ms) \n", GetTimerValue(start,end));
     
      /* =====================================================================
	 Check the factorization
	 =================================================================== */
      double error = get_LU_error(h_A, h_R, ipiv, N);
      
      printf("%5d    %6.2f         %6.2f         %e\n",
             size[i], cpu_perf, gpu_perf, error);

      if (argc != 1)
	break;
    }

    /* Memory clean up */
    free(h_A);
    free(ipiv);
    cublasFree(h_work);
    cublasFree(h_R);
    cublasFree(d_A);

    /* Shutdown */
    status = cublasShutdown();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! shutdown error (A)\n");
    }
}
