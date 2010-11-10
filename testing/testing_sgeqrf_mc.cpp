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

#include <quark.h>

// includes, project
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cublas.h"
#include "magma.h"

int EN_BEE;

int TRACE;


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing sgeqrf
*/
int main( int argc, char** argv) 
{
    //cuInit( 0 );
    //cublasInit( );
    //printout_devices( );

    EN_BEE = 128;

	TRACE = 0;

    float *h_A, *h_R, *h_A2, *h_A3, *h_work, *h_work2, *tau, *d_work2;
    float *d_A, *d_work;
    float gpu_perf, cpu_perf, cpu2_perf;

    TimeStruct start, end;

    /* Matrix size */
    int N=0, n2, lda, M=0;
    int size[10] = {1024,2048,3072,4032,5184,6016,7040,8064,9088,10112};
    
    ////cublasStatus status;
    int i, j, info[1];

    int loop = argc;

    if (argc != 1){
      for(i = 1; i<argc; i++){      
        if (strcmp("-N", argv[i])==0)
          N = atoi(argv[++i]);
        else if (strcmp("-M", argv[i])==0)
          M = atoi(argv[++i]);
        else if (strcmp("-T", argv[i])==0)
          TRACE = atoi(argv[++i]);
        else if (strcmp("-B", argv[i])==0)
          EN_BEE = atoi(argv[++i]);
      }
      if ((M>0 && N>0) || (M==0 && N==0)) {
        printf("  testing_sgetrf_gpu -M %d -N %d -B %d -T %d\n\n", M, N, EN_BEE, TRACE);
        if (M==0 && N==0) {
          M = N = size[9];
          loop = 1;
        }
      } else {
        printf("\nUsage: \n");
        printf("  testing_sgetrf_gpu -M %d -N %d -B 128 -T 1\n\n", 1024, 1024);
        exit(1);
      }
    } else {
      printf("\nUsage: \n");
      printf("  testing_sgetrf_gpu -M %d -N %d -B 128 -T 1\n\n", 1024, 1024);
      M = N = size[9];
    }



    /* Initialize CUBLAS */
    ////status = cublasInit();
    ////if (status != CUBLAS_STATUS_SUCCESS) {
        ////fprintf (stderr, "!!!! CUBLAS initialization error\n");
    ////}

    n2 = M * N;

    int min_mn = min(M,N);

    /* Allocate host memory for the matrix */
    h_A = (float*)malloc(n2 * sizeof(h_A[0]));
    if (h_A == 0) {
        fprintf (stderr, "!!!! host memory allocation error (A)\n");
    }

    h_A2 = (float*)malloc(n2 * sizeof(h_A2[0]));
    if (h_A2 == 0) {
        fprintf (stderr, "!!!! host memory allocation error (A2)\n");
    }

    ////int lwork = 2*size[9]*nb;
    int lwork = n2;

    ////d_work2 = (float*)malloc(size[9]*(size[9]+nb)+nb*nb * sizeof(float));
    ////if (d_work2 == 0) {
        ////fprintf (stderr, "!!!! host memory allocation error (d_work2)\n");
    ////}

    h_work2 = (float*)malloc(lwork * sizeof(float));
    if (h_work2 == 0) {
        fprintf (stderr, "!!!! host memory allocation error (h_work2)\n");
    }

    h_A3 = (float*)malloc(n2 * sizeof(h_A3[0]));
    if (h_A3 == 0) {
        fprintf (stderr, "!!!! host memory allocation error (A3)\n");
    }

    tau = (float*)malloc(min_mn * sizeof(float));
    if (tau == 0) {
      fprintf (stderr, "!!!! host memory allocation error (tau)\n");
    }
    ////cudaMallocHost( (void**)&h_R,  n2*sizeof(float) );
    ////if (h_R == 0) {
        ////fprintf (stderr, "!!!! host memory allocation error (R)\n");
    ////}

    ////int lwork = 2*size[9]*magma_get_sgeqrf_nb(size[9]);
    ////status = cublasAlloc(n2, sizeof(float), (void**)&d_A);
    ////if (status != CUBLAS_STATUS_SUCCESS) {
      ////fprintf (stderr, "!!!! device memory allocation error (d_A)\n");
    ////}

	////int nb = magma_get_sgeqrf_nb(size[9]);

    //status = cublasAlloc(lwork/2, sizeof(float), (void**)&d_work);
    //status = cublasAlloc(size[9]*(size[9]+nb)+nb*nb, sizeof(float), (void**)&d_work);
    ////status = cublasAlloc(size[9]*(size[9]+nb)+nb*nb, sizeof(float), (void**)&d_work);
    ////if (status != CUBLAS_STATUS_SUCCESS) {
      ////fprintf (stderr, "!!!! device memory allocation error (d_work)\n");
    ////}

    ////cudaMallocHost( (void**)&h_work, lwork*sizeof(float) );
    ////if (h_work == 0) {
      ////fprintf (stderr, "!!!! host memory allocation error (work)\n");
    ////}

    printf("\n\n");
    printf("  N    magma_sgeqrf_mc Gflop/s    ||R||_F / ||A||_F\n");
    printf("===================================================\n");
    for(i=0; i<10; i++){

      if (loop == 1) {
        M = N = size[i];
        n2 = M*N;
      }

      for(j = 0; j < n2; j++){
	    h_A[j] = rand() / (float)RAND_MAX;
	    h_A2[j] = h_A[j];
	    h_A3[j] = h_A[j];
	  }

      ////cublasSetVector(n2, sizeof(float), h_A, 1, d_A, 1);
      //magma_sgeqrf_gpu(&N, &N, d_A, &N, tau, h_work, &lwork, d_work, info);
      ////cublasSetVector(n2, sizeof(float), h_A, 1, d_A, 1);
  
      /* ====================================================================
         Performs operation using MAGMA
	 =================================================================== */

      ////start = get_current_time();
      ////magma_sgeqrf_gpu(&N, &N, d_A,&N, tau, h_work, &lwork, d_work, info);
      ////end = get_current_time();
   
	  ////gpu_perf=0.0;
      ////gpu_perf = 4.*N*N*N/(3.*1000000*GetTimerValue(start,end));
      // printf("GPU Processing time: %f (ms) \n", GetTimerValue(start,end));

      /* =====================================================================
         Performs operation using LAPACK 
	 =================================================================== */

      start = get_current_time();
      sgeqrf_(&M, &N, h_A3, &M, tau, h_work2, &lwork, info);
      end = get_current_time();

      if (info[0] < 0)  
        printf("Argument %d of sgeqrf had an illegal value.\n", -info[0]);
 
      cpu2_perf = 4.*N*N*N/(3.*1000000*GetTimerValue(start,end));
      /* =====================================================================
         Performs operation using multicore 
	 =================================================================== */

      start = get_current_time();
      magma_sgeqrf_mc(&M, &N, h_A2, &M, tau, h_work2, &lwork, info);
      end = get_current_time();

      if (info[0] < 0)  
        printf("Argument %d of sgeqrf had an illegal value.\n", -info[0]);
  
      cpu_perf = 4.*N*N*N/(3.*1000000*GetTimerValue(start,end));
      // printf("CPU Processing time: %f (ms) \n", GetTimerValue(start,end));
      
      /* =====================================================================
         Check the result compared to LAPACK
         =================================================================== */
      ////cublasGetVector(n2, sizeof(float), d_A, 1, h_R, 1);

      float work[1], matnorm = 1., mone = -1.;
      int one = 1;
      matnorm = slange_("f", &M, &N, h_A2, &M, work);

      saxpy_(&n2, &mone, h_A2, &one, h_A3, &one);
      printf("%5d         %6.2f                          %e\n", 
	     size[i], cpu_perf,
	     slange_("f", &M, &N, h_A3, &M, work) / matnorm);
      /* =====================================================================
         Check the factorization
         =================================================================== */
      /*
      float result[2];
      float *hwork_Q = (float*)malloc( N * N * sizeof(float));
      float *hwork_R = (float*)malloc( N * N * sizeof(float));
      float *rwork   = (float*)malloc( N * sizeof(float));

      sqrt02(&N, &N, &N, h_A, h_R, hwork_Q, hwork_R, &N, tau,
             h_work, &lwork, rwork, result);

      printf("norm( R - Q'*A ) / ( M * norm(A) * EPS ) = %f\n", result[0]);
      printf("norm( I - Q'*Q ) / ( M * EPS )           = %f\n", result[1]);
      free(hwork_Q);
      free(hwork_R);
      free(rwork);
      */

      if (loop != 1)
	break;
    }

    /* Memory clean up */
    free(h_A);
    free(tau);
    ////cublasFree(h_work);
    ////cublasFree(d_work);
    ////cublasFree(h_R);
    ////cublasFree(d_A);

    /* Shutdown */
    ////status = cublasShutdown();
    ////if (status != CUBLAS_STATUS_SUCCESS) {
        ////fprintf (stderr, "!!!! shutdown error (A)\n");
    ////}
}
