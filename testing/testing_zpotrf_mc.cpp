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

#include <quark.h>

// includes, project
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cublas.h"
#include "magma.h"
#include "magmablas.h"

//#include "cblas.h"

int EN_BEE;

int TRACE;

Quark *quark;

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing spotrf
*/
int main( int argc, char** argv) 
{
    ////cuInit( 0 );
    ////cublasInit( );
    ////printout_devices( );

    cuDoubleComplex *h_A, *h_R, *h_work, *h_A2;
    cuDoubleComplex *d_A;
    float gpu_perf, cpu_perf, cpu_perf2;

    TimeStruct start, end;

    /* Matrix size */
    int N=0, n2, lda;
    int size[10] = {1024,2048,3072,4032,5184,6048,7200,8064,8928,10080};
    
    int i, j, info[1];

    int ione     = 1;
    int ISEED[4] = {0,0,0,1};

    int cores = 4;

    EN_BEE = 128;

    int loop = argc;
    
    if (argc != 1){
      for(i = 1; i<argc; i++){      
        if (strcmp("-N", argv[i])==0)
          N = atoi(argv[++i]);
        else if (strcmp("-T", argv[i])==0)
          TRACE = atoi(argv[++i]);
        else if (strcmp("-C", argv[i])==0)
          cores = atoi(argv[++i]);
        else if (strcmp("-B", argv[i])==0)
          EN_BEE = atoi(argv[++i]);
      }
      if (N==0) {
        N = size[9];
        loop = 1;
      } else {
        size[0] = size[9] = N;
      }
    } else {
      printf("\nUsage: \n");
      printf("  testing_spotrf_gpu -N %d -B 128 -T 1\n\n", 1024);
      N = size[9];
    }

    ////cublasStatus status;

    /* Initialize CUBLAS */
    ////status = cublasInit();
    ////if (status != CUBLAS_STATUS_SUCCESS) {
        ////fprintf (stderr, "!!!! CUBLAS initialization error\n");
    ////}

    lda = N;
    n2 = size[9] * size[9];

    ////int maxNB = magma_get_spotrf_nb(size[9]);
    ////cudaMallocHost( (void**)&h_work,  maxNB*maxNB*sizeof(float) );

    /* Allocate host memory for the matrix */
    h_A = (cuDoubleComplex*)malloc(n2 * sizeof(h_A[0]));
    if (h_A == 0) {
        fprintf (stderr, "!!!! host memory allocation error (A)\n");
    }

    /* Allocate host memory for the matrix */
    h_A2 = (cuDoubleComplex*)malloc(n2 * sizeof(h_A2[0]));
    if (h_A2 == 0) {
        fprintf (stderr, "!!!! host memory allocation error (A2)\n");
    }
    ////h_R = (float*)malloc(n2 * sizeof(h_R[0]));
    ////cudaMallocHost( (void**)&h_R,  n2*sizeof(float) );
    ////if (h_R == 0) {
        ////fprintf (stderr, "!!!! host memory allocation error (R)\n");
    ////}

    ////status = cublasAlloc(n2, sizeof(float), (void**)&d_A);
    ////if (status != CUBLAS_STATUS_SUCCESS) {
      ////fprintf (stderr, "!!!! device memory allocation error (d_A)\n");
    ////}

    printf("\n\n");
    printf("  N    Multicore GFlop/s    ||R||_F / ||A||_F\n");
    printf("=============================================\n");
    for(i=0; i<10; i++){
      N = lda = size[i];
      n2 = N*N;



    lapackf77_zlarnv( &ione, ISEED, &n2, h_A );

    {      
      int i, j;
      for(i=0; i<N; i++) {
        MAGMA_Z_SET2REAL( h_A[i*lda+i], ( MAGMA_Z_GET_X(h_A[i*lda+i]) + 2000. ) );

        for(j=0; j<i; j++)
          h_A[i*lda+j] = h_A[j*lda+i];
      }
    }

    for(j=0; j<n2; j++)
      h_A2[j] = h_A[j];



      ////cublasSetVector(n2, sizeof(float), h_A, 1, d_A, 1);
      ////magma_spotrf_gpu("U", &N, d_A, &lda, h_work, info);
      ////cublasSetVector(n2, sizeof(float), h_A, 1, d_A, 1);
      
      /* ====================================================================
         Performs operation using MAGMA 
       =================================================================== */
      ////start = get_current_time();
      ////magma_spotrf_gpu("L", &N, d_A, &lda, h_work, info);
      ////magma_spotrf_gpu("U", &N, d_A, &lda, h_work, info);
      ////end = get_current_time();
    
      ////gpu_perf = 1.*N*N*N/(3.*1000000*GetTimerValue(start,end));
      // printf("GPU Processing time: %f (ms) \n", GetTimerValue(start,end));
      // printf("Speed: %f GFlops \n", gpu_perf);

      /* =====================================================================
         Performs operation using LAPACK 
      =================================================================== */
      //start = get_current_time();
      //lapackf77_zpotrf("L", &N, h_A, &lda, info);
      lapackf77_zpotrf("U", &N, h_A, &lda, info);
      //end = get_current_time();

      if (info[0] < 0)  
        printf("Argument %d of spotrf had an illegal value.\n", -info[0]);     

      //cpu_perf = 1.*N*N*N/(3.*1000000*GetTimerValue(start,end));
      /* =====================================================================
         Performs operation using multi-core 
      =================================================================== */

quark = QUARK_New(cores);

      start = get_current_time();
      //magma_zpotrf_mc("L", &N, h_A2, &lda, info);
      magma_zpotrf_mc("U", &N, h_A2, &lda, info);
      end = get_current_time();

QUARK_Delete(quark);

      if (info[0] < 0)  
        printf("Argument %d of magma_spotrf_mc had an illegal value.\n", -info[0]);     
  
      cpu_perf2 = 1.*N*N*N/(3.*1000000*GetTimerValue(start,end));
      // printf("CPU Processing time: %f (ms) \n", GetTimerValue(start,end));
      // printf("Speed: %f GFlops \n", cpu_perf);
      
      /* =====================================================================
         Check the result compared to LAPACK
         =================================================================== */
      ////cublasGetVector(n2, sizeof(float), d_A, 1, h_R, 1);

      double work[1], matnorm = 1.;
      cuDoubleComplex mone = MAGMA_Z_NEG_ONE;
      int one = 1;

      matnorm = lapackf77_zlange("f", &N, &N, h_A, &N, work);
      blasf77_zaxpy(&n2, &mone, h_A, &one, h_A2, &one);
      printf("%5d     %6.2f                %e\n", 
      size[i], cpu_perf2,  
      lapackf77_zlange("f", &N, &N, h_A2, &N, work) / matnorm);

      if (loop != 1)
        break;
    }

    /* Memory clean up */
    free(h_A);
    free(h_A2);
    ////cublasFree(h_work);
    ////cublasFree(h_R);
    ////cublasFree(d_A);

    /* Shutdown */
    ////status = cublasShutdown();
    ////if (status != CUBLAS_STATUS_SUCCESS) {
        ////fprintf (stderr, "!!!! shutdown error (A)\n");
    ////}
}
