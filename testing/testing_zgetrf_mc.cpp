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

#ifndef min
#define min(a,b)  (((a)<(b))?(a):(b))
#endif

// block size
int EN_BEE;

// QUARK scheduler intialized here
Quark *quark;

double get_LU_error(int M, int N, cuDoubleComplex *A, int lda, cuDoubleComplex *LU, int *IPIV){
  int min_mn = min(M,N), intONE = 1, i, j;

  lapackf77_zlaswp( &N, A, &lda, &intONE, &min_mn, IPIV, &intONE);

  cuDoubleComplex *L = (cuDoubleComplex *) calloc (M*min_mn, sizeof(cuDoubleComplex));
  cuDoubleComplex *U = (cuDoubleComplex *) calloc (min_mn*N, sizeof(cuDoubleComplex));
  double  *work = (double *) calloc (M+1, sizeof(cuDoubleComplex));

  memset( L, 0, M*min_mn*sizeof(cuDoubleComplex) );
  memset( U, 0, min_mn*N*sizeof(cuDoubleComplex) );

  lapackf77_zlacpy( MagmaLowerStr, &M, &min_mn, LU, &lda, L, &M      );
  lapackf77_zlacpy( MagmaUpperStr, &min_mn, &N, LU, &lda, U, &min_mn );

  for(j=0; j<min_mn; j++)
    L[j+j*M] = MAGMA_Z_MAKE( 1., 0. );

  double matnorm = lapackf77_zlange("f", &M, &N, A, &lda, work);
  cuDoubleComplex alpha = MAGMA_Z_ONE;
  cuDoubleComplex beta  = MAGMA_Z_ZERO;

  blasf77_zgemm("N", "N", &M, &N, &min_mn,
                &alpha, L, &M, U, &min_mn, &beta, LU, &lda);

  for( j = 0; j < N; j++ ) {
    for( i = 0; i < M; i++ ) {
      MAGMA_Z_OP_NEG( LU[i+j*lda], LU[i+j*lda], A[i+j*lda]);
    }
  }
  double residual = lapackf77_zlange("f", &M, &N, LU, &lda, work);

  free(L);
  free(work);

  return residual / (matnorm * N);
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgetrf_mc
*/
int main( int argc, char** argv) 
{
    cuDoubleComplex *h_A, *h_A2, *h_R, *h_work;
    cuDoubleComplex *d_A;
    int *ipiv, *dipiv;
    float gpu_perf, cpu_perf, cpu2_perf;

    TimeStruct start, end;

    /* Matrix size */
    int N=0, n2, lda, M=0;
    int size[10] = {1024,2048,3072,4032,5184,6016,7040,8064,9088,10112};
    
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
        else if (strcmp("-M", argv[i])==0)
          M = atoi(argv[++i]);
        else if (strcmp("-C", argv[i])==0)
          cores = atoi(argv[++i]);
        else if (strcmp("-B", argv[i])==0)
          EN_BEE = atoi(argv[++i]);
      }
      if ((M>0 && N>0) || (M==0 && N==0)) {
        printf("  testing_zgetrf_mc -M %d -N %d -B %d\n\n", M, N, EN_BEE);
        if (M==0 && N==0) {
          M = N = size[9];
          loop = 1;
        }
      } else {
        printf("\nUsage: \n");
        printf("  testing_zgetrf_mc -M %d -N %d -B 128 -T 1\n\n", 1024, 1024);
        exit(1);
      }
    } else {
      printf("\nUsage: \n");
      printf("  testing_zgetrf_mc -M %d -N %d -B 128 -T 1\n\n", 1024, 1024);
      M = N = size[9];
    }

    n2 = M * N;

    int min_mn = min(M, N);

    /* Allocate host memory for the matrix */
    h_A2 = (cuDoubleComplex*)malloc(n2 * sizeof(h_A2[0]));
    if (h_A2 == 0) {
        fprintf (stderr, "!!!! host memory allocation error (h_A2)\n");
    }

    h_A = (cuDoubleComplex*)malloc(n2 * sizeof(h_A[0]));
    if (h_A == 0) {
        fprintf (stderr, "!!!! host memory allocation error (h_A)\n");
    }

    ipiv = (int*)malloc(min_mn * sizeof(int));
    if (ipiv == 0) {
      fprintf (stderr, "!!!! host memory allocation error (ipiv)\n");
    }

    printf("\n\n");
    printf("  M    N   magma_sgetrf_mc GFlop/s     ||PA-LU|| / (||A||*N)\n");
    printf("========================================================\n");
    for(i=0; i<10; i++){

      if (loop == 1) {
        M = N = min_mn = size[i];
        n2 = M*N;
      }

      /* Initialize the matrix */
      lapackf77_zlarnv( &ione, ISEED, &n2, h_A2 );
      lapackf77_zlacpy( MagmaUpperLowerStr, &M, &N, h_A2, &M, h_A, &M );

      /* =====================================================================
         Performs operation using multi-core
         =================================================================== */

	  // initialize scheduler
      quark = QUARK_New(cores);

      start = get_current_time();
      magma_zgetrf_mc(&M, &N, h_A2, &M, ipiv, info);
      end = get_current_time();

      // shut down scheduler
      QUARK_Delete(quark);

      if (info[0] < 0)      
        printf("Argument %d of magma_sgeqrf_mc had an illegal value.\n", -info[0]);

      cpu2_perf = 2.*M*N*min(M,N)/(3.*1000000*GetTimerValue(start,end));
  
      double error = get_LU_error(M, N, h_A, M, h_A2, ipiv);

      printf("%5d %5d       %6.2f                  %e\n",
             M, N, cpu2_perf, error);

      if (loop != 1)
        break;
    }

    /* Memory clean up */
    free(h_A2);
    free(h_A);
    free(ipiv);
}
