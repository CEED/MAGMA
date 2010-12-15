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

int EN_BEE;

Quark *quark;


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgeqrf
*/
int main( int argc, char** argv) 
{
    EN_BEE = 128;

    cuDoubleComplex *h_A, *h_R, *h_A2, *h_A3, *h_work, *h_work2, *tau, *d_work2;
    cuDoubleComplex *d_A, *d_work;
    float gpu_perf, cpu_perf, cpu2_perf;

    TimeStruct start, end;

    /* Matrix size */
    int N=0, n2, lda, M=0;
    int size[10] = {1024,2048,3072,4032,5184,6016,7040,8064,9088,10112};
    
    int i, j, info[1];

    int loop = argc;

    int ione     = 1;
    int ISEED[4] = {0,0,0,1};

    int cores = 4;

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
        printf("  testing_zgeqrf_mc -M %d -N %d -B %d\n\n", M, N, EN_BEE);
        if (M==0 && N==0) {
          M = N = size[9];
          loop = 1;
        }
      } else {
        printf("\nUsage: \n");
        printf("  testing_zgeqrf_mc -M %d -N %d -B 128 -T 1\n\n", 1024, 1024);
        exit(1);
      }
    } else {
      printf("\nUsage: \n");
      printf("  testing_zgeqrf_mc -M %d -N %d -B 128 -T 1\n\n", 1024, 1024);
      M = N = size[9];
    }

    n2 = M * N;

    int min_mn = min(M,N);

    /* Allocate host memory for the matrix */
    h_A2 = (cuDoubleComplex*)malloc(n2 * sizeof(h_A2[0]));
    if (h_A2 == 0) {
        fprintf (stderr, "!!!! host memory allocation error (A2)\n");
    }

    int lwork = n2;

    h_work2 = (cuDoubleComplex*)malloc(lwork * sizeof(cuDoubleComplex));
    if (h_work2 == 0) {
        fprintf (stderr, "!!!! host memory allocation error (h_work2)\n");
    }

    h_A3 = (cuDoubleComplex*)malloc(n2 * sizeof(h_A3[0]));
    if (h_A3 == 0) {
        fprintf (stderr, "!!!! host memory allocation error (A3)\n");
    }

    tau = (cuDoubleComplex*)malloc(min_mn * sizeof(cuDoubleComplex));
    if (tau == 0) {
      fprintf (stderr, "!!!! host memory allocation error (tau)\n");
    }

    printf("\n\n");
    printf("   M     N          sgeqrf      magma_sgeqrf_mc Gflop/s    ||R||_F / ||A||_F\n");
    printf("==============================================================================\n");
    for(i=0; i<10; i++){

      if (loop == 1) {
        M = N = size[i];
        n2 = M*N;
      }

      /* Initialize the matrix */
      lapackf77_zlarnv( &ione, ISEED, &n2, h_A2 );
      lapackf77_zlacpy( MagmaUpperLowerStr, &M, &N, h_A2, &M, h_A3, &M );

	  /* =====================================================================
         Performs operation using LAPACK 
	 =================================================================== */

      start = get_current_time();
      lapackf77_zgeqrf(&M, &N, h_A3, &M, tau, h_work2, &lwork, info);
      end = get_current_time();

      if (info[0] < 0)  
        printf("Argument %d of sgeqrf had an illegal value.\n", -info[0]);
 
      cpu2_perf = 4.*M*N*min(M,N)/(3.*1000000*GetTimerValue(start,end));

	  /* =====================================================================
         Performs operation using multicore 
	 =================================================================== */

      // initialize QUARK
      quark = QUARK_New(cores);

      start = get_current_time();
      magma_zgeqrf_mc(&M, &N, h_A2, &M, tau, h_work2, &lwork, info);
      end = get_current_time();

      // shut down QUARK
      QUARK_Delete(quark);

      if (info[0] < 0)  
        printf("Argument %d of sgeqrf had an illegal value.\n", -info[0]);
  
      cpu_perf = 4.*M*N*min(M,N)/(3.*1000000*GetTimerValue(start,end));
      
      /* =====================================================================
         Check the result compared to LAPACK
         =================================================================== */

      double work[1], matnorm = 1.;
	  cuDoubleComplex mone = MAGMA_Z_NEG_ONE;
      int one = 1;
      matnorm = lapackf77_zlange("f", &M, &N, h_A2, &M, work);

      blasf77_zaxpy(&n2, &mone, h_A2, &one, h_A3, &one);
      printf("%5d  %5d      %6.2f                %6.2f           %e\n", 
	     M,  N, cpu2_perf, cpu_perf,
	     lapackf77_zlange("f", &M, &N, h_A3, &M, work) / matnorm);

      if (loop != 1)
	break;
    }

    /* Memory clean up */
    free(h_A2);
    free(tau);
	free(h_A3);
	free(h_work2);
}
