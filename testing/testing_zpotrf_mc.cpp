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
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas.h>

// includes, project
#include "magma.h"
#include "magmablas.h"

// block size
int EN_BEE;

// QUARK scheduler initialized here
Quark *quark;

// Flops formula
#define PRECISION_z
#define FMULS(n) ((n) * (1.0 / 6.0 * (n) + 0.5) * (n))
#define FADDS(n) ((n) * (1.0 / 6.0 * (n) )      * (n))
#if defined(PRECISION_z) || defined(PRECISION_c)
#define FLOPS(n) ( 6. * FMULS(n) + 2. * FADDS(n) )
#else
#define FLOPS(n) (      FMULS(n) +      FADDS(n) )
#endif


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zpotrf_mc
*/
int main( int argc, char** argv) 
{
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
      printf("  testing_zpotrf_mc -N %d -B 128 \n\n", 1024);
      N = size[9];
    }

    lda = N;
    n2 = size[9] * size[9];

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

    /* Initialize the Quark scheduler */
    quark = QUARK_New(cores);
    
    printf("\n\n");
    printf("  N    Multicore GFlop/s    ||R||_F / ||A||_F\n");
    printf("=============================================\n");
    for(i=0; i<10; i++)
      {
	N = lda = size[i];
	n2 = N*N;

	lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
	
	for(j=0; j<N; j++) 
	  MAGMA_Z_SET2REAL( h_A[j*lda+j], ( MAGMA_Z_GET_X(h_A[j*lda+j]) + 2000. ) );

	for(j=0; j<n2; j++)
	  h_A2[j] = h_A[j];

	/* =====================================================================
	   Performs operation using LAPACK 
	   =================================================================== */

	//lapackf77_zpotrf("L", &N, h_A, &lda, info);
	lapackf77_zpotrf("U", &N, h_A, &lda, info);
	
	if (info[0] < 0)  
	  printf("Argument %d of zpotrf had an illegal value.\n", -info[0]);     

	/* =====================================================================
	   Performs operation using multi-core 
	   =================================================================== */
	start = get_current_time();
	//magma_zpotrf_mc("L", &N, h_A2, &lda, info);
	magma_zpotrf_mc("U", &N, h_A2, &lda, info);
	end = get_current_time();
	
	if (info[0] < 0)  
	  printf("Argument %d of magma_zpotrf_mc had an illegal value.\n", -info[0]);     
  
	cpu_perf2 = FLOPS( (double)N ) / (1000000.*GetTimerValue(start,end));
	
	/* =====================================================================
	   Check the result compared to LAPACK
	   =================================================================== */
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
    
    /* Shut down the Quark scheduler */
    QUARK_Delete(quark);
    
    /* Memory clean up */
    free(h_A);
    free(h_A2);
}
