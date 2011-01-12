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
#include "flops.h"
#include "magma.h"
#include "testings.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgeev
*/

#define CHECK_ERROR
#define PRECISION_z

int main( int argc, char** argv) 
{
    TESTING_CUDA_INIT();

    TimeStruct start, end;
    cuDoubleComplex *h_A, *h_R, *VL, *VR, *h_work, *w1, *w2;
    double *rwork;
    double gpu_time, cpu_time;

    /* Matrix size */
    magma_int_t N=0, n2;
    magma_int_t size[8] = {1024,2048,3072,4032,5184,6016,7040,8064};

    magma_int_t i, j, info;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};

    if (argc != 1){
        for(i = 1; i<argc; i++){
            if (strcmp("-N", argv[i])==0)
                N = atoi(argv[++i]);
        }
        if (N>0)
            printf("  testing_zgeev -N %d\n\n", N);
        else
            {
                printf("\nUsage: \n");
                printf("  testing_zgeev -N %d\n\n", N);
		
		/* Shutdown */
		TESTING_CUDA_FINALIZE();
                exit(1);
            }
    }
    else {
        printf("\nUsage: \n");
        printf("  testing_zgeev -N %d\n\n", 1024);
        N = size[7];
    }

    n2  = N * N;

    w1 = (cuDoubleComplex*)malloc(N * sizeof(cuDoubleComplex));
    if (w1 == 0) {
        fprintf (stderr, "!!!! host memory allocation error (w1)\n");
    }
    w2 = (cuDoubleComplex*)malloc(N * sizeof(cuDoubleComplex));
    if (w1 == 0) {
      fprintf (stderr, "!!!! host memory allocation error (w2)\n");
    }

    #if (defined(PRECISION_s) || defined(PRECISION_d))
    cuDoubleComplex *w1i = (cuDoubleComplex*)malloc(N * sizeof(cuDoubleComplex));
    if (w1i == 0) {
      fprintf (stderr, "!!!! host memory allocation error (w1i)\n");
    }
    cuDoubleComplex *w2i = (cuDoubleComplex*)malloc(N * sizeof(cuDoubleComplex));
    if (w1i == 0) {
      fprintf (stderr, "!!!! host memory allocation error (w2i)\n");
    }
    #endif

    rwork = (double*)malloc(2 * N * sizeof(double));
    if (rwork == 0) {
      fprintf (stderr, "!!!! host memory allocation error (rwork)\n");
    }

    TESTING_MALLOC   ( h_A, cuDoubleComplex, n2);
    TESTING_HOSTALLOC( h_R, cuDoubleComplex, n2);
    TESTING_HOSTALLOC( VL , cuDoubleComplex, n2);
    TESTING_HOSTALLOC( VR , cuDoubleComplex, n2);

    magma_int_t nb = 128;//magma_get_zgeev_nb(N);
    magma_int_t lwork = N*nb;

    cudaMallocHost( (void**)&h_work, lwork*sizeof(cuDoubleComplex) );
    if (h_work == 0) {
        fprintf (stderr, "!!!! host memory allocation error (work)\n");
    }

    printf("\n\n");
    printf("  N     CPU Time(s)    GPU Time(s)     ||R||_F / ||A||_F\n");
    printf("==========================================================\n");
    for(i=0; i<8; i++){
        if (argc==1){
            N = size[i];
            n2 = N*N;
        }

        /* Initialize the matrix */
        lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
        lapackf77_zlacpy( MagmaUpperLowerStr, &N, &N, h_A, &N, h_R, &N );

        /* ====================================================================
           Performs operation using MAGMA
           =================================================================== */
        start = get_current_time();
        #if (defined(PRECISION_c) || defined(PRECISION_z))
        magma_zgeev("V", "V",
        //magma_zgeev("N", "N",
		    &N, h_R, &N, w1, VL, &N, VR, &N,
                    h_work, &lwork, rwork, &info);
        #else
	magma_zgeev("V", "V",
		    //magma_zgeev("N", "N",
                    &N, h_R, &N, w1, w1i, VL, &N, VR, &N,
                    h_work, &lwork, rwork, &info);
	#endif
        end = get_current_time();

        gpu_time = GetTimerValue(start,end)/1000.;

        /* =====================================================================
           Performs operation using LAPACK
           =================================================================== */
        start = get_current_time();
        #if (defined(PRECISION_c) || defined(PRECISION_z))
        lapackf77_zgeev("V", "V",
	//lapackf77_zgeev("N", "N",
			&N, h_A, &N, w2, VL, &N, VR, &N,
			h_work, &lwork, rwork, &info);
	#else
	lapackf77_zgeev("V", "V",
	//lapackf77_zgeev("N", "N",
                        &N, h_A, &N, w2, w2i, VL, &N, VR, &N,
                        h_work, &lwork, rwork, &info);
        #endif
        end = get_current_time();
        if (info < 0)
            printf("Argument %d of zgeev had an illegal value.\n", -info);

        cpu_time = GetTimerValue(start,end)/1000.;

        /* =====================================================================
           Check the result compared to LAPACK
           =================================================================== */
        double work[1], matnorm = 1., result = 0.;
        cuDoubleComplex mone = MAGMA_Z_NEG_ONE;
        magma_int_t one = 1;

#ifdef CHECK_ERROR
        matnorm = lapackf77_zlange("f", &N, &one, w1, &N, work);
	printf("norm = %f\n", matnorm);
        blasf77_zaxpy(&N, &mone, w1, &one, w2, &one);

	result = lapackf77_zlange("f", &N, &one, w2, &N, work) / matnorm;
#endif

        printf("%5d     %6.2f         %6.2f         %e\n",
               N, cpu_time, gpu_time,
               result);

        if (argc != 1)
            break;
    }

    /* Memory clean up */
    free(w1);
    free(w2);
    #if (defined(PRECISION_s) || defined(PRECISION_d))
    free(w1i);
    free(w2i);
    #endif
    free(rwork);
    cublasFree(h_work);

    TESTING_FREE    ( h_A );
    TESTING_HOSTFREE( h_R );
    TESTING_HOSTFREE( VL  );
    TESTING_HOSTFREE( VR  );

    /* Shutdown */
    TESTING_CUDA_FINALIZE();
    return EXIT_SUCCESS;
}
