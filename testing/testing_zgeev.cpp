/*
    -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

       to go to real 'w' must be changed to 'wr', 'wi' everywhere

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

#ifndef min
#define min(a,b)  (((a)<(b))?(a):(b))
#endif


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgeev
*/
int main( int argc, char** argv) 
{
    cuInit( 0 );
    cublasInit( );
    printout_devices( );

    cuDoubleComplex *h_A, *h_R, *VL, *VR, *h_work, *w1, *w2;
    double *rwork;
    double gpu_time, cpu_time;

    TimeStruct start, end;

    /* Matrix size */
    magma_int_t N=0, n2;
    magma_int_t size[8] = {1024,2048,3072,4032,5184,6016,7040,8064};

    cublasStatus status;
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
                exit(1);
            }
    }
    else {
        printf("\nUsage: \n");
        printf("  testing_zgeev -N %d\n\n", 1024);
        N = size[7];
    }

    /* Initialize CUBLAS */
    status = cublasInit();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! CUBLAS initialization error\n");
    }

    n2  = N * N;

    /* Allocate host memory for the matrix */
    h_A = (cuDoubleComplex*)malloc(n2 * sizeof(h_A[0]));
    if (h_A == 0) {
        fprintf (stderr, "!!!! host memory allocation error (A)\n");
    }
    VL = (cuDoubleComplex*)malloc(n2 * sizeof(h_A[0]));
    if (VL == 0) {
      fprintf (stderr, "!!!! host memory allocation error (VL)\n");
    }
    VR = (cuDoubleComplex*)malloc(n2 * sizeof(h_A[0]));
    if (VR == 0) {
      fprintf (stderr, "!!!! host memory allocation error (VR)\n");
    }

    w1 = (cuDoubleComplex*)malloc(N * sizeof(cuDoubleComplex));
    if (w1 == 0) {
        fprintf (stderr, "!!!! host memory allocation error (w1)\n");
    }
    w2 = (cuDoubleComplex*)malloc(N * sizeof(cuDoubleComplex));
    if (w1 == 0) {
      fprintf (stderr, "!!!! host memory allocation error (w2)\n");
    }

    rwork = (double*)malloc(2 * N * sizeof(double));
    if (rwork == 0) {
      fprintf (stderr, "!!!! host memory allocation error (rwork)\n");
    }


    cudaMallocHost( (void**)&h_R,  n2*sizeof(cuDoubleComplex) );
    if (h_R == 0) {
        fprintf (stderr, "!!!! host memory allocation error (R)\n");
    }

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

        magma_zgeev("V", "V", &N, h_R, &N, w1, VL, &N, VR, &N, 
		    h_work, &lwork, rwork, &info);

        for(j=0; j<n2; j++)
            h_R[j] = h_A[j];

        /* ====================================================================
           Performs operation using MAGMA
           =================================================================== */
        start = get_current_time();
        magma_zgeev("V", "V", &N, h_R, &N, w1, VL, &N, VR, &N,
                    h_work, &lwork, rwork, &info);
        end = get_current_time();

        gpu_time = GetTimerValue(start,end)/1000.;

        /* =====================================================================
           Performs operation using LAPACK
           =================================================================== */
        start = get_current_time();
        lapackf77_zgeev("V", "V", &N, h_A, &N, w2, VL, &N, VR, &N,
			h_work, &lwork, rwork, &info);
        end = get_current_time();
        if (info < 0)
            printf("Argument %d of zgeev had an illegal value.\n", -info);

        cpu_time = GetTimerValue(start,end)/1000.;

        /* =====================================================================
           Check the result compared to LAPACK
           =================================================================== */
        double work[1], matnorm = 1.;
        cuDoubleComplex mone = MAGMA_Z_NEG_ONE;
        magma_int_t one = 1;

        matnorm = lapackf77_zlange("f", &N, &one, w1, &N, work);
        blasf77_zaxpy(&N, &mone, w1, &one, w2, &one);

        printf("%5d     %6.2f         %6.2f         %e\n",
               N, cpu_time, gpu_time,
               lapackf77_zlange("f", &N, &one, w2, &N, work) / matnorm);

        if (argc != 1)
            break;
    }

    /* Memory clean up */
    free(h_A);
    free(VL);
    free(VR);
    free(w1);
    free(w2);
    free(rwork);
    cublasFree(h_work);
    cublasFree(h_R);

    /* Shutdown */
    status = cublasShutdown();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! shutdown error (A)\n");
    }
}
