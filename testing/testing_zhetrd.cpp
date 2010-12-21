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

// Flops formula
#define PRECISION_z
#define CHECK_ERROR
#if defined(PRECISION_z) || defined(PRECISION_c)
#define FLOPS(n) ( 4.* 4. * n * n * n / 3. )
#else
#define FLOPS(n) (     4. * n * n * n / 3. )
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zhetrd
*/
int main( int argc, char** argv)
{
    cuInit( 0 );
    cublasInit( );
    printout_devices( );

    cuDoubleComplex *h_A, *h_R, *h_work;
    cuDoubleComplex *tau, *tau2;
    double          *diag, *offdiag, *diag2, *offdiag2;
    double gpu_perf, cpu_perf, eps;

    TimeStruct start, end;

    /* Matrix size */
    int N=0, n2, lda;
    int size[10] = {1024,2048,3072,4032,5184,6016,7040,8064,9088,10112};

    cublasStatus status;
    int i, info;
    int ione     = 1;
    int ISEED[4] = {0,0,0,1};
    
    if (argc != 1){
        for(i = 1; i<argc; i++){
            if (strcmp("-N", argv[i])==0)
                N = atoi(argv[++i]);
        }
        if (N>0) size[0] = size[9] = N;
        else exit(1);
    }
    else {
        printf("\nUsage: \n");
        printf("  testing_zhetrd -N %d\n\n", 1024);
        N = size[9];
    }

    /* Initialize CUBLAS */
    status = cublasInit();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! CUBLAS initialization error\n");
    }

    eps = lapackf77_dlamch( "E" );

    lda = N;
    if (N%32!=0)
        lda = (N/32)*32 + 32;
    n2 = size[9] * lda;

    /* Allocate host memory for the matrix */
    h_A = (cuDoubleComplex*)malloc(n2 * sizeof(h_A[0]));
    if (h_A == 0) {
        fprintf (stderr, "!!!! host memory allocation error (A)\n");
    }

    tau = (cuDoubleComplex*)malloc(size[9] * sizeof(cuDoubleComplex));
    tau2= (cuDoubleComplex*)malloc(size[9] * sizeof(cuDoubleComplex));
    if (tau == 0) {
        fprintf (stderr, "!!!! host memory allocation error (tau)\n");
    }


    diag = (double*)malloc(size[9] * sizeof(double));
    diag2= (double*)malloc(size[9] * sizeof(double));
    if (diag == 0) {
        fprintf (stderr, "!!!! host memory allocation error (diag)\n");
    }

    offdiag = (double*)malloc(size[9] * sizeof(double));
    offdiag2= (double*)malloc(size[9] * sizeof(double));
    if (offdiag == 0) {
        fprintf (stderr, "!!!! host memory allocation error (offdiag)\n");
    }

    cudaMallocHost( (void**)&h_R,  n2*sizeof(cuDoubleComplex) );
    if (h_R == 0) {
        fprintf (stderr, "!!!! host memory allocation error (R)\n");
    }

    int nb = magma_get_zhetrd_nb(size[9]);
    int lwork = 2*size[9]*nb;
    //int lwork = 2*size[9]*lda/nb;

    cudaMallocHost( (void**)&h_work, (lwork)*sizeof(cuDoubleComplex) );
    if (h_work == 0) {
        fprintf (stderr, "!!!! host memory allocation error (work)\n");
    }

    printf("\n\n");
    printf("  N    CPU GFlop/s    GPU GFlop/s   |A-QHQ'|/N|A|  |I-QQ'|/N \n");
    printf("=============================================================\n");
    for(i=0; i<10; i++){
        N = size[i];

        if (N%32==0)
            lda = N;
        else
            lda = (N/32)*32+32;

        n2 = N*lda;

        /* Initialize the matrices */
        lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
        lapackf77_zlacpy( MagmaUpperLowerStr, &N, &N, h_A, &lda, h_R, &lda );

        /* ====================================================================
           Performs operation using MAGMA
           =================================================================== */
        start = get_current_time();
        magma_zhetrd('L', N, h_R, lda, diag, offdiag,
                     tau, h_work, &lwork, &info);
        end = get_current_time();

        gpu_perf = FLOPS(N)/(1000000.*GetTimerValue(start,end));
        // printf("GPU Processing time: %f (ms) \n", GetTimerValue(start,end));

        /* =====================================================================
           Check the factorization
           =================================================================== */
	double result[2] = {0., 0.};
#ifdef CHECK_ERROR
        cuDoubleComplex *hwork_Q = 
	  (cuDoubleComplex*)malloc( N * N * sizeof(cuDoubleComplex));
        cuDoubleComplex *work    = 
	  (cuDoubleComplex*)malloc( 2 * N * N * sizeof(cuDoubleComplex));

        int test;

        lapackf77_zlacpy("L", &N, &N, h_R, &lda, hwork_Q, &N);
        lapackf77_zungtr("L", &N, hwork_Q, &N, tau, h_work, &lwork, &info);

#if defined(PRECISION_z) || defined(PRECISION_c) 
        double *rwork   = (double*)malloc( N * sizeof(double));

        test = 2;
        lapackf77_zhet21(&test, "L", &N, &ione, h_A, &lda, diag, offdiag,
                         hwork_Q, &N, h_R, &lda, tau, work, rwork, &result[0]);

        test = 3;
        lapackf77_zhet21(&test, "L", &N, &ione, h_A, &lda, diag, offdiag,
                         hwork_Q, &N, h_R, &lda, tau, work, rwork, &result[1]);

        free(rwork);
#else
        test = 2;
        lapackf77_zhet21(&test, "L", &N, &ione, h_A, &lda, diag, offdiag,
                         hwork_Q, &N, h_R, &lda, tau, work, &result[0]);

        test = 3;
        lapackf77_zhet21(&test, "L", &N, &ione, h_A, &lda, diag, offdiag,
                         hwork_Q, &N, h_R, &lda, tau, work, &result[1]);

#endif

        free(hwork_Q);
        free(work);
        /* =====================================================================
           Performs operation using LAPACK
           =================================================================== */
        start = get_current_time();
        lapackf77_zhetrd("L", &N, h_A, &lda, diag2, offdiag2, tau2, 
			 h_work, &lwork, &info);
        end = get_current_time();
#endif
        if (info < 0)
            printf("Argument %d of zhetrd had an illegal value.\n", -info);

        cpu_perf = FLOPS(N)/(1000000.*GetTimerValue(start,end));
        // printf("CPU Processing time: %f (ms) \n", GetTimerValue(start,end));

        /* =====================================================================
           Print performance and error.
           =================================================================== */
        printf("%5d    %6.2f         %6.2f      %e %e\n",
               size[i], cpu_perf, gpu_perf,
               result[0]*eps, result[1]*eps);

        if (argc != 1)
            break;
    }

    /* Memory clean up */
    free(h_A);
    free(tau);     free(tau2);
    free(diag);    free(diag2);
    free(offdiag); free(offdiag2);
    cublasFree(h_work);
    cublasFree(h_R);

    /* Shutdown */
    status = cublasShutdown();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! shutdown error (A)\n");
    }
}
