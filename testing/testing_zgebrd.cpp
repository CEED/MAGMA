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

#define PRECISION_z
/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgebrd
*/
int main( int argc, char** argv)
{
    cuInit( 0 );
    cublasInit( );
    printout_devices( );

    cuDoubleComplex *h_A, *h_R, *h_work;
    cuDoubleComplex *taup, *tauq;
    double          *diag, *offdiag, *diag2, *offdiag2;
    cuDoubleComplex *d_A;
    double           gpu_perf, cpu_perf, eps;

    TimeStruct start, end;

    /* Matrix size */
    int M, N=0, n2, lda;
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
        printf("  testing_zgebrd -N %d\n\n", 1024);
    }

    /* Initialize CUBLAS */
    status = cublasInit();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! CUBLAS initialization error\n");
        return 127;
    }

    eps = lapackf77_dlamch( "E" );

    lda = N;
    n2 = size[9] * size[9];

    /* Allocate host memory for the matrix */
    h_A = (cuDoubleComplex*)malloc(n2 * sizeof(h_A[0]));
    if (h_A == 0) {
        fprintf (stderr, "!!!! host memory allocation error (A)\n");
        return 127;
    }

    taup = (cuDoubleComplex*)malloc(size[9] * sizeof(cuDoubleComplex));
    tauq = (cuDoubleComplex*)malloc(size[9] * sizeof(cuDoubleComplex));
    if (taup == 0) {
        fprintf (stderr, "!!!! host memory allocation error (taup)\n");
        return 127;
    }


    diag = (double*)malloc(size[9] * sizeof(double));
    diag2= (double*)malloc(size[9] * sizeof(double));
    if (diag == 0) {
        fprintf (stderr, "!!!! host memory allocation error (diag)\n");
        return 127;
    }

    offdiag = (double*)malloc(size[9] * sizeof(double));
    offdiag2= (double*)malloc(size[9] * sizeof(double));
    if (offdiag == 0) {
        fprintf (stderr, "!!!! host memory allocation error (offdiag)\n");
        return 127;
    }

    cudaMallocHost( (void**)&h_R,  n2*sizeof(cuDoubleComplex) );
    if (h_R == 0) {
        fprintf (stderr, "!!!! host memory allocation error (R)\n");
        return 127;
    }

    int nb = magma_get_zgebrd_nb(size[9]);
    int lwork = 2*size[9]*nb;
    status = cublasAlloc(n2+lwork, sizeof(cuDoubleComplex), (void**)&d_A);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! device memory allocation error (d_A)\n");
        return 127;
    }

    cudaMallocHost( (void**)&h_work, (lwork)*sizeof(cuDoubleComplex) );
    //h_work = (cuDoubleComplex*)malloc( nb *lwork * sizeof(cuDoubleComplex) );
    if (h_work == 0) {
        fprintf (stderr, "!!!! host memory allocation error (work)\n");
        return 127;
    }

    printf("\n\n");
    printf("  N    CPU GFlop/s    GPU GFlop/s   |A-QHQ'|/N|A|  |I-QQ'|/N \n");
    printf("=============================================================\n");
    for(i=0; i<10; i++){
        M = N = lda = size[i];
        n2 = M*N;

        /* Initialize the matrices */
        lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
        lapackf77_zlacpy( MagmaUpperLowerStr, &M, &N, h_A, &lda, h_R, &lda );

        /* ====================================================================
           Performs operation using MAGMA
           =================================================================== */
        if (getenv("MAGMA_USE_LAPACK")) {
          start = get_current_time();
          lapackf77_zgebrd( &M, &N, h_R, &N, diag, offdiag,
                      tauq, taup, h_work, &lwork, &info);
          end = get_current_time();
        } else {
          start = get_current_time();
          magma_zgebrd( M, N, h_R, N, diag, offdiag,
                      tauq, taup, h_work, lwork, d_A, &info);
          end = get_current_time();
	}
        if (getenv("MAGMA_SHOW_INFO"))
          printf("zgebrd_INFO=%d\n", info);

        gpu_perf =(4.*M*N*N-4.*N*N*N/3.)/(1000000.*GetTimerValue(start,end));
        // printf("GPU Processing time: %f (ms) \n", GetTimerValue(start,end));

        /* =====================================================================
           Check the factorization
           =================================================================== */
        int lwork = nb * N * N;
        cuDoubleComplex *PT      = (cuDoubleComplex*)malloc( N * N * sizeof(cuDoubleComplex));
        cuDoubleComplex *work    = (cuDoubleComplex*)malloc( lwork * sizeof(cuDoubleComplex));

        double result[3] = {0., 0., 0.};

        lapackf77_zlacpy(MagmaUpperLowerStr, &N, &N, h_R, &N, PT, &N);

        // generate Q & P'
        lapackf77_zungbr("Q", &M, &M, &M, h_R, &N, tauq, work, &lwork, &info);
        lapackf77_zungbr("P", &M, &M, &M,  PT, &N, taup, work, &lwork, &info);

        // Test 1:  Check the decomposition A := Q * B * PT
        //      2:  Check the orthogonality of Q
        //      3:  Check the orthogonality of PT
#if defined(PRECISION_z) || defined(PRECISION_c) 
        double *rwork   = (double*)malloc( M * sizeof(double));
        lapackf77_zbdt01(&M, &N, &ione, h_A, &M, h_R, &M, diag, offdiag, PT, &M,
                         work, rwork, &result[0]);
        lapackf77_zunt01("Columns", &M, &M, h_R, &M, work, &lwork, rwork, &result[1]);
        lapackf77_zunt01("Rows", &M, &N, PT, &M, work, &lwork, rwork, &result[2]);
        free(rwork);
#else
        lapackf77_zbdt01(&M, &N, &ione, h_A, &M, h_R, &M, diag, offdiag, PT, &M,
                         work, &result[0]);
        lapackf77_zunt01("Columns", &M, &M, h_R, &M, work, &lwork, &result[1]);
        lapackf77_zunt01("Rows", &M, &N, PT, &M, work, &lwork, &result[2]);
#endif
        //printf("N = %d\n", N);
        //printf("norm(A -  Q  B  PT) / ( N * norm(A) * EPS ) = %f\n", result[0]);
        //printf("norm(I -  Q' *  Q ) / ( N * EPS )           = %f\n", result[1]);
        //printf("norm(I - PT' * PT ) / ( N * EPS )           = %f\n", result[2]);
        //printf("\n");

        free(PT);
        free(work);
        /* =====================================================================
           Performs operation using LAPACK
           =================================================================== */
        start = get_current_time();
        lapackf77_zgebrd(&M, &N, h_A, &N, diag2, offdiag2, tauq, taup,
                         h_work, &lwork, &info);
        end = get_current_time();

        if (info < 0)
            printf("Argument %d of zgebrd had an illegal value.\n", -info);

        cpu_perf = (4.*M*N*N-4.*N*N*N/3.)/(1000000.*GetTimerValue(start,end));
        // printf("CPU Processing time: %f (ms) \n", GetTimerValue(start,end));

        /* =====================================================================
           Print performance and error.
           =================================================================== */
        printf("%5d   %6.2f        %6.2f       %4.2e %4.2e %4.2e\n",
               size[i], cpu_perf, gpu_perf,
               result[0]*eps, result[1]*eps, result[2]*eps );
        
        if (argc != 1)
            break;
    }

    /* Memory clean up */
    free(h_A);
    free(taup);
    free(tauq);
    free(diag);    free(diag2);
    free(offdiag); free(offdiag2);
    cublasFree(h_work);
    //free(h_work);
    cublasFree(h_R);
    cublasFree(d_A);

    /* Shutdown */
    status = cublasShutdown();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! shutdown error (A)\n");
        return 127;
    }

    return 0;
}
