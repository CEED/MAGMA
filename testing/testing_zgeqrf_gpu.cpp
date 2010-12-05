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

#ifndef min
#define min(a,b)  (((a)<(b))?(a):(b))
#endif

// define pfactor for number of flops in complex
#define PRECISION_z
#if (defined(PRECISION_s) || defined(PRECISION_d))
   #define pfactor 1.
#else
   #define pfactor 4.
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgeqrf
*/
int main( int argc, char** argv)
{
    cuInit( 0 );
    cublasInit( );
    printout_devices( );

    cuDoubleComplex *h_A, *h_R, *h_work, *tau, *d_A;
    double gpu_perf, cpu_perf;

    TimeStruct start, end;

    /* Matrix size */
    magma_int_t M=0, N=0, n2, lda;
    magma_int_t size[10] = {1024,2048,3072,4032,5184,6016,7040,8064,9088,9984};

    cublasStatus status;
    magma_int_t i, info;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};

    if (argc != 1){
        for(i = 1; i<argc; i++){
            if (strcmp("-N", argv[i])==0)
                N = atoi(argv[++i]);
            else if (strcmp("-M", argv[i])==0)
                M = atoi(argv[++i]);
        }
        if (M>0 && N>0)
            printf("  testing_zgeqrf_gpu -M %d -N %d\n\n", M, N);
        else
            {
                printf("\nUsage: \n");
                printf("  testing_zgeqrf_gpu -M %d -N %d\n\n", 1024, 1024);
                exit(1);
            }
    }
    else {
        printf("\nUsage: \n");
        printf("  testing_zgeqrf_gpu -M %d -N %d\n\n", 1024, 1024);
        M = N = size[9];
    }

    /* Initialize CUBLAS */
    status = cublasInit();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! CUBLAS initialization error\n");
    }

    lda = (M/32)*32;
    if (lda<M) lda+=32;

    n2  = M * N;

    magma_int_t min_mn = min(M, N);

    /* Allocate host memory for the matrix */
    h_A = (cuDoubleComplex*)malloc(n2 * sizeof(h_A[0]));
    if (h_A == 0) {
        fprintf (stderr, "!!!! host memory allocation error (A)\n");
    }

    tau = (cuDoubleComplex*)malloc(min_mn * sizeof(cuDoubleComplex));
    if (tau == 0) {
        fprintf (stderr, "!!!! host memory allocation error (tau)\n");
    }

    cudaMallocHost( (void**)&h_R,  n2*sizeof(cuDoubleComplex) );
    if (h_R == 0) {
        fprintf (stderr, "!!!! host memory allocation error (R)\n");
    }

    magma_int_t nb = magma_get_zgeqrf_nb(min_mn);
    magma_int_t lwork = (M+N)*nb;

    status = cublasAlloc(lda*N, sizeof(cuDoubleComplex), (void**)&d_A);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! device memory allocation error (d_A)\n");
    }

    cudaMallocHost( (void**)&h_work, lwork*sizeof(cuDoubleComplex) );
    if (h_work == 0) {
        fprintf (stderr, "!!!! host memory allocation error (work)\n");
    }

    printf("\n\n");
    printf("  M     N   CPU GFlop/s   GPU GFlop/s    ||R||_F / ||A||_F\n");
    printf("==========================================================\n");
    for(i=0; i<10; i++){
        if (argc==1){
            M = N = min_mn = size[i];
            n2 = M*N;

            lda = (M/32)*32;
            if (lda<M) lda+=32;
        }

        /* Initialize the matrix */
        lapackf77_zlarnv( &ione, ISEED, &n2, h_A );

        cublasSetMatrix( M, N, sizeof(cuDoubleComplex), h_A, M, d_A, lda);
        magma_zgeqrf_gpu( M, N, d_A, lda, tau, &info);
        cublasSetMatrix( M, N, sizeof(cuDoubleComplex), h_A, M, d_A, lda);

        /* ====================================================================
           Performs operation using MAGMA
           =================================================================== */
        start = get_current_time();
        magma_zgeqrf_gpu( M, N, d_A, lda, tau, &info);
        end = get_current_time();

        gpu_perf = pfactor*4.*M*N*min_mn/(3.*1000000*GetTimerValue(start,end));
        // printf("GPU Processing time: %f (ms) \n", GetTimerValue(start,end));

        /* =====================================================================
           Performs operation using LAPACK
           =================================================================== */
        start = get_current_time();
        lapackf77_zgeqrf(&M, &N, h_A, &M, tau, h_work, &lwork, &info);
        end = get_current_time();
        if (info < 0)
            printf("Argument %d of zgeqrf had an illegal value.\n", -info);

        cpu_perf = pfactor*4.*M*N*min_mn/(3.*1000000*GetTimerValue(start,end));
        // printf("CPU Processing time: %f (ms) \n", GetTimerValue(start,end));

        /* =====================================================================
           Check the result compared to LAPACK
           =================================================================== */
        cublasGetMatrix( M, N, sizeof(cuDoubleComplex), d_A, lda, h_R, M);

        double work[1], matnorm = 1.;
        cuDoubleComplex mone = MAGMA_Z_NEG_ONE;
        magma_int_t one = 1;
        matnorm = lapackf77_zlange("f", &M, &N, h_A, &M, work);
        blasf77_zaxpy(&n2, &mone, h_A, &one, h_R, &one);
        printf("%5d %5d  %6.2f         %6.2f        %e\n",
               M, N, cpu_perf, gpu_perf,
               lapackf77_zlange("f", &M, &N, h_R, &M, work) / matnorm);

        /* =====================================================================
           Check the factorization
           =================================================================== */
        /*
          cuDoubleComplex result[2];
          cuDoubleComplex *hwork_Q = (cuDoubleComplex*)malloc( M * N * sizeof(cuDoubleComplex));
          cuDoubleComplex *hwork_R = (cuDoubleComplex*)malloc( M * N * sizeof(cuDoubleComplex));
          cuDoubleComplex *rwork   = (cuDoubleComplex*)malloc( N * sizeof(cuDoubleComplex));

          lapackf77_zqrt02(&M, &min_mn, &min_mn, h_A, h_R, hwork_Q, hwork_R, &lda, tau,
          h_work, &lwork, rwork, result);

          printf("norm( R - Q'*A ) / ( M * norm(A) * EPS ) = %f\n", result[0]);
          printf("norm( I - Q'*Q ) / ( M * EPS )           = %f\n", result[1]);
          free(hwork_Q);
          free(hwork_R);
          free(rwork);
        */

        if (argc != 1)
            break;
    }

    /* Memory clean up */
    free(h_A);
    free(tau);
    cublasFree(h_work);
    cublasFree(h_R);
    cublasFree(d_A);

    /* Shutdown */
    status = cublasShutdown();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! shutdown error (A)\n");
    }
}
