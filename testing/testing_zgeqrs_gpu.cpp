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

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgeqrs
*/
int main( int argc, char** argv)
{
    cuInit( 0 );
    cublasInit( );
    printout_devices( );

    cuDoubleComplex *h_A, *h_R, *h_work, *tau;
    cuDoubleComplex *d_A, *d_work, *d_x;
    double gpu_perf, cpu_perf;

    cuDoubleComplex *x, *b, *r;
    cuDoubleComplex *d_b;

    int nrhs = 3;

    TimeStruct start, end;

    /* Matrix size */
    int M=0, N=0, n2, lda, szeB;
    int size[10] = {1024,2048,3072,4032,5184,6016,7040,8064,9088,10112};

    cublasStatus status;
    int i, j, info;
    int ione     = 1;
    int ISEED[4] = {0,0,0,1};
    
    if (argc != 1){
        for(i = 1; i<argc; i++){
            if (strcmp("-N", argv[i])==0)
                N = atoi(argv[++i]);
            else if (strcmp("-M", argv[i])==0)
                M = atoi(argv[++i]);
            else if (strcmp("-nrhs", argv[i])==0)
                nrhs = atoi(argv[++i]);
        }
        if (N>0 && M>0 && M >= N)
            printf("  testing_zgeqrs_gpu -nrhs %d -M %d -N %d\n\n", nrhs, M, N);
        else
            {
                printf("\nUsage: \n");
                printf("  testing_zgeqrs_gpu -nrhs %d  -M %d  -N %d\n\n", nrhs, M, N);
                printf("  M has to be >= N, exit.\n");
                exit(1);
            }
    }
    else {
        printf("\nUsage: \n");
        printf("  testing_zgeqrs_gpu -nrhs %d  -M %d  -N %d\n\n", nrhs, 1024, 1024);
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

    int min_mn = min(M, N);

    /* Allocate host memory for the matrix */
    h_A = (cuDoubleComplex*)malloc(n2 * sizeof(h_A[0]));
    if (h_A == 0) {
        fprintf (stderr, "!!!! host memory allocation error (A)\n");
    }

    tau = (cuDoubleComplex*)malloc(min_mn * sizeof(cuDoubleComplex));
    if (tau == 0) {
        fprintf (stderr, "!!!! host memory allocation error (tau)\n");
    }

    x = (cuDoubleComplex*)malloc(nrhs* M * sizeof(cuDoubleComplex));
    b = (cuDoubleComplex*)malloc(nrhs* M * sizeof(cuDoubleComplex));
    r = (cuDoubleComplex*)malloc(nrhs* M * sizeof(cuDoubleComplex));

    cudaMallocHost( (void**)&h_R,  n2*sizeof(cuDoubleComplex) );
    if (h_R == 0) {
        fprintf (stderr, "!!!! host memory allocation error (R)\n");
    }

    int nb = magma_get_zgeqrf_nb(M);
    // int lwork = (3*size[9]+nb)*nb;
    int lwork = (M+2*N+nb)*nb;

    if (nrhs > nb)
        lwork = (M+2*N+nb)*nrhs;

    status = cublasAlloc(lda*N, sizeof(cuDoubleComplex), (void**)&d_A);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! device memory allocation error (d_A)\n");
    }

    status = cublasAlloc(nrhs * M, sizeof(cuDoubleComplex), (void**)&d_b);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! device memory allocation error (d_b)\n");
    }

    status = cublasAlloc(lwork, sizeof(cuDoubleComplex), (void**)&d_work);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! device memory allocation error (d_work)\n");
    }

    status = cublasAlloc(nrhs * N, sizeof(cuDoubleComplex), (void**)&d_x);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! device memory allocation error (d_x)\n");
    }

    cudaMallocHost( (void**)&h_work, lwork*sizeof(cuDoubleComplex) );
    if (h_work == 0) {
        fprintf (stderr, "!!!! host memory allocation error (work)\n");
    }

    printf("\n");
    printf("                                         ||b-Ax|| / (N||A||)\n");
    printf("  M     N    CPU GFlop/s   GPU GFlop/s      GPU      CPU    \n");
    printf("============================================================\n");
    for(i=0; i<10; i++){
        if (argc == 1){
            M = N = min_mn = size[i];
            n2 = M*N;

            lda = (M/32)*32;
            if (lda<M) lda+=32;
        }

        /* Initialize the matrices */
        lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
        lapackf77_zlacpy( MagmaUpperLowerStr, &M, &N, h_A, &M, h_R, &M );

        szeB = M*nrhs;
        lapackf77_zlarnv( &ione, ISEED, &szeB, b );
        lapackf77_zlacpy( MagmaUpperLowerStr, &M, &nrhs, b, &M, r, &M );

        cublasSetMatrix( M, N, sizeof(cuDoubleComplex), h_A, M, d_A, lda);
        magma_zgeqrf_gpu( M, N, d_A, lda, tau, &info);
        cublasSetMatrix( M, N, sizeof(cuDoubleComplex), h_A, M, d_A, lda);
        cublasSetMatrix( M, nrhs, sizeof(cuDoubleComplex), b, M, d_b, M);

        /* ====================================================================
           Performs operation using MAGMA
           =================================================================== */
        start = get_current_time();
        magma_zgeqrf_gpu2( M, N, d_A, lda, tau, d_work, &info);

        // Solve the least-squares problem min || A * X - B ||
        magma_zgeqrs_gpu( M, N, nrhs, d_A, lda, tau,
                          d_b, M, h_work, &lwork, d_work, &info);
        end = get_current_time();

        gpu_perf=(4.*M*N*min_mn/3. + 3.*nrhs*N*N)/(1000000.*
                                                   GetTimerValue(start,end));

        double work[1], matnorm;
        cuDoubleComplex zone  = MAGMA_Z_ONE;
        cuDoubleComplex mzone = MAGMA_Z_NEG_ONE;

        // get the solution in x
        cublasGetMatrix(N, nrhs, sizeof(cuDoubleComplex), d_b, M, x, N);

        // compute the residual
        if (nrhs == 1)
            blasf77_zgemv("n", &M, &N, &mzone, h_A, &M, x, &ione, &zone, r, &ione);
        else
            blasf77_zgemm("n","n", &M, &nrhs, &N, &mzone, h_A, &M, x, &N, &zone, r, &M);
        matnorm = lapackf77_zlange("f", &M, &N, h_A, &M, work);

        /* =====================================================================
           Performs operation using LAPACK
           =================================================================== */
        for(int k=0; k<nrhs; k++)
            for(j=0; j<M; j++)
                x[j+k*M] = b[j+k*M];

        start = get_current_time();
        lapackf77_zgeqrf(&M, &N, h_R, &M, tau, h_work, &lwork, &info);

        if (info < 0)
            printf("Argument %d of zgeqrf had an illegal value.\n", -info);

        // Solve the least-squares problem: min || A * X - B ||
        // 1. B(1:M,1:NRHS) = Q^T B(1:M,1:NRHS)
        lapackf77_zunmqr("l", "t", &M, &nrhs, &min_mn, h_R, &M,
                         tau, x, &M, h_work, &lwork, &info);

        // 2. B(1:N,1:NRHS) := inv(R) * B(1:M,1:NRHS)
        blasf77_ztrsm("l", "u", "n", "n", &N, &nrhs, &zone, h_R, &M, x, &M);

        end = get_current_time();
        cpu_perf = (4.*M*N*min_mn/3.+3.*nrhs*N*N)/(1000000.*
                                                   GetTimerValue(start,end));

        if (nrhs == 1)
            blasf77_zgemv("n", &M, &N, &mzone, h_A, &M, x, &ione, &zone, b, &ione);
        else
            blasf77_zgemm("n","n", &M, &nrhs, &N, &mzone, h_A, &M, x, &M, &zone, b, &M);

        printf("%5d %5d   %6.1f       %6.1f       %7.2e   %7.2e\n",
               M, N, cpu_perf, gpu_perf,
               lapackf77_zlange("f", &M, &nrhs, r, &M, work)/(min_mn*matnorm),
               lapackf77_zlange("f", &M, &nrhs, b, &M, work)/(min_mn*matnorm) );

        if (argc != 1)
            break;
    }

    /* Memory clean up */
    free(h_A);
    free(tau);
    free(x);
    free(b);
    free(r);
    cublasFree(h_work);
    cublasFree(d_work);
    cublasFree(d_x);
    cublasFree(h_R);
    cublasFree(d_A);
    cublasFree(d_b);

    /* Shutdown */
    status = cublasShutdown();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! shutdown error (A)\n");
    }
}
