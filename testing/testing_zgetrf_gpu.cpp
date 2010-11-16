/*
 *  -- MAGMA (version 1.0) --
 *     Univ. of Tennessee, Knoxville
 *     Univ. of California, Berkeley
 *     Univ. of Colorado, Denver
 *     November 2010
 *
 * @precisions normal z -> c d s
 *
 **/
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
#ifndef max
#define max(a,b)  (((a)<(b))?(b):(a))
#endif

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
            MAGMA_Z_OP_NEG( LU[i+j*lda], A[i+j*lda]);
        }
    }
    double residual = lapackf77_zlange("f", &M, &N, LU, &lda, work);

    free(L);
    free(work);

    return residual / (matnorm * N);
}


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgetrf
*/
int main( int argc, char** argv)
{
    cuInit( 0 );
    cublasInit( );
    printout_devices( );

    cuDoubleComplex *h_A, *h_R, *h_work;
    cuDoubleComplex *d_A;
    int *ipiv;
    double gpu_perf, cpu_perf;

    TimeStruct start, end;

    /* Matrix size */
    int M = 0, N = 0, n2, lda;
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
        }
        if (M>0 && N>0)
            printf("  testing_zgetrf -M %d -N %d\n\n", M, N);
        else
            {
                printf("\nUsage: \n");
                printf("  testing_zgetrf -M %d -N %d\n\n", 1024, 1024);
                exit(1);
            }
    }
    else {
        printf("\nUsage: \n");
        printf("  testing_zgetrf_gpu -M %d -N %d\n\n", 1024, 1024);
        M = N = size[9];
    }

    /* Initialize CUBLAS */
    status = cublasInit();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! CUBLAS initialization error\n");
    }

    lda = ((M+31)/32)*32;
    n2 = M * N;

    int min_mn = min(M, N);

    /* Allocate host memory for the matrix */
    h_A = (cuDoubleComplex*)malloc(n2 * sizeof(h_A[0]));
    if (h_A == 0) {
        fprintf (stderr, "!!!! host memory allocation error (A)\n");
    }

    ipiv = (int*)malloc(min_mn * sizeof(int));
    if (ipiv == 0) {
        fprintf (stderr, "!!!! host memory allocation error (ipiv)\n");
    }

    cudaMallocHost( (void**)&h_R,  n2*sizeof(cuDoubleComplex) );
    if (h_R == 0) {
        fprintf (stderr, "!!!! host memory allocation error (R)\n");
    }

    int nb = magma_get_zgetrf_nb(min_mn);
    int lwork = (M+32) * nb;
    status = cublasAlloc(lda *N, sizeof(cuDoubleComplex), (void**)&d_A);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! device memory allocation error (d_A)\n");
    }

    cudaMallocHost( (void**)&h_work, lwork*sizeof(cuDoubleComplex) );
    if (h_work == 0) {
        fprintf (stderr, "!!!! host memory allocation error (work)\n");
    }

    printf("\n\n");
    printf("  M     N   CPU GFlop/s    GPU GFlop/s   ||PA-LU||/(||A||*N)\n");
    printf("============================================================\n");
    for(i=0; i<10; i++){
        if (argc==1){
            M = N = min_mn = size[i];
            n2 = M*N;

            lda = ((M+31)/32)*32;
        }

        /* Initialize the matrix */
        lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
        lapackf77_zlacpy( MagmaUpperLowerStr, &M, &N, h_A, &lda, h_R, &lda );

        cublasSetMatrix( M, N, sizeof(cuDoubleComplex), h_A, M, d_A, lda);
        magma_zgetrf_gpu( M, N, d_A, lda, ipiv, &info);
        cublasSetMatrix( M, N, sizeof(cuDoubleComplex), h_A, M, d_A, lda);

        /* =====================================================================
           Performs operation using LAPACK
           =================================================================== */
        start = get_current_time();
        lapackf77_zgetrf(&M, &N, h_A, &M, ipiv, &info);
        end = get_current_time();
        if (info < 0)
            printf("Argument %d of zgetrf had an illegal value.\n", -info);

        cpu_perf = 2.*M*N*min_mn/(3.*1000000*GetTimerValue(start,end));
        // printf("CPU Processing time: %f (ms) \n", GetTimerValue(start,end));

        for(j=0; j<n2; j++)
            h_A[j] = h_R[j];

        /* ====================================================================
           Performs operation using MAGMA
           =================================================================== */
        start = get_current_time();
        magma_zgetrf_gpu( M, N, d_A, lda, ipiv, &info);
        end = get_current_time();
        cublasGetMatrix( M, N, sizeof(cuDoubleComplex), d_A, lda, h_R, M);

        gpu_perf = 2.*M*N*min_mn/(3.*1000000*GetTimerValue(start,end));
        // printf("GPU Processing time: %f (ms) \n", GetTimerValue(start,end));

        /* =====================================================================
           Check the factorization
           =================================================================== */
        double error = get_LU_error(M, N, h_A, M, h_R, ipiv);

        printf("%5d %5d  %6.2f         %6.2f         %e\n",
               M, N, cpu_perf, gpu_perf, error);

        if (argc != 1)
            break;
    }

    /* Memory clean up */
    free(h_A);
    free(ipiv);
    cublasFree(h_work);
    cublasFree(h_R);
    cublasFree(d_A);

    /* Shutdown */
    status = cublasShutdown();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! shutdown error (A)\n");
    }
}
