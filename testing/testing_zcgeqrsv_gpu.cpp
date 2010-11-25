/*
  -- MAGMA (version 0.1) --
  Univ. of Tennessee, Knoxville
  Univ. of California, Berkeley
  Univ. of Colorado, Denver
  November 2010

  @precisions mixed zc -> ds

*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include "magma.h"

#define PRECISION_z

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing dgeqrs
*/
int main( int argc, char** argv)
{
#if defined(PRECISION_z) && (GPUSHMEM < 200)
    fprintf(stderr, "This functionnality is not available in MAGMA for this precisions actually\n");
    return EXIT_SUCCESS;
#else

    printf("Iterative Refinement- QR \n");
    printf("\n");

    cuInit( 0 );
    cublasInit( );
    printout_devices( );

    cuDoubleComplex *h_A, *h_R, *h_work_d, *tau_d;
    cuDoubleComplex *d_A, *d_work_d, *d_x;
    cuFloatComplex *tau , *h_work  , *d_work ;
    cuDoubleComplex *x, *b, *rr;
    cuDoubleComplex *d_b;
    cuDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    cuDoubleComplex c_one     = MAGMA_Z_ONE;
    double          cpu_perf, mperf, dperf, sperf;
    double          wnorm, matnorm;
    int nrhs = 1;
    int ione     = 1;
    int ISEED[4] = {0,0,0,1};

    TimeStruct start, end;

    /* Matrix size */
    int M, N=0, n2, lda;
    int size[10] = {1024,2048,3072,4032,5184,6016,7040,8000, 9088,10112};


    
    cublasStatus status;
    int i, info;

    if (argc != 1){
        for(i = 1; i<argc; i++){
            if (strcmp("-N", argv[i])==0)
                N = atoi(argv[++i]);
        }
        if (N>0) size[0] = size[5] = N;
        else exit(1);
    }
    else {
        printf("\nUsage: \n");
        printf("  testing_zcgeqrsv_gpu -N %d\n\n", 1024);
    }

    /* Initialize CUBLAS */
    status = cublasInit();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! CUBLAS initialization error\n");
        exit(1);;
    }
    int size5 = size[7];
    lda = N;
    n2 = size5 * size5;
    cuFloatComplex *  h_AA ;
    /* Allocate host memory for the matrix */
    h_AA = (cuFloatComplex*)malloc(n2 * sizeof(h_AA[0]));
    if (h_AA == 0) {
        fprintf (stderr, "!!!! host memory allocation error (A)\n");
        exit(1);;
    }

    h_A = (cuDoubleComplex*)malloc(n2 * sizeof(h_A[0]));
    if (h_A == 0) {
        fprintf (stderr, "!!!! host memory allocation error (A)\n");
        exit(1);;
    }

    tau_d = (cuDoubleComplex*)malloc(size5 * sizeof(cuDoubleComplex));
    if (tau_d == 0) {
        fprintf (stderr, "!!!! host memory allocation error (tau_d)\n");
        exit(1);;
    }
    tau = (cuFloatComplex*)malloc(size5 * sizeof(cuFloatComplex));
    if (tau == 0) {
        fprintf (stderr, "!!!! host memory allocation error (tau)\n");
        exit(1);;
    }

    x = (cuDoubleComplex*)malloc(size5 * sizeof(cuDoubleComplex));
    b = (cuDoubleComplex*)malloc(size5 * sizeof(cuDoubleComplex));
    rr = (cuDoubleComplex*)malloc(size5 * sizeof(cuDoubleComplex));

    cudaMallocHost( (void**)&h_R,  n2*sizeof(cuDoubleComplex) );
    if (h_R == 0) {
        fprintf (stderr, "!!!! host memory allocation error (R)\n");
        exit(1);;
    }

    int nb = magma_get_zgeqrf_nb(size5);
    int nb_s = magma_get_cgeqrf_nb(size5);
    int lwork_d = (3*size5+nb)*nb;
    int lwork = (3*size5+nb)*nb_s;
    cuFloatComplex *SWORK ;

    status = cublasAlloc(n2, sizeof(cuDoubleComplex), (void**)&d_A);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! device memory allocation error (d_A)\n");
        exit(1);;
    }

    status = cublasAlloc(n2, sizeof(cuFloatComplex), (void**)&SWORK);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! device memory allocation error (SWORK)\n");
        exit(1);;
    }

    cuDoubleComplex *X , *WORK ;
    status = cublasAlloc(size5, sizeof(cuDoubleComplex), (void**)&WORK);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! device memory allocation error (WORK)\n");
        exit(1);;
    }
    status = cublasAlloc(size5, sizeof(cuDoubleComplex), (void**)&X);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! device memory allocation error (X)\n");
        exit(1);;
    }
    int ITER[1] ;
    status = cublasAlloc(size5, sizeof(cuDoubleComplex), (void**)&d_b);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! device memory allocation error (d_b)\n");
        exit(1);;
    }

    status = cublasAlloc(lwork_d, sizeof(cuDoubleComplex), (void**)&d_work_d);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! device memory allocation error (d_work_d)\n");
        exit(1);;
    }
    status = cublasAlloc(lwork, sizeof(cuFloatComplex), (void**)&d_work);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! device memory allocation error (d_work)\n");
        exit(1);
    }

    //status = cublasAlloc(nb, sizeof(cuDoubleComplex), (void**)&d_x);
    status = cublasAlloc(size5, sizeof(cuDoubleComplex), (void**)&d_x);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! device memory allocation error (d_x)\n");
        exit(1);;
    }

    cudaMallocHost( (void**)&h_work_d, lwork_d*sizeof(cuDoubleComplex) );
    if (h_work_d == 0) {
        fprintf (stderr, "!!!! host memory allocation error (work)\n");
        exit(1);;
    }
    cudaMallocHost( (void**)&h_work, lwork_d*sizeof(cuFloatComplex) );
    if (h_work == 0) {
        fprintf (stderr, "!!!! host memory allocation error (work)\n");
        exit(1);;
    }

    printf("\n\n");
    printf("           CPU GFlop/s                 GPU GFlop/s   \n");
    printf("  N          Doule           Double\tSingle\t Mixed    || b-Ax || / ||A||\n");
    printf("=========================================================================================\n");

    for(i=0; i<8; i++){
        M = N = lda = size[i]  ;
        n2 = N*N;

        lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
        lapackf77_zlarnv( &ione, ISEED, &N,  b   );
        lapackf77_zlacpy( MagmaUpperLowerStr, &N, &ione, b, &N, rr, &N );

        cublasSetVector(n2, sizeof(cuDoubleComplex), h_A, 1, d_A, 1);
        cublasSetVector(N,  sizeof(cuDoubleComplex), b,   1, d_b, 1);

        //=====================================================================
        //              Mixed Precision Iterative Refinement - GPU
        //=====================================================================
        start = get_current_time();
        magma_zcgeqrsv_gpu( M, N, nrhs, 
                            d_A, N, d_b, N, X, N, 
                            WORK, SWORK, ITER, &info, 
                            tau, lwork, h_work, d_work, tau_d, lwork_d,
                            h_work_d, d_work_d);
        end = get_current_time();
        mperf = (4.*N*N*N/3.+2.*N*N)/(1000000.*GetTimerValue(start,end));
        
        //=====================================================================
        //                 Error Computation
        //=====================================================================
        cublasGetVector(N, sizeof(cuDoubleComplex), X, 1, x, 1);
        blasf77_zgemv( MagmaNoTransStr, &N, &N, 
                       &c_neg_one, h_A, &N, 
                                   x,   &ione, 
                       &c_one,     rr,  &ione);
        matnorm = lapackf77_zlange("f", &N, &N, h_A, &N, &wnorm);

        //=====================================================================
        //                 Double Precision Solve
        //=====================================================================

        start = get_current_time();
        magma_zgeqrf_gpu2(M, N, d_A, N, tau_d, d_work_d, &info);
        magma_zgeqrs_gpu( M, N, nrhs, d_A, N, tau_d,
                          d_b, M, h_work_d, lwork_d, d_work_d, &info);
        end = get_current_time();
        dperf = (4.*N*N*N/3.+2.*N*N)/(1000000.*GetTimerValue(start,end));

        //=====================================================================
        //                 Single Precision Solve
        //=====================================================================
        start = get_current_time();
        magma_cgeqrf_gpu2(M, N, SWORK, N, tau, d_work, &info);
        magma_cgeqrs_gpu( M, N, nrhs, SWORK, N, tau,
                          SWORK + M * N , M, h_work, lwork, d_work, &info);
        end = get_current_time();
        sperf = (4.*N*N*N/3.+2.*N*N)/(1000000.*GetTimerValue(start,end));

        /* =====================================================================
           Performs operation using LAPACK
           =================================================================== */
        start = get_current_time();
        lapackf77_zgeqrf(&M, &N, h_A, &lda, tau_d, h_work_d, &lwork_d, &info);
        if (info < 0)
            printf("Argument %d of sgeqrf had an illegal value.\n", -info);
        
        // Solve the least-squares problem: min || A * X - B ||
        lapackf77_zunmqr( MagmaLeftStr, MagmaConjTransStr, &M, &nrhs, &M, h_A, &lda,
                          tau_d, b, &M, h_work_d, &lwork_d, &info);

        // B(1:N,1:NRHS) := inv(R) * B(1:N,1:NRHS)
        blasf77_ztrsm( MagmaLeftStr, MagmaUpperStr, MagmaNoTransStr, MagmaNonUnitStr, 
                       &M, &nrhs, &c_one, h_A, &lda, b, &M);
        end = get_current_time();
        cpu_perf = (4.*N*N*N/3.+2.*N*N)/(1000000.*GetTimerValue(start,end));


        printf("%5d \t%8.2f\t%9.2f\t%6.2f\t%6.2f  \t %e",
               size[i], cpu_perf, dperf, sperf, mperf ,
               lapackf77_zlange("f", &N, &nrhs, rr, &N, &wnorm) / matnorm );

        printf(" %2d \n", ITER[0]);

        if (argc != 1)
            break;
    }

    /* Memory clean up */
    free(h_A);
    free(h_AA);
    free(tau_d);
    free(tau);
    free(x);
    free(b);
    free(rr);
    cublasFree(h_work_d);
    cublasFree(d_work_d);
    cublasFree(h_work);
    cublasFree(d_work);
    cublasFree(d_x);
    cublasFree(h_R);
    cublasFree(d_A);
    cublasFree(d_b);
    cublasFree(WORK);
    cublasFree(SWORK);
    cublasFree(X);

    /* Shutdown */
    status = cublasShutdown();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! shutdown error (A)\n");
    }

#endif /*defined(PRECISION_z) && (GPUSHMEM < 200)*/
}
