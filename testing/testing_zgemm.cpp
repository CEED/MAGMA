/*
 *  -- MAGMA (version 1.1) --
 *     Univ. of Tennessee, Knoxville
 *     Univ. of California, Berkeley
 *     Univ. of Colorado, Denver
 *     November 2011
 *
 * @precisions normal z -> c d s
 *
 **/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <cublas.h>

#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

int main( int argc, char** argv)
{
    TESTING_CUDA_INIT();

    real_Double_t   gflops, magma_perf, magma_time, cublas_perf, cublas_time, cpu_perf, cpu_time;
    double      magma_error, cublas_error, work[1];
    char        transA = MagmaNoTrans;
    char        transB = MagmaNoTrans;

    magma_int_t istart = 1024;
    magma_int_t iend   = 6240;
    magma_int_t M, M0 = 0;
    magma_int_t N, N0 = 0;
    magma_int_t K, K0 = 0;
    magma_int_t i;
    magma_int_t Am, An, Bm, Bn;
    magma_int_t szeA, szeB, szeC;
    magma_int_t lda, ldb, ldc, ldda, lddb, lddc;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    
    cuDoubleComplex *h_A, *h_B, *h_C, *h_C2, *h_C3;
    cuDoubleComplex *d_A, *d_B, *d_C;
    cuDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    cuDoubleComplex alpha = MAGMA_Z_MAKE(  0.29, -0.86 );
    cuDoubleComplex beta  = MAGMA_Z_MAKE( -0.48,  0.38 );
    
    int lapack = getenv("MAGMA_RUN_LAPACK") != NULL;
    int count = 1;

    printf("\nUsage: testing_zgemm [-NN|NT|TN|TT|NC|CN|TC|CT|CC] -M m -N n -K k -count c -l\n"
            "  -l  or setting $MAGMA_RUN_LAPACK runs CPU BLAS,\n"
            "      and computes both MAGMA and CUBLAS error using CPU BLAS result.\n"
            "      Else, MAGMA error is computed using CUBLAS result.\n\n");

    for( int i = 1; i < argc; ++i ) {
        if ( strcmp("-N", argv[i]) == 0 && i+1 < argc ){
            N0 = atoi(argv[++i]);
        }
        else if ( strcmp("-M", argv[i]) == 0 && i+1 < argc ){
            M0 = atoi(argv[++i]);
        }
        else if ( strcmp("-K", argv[i]) == 0 && i+1 < argc ){
            K0 = atoi(argv[++i]);
        }
        else if (strcmp("-NN", argv[i])==0){
            transA = transB = MagmaNoTrans;
        }
        else if (strcmp("-TT", argv[i])==0){
            transA = transB = MagmaTrans;
        }
        else if (strcmp("-NT", argv[i])==0){
            transA = MagmaNoTrans;
            transB = MagmaTrans;
        }
        else if (strcmp("-TN", argv[i])==0){
            transA = MagmaTrans;
            transB = MagmaNoTrans;
        }
        else if (strcmp("-NC", argv[i])==0){
            transA = MagmaNoTrans;
            transB = MagmaConjTrans;
        }
        else if (strcmp("-TC", argv[i])==0){
            transA = MagmaTrans;
            transB = MagmaConjTrans;
        }
        else if (strcmp("-CN", argv[i])==0){
            transA = MagmaConjTrans;
            transB = MagmaNoTrans;
        }
        else if (strcmp("-CT", argv[i])==0){
            transA = MagmaConjTrans;
            transB = MagmaTrans;
        }
        else if (strcmp("-CC", argv[i])==0){
            transA = transB = MagmaConjTrans;
        }
        else if (strcmp("-l", argv[i])==0) {
            lapack = true;
        }
        else if ( strcmp("-count", argv[i]) == 0 && i+1 < argc ){
            count = atoi(argv[++i]);
        }
        else {
            printf( "invalid argument: %s\n", argv[i] );
            exit(1);
        }
    }

    if ( (M0 != 0) && (N0 != 0) && (K0 != 0) )
        iend = istart + 1;
    
    M = N = K = iend;
    if ( M0 != 0 ) M = M0;
    if ( N0 != 0 ) N = N0;
    if ( K0 != 0 ) K = K0;
    
    if( transA == MagmaNoTrans ) {
        Am = M;
        An = K;
    }  else {
        Am = K;
        An = M;
    }
    
    if( transB == MagmaNoTrans ) {
        Bm = K;
        Bn = N;
    }  else {
        Bm = N;
        Bn = K;
    }
    
    lda = ldc = M;
    ldb = Bm;
    
    ldda = ((M+31)/32)*32;
    lddb = ((ldb+31)/32)*32;
    lddc = ldda;

    K += 32;
    M += 32;
    N += 32;

    TESTING_MALLOC( h_A,  cuDoubleComplex, lda*K );
    TESTING_MALLOC( h_B,  cuDoubleComplex, ldb*Bn );
    TESTING_MALLOC( h_C,  cuDoubleComplex, ldc*N );
    TESTING_MALLOC( h_C2, cuDoubleComplex, ldc*N );
    TESTING_MALLOC( h_C3, cuDoubleComplex, ldc*N );

    TESTING_DEVALLOC( d_A, cuDoubleComplex, ldda*K );
    TESTING_DEVALLOC( d_B, cuDoubleComplex, lddb*Bn );
    TESTING_DEVALLOC( d_C, cuDoubleComplex, lddc*N );

    printf("Testing transA = %c  transB = %c\n", transA, transB);
    printf("    M     N     K   MAGMA Gflop/s (sec)  CUBLAS Gflop/s (sec)  CPU Gflop/s (sec)  MAGMA error  CUBLAS error\n");
    printf("===========================================================================================================\n");
    for( i=istart; i<iend; i = (int)(i*1.25) ) {
        for( int cnt = 0; cnt < count; ++cnt ) {
            M = N = K = i;
            if ( M0 != 0 ) M = M0;
            if ( N0 != 0 ) N = N0;
            if ( K0 != 0 ) K = K0;
    
            if( transA == MagmaNoTrans ) {
                lda = Am = M;
                An = K;
            }  else {
                lda = Am = K;
                An = M;
            }
    
            if( transB == MagmaNoTrans ) {
                ldb = Bm = K;
                Bn = N;
            }  else {
                ldb = Bm = N;
                Bn = K;
            }
            gflops = FLOPS_ZGEMM( M, N, K ) / 1e9;
            ldc = M;
    
            ldda = ((lda+31)/32)*32;
            lddb = ((ldb+31)/32)*32;
            lddc = ((ldc+31)/32)*32;
    
            szeA = lda * An;
            szeB = ldb * Bn;
            szeC = ldc * N;
    
            /* Initialize the matrices */
            lapackf77_zlarnv( &ione, ISEED, &szeA, h_A );
            lapackf77_zlarnv( &ione, ISEED, &szeB, h_B );
            lapackf77_zlarnv( &ione, ISEED, &szeC, h_C );
            
            /* =====================================================================
               Performs operation using MAGMA-BLAS
               =================================================================== */
            magma_zsetmatrix( Am, An, h_A, lda, d_A, ldda );
            magma_zsetmatrix( Bm, Bn, h_B, ldb, d_B, lddb );
            magma_zsetmatrix( M, N, h_C, ldc, d_C, lddc );
    
            magma_time = magma_sync_wtime( NULL );
            magmablas_zgemm( transA, transB, M, N, K,
                             alpha, d_A, ldda,
                                    d_B, lddb,
                             beta,  d_C, lddc );
            magma_time = magma_sync_wtime( NULL ) - magma_time;
            magma_perf = gflops / magma_time;
            
            magma_zgetmatrix( M, N, d_C, lddc, h_C2, ldc );
            
            /* =====================================================================
               Performs operation using CUDA-BLAS
               =================================================================== */
            magma_zsetmatrix( M, N, h_C, ldc, d_C, lddc );
            
            cublas_time = magma_sync_wtime( NULL );
            cublasZgemm( transA, transB, M, N, K,
                         alpha, d_A, ldda,
                                d_B, lddb,
                         beta,  d_C, lddc );
            cublas_time = magma_sync_wtime( NULL ) - cublas_time;
            cublas_perf = gflops / cublas_time;
            
            magma_zgetmatrix( M, N, d_C, lddc, h_C3, ldc );
            
            /* =====================================================================
               Performs operation using BLAS
               =================================================================== */
            if ( lapack ) {
                cpu_time = magma_wtime();
                blasf77_zgemm( &transA, &transB, &M, &N, &K,
                               &alpha, h_A, &lda,
                                       h_B, &ldb,
                               &beta,  h_C, &ldc );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
            }
            
            /* =====================================================================
               Error Computation and Performance Compariosn
               =================================================================== */
            if ( lapack ) {
                // compare both magma & cublas to lapack
                blasf77_zaxpy(&szeC, &c_neg_one, h_C, &ione, h_C2, &ione);
                magma_error = lapackf77_zlange("M", &M, &N, h_C2, &ldc, work);
                
                blasf77_zaxpy(&szeC, &c_neg_one, h_C, &ione, h_C3, &ione);
                cublas_error = lapackf77_zlange("M", &M, &N, h_C3, &ldc, work);
                
                printf("%5d %5d %5d   %7.2f (%7.4f)    %7.2f (%7.4f)   %7.2f (%7.4f)    %8.2e     %8.2e\n",
                       (int) M, (int) N, (int) K,
                       magma_perf, magma_time, cublas_perf, cublas_time, cpu_perf, cpu_time,
                       magma_error, cublas_error );
            }
            else {
                // compare magma to cublas
                blasf77_zaxpy(&szeC, &c_neg_one, h_C3, &ione, h_C2, &ione);
                magma_error = lapackf77_zlange("M", &M, &N, h_C2, &ldc, work);
                
                printf("%5d %5d %5d   %7.2f (%7.4f)    %7.2f (%7.4f)     ---   (  ---  )    %8.2e     ---\n",
                       (int) M, (int) N, (int) K,
                       magma_perf, magma_time, cublas_perf, cublas_time,
                       magma_error );
            }
        }
        if ( count > 1 ) {
            printf( "\n" );
        }
    }

    /* Memory clean up */
    TESTING_FREE( h_A );
    TESTING_FREE( h_B );
    TESTING_FREE( h_C );
    TESTING_FREE( h_C2 );
    TESTING_FREE( h_C3 );

    TESTING_DEVFREE( d_A );
    TESTING_DEVFREE( d_B );
    TESTING_DEVFREE( d_C );

    TESTING_CUDA_FINALIZE();
}
