/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> c d s
*/
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

#define PRECISION_z

#if (GPUSHMEM >= 200)
// todo: put prototypes in include header file
extern "C" magma_int_t
magmablas_zhemv2( char uplo, magma_int_t n,
                      cuDoubleComplex alpha,
                      cuDoubleComplex *A, magma_int_t lda,
                      cuDoubleComplex *X, magma_int_t incx,
                      cuDoubleComplex beta,
                      cuDoubleComplex *Y, magma_int_t incy,
                      cuDoubleComplex *work, magma_int_t lwork);
#endif

int main(int argc, char **argv)
{
    TESTING_INIT();

    real_Double_t   gflops, magma_perf, magma_time, cublas_perf, cublas_time, cpu_perf, cpu_time;
    double          magma_error, cublas_error, work[1];
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t N, lda, sizeA, sizeX, sizeY, blocks, ldwork;
    magma_int_t incx = 1;
    magma_int_t incy = 1;
    magma_int_t nb   = 64;
    cuDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    cuDoubleComplex alpha = MAGMA_Z_MAKE(  1.5, -2.3 );
    cuDoubleComplex beta  = MAGMA_Z_MAKE( -0.6,  0.8 );
    cuDoubleComplex *A, *X, *Y, *Ycublas, *Ymagma;
    cuDoubleComplex *dA, *dX, *dY, *dC_work;
    
    magma_opts opts;
    parse_opts( argc, argv, &opts );

    printf("    N   MAGMA Gflop/s (ms)  CUBLAS Gflop/s (ms)   CPU Gflop/s (ms)  MAGMA error  CUBLAS error\n");
    printf("=============================================================================================\n");
    for( int i = 0; i < opts.ntest; ++i ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[i];
            lda    = ((N + 31)/32)*32;
            sizeA  = N*lda;
            sizeX  = N*incx;
            sizeY  = N*incy;
            gflops = FLOPS_ZHEMV( N ) / 1e9;
            
            TESTING_MALLOC( A,       cuDoubleComplex, sizeA );
            TESTING_MALLOC( X,       cuDoubleComplex, sizeX );
            TESTING_MALLOC( Y,       cuDoubleComplex, sizeY );
            TESTING_MALLOC( Ycublas, cuDoubleComplex, sizeY );
            TESTING_MALLOC( Ymagma,  cuDoubleComplex, sizeY );
            
            TESTING_DEVALLOC( dA, cuDoubleComplex, sizeA );
            TESTING_DEVALLOC( dX, cuDoubleComplex, sizeX );
            TESTING_DEVALLOC( dY, cuDoubleComplex, sizeY );
            
            blocks = (N + nb - 1) / nb;
            ldwork = lda * (blocks + 1);
            TESTING_DEVALLOC( dC_work, cuDoubleComplex, ldwork );
            
            /* Initialize the matrix */
            lapackf77_zlarnv( &ione, ISEED, &sizeA, A );
            magma_zmake_hermitian( N, A, lda );
            lapackf77_zlarnv( &ione, ISEED, &sizeX, X );
            lapackf77_zlarnv( &ione, ISEED, &sizeY, Y );
            
            /* =====================================================================
               Performs operation using CUBLAS
               =================================================================== */
            magma_zsetmatrix( N, N, A, lda, dA, lda );
            magma_zsetvector( N, X, incx, dX, incx );
            magma_zsetvector( N, Y, incy, dY, incy );
            
            cublas_time = magma_sync_wtime( 0 );
            cublasZhemv( opts.uplo, N, alpha, dA, lda, dX, incx, beta, dY, incy );
            cublas_time = magma_sync_wtime( 0 ) - cublas_time;
            cublas_perf = gflops / cublas_time;
            
            magma_zgetvector( N, dY, incy, Ycublas, incy );
            
            /* =====================================================================
               Performs operation using MAGMA BLAS
               =================================================================== */
            magma_zsetvector( N, Y, incy, dY, incy );
            
            magma_time = magma_sync_wtime( 0 );
            #if (GPUSHMEM >= 200)
            magmablas_zhemv2( opts.uplo, N, alpha, dA, lda, dX, incx, beta, dY, incy, dC_work, ldwork );
            #else
            magmablas_zhemv( opts.uplo, N, alpha, dA, lda, dX, incx, beta, dY, incy );
            #endif
            magma_time = magma_sync_wtime( 0 ) - magma_time;
            magma_perf = gflops / magma_time;
            
            magma_zgetvector( N, dY, incy, Ymagma, incy );
            
            /* =====================================================================
               Performs operation using CPU BLAS
               =================================================================== */
            cpu_time = magma_wtime();
            blasf77_zhemv( &opts.uplo, &N, &alpha, A, &lda, X, &incx, &beta, Y, &incy );
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gflops / cpu_time;
            
            /* =====================================================================
               Check the result
               =================================================================== */
            blasf77_zaxpy( &N, &c_neg_one, Y, &incy, Ymagma, &incy );
            magma_error = lapackf77_zlange( "M", &N, &ione, Ymagma, &N, work ) / N;
            
            blasf77_zaxpy( &N, &c_neg_one, Y, &incy, Ycublas, &incy );
            cublas_error = lapackf77_zlange( "M", &N, &ione, Ycublas, &N, work ) / N;
            
            printf("%5d   %7.2f (%7.2f)    %7.2f (%7.2f)   %7.2f (%7.2f)    %8.2e     %8.2e\n",
                   (int) N,
                   magma_perf,  1000.*magma_time,
                   cublas_perf, 1000.*cublas_time,
                   cpu_perf,    1000.*cpu_time,
                   magma_error, cublas_error );
            
            TESTING_FREE( A );
            TESTING_FREE( X );
            TESTING_FREE( Y );
            TESTING_FREE( Ycublas );
            TESTING_FREE( Ymagma );
            
            TESTING_DEVFREE( dA );
            TESTING_DEVFREE( dX );
            TESTING_DEVFREE( dY );
            TESTING_DEVFREE( dC_work );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    TESTING_FINALIZE();
    return 0;
}
