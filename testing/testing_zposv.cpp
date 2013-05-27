/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> c d s
*/
// includes, system
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <cublas.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zposv
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t   gflops, gpu_perf, gpu_time;
    double          Rnorm, Anorm, Xnorm, *work;
    cuDoubleComplex c_one     = MAGMA_Z_ONE;
    cuDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    cuDoubleComplex *h_A, *h_R, *h_B, *h_X;
    magma_int_t N, lda, ldb, info, sizeA, sizeB;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    
    magma_opts opts;
    parse_opts( argc, argv, &opts );
    
    printf("    N  NRHS   GPU GFlop/s (sec)   ||B - AX|| / ||A||*||X||\n");
    printf("==========================================================\n");
    for( int i = 0; i < opts.ntest; ++i ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N   = opts.nsize[i];
            lda = ldb = N;
            gflops = ( FLOPS_ZPOTRF( N ) + FLOPS_ZPOTRS( N, opts.nrhs ) ) / 1e9;
            
            TESTING_MALLOC( h_A, cuDoubleComplex, lda*N         );
            TESTING_MALLOC( h_R, cuDoubleComplex, lda*N         );
            TESTING_MALLOC( h_B, cuDoubleComplex, ldb*opts.nrhs );
            TESTING_MALLOC( h_X, cuDoubleComplex, ldb*opts.nrhs );
            TESTING_MALLOC( work, double,         N             );
            
            /* ====================================================================
               Initialize the matrix
               =================================================================== */
            sizeA = lda*N;
            sizeB = ldb*opts.nrhs;
            lapackf77_zlarnv( &ione, ISEED, &sizeA, h_A );
            lapackf77_zlarnv( &ione, ISEED, &sizeB, h_B );
            magma_zmake_hpd( N, h_A, lda );
            
            // copy A to R and B to X; save A and B for residual
            lapackf77_zlacpy( MagmaUpperLowerStr, &N, &N,         h_A, &lda, h_R, &lda );
            lapackf77_zlacpy( MagmaUpperLowerStr, &N, &opts.nrhs, h_B, &ldb, h_X, &ldb );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            magma_zposv( opts.uplo, N, opts.nrhs, h_R, lda, h_X, ldb, &info );
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0)
                printf("magma_zpotrf returned error %d.\n", (int) info);
            
            /* =====================================================================
               Residual
               =================================================================== */
            Anorm = lapackf77_zlange("I", &N, &N,         h_A, &lda, work);
            Xnorm = lapackf77_zlange("I", &N, &opts.nrhs, h_X, &ldb, work);
            
            blasf77_zgemm( MagmaNoTransStr, MagmaNoTransStr, &N, &opts.nrhs, &N,
                           &c_one,     h_A, &lda,
                                       h_X, &ldb,
                           &c_neg_one, h_B, &ldb );
            
            Rnorm = lapackf77_zlange("I", &N, &opts.nrhs, h_B, &ldb, work);
            
            printf( "%5d %5d   %7.2f (%7.2f)   %8.2e\n",
                    (int) N, (int) opts.nrhs, gpu_perf, gpu_time, Rnorm/(Anorm*Xnorm) );
            
            TESTING_FREE( h_A );
            TESTING_FREE( h_R );
            TESTING_FREE( h_B );
            TESTING_FREE( h_X );
            TESTING_FREE( work );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    TESTING_FINALIZE();
    return 0;
}
