/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> c d s
       @author Mark Gates
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
   -- Testing zgesv_gpu
*/
int main(int argc , char **argv)
{
    TESTING_INIT();

    real_Double_t   gflops, gpu_perf, gpu_time;
    double          Rnorm, Anorm, Xnorm, *work;
    cuDoubleComplex c_one     = MAGMA_Z_ONE;
    cuDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    cuDoubleComplex *h_A, *h_B, *h_X;
    cuDoubleComplex *d_A, *d_B;
    magma_int_t *ipiv;
    magma_int_t N, NRHS, lda, ldb, ldda, lddb, info, sizeA, sizeB;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    
    magma_opts opts;
    parse_opts( argc, argv, &opts );
    
    NRHS = opts.nrhs;
    
    printf("    N   NRHS   GPU GFlop/s (sec)   ||B - AX|| / ||A||*||X||\n");
    printf("===========================================================\n");
    for( int i = 0; i < opts.ntest; ++i ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[i];
            lda    = N;
            ldb    = lda;
            ldda   = ((N+31)/32)*32;
            lddb   = ldda;
            gflops = ( FLOPS_ZGETRF( N, N ) + FLOPS_ZGETRS( N, NRHS ) ) / 1e9;
            
            TESTING_MALLOC( h_A, cuDoubleComplex, lda*N    );
            TESTING_MALLOC( h_B, cuDoubleComplex, ldb*NRHS );
            TESTING_MALLOC( h_X, cuDoubleComplex, ldb*NRHS );
            TESTING_MALLOC( work, double,         N        );
            TESTING_MALLOC( ipiv, magma_int_t,    N        );
            
            TESTING_DEVALLOC( d_A, cuDoubleComplex, ldda*N    );
            TESTING_DEVALLOC( d_B, cuDoubleComplex, lddb*NRHS );
            
            /* Initialize the matrices */
            sizeA = lda*N;
            sizeB = ldb*NRHS;
            lapackf77_zlarnv( &ione, ISEED, &sizeA, h_A );
            lapackf77_zlarnv( &ione, ISEED, &sizeB, h_B );
            
            magma_zsetmatrix( N, N,    h_A, lda, d_A, ldda );
            magma_zsetmatrix( N, NRHS, h_B, ldb, d_B, lddb );
            
            //=====================================================================
            // Solve Ax = b through an LU factorization, using MAGMA
            //=====================================================================
            gpu_time = magma_wtime();
            magma_zgesv_gpu( N, NRHS, d_A, ldda, ipiv, d_B, lddb, &info );
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0)
                printf("magma_zgesv_gpu returned error %d.\n", (int) info);
            
            //=====================================================================
            // Residual
            //=====================================================================
            magma_zgetmatrix( N, NRHS, d_B, lddb, h_X, ldb );
            
            Anorm = lapackf77_zlange("I", &N, &N,    h_A, &lda, work);
            Xnorm = lapackf77_zlange("I", &N, &NRHS, h_X, &ldb, work);
            
            blasf77_zgemm( MagmaNoTransStr, MagmaNoTransStr, &N, &NRHS, &N,
                           &c_one,     h_A, &lda,
                                       h_X, &ldb,
                           &c_neg_one, h_B, &ldb);
            
            Rnorm = lapackf77_zlange("I", &N, &NRHS, h_B, &ldb, work);
            
            printf( "%5d  %5d   %7.2f (%7.2f)   %8.2e\n",
                    (int) N, (int) NRHS, gpu_perf, gpu_time, Rnorm/(Anorm*Xnorm) );
            
            TESTING_FREE( h_A );
            TESTING_FREE( h_B );
            TESTING_FREE( h_X );
            TESTING_FREE( work );
            TESTING_FREE( ipiv );
            
            TESTING_DEVFREE( d_A );
            TESTING_DEVFREE( d_B );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    TESTING_FINALIZE();
    return 0;
}
