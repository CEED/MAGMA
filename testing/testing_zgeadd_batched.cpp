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
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <cublas.h>

// includes, project
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgeadd_batched
   Code is very similar to testing_zlacpy_batched.cpp
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t    gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    double           error, work[1];
    magmaDoubleComplex  c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex *h_A, *h_B;
    magmaDoubleComplex *d_A, *d_B;
    magmaDoubleComplex **hAarray, **hBarray, **dAarray, **dBarray;
    magmaDoubleComplex alpha = MAGMA_Z_MAKE( 3.1415, 2.718 );
    magma_int_t M, N, mb, nb, size, lda, ldda, mstride, nstride, ntile;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    
    magma_opts opts;
    parse_opts( argc, argv, &opts );
    mb = (opts.nb == 0 ? 32 : opts.nb);
    nb = (opts.nb == 0 ? 64 : opts.nb);
    mstride = 2*mb;
    nstride = 3*nb;
    
    printf("mb=%d, nb=%d, mstride=%d, nstride=%d\n", mb, nb, mstride, nstride );
    printf("    M     N ntile   CPU GFlop/s (sec)   GPU GFlop/s (sec)   error   \n");
    printf("====================================================================\n");
    for( int i = 0; i < opts.ntest; ++i ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[i];
            N = opts.nsize[i];
            lda    = M;
            ldda   = ((M+31)/32)*32;
            size   = lda*N;
            
            if ( N < nb || M < nb ) {
                ntile = 0;
            } else {
                ntile = min( (M - nb)/mstride + 1,
                             (N - nb)/nstride + 1 );
            }
            gflops = 2.*mb*nb*ntile / 1e9;
            
            TESTING_MALLOC(   h_A, magmaDoubleComplex, lda *N );
            TESTING_MALLOC(   h_B, magmaDoubleComplex, lda *N );
            TESTING_DEVALLOC( d_A, magmaDoubleComplex, ldda*N );
            TESTING_DEVALLOC( d_B, magmaDoubleComplex, ldda*N );
            
            TESTING_MALLOC(   hAarray, magmaDoubleComplex*, ntile );
            TESTING_MALLOC(   hBarray, magmaDoubleComplex*, ntile );
            TESTING_DEVALLOC( dAarray, magmaDoubleComplex*, ntile );
            TESTING_DEVALLOC( dBarray, magmaDoubleComplex*, ntile );
            
            lapackf77_zlarnv( &ione, ISEED, &size, h_A );
            lapackf77_zlarnv( &ione, ISEED, &size, h_B );

            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            magma_zsetmatrix( M, N, h_A, lda, d_A, ldda );
            magma_zsetmatrix( M, N, h_B, lda, d_B, ldda );
            
            // setup pointers
            for( int tile = 0; tile < ntile; ++tile ) {
                int offset = tile*mstride + tile*nstride*ldda;
                hAarray[tile] = &d_A[offset];
                hBarray[tile] = &d_B[offset];
            }
            magma_setvector( ntile, sizeof(magmaDoubleComplex*), hAarray, 1, dAarray, 1 );
            magma_setvector( ntile, sizeof(magmaDoubleComplex*), hBarray, 1, dBarray, 1 );
            
            gpu_time = magma_sync_wtime( 0 );
            magmablas_zgeadd_batched( mb, nb, alpha, dAarray, ldda, dBarray, ldda, ntile );
            gpu_time = magma_sync_wtime( 0 ) - gpu_time;
            gpu_perf = gflops / gpu_time;
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            cpu_time = magma_wtime();
            for( int tile = 0; tile < ntile; ++tile ) {
                int offset = tile*mstride + tile*nstride*lda;
                for( int j = 0; j < nb; ++j ) {
                    blasf77_zaxpy( &mb, &alpha,
                                   &h_A[offset + j*lda], &ione,
                                   &h_B[offset + j*lda], &ione );
                }
            }
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gflops / cpu_time;
            
            /* =====================================================================
               Check the result
               =================================================================== */
            magma_zgetmatrix( M, N, d_B, ldda, h_A, lda );
            
            error = lapackf77_zlange( "F", &M, &N, h_B, &lda, work );
            blasf77_zaxpy(&size, &c_neg_one, h_A, &ione, h_B, &ione);
            error = lapackf77_zlange("f", &M, &N, h_B, &lda, work) / error;

            printf("%5d %5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e\n",
                   (int) M, (int) N, (int) ntile,
                   cpu_perf, cpu_time, gpu_perf, gpu_time, error );
            
            TESTING_FREE( h_A );
            TESTING_FREE( h_B );
            TESTING_DEVFREE( d_A );
            TESTING_DEVFREE( d_B );
            
            TESTING_FREE( hAarray );
            TESTING_FREE( hBarray );
            TESTING_DEVFREE( dAarray );
            TESTING_DEVFREE( dBarray );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    TESTING_FINALIZE();
    return 0;
}
