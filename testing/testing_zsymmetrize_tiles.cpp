/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> s d c

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
   -- Testing zsymmetrize
   Code is very similar to testing_ztranspose.cpp
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t    gbytes, gpu_perf, gpu_time, cpu_perf, cpu_time;
    double           error, work[1];
    magmaDoubleComplex  c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex *h_A, *h_R;
    magmaDoubleComplex *d_A;
    magma_int_t N, nb, size, lda, ldda, mstride, nstride, ntile;
    magma_int_t ione     = 1;
    
    magma_opts opts;
    parse_opts( argc, argv, &opts );
    nb = (opts.nb == 0 ? 64 : opts.nb);
    mstride = 2*nb;
    nstride = 3*nb;
    
    printf("nb=%d, mstride=%d, nstride=%d\n", (int) nb, (int) mstride, (int) nstride );
    printf("    N ntile   CPU GByte/s (sec)   GPU GByte/s (sec)   check\n");
    printf("===========================================================\n");
    for( int i = 0; i < opts.ntest; ++i ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[i];
            lda    = N;
            ldda   = ((N+31)/32)*32;
            size   = lda*N;
            
            if ( N < nb ) {
                ntile = 0;
            } else {
                ntile = min( (N - nb)/mstride + 1,
                             (N - nb)/nstride + 1 );
            }
            // load each tile, save each tile
            gbytes = sizeof(magmaDoubleComplex) * 2.*nb*nb*ntile / 1e9;
            
            TESTING_MALLOC(   h_A, magmaDoubleComplex, size   );
            TESTING_MALLOC(   h_R, magmaDoubleComplex, size   );
            TESTING_DEVALLOC( d_A, magmaDoubleComplex, ldda*N );
            
            /* Initialize the matrix */
            for( int j = 0; j < N; ++j ) {
                for( int i = 0; i < N; ++i ) {
                    h_A[i + j*lda] = MAGMA_Z_MAKE( i + j/10000., j );
                }
            }

            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            magma_zsetmatrix( N, N, h_A, lda, d_A, ldda );
            
            gpu_time = magma_sync_wtime( 0 );
            magmablas_zsymmetrize_tiles( opts.uplo, nb, d_A, ldda, ntile, mstride, nstride );
            gpu_time = magma_sync_wtime( 0 ) - gpu_time;
            gpu_perf = gbytes / gpu_time;
            
            /* =====================================================================
               Performs operation using naive in-place algorithm
               (LAPACK doesn't implement symmetrize)
               =================================================================== */
            cpu_time = magma_wtime();
            for( int tile = 0; tile < ntile; ++tile ) {
                int offset = tile*mstride + tile*nstride*lda;
                for( int j = 0; j < nb; ++j ) {
                    for( int i = 0; i < j; ++i ) {
                        if ( opts.uplo == MagmaLower ) {
                            h_A[offset + i + j*lda] = MAGMA_Z_CNJG( h_A[offset + j + i*lda] );
                        }
                        else {
                            h_A[offset + j + i*lda] = MAGMA_Z_CNJG( h_A[offset + i + j*lda] );
                        }
                    }
                }
            }
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gbytes / cpu_time;
            
            /* =====================================================================
               Check the result
               =================================================================== */
            magma_zgetmatrix( N, N, d_A, ldda, h_R, lda );
            
            blasf77_zaxpy(&size, &c_neg_one, h_A, &ione, h_R, &ione);
            error = lapackf77_zlange("f", &N, &N, h_R, &lda, work);

            printf("%5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %s\n",
                   (int) N, (int) ntile,
                   cpu_perf, cpu_time, gpu_perf, gpu_time,
                   (error == 0. ? "okay" : "fail") );
            
            TESTING_FREE( h_A );
            TESTING_FREE( h_R );
            TESTING_DEVFREE( d_A );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    TESTING_FINALIZE();
    return 0;
}
