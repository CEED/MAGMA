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
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing ztranspose
*/
int main( int argc, char** argv)
{
    TESTING_CUDA_INIT();

    real_Double_t    gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    double           error, work[1];
    cuDoubleComplex  c_neg_one = MAGMA_Z_NEG_ONE;
    cuDoubleComplex *h_A, *h_R, tmp;
    cuDoubleComplex *d_A;
    magma_int_t N, size, lda, ldda, info;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    
    magma_opts opts;
    parse_opts( argc, argv, &opts );
    
    printf("    N   CPU GFlop/s (sec)   GPU GFlop/s (sec)   check\n");
    printf("=====================================================\n");
    for( int i = 0; i < opts.ntest; ++i ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[i];
            lda    = N;
            ldda   = ((N+31)/32)*32;
            size   = lda*N;
            gflops = 2*N*N / 1e9;
    
            TESTING_MALLOC(   h_A, cuDoubleComplex, size   );
            TESTING_MALLOC(   h_R, cuDoubleComplex, size   );
            TESTING_DEVALLOC( d_A, cuDoubleComplex, ldda*N );
            
            /* Initialize the matrix */
            lapackf77_zlarnv( &ione, ISEED, &size, h_A );
            for( int j = 0; j < N; ++j ) {
                for( int i = 0; i < N; ++i ) {
                    h_A[i + j*lda] = MAGMA_Z_MAKE( i + j/1000., j );
                }
            }
            //printf( "A=" );
            //magma_zprint( N, N, h_A, lda );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            magma_zsetmatrix( N, N, h_A, lda, d_A, ldda );
            gpu_time = magma_wtime();
            magmablas_ztranspose_inplace( N, d_A, ldda );
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0)
                printf("magma_ztranspose returned error %d.\n", (int) info);
            
            /* =====================================================================
               Performs operation using naive in-place algorithm
               (LAPACK doesn't implement transpose)
               =================================================================== */
            cpu_time = magma_wtime();
            for( int j = 0; j < N; ++j ) {
                for( int i = 0; i < j; ++i ) {
                    tmp            = h_A[i + j*lda];
                    h_A[i + j*lda] = h_A[j + i*lda];
                    h_A[j + i*lda] = tmp;
                }
            }
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gflops / cpu_time;
            

            /* =====================================================================
               Check the result
               =================================================================== */
            magma_zgetmatrix( N, N, d_A, ldda, h_R, lda );
            
            //printf( "A^T=" );
            //magma_zprint( N, N, h_A, lda );
            //printf( "dA^T=" );
            //magma_zprint( N, N, h_R, lda );
            
            blasf77_zaxpy(&size, &c_neg_one, h_A, &ione, h_R, &ione);
            error = lapackf77_zlange("f", &N, &N, h_R, &lda, work);

            //printf( "diff=" );
            //magma_zprint( N, N, h_R, lda );
            
            printf("%5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %s\n",
                   (int) N, cpu_perf, cpu_time, gpu_perf, gpu_time,
                   (error == 0 ? "okay" : "fail") );
            
            TESTING_FREE( h_A );
            TESTING_FREE( h_R );
            TESTING_DEVFREE( d_A );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    TESTING_CUDA_FINALIZE();
    return 0;
}
