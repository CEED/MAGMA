/*
 *  -- MAGMA (version 1.1) --
 *     Univ. of Tennessee, Knoxville
 *     Univ. of California, Berkeley
 *     Univ. of Colorado, Denver
 *     November 2011
 *
 * @precisions normal z -> c d s
 * @author Chongxiao Cao
 *
 **/
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
   -- Testing ztrmv
*/
int main( int argc, char** argv)
{
    TESTING_CUDA_INIT();

    real_Double_t   gflops, cublas_perf, cublas_time, cpu_perf, cpu_time;
    double          cublas_error, Cnorm, work[1];
    magma_int_t N;
    magma_int_t Ak;
    magma_int_t sizeA;
    magma_int_t lda, ldda;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    
    cuDoubleComplex *h_A, *h_x, *h_xcublas;
    cuDoubleComplex *d_A, *d_x;
    cuDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    
    magma_opts opts;
    parse_opts( argc, argv, &opts );
    
    printf("If running lapack (option --lapack), MAGMA and CUBLAS error are both computed\n"
           "relative to CPU BLAS result. Else, MAGMA error is computed relative to CUBLAS result.\n\n"
           "uplo = %c, transA = %c, diag = %c \n", opts.uplo, opts.transA, opts.diag );
    printf("    N   CUBLAS Gflop/s (ms)   CPU Gflop/s (ms)  CUBLAS error\n");
    printf("==================================================================\n");
    for( int i = 0; i < opts.ntest; ++i ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[i];
            gflops = FLOPS_ZTRMM(opts.side, N, 1) / 1e9;

            lda = N;
            Ak = N;
            
            ldda = ((lda+31)/32)*32;
            
            sizeA = lda*Ak;
            
            TESTING_MALLOC( h_A,  cuDoubleComplex, lda*Ak );
            TESTING_MALLOC( h_x,  cuDoubleComplex, N  );
            TESTING_MALLOC( h_xcublas, cuDoubleComplex, N  );
            
            TESTING_DEVALLOC( d_A, cuDoubleComplex, ldda*Ak );
            TESTING_DEVALLOC( d_x, cuDoubleComplex, N  );
            
            /* Initialize the matrices */
            lapackf77_zlarnv( &ione, ISEED, &sizeA, h_A );
            lapackf77_zlarnv( &ione, ISEED, &N, h_x );
            
            /* =====================================================================
               Performs operation using CUDA-BLAS
               =================================================================== */
            magma_zsetmatrix( Ak, Ak, h_A, lda, d_A, ldda );
            magma_zsetvector( N, h_x, 1, d_x, 1 );
            
            cublas_time = magma_sync_wtime( NULL );
            cublasZtrmv( opts.uplo, opts.transA, opts.diag,
                         N, 
                         d_A, ldda,
                         d_x, 1 );
            cublas_time = magma_sync_wtime( NULL ) - cublas_time;
            cublas_perf = gflops / cublas_time;
            
            magma_zgetvector( N, d_x, 1, h_xcublas, 1 );
            
            /* =====================================================================
               Performs operation using CPU BLAS
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                blasf77_ztrmv( &opts.uplo, &opts.transA, &opts.diag, 
                               &N,
                               h_A, &lda,
                               h_x, &ione );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
            }
            
            /* =====================================================================
               Check the result
               =================================================================== */
            if ( opts.lapack ) {
                // compute relative error for both magma & cublas, relative to lapack,
                // |C_magma - C_lapack| / |C_lapack|
                Cnorm = lapackf77_zlange( "M", &N, &ione, h_x, &N, work );
                
                blasf77_zaxpy( &N, &c_neg_one, h_x, &ione, h_xcublas, &ione );
                cublas_error = lapackf77_zlange( "M", &N, &ione, h_xcublas, &N, work ) / Cnorm;
                
                printf("%5d   %7.2f (%7.2f)   %7.2f (%7.2f)    %8.2e\n",
                       (int) N,
                       cublas_perf, 1000.*cublas_time,
                       cpu_perf,    1000.*cpu_time,
                       cublas_error );
            }
            else {
                printf("%5d   %7.2f (%7.2f)    ---   (  ---  )    ---     ---\n",
                       (int) N,
                       cublas_perf, 1000.*cublas_time);
            }
            
            TESTING_FREE( h_A  );
            TESTING_FREE( h_x  );
            TESTING_FREE( h_xcublas );
            
            TESTING_DEVFREE( d_A );
            TESTING_DEVFREE( d_x );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    TESTING_CUDA_FINALIZE();
    return 0;
}
