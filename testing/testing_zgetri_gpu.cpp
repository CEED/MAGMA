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
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgetrf
*/
int main( int argc, char** argv )
{
    TESTING_CUDA_INIT();

    real_Double_t   gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    cuDoubleComplex *h_A, *h_R, *work;
    cuDoubleComplex *d_A, *dwork;
    cuDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magma_int_t N, n2, lda, ldda, info, lwork, ldwork;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    cuDoubleComplex tmp;
    double error, rwork[1];
    magma_int_t *ipiv;
    
    magma_opts opts;
    parse_opts( argc, argv, &opts );
    
    printf("    N   CPU GFlop/s (sec)   GPU GFlop/s (sec)   ||R||_F / ||A||_F\n");
    printf("=================================================================\n");
    for( int i = 0; i < opts.ntest; ++i ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[i];
            lda    = N;
            n2     = lda*N;
            ldda   = ((N+31)/32)*32;
            ldwork = N * magma_get_zgetri_nb( N );
            gflops = FLOPS_ZGETRI( N ) / 1e9;
            
            /* query for lapack workspace size */
            lwork = -1;
            lapackf77_zgetri( &N, h_A, &lda, ipiv, &tmp, &lwork, &info );
            if (info != 0)
                printf("lapackf77_zgetri returned error %d.\n", (int) info);
            lwork = int( MAGMA_Z_REAL( tmp ));
            
            TESTING_MALLOC(    ipiv,  magma_int_t,     N      );
            TESTING_MALLOC(    work,  cuDoubleComplex, lwork  );
            TESTING_MALLOC(    h_A,   cuDoubleComplex, n2     );
            TESTING_HOSTALLOC( h_R,   cuDoubleComplex, n2     );
            TESTING_DEVALLOC(  d_A,   cuDoubleComplex, ldda*N );
            TESTING_DEVALLOC(  dwork, cuDoubleComplex, ldwork );
            
            /* Initialize the matrix */
            lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
            error = lapackf77_zlange( "f", &N, &N, h_A, &lda, rwork );  // norm(A)
            
            /* Factor the matrix. Both MAGMA and LAPACK will use this factor. */
            magma_zsetmatrix( N, N, h_A, lda, d_A, ldda );
            magma_zgetrf_gpu( N, N, d_A, ldda, ipiv, &info );
            magma_zgetmatrix( N, N, d_A, ldda, h_A, lda );
            
            // check for exact singularity
            //h_A[ 10 + 10*lda ] = MAGMA_Z_MAKE( 0.0, 0.0 );
            //magma_zsetmatrix( N, N, h_A, lda, d_A, ldda );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            magma_zgetri_gpu( N,    d_A, ldda, ipiv, dwork, ldwork, &info );
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0)
                printf("magma_zgetri_gpu returned error %d.\n", (int) info);
            
            magma_zgetmatrix( N, N, d_A, ldda, h_R, lda );
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.check ) {
                cpu_time = magma_wtime();
                lapackf77_zgetri( &N,     h_A, &lda, ipiv, work, &lwork, &info );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0)
                    printf("lapackf77_zgetri returned error %d.\n", (int) info);
                
                /* =====================================================================
                   Check the result compared to LAPACK
                   =================================================================== */
                blasf77_zaxpy( &n2, &c_neg_one, h_A, &ione, h_R, &ione );
                error = lapackf77_zlange( "f", &N, &N, h_R, &lda, rwork ) / error;
                
                printf( "%5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e\n",
                        (int) N, cpu_perf, cpu_time, gpu_perf, gpu_time, error );
            }
            else {
                printf( "%5d     ---   (  ---  )   %7.2f (%7.2f)     ---\n",
                        (int) N, gpu_perf, gpu_time );
            }
            
            TESTING_FREE(     ipiv  );
            TESTING_FREE(     work  );
            TESTING_FREE(     h_A   );
            TESTING_HOSTFREE( h_R   );
            TESTING_DEVFREE(  d_A   );
            TESTING_DEVFREE(  dwork );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    TESTING_CUDA_FINALIZE();
    return 0;
}
