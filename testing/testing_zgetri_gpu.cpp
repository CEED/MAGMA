/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
       @author Mark Gates
*/
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgetri
*/
int main( int argc, char** argv )
{
    TESTING_INIT();

    // constants
    const magmaDoubleComplex c_zero    = MAGMA_Z_ZERO;
    const magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    const magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    
    real_Double_t   gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    magmaDoubleComplex *h_A, *h_Ainv, *h_R, *work;
    magmaDoubleComplex_ptr d_A, dwork;
    magma_int_t N, n2, lda, ldda, info, lwork, ldwork;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magmaDoubleComplex tmp;
    double error, rwork[1];
    magma_int_t *ipiv;
    magma_int_t status = 0;
    
    magma_opts opts;
    opts.parse_opts( argc, argv );
    
    double tol = opts.tolerance * lapackf77_dlamch("E");
    
    printf("%%   N   CPU GFlop/s (sec)   GPU GFlop/s (sec)   ||I - A*A^{-1}||_1 / (N*cond(A))\n");
    printf("%%===============================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            lda    = N;
            n2     = lda*N;
            ldda   = magma_roundup( N, opts.align );  // multiple of 32 by default
            ldwork = N * magma_get_zgetri_nb( N );
            gflops = FLOPS_ZGETRI( N ) / 1e9;
            
            // query for workspace size
            lwork = -1;
            lapackf77_zgetri( &N, NULL, &lda, NULL, &tmp, &lwork, &info );
            if (info != 0)
                printf("lapackf77_zgetri returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            lwork = magma_int_t( MAGMA_Z_REAL( tmp ));
            
            TESTING_MALLOC_CPU( ipiv,   magma_int_t,        N      );
            TESTING_MALLOC_CPU( work,   magmaDoubleComplex, lwork  );
            TESTING_MALLOC_CPU( h_A,    magmaDoubleComplex, n2     );
            TESTING_MALLOC_CPU( h_Ainv, magmaDoubleComplex, n2     );
            TESTING_MALLOC_CPU( h_R,    magmaDoubleComplex, n2     );
            
            TESTING_MALLOC_DEV( d_A,    magmaDoubleComplex, ldda*N );
            TESTING_MALLOC_DEV( dwork,  magmaDoubleComplex, ldwork );
            
            /* Initialize the matrix */
            lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
            
            /* Factor the matrix. Both MAGMA and LAPACK will use this factor. */
            magma_zsetmatrix( N, N, h_A, lda, d_A, ldda );
            magma_zgetrf_gpu( N, N, d_A, ldda, ipiv, &info );
            magma_zgetmatrix( N, N, d_A, ldda, h_Ainv, lda );
            if ( info != 0 )
                printf("magma_zgetrf_gpu returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            // check for exact singularity
            //h_Ainv[ 10 + 10*lda ] = MAGMA_Z_MAKE( 0.0, 0.0 );
            //magma_zsetmatrix( N, N, h_Ainv, lda, d_A, ldda );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            magma_zgetri_gpu( N, d_A, ldda, ipiv, dwork, ldwork, &info );
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0)
                printf("magma_zgetri_gpu returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                lapackf77_zgetri( &N, h_Ainv, &lda, ipiv, work, &lwork, &info );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0)
                    printf("lapackf77_zgetri returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                printf( "%5d   %7.2f (%7.2f)   %7.2f (%7.2f)",
                        (int) N, cpu_perf, cpu_time, gpu_perf, gpu_time );
            }
            else {
                printf( "%5d     ---   (  ---  )   %7.2f (%7.2f)",
                        (int) N, gpu_perf, gpu_time );
            }
            
            /* =====================================================================
               Check the result
               =================================================================== */
            if ( opts.check ) {
                magma_zgetmatrix( N, N, d_A, ldda, h_Ainv, lda );
                
                // compute 1-norm condition number estimate, following LAPACK's zget03
                double normA, normAinv, rcond;
                normA    = lapackf77_zlange( "1", &N, &N, h_A,    &lda, rwork );
                normAinv = lapackf77_zlange( "1", &N, &N, h_Ainv, &lda, rwork );
                if ( normA <= 0 || normAinv <= 0 ) {
                    rcond = 0;
                    error = 1 / (tol/opts.tolerance);  // == 1/eps
                }
                else {
                    rcond = (1 / normA) / normAinv;
                    // R = I
                    // R -= A*A^{-1}
                    // err = ||I - A*A^{-1}|| / ( N ||A||*||A^{-1}|| ) = ||R|| * rcond / N, using 1-norm
                    lapackf77_zlaset( "full", &N, &N, &c_zero, &c_one, h_R, &lda );
                    blasf77_zgemm( "no", "no", &N, &N, &N, &c_neg_one,
                                   h_A,    &lda,
                                   h_Ainv, &lda, &c_one,
                                   h_R,    &lda );
                    error = lapackf77_zlange( "1", &N, &N, h_R, &lda, rwork );
                    error = error * rcond / N;
                }
                
                bool okay = (error < tol);
                status += ! okay;
                printf( "   %8.2e   %s\n",
                        error, (okay ? "ok" : "failed"));
            }
            else {
                printf( "\n" );
            }
            
            TESTING_FREE_CPU( ipiv   );
            TESTING_FREE_CPU( work   );
            TESTING_FREE_CPU( h_A    );
            TESTING_FREE_CPU( h_Ainv );
            TESTING_FREE_CPU( h_R    );
            
            TESTING_FREE_DEV( d_A    );
            TESTING_FREE_DEV( dwork  );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    TESTING_FINALIZE();
    return status;
}
