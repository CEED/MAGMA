/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

       @author Stan Tomov
       @author Mathieu Faverge
       @author Mark Gates
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zunglq
*/
int main( int argc, char** argv )
{
    TESTING_INIT();

    real_Double_t    gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    double           Anorm, error, work[1];
    magmaDoubleComplex  c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex *hA, *hR, *tau, *h_work;
    magmaDoubleComplex_ptr dA, dT;
    magma_int_t m, n, k;
    magma_int_t n2, lda, ldda, lwork, min_mn, nb, info;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t status = 0;
    
    magma_opts opts;
    opts.parse_opts( argc, argv );

    double tol = opts.tolerance * lapackf77_dlamch("E");
    opts.lapack |= opts.check;  // check (-c) implies lapack (-l)
    
    printf("%%   m     n     k   CPU GFlop/s (sec)   GPU GFlop/s (sec)   ||R|| / ||A||\n");
    printf("%%========================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            m = opts.msize[itest];
            n = opts.nsize[itest];
            k = opts.ksize[itest];
            
            if ( n < m || m < k ) {
                printf( "%5d %5d %5d   skipping because n < m or m < k\n", (int) m, (int) n, (int) k );
                continue;
            }
            
            lda  = m;
            ldda = magma_roundup( m, opts.align );  // multiple of 32 by default
            n2 = lda*n;
            min_mn = min(m, n);
            nb = magma_get_zgelqf_nb( m );
            lwork  = max( m*nb + nb*nb, nb*nb );
            gflops = FLOPS_ZUNGLQ( m, n, k ) / 1e9;
            
            TESTING_MALLOC_PIN( h_work, magmaDoubleComplex, lwork  );
            TESTING_MALLOC_PIN( hR,     magmaDoubleComplex, lda*n  );
            
            TESTING_MALLOC_CPU( hA,     magmaDoubleComplex, lda*n  );
            TESTING_MALLOC_CPU( tau,    magmaDoubleComplex, min_mn );
            
            TESTING_MALLOC_DEV( dA,     magmaDoubleComplex, ldda*n );
            TESTING_MALLOC_DEV( dT,     magmaDoubleComplex, ( 2*min_mn + magma_roundup( n, 32 ) )*nb );
            
            lapackf77_zlarnv( &ione, ISEED, &n2, hA );
            lapackf77_zlacpy( MagmaFullStr, &m, &n, hA, &lda, hR, &lda );
            
            Anorm = lapackf77_zlange("f", &m, &n, hA, &lda, work );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            // first, get LQ factors in both hA and hR
            magma_zsetmatrix( m, n, hA, lda, dA, ldda );
            magma_zgelqf_gpu( m, n, dA, ldda, tau, h_work, lwork, &info );
            if (info != 0)
                printf("magma_zgelqf_gpu returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            magma_zgetmatrix( m, n, dA, ldda, hA, lda );
            lapackf77_zlacpy( MagmaFullStr, &m, &n, hA, &lda, hR, &lda );
            
            gpu_time = magma_wtime();
            magma_zunglq( m, n, k, hR, lda, tau, h_work, lwork, &info );
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0)
                printf("magma_zunglq returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                lapackf77_zunglq( &m, &n, &k, hA, &lda, tau, h_work, &lwork, &info );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0)
                    printf("lapackf77_zunglq returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                
                if ( opts.verbose ) {
                    printf( "R=" );  magma_zprint( m, n, hR, lda );
                    printf( "A=" );  magma_zprint( m, n, hA, lda );
                }
                
                // compute relative error |R|/|A| := |Q_magma - Q_lapack|/|A|
                blasf77_zaxpy( &n2, &c_neg_one, hA, &ione, hR, &ione );
                error = lapackf77_zlange("f", &m, &n, hR, &lda, work) / Anorm;
                
                if ( opts.verbose ) {
                    printf( "diff=" );  magma_zprint( m, n, hR, lda );
                }
                
                bool okay = (error < tol);
                status += ! okay;
                printf("%5d %5d %5d   %7.1f (%7.2f)   %7.1f (%7.2f)   %8.2e   %s\n",
                       (int) m, (int) n, (int) k,
                       cpu_perf, cpu_time, gpu_perf, gpu_time,
                       error, (okay ? "ok" : "failed"));
            }
            else {
                printf("%5d %5d %5d     ---   (  ---  )   %7.1f (%7.2f)     ---  \n",
                       (int) m, (int) n, (int) k,
                       gpu_perf, gpu_time );
            }
            
            TESTING_FREE_PIN( h_work );
            TESTING_FREE_PIN( hR     );
            
            TESTING_FREE_CPU( hA  );
            TESTING_FREE_CPU( tau );
            
            TESTING_FREE_DEV( dA );
            TESTING_FREE_DEV( dT );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }
    
    TESTING_FINALIZE();
    return status;
}
