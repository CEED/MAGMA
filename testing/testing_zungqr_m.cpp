/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

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
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <assert.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zungqr
*/
int main( int argc, char** argv )
{
    TESTING_INIT();

    real_Double_t    gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    double           error, work[1];
    magmaDoubleComplex  c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex *hA, *hR, *hT, *tau, *h_work;
    magmaDoubleComplex *dA, *dT;
    magma_int_t m, n, k;
    magma_int_t n2, lda, ldda, lwork, min_mn, nb, info;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    
    magma_opts opts;
    parse_opts( argc, argv, &opts );
    opts.lapack |= opts.check;  // check (-c) implies lapack (-l)
    
    printf("    m     n     k   CPU GFlop/s (sec)   GPU GFlop/s (sec)   ||R|| / ||A||\n");
    printf("=========================================================================\n");
    for( int i = 0; i < opts.ntest; ++i ){
        for( int iter = 0; iter < opts.niter; ++iter ) {
            m = opts.msize[i];
            n = opts.nsize[i];
            k = opts.ksize[i];
            if ( m < n || n < k ) {
                printf( "skipping m %d, n %d, k %d because m < n or n < k\n", (int) m, (int) n, (int) k );
                continue;
            }
            
            lda  = m;
            ldda = ((m + 31)/32)*32;
            n2 = lda*n;
            min_mn = min(m, n);
            nb = magma_get_zgeqrf_nb( m );
            lwork  = (m + 2*n+nb)*nb;
            gflops = FLOPS_ZUNGQR( m, n, k ) / 1e9;
            
            TESTING_HOSTALLOC( hA,     magmaDoubleComplex, lda*n  );
            TESTING_HOSTALLOC( h_work, magmaDoubleComplex, lwork  );
            TESTING_MALLOC(    hR,     magmaDoubleComplex, lda*n  );
            TESTING_MALLOC(    hT,     magmaDoubleComplex, min_mn*nb );
            TESTING_MALLOC(    tau,    magmaDoubleComplex, min_mn );
            TESTING_DEVALLOC(  dA,     magmaDoubleComplex, ldda*n );
            TESTING_DEVALLOC(  dT,     magmaDoubleComplex, ( 2*min_mn + ((n + 31)/32)*32 )*nb );
            
            lapackf77_zlarnv( &ione, ISEED, &n2, hA );
            lapackf77_zlacpy( MagmaUpperLowerStr, &m, &n, hA, &lda, hR, &lda );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            // first, get QR factors
            magma_zsetmatrix( m, n, hA, lda, dA, ldda );
            magma_zgeqrf_gpu( m, n, dA, ldda, tau, dT, &info );
            if ( info != 0 )
                printf("magma_zgeqrf_gpu returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            magma_zgetmatrix( m, n, dA, ldda, hR, lda );
            magma_zgetmatrix( nb, min_mn, dT, nb, hT, nb );
            
            gpu_time = magma_wtime();
            magma_zungqr_m( m, n, k, hR, lda, tau, hT, nb, &info );
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if ( info != 0 )
                printf("magma_zungqr_gpu returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                error = lapackf77_zlange("f", &m, &n, hA, &lda, work );
                
                lapackf77_zgeqrf( &m, &n, hA, &lda, tau, h_work, &lwork, &info );
                if ( info != 0 )
                    printf("lapackf77_zgeqrf returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                
                cpu_time = magma_wtime();
                lapackf77_zungqr( &m, &n, &k, hA, &lda, tau, h_work, &lwork, &info );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if ( info != 0 )
                    printf("lapackf77_zungqr returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                
                // compute relative error |R|/|A| := |Q_magma - Q_lapack|/|A|
                blasf77_zaxpy( &n2, &c_neg_one, hA, &ione, hR, &ione );
                error = lapackf77_zlange("f", &m, &n, hR, &lda, work) / error;
                
                printf("%5d %5d %5d   %7.1f (%7.2f)   %7.1f (%7.2f)   %8.2e\n",
                       (int) m, (int) n, (int) k,
                       cpu_perf, cpu_time, gpu_perf, gpu_time, error );
            }
            else {
                printf("%5d %5d %5d     ---   (  ---  )   %7.1f (%7.2f)     ---  \n",
                       (int) m, (int) n, (int) k,
                       gpu_perf, gpu_time );
            }
            
            TESTING_HOSTFREE( hA     );
            TESTING_HOSTFREE( h_work );
            TESTING_FREE(     hR     );
            TESTING_FREE(     hT     );
            TESTING_FREE(     tau    );
            TESTING_DEVFREE(  dA     );
            TESTING_DEVFREE(  dT     );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }
    
    TESTING_FINALIZE();
    return 0;
}
