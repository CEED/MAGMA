/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Stan Tomov
       @author Mark Gates

       @precisions normal z -> s d c

*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "flops.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"

#define COMPLEX

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zhetrd
*/

int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t    gflops, gpu_perf, cpu_perf, gpu_time, cpu_time;
    magmaDoubleComplex *h_A, *h_R, *h_Q, *h_work, *work;
    magmaDoubleComplex *tau;
    double *diag, *offdiag;
    #ifdef COMPLEX
    double *rwork = NULL;
    #endif
    double result[2] = {0., 0.};
    magma_int_t N, n2, lda, lwork, info, nb;
    magma_int_t ione     = 1;
    magma_int_t itwo     = 2;
    magma_int_t ithree   = 3;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t status = 0;
    magma_int_t k = 1;  // TODO: UNKNOWN, UNDOCUMENTED VARIABLE (number of streams?)

    magma_opts opts;
    opts.parse_opts( argc, argv );
    opts.ngpu = abs( opts.ngpu );  // always uses multi-GPU code
    
    double tol = opts.tolerance * lapackf77_dlamch("E");
    double eps = lapackf77_dlamch( "E" );

    /* To avoid uninitialized variable warning */
    h_Q   = NULL;
    work  = NULL;

    printf("%% uplo = %s, ngpu %d\n", lapack_uplo_const(opts.uplo), (int) opts.ngpu );
    printf("%% N     CPU Gflop/s (sec)   GPU Gflop/s (sec)   |A-QHQ^H|/N|A|   |I-QQ^H|/N\n");
    printf("%%==========================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N      = opts.nsize[itest];
            lda    = N;
            n2     = lda*N;
            nb     = magma_get_zhetrd_nb(N);
            /* We suppose the magma nb is bigger than lapack nb */
            lwork  = N*nb;
            gflops = FLOPS_ZHETRD( N ) / 1e9;
            
            /* Allocate host memory for the matrix */
            TESTING_MALLOC_PIN( h_R,     magmaDoubleComplex, lda*N );
            TESTING_MALLOC_PIN( h_work,  magmaDoubleComplex, lwork );
            
            TESTING_MALLOC_CPU( h_A,     magmaDoubleComplex, lda*N );
            TESTING_MALLOC_CPU( tau,     magmaDoubleComplex, N     );
            TESTING_MALLOC_CPU( diag,    double, N   );
            TESTING_MALLOC_CPU( offdiag, double, N-1 );
            
            /* ====================================================================
               Initialize the matrix
               =================================================================== */
            lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
            magma_zmake_hermitian( N, h_A, lda );
            
            lapackf77_zlacpy( MagmaFullStr, &N, &N, h_A, &lda, h_R, &lda );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            if ( opts.ngpu == 0 ) {
                magma_zhetrd(opts.uplo, N, h_R, lda, diag, offdiag,
                             tau, h_work, lwork, &info);
            } else {
                magma_zhetrd_mgpu(opts.ngpu, k, opts.uplo, N, h_R, lda, diag, offdiag,
                                  tau, h_work, lwork, &info);
            }
            gpu_time = magma_wtime() - gpu_time;
            if (info != 0) {
                printf("magma_zhetrd_mgpu returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            }
            
            gpu_perf = gflops / gpu_time;
            
            /* =====================================================================
               Check the factorization
               =================================================================== */
            if ( opts.check ) {
                TESTING_MALLOC_CPU( h_Q,  magmaDoubleComplex, lda*N );
                TESTING_MALLOC_CPU( work, magmaDoubleComplex, 2*N*N );
                #ifdef COMPLEX
                TESTING_MALLOC_CPU( rwork, double, N );
                #endif
                
                lapackf77_zlacpy( lapack_uplo_const(opts.uplo), &N, &N, h_R, &lda, h_Q, &lda);
                lapackf77_zungtr( lapack_uplo_const(opts.uplo), &N, h_Q, &lda, tau, h_work, &lwork, &info);

                lapackf77_zhet21( &itwo, lapack_uplo_const(opts.uplo), &N, &ione,
                                  h_A, &lda, diag, offdiag,
                                  h_Q, &lda, h_R, &lda,
                                  tau, work,
                                  #ifdef COMPLEX
                                  rwork,
                                  #endif
                                  &result[0] );
                
                lapackf77_zhet21( &ithree, lapack_uplo_const(opts.uplo), &N, &ione,
                                  h_A, &lda, diag, offdiag,
                                  h_Q, &lda, h_R, &lda,
                                  tau, work,
                                  #ifdef COMPLEX
                                  rwork,
                                  #endif
                                  &result[1] );
                result[0] *= eps;
                result[1] *= eps;

                TESTING_FREE_CPU( h_Q );
                TESTING_FREE_CPU( work );
                #ifdef COMPLEX
                TESTING_FREE_CPU( rwork );
                #endif
            }
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                lapackf77_zhetrd(lapack_uplo_const(opts.uplo), &N, h_A, &lda, diag, offdiag, tau,
                                 h_work, &lwork, &info);
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0) {
                    printf("lapackf77_zhetrd returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                }
            }
            
            /* =====================================================================
               Print performance and error.
               =================================================================== */
            if ( opts.lapack ) {
                printf("%5d   %7.2f (%7.2f)", (int) N, cpu_perf, cpu_time );
            }
            else {
                printf("%5d     ---   (  ---  )", (int) N );
            }
            printf("   %7.2f (%7.2f)", gpu_perf, gpu_time );
            if ( opts.check ) {
                printf("     %8.2e        %8.2e   %s\n",
                       result[0], result[1], (result[0] < tol && result[1] < tol ? "ok" : "failed") );
                status += ! (result[0] < tol && result[1] < tol);
            }
            else {
                printf("       ---  \n");
            }

            TESTING_FREE_PIN( h_R );
            TESTING_FREE_PIN( h_work );

            TESTING_FREE_CPU( h_A );
            TESTING_FREE_CPU( tau );
            TESTING_FREE_CPU( diag );
            TESTING_FREE_CPU( offdiag );
            
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    opts.cleanup();
    TESTING_FINALIZE();
    return status;
}
