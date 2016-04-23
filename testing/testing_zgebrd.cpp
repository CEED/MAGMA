/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

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
   -- Testing zgebrd
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t    gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    magmaDoubleComplex *h_A, *h_Q, *h_PT, *h_work;
    magmaDoubleComplex *taup, *tauq;
    double      *diag, *offdiag;
    double      eps, result[3] = {0., 0., 0.};
    magma_int_t M, N, n2, lda, lhwork, info, minmn, nb;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};

    eps = lapackf77_dlamch( "E" );
    magma_int_t status = 0;

    magma_opts opts;
    opts.parse_opts( argc, argv );

    double tol = opts.tolerance * lapackf77_dlamch("E");
    
    printf("%%   M     N   CPU Gflop/s (sec)   GPU Gflop/s (sec)   |A-QBP^H|/N|A|   |I-QQ^H|/N   |I-PP^H|/N\n");
    printf("%%=============================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            minmn  = min(M, N);
            nb     = magma_get_zgebrd_nb( M, N );
            lda    = M;
            n2     = lda*N;
            lhwork = (M + N)*nb;
            gflops = FLOPS_ZGEBRD( M, N ) / 1e9;

            TESTING_CHECK( magma_zmalloc_cpu( &h_A,     lda*N ));
            TESTING_CHECK( magma_zmalloc_cpu( &tauq,    minmn ));
            TESTING_CHECK( magma_zmalloc_cpu( &taup,    minmn ));
            TESTING_CHECK( magma_dmalloc_cpu( &diag,    minmn   ));
            TESTING_CHECK( magma_dmalloc_cpu( &offdiag, minmn-1 ));
            
            TESTING_CHECK( magma_zmalloc_pinned( &h_Q,     lda*N  ));
            TESTING_CHECK( magma_zmalloc_pinned( &h_work,  lhwork ));
            
            /* Initialize the matrices */
            lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
            lapackf77_zlacpy( MagmaFullStr, &M, &N, h_A, &lda, h_Q, &lda );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            magma_zgebrd( M, N, h_Q, lda,
                              diag, offdiag, tauq, taup,
                          h_work, lhwork, &info );
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0) {
                printf("magma_zgebrd returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            }
            
            /* =====================================================================
               Check the factorization
               =================================================================== */
            if ( opts.check ) {
                // zungbr prefers minmn*NB
                // zbdt01 needs M+N
                // zunt01 prefers minmn*(minmn+1) to check Q and P
                magma_int_t lwork_err;
                magmaDoubleComplex *h_work_err;
                lwork_err = max( minmn * nb, M+N );
                lwork_err = max( lwork_err, minmn*(minmn+1) );
                TESTING_CHECK( magma_zmalloc_cpu( &h_PT,       lda*N     ));
                TESTING_CHECK( magma_zmalloc_cpu( &h_work_err, lwork_err ));
                
                // zbdt01 needs M
                // zunt01 needs minmn
                #ifdef COMPLEX
                double *rwork_err;
                TESTING_CHECK( magma_dmalloc_cpu( &rwork_err, M ));
                #endif

                lapackf77_zlacpy( MagmaFullStr, &M, &N, h_Q, &lda, h_PT, &lda );
                
                // generate Q & P^H
                lapackf77_zungbr("Q", &M, &minmn, &N, h_Q,  &lda, tauq, h_work_err, &lwork_err, &info);
                if (info != 0) {
                    printf("lapackf77_zungbr #1 returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                }
                lapackf77_zungbr("P", &minmn, &N, &M, h_PT, &lda, taup, h_work_err, &lwork_err, &info);
                if (info != 0) {
                    printf("lapackf77_zungbr #2 returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                }
                
                // Test 1:  Check the decomposition A := Q * B * PT
                //      2:  Check the orthogonality of Q
                //      3:  Check the orthogonality of PT
                lapackf77_zbdt01( &M, &N, &ione,
                                  h_A, &lda, h_Q, &lda,
                                  diag, offdiag, h_PT, &lda,
                                  h_work_err,
                                  #ifdef COMPLEX
                                  rwork_err,
                                  #endif
                                  &result[0] );
                // LAPACK normalizes by N*|A|, but that fails for very tall matrices,
                // so normalize by max(M*N)*|A|. TODO: is there justification for that change?
                result[0] = N*result[0] / max(M,N);
                
                lapackf77_zunt01( "Columns", &M, &minmn, h_Q,  &lda, h_work_err, &lwork_err,
                                  #ifdef COMPLEX
                                  rwork_err,
                                  #endif
                                  &result[1]);
                
                lapackf77_zunt01( "Rows",    &minmn, &N, h_PT, &lda, h_work_err, &lwork_err,
                                  #ifdef COMPLEX
                                  rwork_err,
                                  #endif
                                  &result[2]);
                
                magma_free_cpu( h_PT );
                magma_free_cpu( h_work_err );
                #ifdef COMPLEX
                magma_free_cpu( rwork_err );
                #endif
            }
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                lapackf77_zgebrd( &M, &N, h_A, &lda,
                                  diag, offdiag, tauq, taup,
                                  h_work, &lhwork, &info );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0) {
                    printf("lapackf77_zgebrd returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                }
            }
            
            /* =====================================================================
               Print performance and error.
               =================================================================== */
            if ( opts.lapack ) {
                printf("%5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)",
                       (int) M, (int) N, cpu_perf, cpu_time, gpu_perf, gpu_time );
            }
            else {
                printf("%5d %5d     ---   (  ---  )   %7.2f (%7.2f)",
                       (int) M, (int) N, gpu_perf, gpu_time );
            }
            if ( opts.check ) {
                bool okay = (result[0]*eps < tol) && (result[1]*eps < tol) && (result[2]*eps < tol);
                status += ! okay;
                printf("   %8.2e         %8.2e     %8.2e   %s\n",
                       result[0]*eps, result[1]*eps, result[2]*eps,
                       (okay ? "ok" : "failed") );
            } else {
                printf("     ---            --- \n");
            }
            
            magma_free_cpu( h_A     );
            magma_free_cpu( tauq    );
            magma_free_cpu( taup    );
            magma_free_cpu( diag    );
            magma_free_cpu( offdiag );
            
            magma_free_pinned( h_Q    );
            magma_free_pinned( h_work );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
