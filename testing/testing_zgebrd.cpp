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

#define PRECISION_z

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgebrd
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t    gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    magmaDoubleComplex *h_A, *h_Q, *h_PT, *h_work, *chkwork;
    magmaDoubleComplex *taup, *tauq;
    #if defined(PRECISION_z) || defined(PRECISION_c)
    double      *rwork;
    #endif
    double      *diag, *offdiag;
    double      eps, result[3] = {0., 0., 0.};
    magma_int_t M, N, n2, lda, lhwork, lchkwork, info, minmn, nb;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};

    eps = lapackf77_dlamch( "E" );

    magma_opts opts;
    parse_opts( argc, argv, &opts );
    double tol = opts.tolerance * lapackf77_dlamch("E");
    
    printf("    M     N   CPU GFlop/s (sec)   GPU GFlop/s (sec)   |A-QBP'|/N|A|  |I-QQ'|/N  |I-PP'|/N\n");
    printf("=========================================================================================\n");
    for( int i = 0; i < opts.ntest; ++i ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[i];
            N = opts.nsize[i];
            minmn  = min(M, N);
            nb     = magma_get_zgebrd_nb(N);
            lda    = M;
            n2     = lda*N;
            lhwork = (M + N)*nb;
            gflops = FLOPS_ZGEBRD( M, N ) / 1e9;

            TESTING_MALLOC_CPU( h_A,     magmaDoubleComplex, lda*N );
            TESTING_MALLOC_CPU( tauq,    magmaDoubleComplex, minmn );
            TESTING_MALLOC_CPU( taup,    magmaDoubleComplex, minmn );
            TESTING_MALLOC_CPU( diag,    double, minmn   );
            TESTING_MALLOC_CPU( offdiag, double, minmn-1 );
            
            TESTING_MALLOC_PIN( h_Q,     magmaDoubleComplex, lda*N  );
            TESTING_MALLOC_PIN( h_work,  magmaDoubleComplex, lhwork );
            
            /* Initialize the matrices */
            lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
            lapackf77_zlacpy( MagmaUpperLowerStr, &M, &N, h_A, &lda, h_Q, &lda );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            magma_zgebrd( M, N, h_Q, lda,
                          diag, offdiag, tauq, taup,
                          h_work, lhwork, &info);
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0)
                printf("magma_zgebrd returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            /* =====================================================================
               Check the factorization
               =================================================================== */
            if ( opts.check ) {
                lchkwork = max( minmn * nb, M+N );
                /* For optimal performance in zunt01 */
                lchkwork = max( lchkwork, minmn*minmn );
                TESTING_MALLOC_CPU( h_PT,    magmaDoubleComplex, lda*N   );
                TESTING_MALLOC_CPU( chkwork, magmaDoubleComplex, lchkwork );
                #if defined(PRECISION_z) || defined(PRECISION_c)
                TESTING_MALLOC_CPU( rwork, double, 5*minmn );
                #endif

                lapackf77_zlacpy(MagmaUpperLowerStr, &M, &N, h_Q, &lda, h_PT, &lda);
                
                // generate Q & P'
                lapackf77_zungbr("Q", &M, &minmn, &N, h_Q,  &lda, tauq, chkwork, &lchkwork, &info);
                if (info != 0)
                    printf("lapackf77_zungbr #1 returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                lapackf77_zungbr("P", &minmn, &N, &M, h_PT, &lda, taup, chkwork, &lchkwork, &info);
                if (info != 0)
                    printf("lapackf77_zungbr #2 returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                
                // Test 1:  Check the decomposition A := Q * B * PT
                //      2:  Check the orthogonality of Q
                //      3:  Check the orthogonality of PT
                #if defined(PRECISION_z) || defined(PRECISION_c)
                lapackf77_zbdt01(&M, &N, &ione,
                                 h_A, &lda, h_Q, &lda,
                                 diag, offdiag, h_PT, &lda,
                                 chkwork, rwork, &result[0]);
                lapackf77_zunt01("Columns", &M, &minmn, h_Q,  &lda, chkwork, &lchkwork, rwork, &result[1]);
                lapackf77_zunt01("Rows",    &minmn, &N, h_PT, &lda, chkwork, &lchkwork, rwork, &result[2]);
                #else
                lapackf77_zbdt01(&M, &N, &ione,
                                 h_A, &lda, h_Q, &lda,
                                 diag, offdiag, h_PT, &lda,
                                 chkwork, &result[0]);
                lapackf77_zunt01("Columns", &M, &minmn, h_Q,  &lda, chkwork, &lchkwork, &result[1]);
                lapackf77_zunt01("Rows",    &minmn, &N, h_PT, &lda, chkwork, &lchkwork, &result[2]);
                #endif
                
                TESTING_FREE_CPU( h_PT );
                TESTING_FREE_CPU( chkwork );
                #if defined(PRECISION_z) || defined(PRECISION_c)
                TESTING_FREE_CPU( rwork );
                #endif
            }
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                lapackf77_zgebrd(&M, &N, h_A, &lda,
                                 diag, offdiag, tauq, taup,
                                 h_work, &lhwork, &info);
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0)
                    printf("lapackf77_zgebrd returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
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
                printf("   %8.2e       %8.2e   %8.2e%s\n",
                       result[0]*eps, result[1]*eps, result[2]*eps,
                       ( ( (result[0]*eps < tol) && (result[1]*eps < tol) ) ? "" : "  failed") );
            } else {
                printf("     ---            --- \n");
            }
            
            TESTING_FREE_CPU( h_A     );
            TESTING_FREE_CPU( tauq    );
            TESTING_FREE_CPU( taup    );
            TESTING_FREE_CPU( diag    );
            TESTING_FREE_CPU( offdiag );
            
            TESTING_FREE_PIN( h_Q    );
            TESTING_FREE_PIN( h_work );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    TESTING_FINALIZE();
    return 0;
}
