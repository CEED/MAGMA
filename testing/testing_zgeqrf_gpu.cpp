/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> c d s
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
   -- Testing zgeqrf
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t    gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    double           error, work[1];
    magmaDoubleComplex  c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex *h_A, *d_A, *h_R, *tau, *h_work, tmp[1];
    magma_int_t M, N, n2, lda, ldda, lwork, info, min_mn, nb, size;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1}, ISEED2[4];
    
    magma_opts opts;
    parse_opts( argc, argv, &opts );
 
    magma_int_t status = 0;
    double tol;
    opts.lapack |= (opts.check == 2);  // check (-c2) implies lapack (-l)

    if ( opts.version != 2 && opts.check != 0 ) {
        printf( "  ===================================================================\n"
                "  NOTE: residual checks for this version will be wrong!\n"
                "  Because tester ignores the special structure of MAGMA zgeqrf resuls.\n"
                "  Use --version 2 to see correct residuals.\n" 
                "  ===================================================================\n\n");
        opts.check = 0;
    }
    if ( opts.check == 1 ) {
        printf("  M     N     CPU GFlop/s (sec)   GPU GFlop/s (sec)   ||R-Q'A||_1 / (M*||A||_1*eps) ||I-Q'Q||_1 / (M*eps)\n");
        printf("=========================================================================================================\n");
        tol = 1.0;
    } else {
        printf("  M     N     CPU GFlop/s (sec)   GPU GFlop/s (sec)   ||R||_F / ||A||_F\n");
        printf("=======================================================================\n");
        tol = opts.tolerance * lapackf77_dlamch("E");
    }
    for( int i = 0; i < opts.ntest; ++i ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[i];
            N = opts.nsize[i];
            min_mn = min(M, N);
            lda    = M;
            n2     = lda*N;
            ldda   = ((M+31)/32)*32;
            gflops = FLOPS_ZGEQRF( M, N ) / 1e9;
            
            lwork = -1;
            lapackf77_zgeqrf(&M, &N, h_A, &M, tau, tmp, &lwork, &info);
            lwork = (magma_int_t)MAGMA_Z_REAL( tmp[0] );
            
            TESTING_MALLOC(    tau, magmaDoubleComplex, min_mn );
            TESTING_MALLOC(    h_A, magmaDoubleComplex, n2     );
            TESTING_HOSTALLOC( h_R, magmaDoubleComplex, n2     );
            TESTING_DEVALLOC(  d_A, magmaDoubleComplex, ldda*N );
            TESTING_MALLOC( h_work, magmaDoubleComplex, lwork  );
            
            /* Initialize the matrix */
            for ( int j=0; j<4; j++ ) ISEED2[j] = ISEED[j]; // saving seeds
            lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
            lapackf77_zlacpy( MagmaUpperLowerStr, &M, &N, h_A, &lda, h_R, &lda );
            magma_zsetmatrix( M, N, h_R, lda, d_A, ldda );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            if ( opts.version == 2 ) {
                magma_zgeqrf2_gpu( M, N, d_A, ldda, tau, &info);
            }
            else {
                magmaDoubleComplex *dT;
                nb = magma_get_zgeqrf_nb( M );
                size = (2*min(M, N) + (N+31)/32*32 )*nb;
                magma_zmalloc( &dT, size );
                if ( opts.version == 3 ) {
                    magma_zgeqrf3_gpu( M, N, d_A, ldda, tau, dT, &info);
                }
                else {
                    magma_zgeqrf_gpu( M, N, d_A, ldda, tau, dT, &info);
                }
                magma_free( dT );
            }
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0)
                printf("magma_zgeqrf returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            if ( opts.lapack ) {
                /* =====================================================================
                   Performs operation using LAPACK
                   =================================================================== */
                magmaDoubleComplex *tau;
                TESTING_MALLOC( tau, magmaDoubleComplex, min_mn );
                cpu_time = magma_wtime();
                lapackf77_zgeqrf(&M, &N, h_A, &lda, tau, h_work, &lwork, &info);
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0)
                    printf("lapackf77_zgeqrf returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                TESTING_FREE( tau );
            }

            if ( opts.check == 1 ) {
                /* =====================================================================
                   Check the result 
                   =================================================================== */
                magma_int_t i, lwork = n2+N;
                magmaDoubleComplex *h_W1, *h_W2, *h_W3;
                double *h_RW, results[2];
                magma_zgetmatrix( M, N, d_A, ldda, h_R, M );

                TESTING_MALLOC( h_W1, magmaDoubleComplex, n2 ); // Q
                TESTING_MALLOC( h_W2, magmaDoubleComplex, n2 ); // R
                TESTING_MALLOC( h_W3, magmaDoubleComplex, lwork ); // WORK
                TESTING_MALLOC( h_RW, double, M );  // RWORK
                lapackf77_zlarnv( &ione, ISEED2, &n2, h_A );
                lapackf77_zqrt02( &M, &N, &min_mn, h_A, h_R, h_W1, h_W2, &lda, tau, h_W3, &lwork,
                                  h_RW, results );

                if ( opts.lapack ) {
                    printf("%5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e                      %8.2e",
                           (int) M, (int) N, cpu_perf, cpu_time, gpu_perf, gpu_time, results[0],results[1] );
                    printf("%s\n", (results[0] > tol ? "  fail" : ""));
                } else {
                    printf("%5d %5d     ---   (  ---  )   %7.2f (%7.2f)    %8.2e                      %8.2e",
                           (int) M, (int) N, gpu_perf, gpu_time, results[0],results[1] );
                    printf("%s\n", (results[0] > tol ? "  fail" : ""));
                } 
                status |= (error > tol);
            
                TESTING_FREE( h_W1 );
                TESTING_FREE( h_W2 );
                TESTING_FREE( h_W3 );
                TESTING_FREE( h_RW );

            } else if ( opts.check == 2 ) {
                /* =====================================================================
                   Check the result compared to LAPACK
                   =================================================================== */
                magma_zgetmatrix( M, N, d_A, ldda, h_R, M );
                error = lapackf77_zlange("f", &M, &N, h_A, &lda, work);
                blasf77_zaxpy(&n2, &c_neg_one, h_A, &ione, h_R, &ione);
                error = lapackf77_zlange("f", &M, &N, h_R, &lda, work) / error;
              
                if ( opts.lapack ) {
                    printf("%5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e",
                           (int) M, (int) N, cpu_perf, cpu_time, gpu_perf, gpu_time, error );
                } else {
                    printf("%5d %5d     ---   (  ---  )   %7.2f (%7.2f)    %8.2e",
                           (int) M, (int) N, gpu_perf, gpu_time, error );
                }
                printf("%s\n", (error > tol ? "  fail" : ""));
                status |= (error > tol);
            }
            else {
                if ( opts.lapack ) {
                    printf("%5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)   ---\n",
                           (int) M, (int) N, cpu_perf, cpu_time, gpu_perf, gpu_time );
                } else {
                    printf("%5d %5d     ---   (  ---  )   %7.2f (%7.2f)     ---  \n",
                           (int) M, (int) N, gpu_perf, gpu_time);
                }

            }
            
            TESTING_FREE( tau );
            TESTING_FREE( h_A );
            TESTING_FREE( h_work );
            TESTING_HOSTFREE( h_R );
            TESTING_DEVFREE( d_A );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }
    
    TESTING_FINALIZE();
    return status;
}
