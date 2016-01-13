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
#include "magma.h"
#include "magma_lapack.h"
#include "magma_operators.h"
#include "testings.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgeqrf
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    /* Constants */
    magmaDoubleComplex c_zero    = MAGMA_Z_ZERO;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    double d_one     = MAGMA_D_ONE;
    double d_neg_one = MAGMA_D_NEG_ONE;
    
    /* Local variables */
    real_Double_t    gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    double           Anorm, error, error2, diff, terr, rwork[1];
    magmaDoubleComplex *h_A, *h_T, *h_R, *tau, *h_work, tmp[1];
    magmaDoubleComplex_ptr d_A, d_T, ddA, dtau;
    magmaDouble_ptr dwork;

    magma_int_t M, N, lda, ldda, lwork, size, info, min_mn;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t status = 0;

    #define BLOCK_SIZE 64

    magma_opts opts;
    opts.parse_opts( argc, argv );
    
    double eps = lapackf77_dlamch("E");
    double tol = opts.tolerance * lapackf77_dlamch("E");
    
    magma_queue_t queues[2];
    magma_queue_create( &queues[0] );
    magma_queue_create( &queues[1] );

    printf("%% version %d\n", (int) opts.version );
    printf("%% It's okay if |Q - Q_lapack| is large; MAGMA and LAPACK\n"
           "%% just chose different Householder reflectors, both valid.\n"
           "%% Number in parenthesis after each error is error/epsilon, for comparison with --tolerance.\n\n");
    
    printf("%%   M     N    CPU Gflop/s (ms)    GPU Gflop/s (ms)   |R - Q^H*A|        |I - Q^H*Q|        |T - T_lapack|     |Q - Q_lapack|  (error/epsilon)\n");
    printf("%%=============================================================================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M     = opts.msize[itest];
            N     = opts.nsize[itest];

            if (N > 128) {
                printf("%5d %5d   skipping because zgeqr2x requires N <= 128\n",
                        (int) M, (int) N);
                continue;
            }
            if (M < N) {
                printf("%5d %5d   skipping because zgeqr2x requires M >= N\n",
                        (int) M, (int) N);
                continue;
            }

            min_mn = min( M, N );
            lda    = M;
            ldda   = magma_roundup( M, opts.align );  // multiple of 32 by default
            // TODO: flops should be GEQRF + LARFT (whatever that is)
            gflops = (FLOPS_ZGEQRF( M, N ) + FLOPS_ZGEQRT( M, N )) / 1e9;

            lwork = -1;
            lapackf77_zgeqrf( &M, &N, NULL, &M, NULL, tmp, &lwork, &info );
            lwork = (magma_int_t)MAGMA_Z_REAL( tmp[0] );
            lwork = max( lwork, N*N );
        
            /* Allocate memory for the matrix */
            TESTING_MALLOC_CPU( tau,    magmaDoubleComplex, min_mn );
            TESTING_MALLOC_CPU( h_A,    magmaDoubleComplex, lda*N  );
            TESTING_MALLOC_CPU( h_T,    magmaDoubleComplex, N*N    );
            TESTING_MALLOC_CPU( h_work, magmaDoubleComplex, lwork );
            
            TESTING_MALLOC_PIN( h_R,    magmaDoubleComplex, lda*N  );
            
            TESTING_MALLOC_DEV( d_A,    magmaDoubleComplex, ldda*N );
            TESTING_MALLOC_DEV( d_T,    magmaDoubleComplex, N*N    );
            TESTING_MALLOC_DEV( ddA,    magmaDoubleComplex, N*N    );
            TESTING_MALLOC_DEV( dtau,   magmaDoubleComplex, min_mn );
            
            TESTING_MALLOC_DEV( dwork,  double, max(5*min_mn, (BLOCK_SIZE*2+2)*min_mn) );
            
            magmablas_zlaset( MagmaFull, N, N, c_zero, c_zero, ddA, N );
            magmablas_zlaset( MagmaFull, N, N, c_zero, c_zero, d_T, N );
            
            /* Initialize the matrix */
            size = lda*N;
            lapackf77_zlarnv( &ione, ISEED, &size, h_A );
            lapackf77_zlacpy( MagmaFullStr, &M, &N, h_A, &lda, h_R, &lda );
            magma_zsetmatrix( M, N, h_R, lda, d_A, ldda );
    
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            magmablasSetKernelStream( opts.queue );
            gpu_time = magma_sync_wtime( opts.queue );
    
            if (opts.version == 1) {
                magma_zgeqr2x_gpu(  M, N, d_A, ldda, dtau, d_T, ddA, dwork, &info );
            }
            else if (opts.version == 2) {
                magma_zgeqr2x2_gpu( M, N, d_A, ldda, dtau, d_T, ddA, dwork, &info );
            }
            else if (opts.version == 3) {
                magma_zgeqr2x3_gpu( M, N, d_A, ldda, dtau, d_T, ddA, dwork, &info );
            }
            else {
                /*
                  Going through NULL stream is faster
                  Going through any stream is slower
                  Doing two streams in parallel is slower than doing them sequentially
                  Queuing happens on the NULL stream - user defined buffers are smaller?
                */
                magma_zgeqr2x4_gpu( M, N, d_A, ldda, dtau, d_T, ddA, dwork, opts.queue, &info );
            }
            gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;
            gpu_perf = gflops / gpu_time;

            if (info != 0) {
                printf("magma_zgeqr2x_gpu version %d returned error %d: %s.\n",
                       (int) opts.version, (int) info, magma_strerror( info ));
            }
            else if ( opts.check ) {
                /* =====================================================================
                   Check the result, following zqrt01 except using the reduced Q.
                   This works for any M,N (square, tall, wide).
                   =================================================================== */
                magma_zgetmatrix( M, N, d_A, ldda, h_R, lda );
                magma_zgetmatrix( N, N, ddA, N,    h_T, N );
                magma_zgetmatrix( min_mn, 1, dtau, min_mn, tau, min_mn );
                // Restore the upper triangular part of A before the check
                lapackf77_zlacpy( "Upper", &N, &N, h_T, &N, h_R, &lda );

                magma_int_t ldq = M;
                magma_int_t ldr = min_mn;
                magmaDoubleComplex *Q, *R;
                double *work;
                TESTING_MALLOC_CPU( Q,    magmaDoubleComplex, ldq*min_mn );  // M by K
                TESTING_MALLOC_CPU( R,    magmaDoubleComplex, ldr*N );       // K by N
                TESTING_MALLOC_CPU( work, double,             min_mn );
                
                // generate M by K matrix Q, where K = min(M,N)
                lapackf77_zlacpy( "Lower", &M, &min_mn, h_R, &lda, Q, &ldq );
                lapackf77_zungqr( &M, &min_mn, &min_mn, Q, &ldq, tau, h_work, &lwork, &info );
                assert( info == 0 );
                
                // copy K by N matrix R
                lapackf77_zlaset( "Lower", &min_mn, &N, &c_zero, &c_zero, R, &ldr );
                lapackf77_zlacpy( "Upper", &min_mn, &N, h_R, &lda,        R, &ldr );
                
                // error = || R - Q^H*A || / (N * ||A||)
                blasf77_zgemm( "Conj", "NoTrans", &min_mn, &N, &M,
                               &c_neg_one, Q, &ldq, h_A, &lda, &c_one, R, &ldr );
                Anorm = lapackf77_zlange( "1", &M,      &N, h_A, &lda, work );
                error = lapackf77_zlange( "1", &min_mn, &N, R,   &ldr, work );
                if ( N > 0 && Anorm > 0 )
                    error /= (N*Anorm);
                
                // set R = I (K by K identity), then R = I - Q^H*Q
                // error = || I - Q^H*Q || / N
                lapackf77_zlaset( "Upper", &min_mn, &min_mn, &c_zero, &c_one, R, &ldr );
                blasf77_zherk( "Upper", "Conj", &min_mn, &M, &d_neg_one, Q, &ldq, &d_one, R, &ldr );
                error2 = safe_lapackf77_zlanhe( "1", "Upper", &min_mn, R, &ldr, work );
                if ( N > 0 )
                    error2 /= N;
                
                TESTING_FREE_CPU( Q    );  Q    = NULL;
                TESTING_FREE_CPU( R    );  R    = NULL;
                TESTING_FREE_CPU( work );  work = NULL;

                /* =====================================================================
                   Performs operation using LAPACK
                   =================================================================== */
                cpu_time = magma_wtime();
                lapackf77_zgeqrf( &M, &N, h_A, &lda, tau, h_work, &lwork, &info );
                lapackf77_zlarft( MagmaForwardStr, MagmaColumnwiseStr,
                                  &M, &N, h_A, &lda, tau, h_work, &N );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0) {
                    printf("lapackf77_zgeqrf returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                }

                /* =====================================================================
                   Check the result compared to LAPACK
                   Okay if these are different -- just chose different Householder reflectors
                   =================================================================== */
                size = lda*N;
                blasf77_zaxpy( &size, &c_neg_one, h_A, &ione, h_R, &ione );
                Anorm = lapackf77_zlange( "M", &M, &N, h_A, &lda, rwork );
                diff =  lapackf77_zlange( "M", &M, &N, h_R, &lda, rwork );
                if ( Anorm > 0) {
                    diff /= (N*Anorm);
                }
                
                /* =====================================================================
                   Check if T is correct
                   =================================================================== */
                // Recompute T in h_work for d_A (magma), in case it is different than h_A (lapack)
                magma_zgetmatrix( M, N, d_A, ldda, h_R, lda );
                magma_zgetmatrix( min_mn, 1, dtau, min_mn, tau, min_mn );
                lapackf77_zlarft( MagmaForwardStr, MagmaColumnwiseStr,
                                  &M, &N, h_R, &lda, tau, h_work, &N );
                
                magma_zgetmatrix( N, N, d_T, N, h_T, N );
                size = N*N;
                blasf77_zaxpy( &size, &c_neg_one, h_work, &ione, h_T, &ione );
                Anorm = lapackf77_zlantr( "F", "U", "N", &N, &N, h_work, &N, rwork );
                terr  = lapackf77_zlantr( "F", "U", "N", &N, &N, h_T,    &N, rwork );
                if (Anorm > 0) {
                    terr /= Anorm;
                }
                
                bool okay = (error < tol) && (error2 < tol) && (terr < tol);
                status += ! okay;
                printf("%5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e (%5.0f)   %8.2e (%5.0f)   %8.2e (%5.0f)   %8.2e (%5.0f)  %s\n",
                       (int) M, (int) N, cpu_perf, 1000.*cpu_time, gpu_perf, 1000.*gpu_time,
                       error,  error /eps,
                       error2, error2/eps,
                       terr,   terr  /eps,
                       diff,   diff  /eps,
                       (okay ? "ok" : "failed"));
            }
            else {
                printf("%5d %5d     ---   (  ---  )   %7.2f (%7.2f)     ---  \n",
                       (int) M, (int) N, gpu_perf, 1000.*gpu_time);
            }
            
            TESTING_FREE_CPU( tau    );
            TESTING_FREE_CPU( h_A    );
            TESTING_FREE_CPU( h_T    );
            TESTING_FREE_CPU( h_work );
            
            TESTING_FREE_PIN( h_R    );
        
            TESTING_FREE_DEV( d_A   );
            TESTING_FREE_DEV( d_T   );
            TESTING_FREE_DEV( ddA   );
            TESTING_FREE_DEV( dtau  );
            TESTING_FREE_DEV( dwork );
        
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }
    
    magma_queue_destroy( queues[0] );
    magma_queue_destroy( queues[1] );

    opts.cleanup();
    TESTING_FINALIZE();
    return status;
}
