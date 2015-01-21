/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
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
   -- Testing zgeqrf
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    const double             d_neg_one = MAGMA_D_NEG_ONE;
    const double             d_one     = MAGMA_D_ONE;
    const magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    const magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    const magmaDoubleComplex c_zero    = MAGMA_Z_ZERO;
    const magma_int_t        ione      = 1;
    
    real_Double_t    gflops, gpu_perf, gpu_time, cpu_perf=0, cpu_time=0;
    double           Anorm, error=0, error2=0;
    magmaDoubleComplex *h_A, *h_R, *tau, *h_work, tmp[1];
    magmaDoubleComplex_ptr d_A, dT;
    magma_int_t M, N, n2, lda, ldda, lwork, info, min_mn, nb, size;
    magma_int_t ISEED[4] = {0,0,0,1};
    
    magma_opts opts;
    parse_opts( argc, argv, &opts );
    
    magma_int_t status = 0;
    double tol = opts.tolerance * lapackf77_dlamch("E");
    
    printf( "version %d\n", (int) opts.version );
    if ( opts.version == 2 ) {
        printf("    M     N   CPU GFlop/s (sec)   GPU GFlop/s (sec)   |R - Q^H*A|   |I - Q^H*Q|\n");
        printf("===============================================================================\n");
    }
    else {
        printf("    M     N   CPU GFlop/s (sec)   GPU GFlop/s (sec)    |b - A*x|\n");
        printf("================================================================\n");
    }
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            min_mn = min(M, N);
            lda    = M;
            n2     = lda*N;
            ldda   = ((M+31)/32)*32;
            gflops = FLOPS_ZGEQRF( M, N ) / 1e9;
            
            // query for workspace size
            lwork = -1;
            lapackf77_zgeqrf(&M, &N, NULL, &M, NULL, tmp, &lwork, &info);
            lwork = (magma_int_t)MAGMA_Z_REAL( tmp[0] );
            
            TESTING_MALLOC_CPU( tau,    magmaDoubleComplex, min_mn );
            TESTING_MALLOC_CPU( h_A,    magmaDoubleComplex, n2     );
            TESTING_MALLOC_CPU( h_work, magmaDoubleComplex, lwork  );
            
            TESTING_MALLOC_PIN( h_R,    magmaDoubleComplex, n2     );
            
            TESTING_MALLOC_DEV( d_A,    magmaDoubleComplex, ldda*N );
            
            /* Initialize the matrix */
            lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
            lapackf77_zlacpy( MagmaUpperLowerStr, &M, &N, h_A, &lda, h_R, &lda );
            magma_zsetmatrix( M, N, h_R, lda, d_A, ldda );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            if ( opts.version == 2 ) {
                // LAPACK complaint arguments
                magma_zgeqrf2_gpu( M, N, d_A, ldda, tau, &info );
            }
            else {
                nb = magma_get_zgeqrf_nb( M );
                size = (2*min(M, N) + (N+31)/32*32 )*nb;
                TESTING_MALLOC_DEV( dT, magmaDoubleComplex, size );
                if ( opts.version == 1 ) {
                    // stores dT, V blocks have zeros, R blocks inverted & stored in dT
                    magma_zgeqrf_gpu( M, N, d_A, ldda, tau, dT, &info );
                }
                #ifdef HAVE_CUBLAS
                else if ( opts.version == 3 ) {
                    // stores dT, V blocks have zeros, R blocks stored in dT
                    magma_zgeqrf3_gpu( M, N, d_A, ldda, tau, dT, &info );
                }
                #endif
                else {
                    printf( "Unknown version %d\n", (int) opts.version );
                    exit(1);
                }
            }
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0)
                printf("magma_zgeqrf returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            if ( opts.check && opts.version == 2 ) {
                /* =====================================================================
                   Check the result, following zqrt01 except using the reduced Q.
                   This works for any M,N (square, tall, wide).
                   Only for version 2, which has LAPACK complaint output.
                   =================================================================== */
                magma_zgetmatrix( M, N, d_A, ldda, h_R, lda );
                
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
                error2 = lapackf77_zlanhe( "1", "Upper", &min_mn, R, &ldr, work );
                if ( N > 0 )
                    error2 /= N;
                
                TESTING_FREE_CPU( Q    );  Q    = NULL;
                TESTING_FREE_CPU( R    );  R    = NULL;
                TESTING_FREE_CPU( work );  work = NULL;
            }
            else if ( opts.check && M >= N ) {
                /* =====================================================================
                   Check the result by solving consistent linear system, A*x = b.
                   Only for versions 1 & 3 with M >= N.
                   =================================================================== */
                magma_int_t lwork;
                magmaDoubleComplex *x, *b, *hwork;
                magmaDoubleComplex_ptr d_B;
                const magmaDoubleComplex c_zero    = MAGMA_Z_ZERO;
                const magmaDoubleComplex c_one     = MAGMA_Z_ONE;
                const magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
                const magma_int_t ione = 1;

                // initialize RHS, b = A*random
                TESTING_MALLOC_CPU( x, magmaDoubleComplex, N );
                TESTING_MALLOC_CPU( b, magmaDoubleComplex, M );
                lapackf77_zlarnv( &ione, ISEED, &N, x );
                blasf77_zgemv( "Notrans", &M, &N, &c_one, h_A, &lda, x, &ione, &c_zero, b, &ione );
                // copy to GPU
                TESTING_MALLOC_DEV( d_B, magmaDoubleComplex, M );
                magma_zsetvector( M, b, 1, d_B, 1 );

                if ( opts.version == 1 ) {
                    // allocate hwork
                    magma_zgeqrs_gpu( M, N, 1,
                                      d_A, ldda, tau, dT,
                                      d_B, M, tmp, -1, &info );
                    lwork = (magma_int_t)MAGMA_Z_REAL( tmp[0] );
                    TESTING_MALLOC_CPU( hwork, magmaDoubleComplex, lwork );

                    // solve linear system
                    magma_zgeqrs_gpu( M, N, 1,
                                      d_A, ldda, tau, dT,
                                      d_B, M, hwork, lwork, &info );
                    if (info != 0)
                        printf("magma_zgeqrs returned error %d: %s.\n",
                               (int) info, magma_strerror( info ));
                    TESTING_FREE_CPU( hwork );
                }
                #ifdef HAVE_CUBLAS
                else if ( opts.version == 3 ) {
                    // allocate hwork
                    magma_zgeqrs3_gpu( M, N, 1,
                                       d_A, ldda, tau, dT,
                                       d_B, M, tmp, -1, &info );
                    lwork = (magma_int_t)MAGMA_Z_REAL( tmp[0] );
                    TESTING_MALLOC_CPU( hwork, magmaDoubleComplex, lwork );

                    // solve linear system
                    magma_zgeqrs3_gpu( M, N, 1,
                                       d_A, ldda, tau, dT,
                                       d_B, M, hwork, lwork, &info );
                    if (info != 0)
                        printf("magma_zgeqrs3 returned error %d: %s.\n",
                               (int) info, magma_strerror( info ));
                    TESTING_FREE_CPU( hwork );
                }
                #endif
                else {
                    printf( "Unknown version %d\n", (int) opts.version );
                    exit(1);
                }
                magma_zgetvector( N, d_B, 1, x, 1 );

                // compute r = Ax - b, saved in b
                blasf77_zgemv( "Notrans", &M, &N, &c_one, h_A, &lda, x, &ione, &c_neg_one, b, &ione );

                // compute residual |Ax - b| / (n*|A|*|x|)
                double norm_x, norm_A, norm_r, work[1];
                norm_A = lapackf77_zlange( "F", &M, &N, h_A, &lda, work );
                norm_r = lapackf77_zlange( "F", &M, &ione, b, &M, work );
                norm_x = lapackf77_zlange( "F", &N, &ione, x, &N, work );

                TESTING_FREE_CPU( x );
                TESTING_FREE_CPU( b );
                TESTING_FREE_DEV( d_B );

                error = norm_r / (N * norm_A * norm_x);
            }
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                lapackf77_zgeqrf(&M, &N, h_A, &lda, tau, h_work, &lwork, &info);
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0)
                    printf("lapackf77_zgeqrf returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
            }
            
            /* =====================================================================
               Print performance and error.
               =================================================================== */
            printf("%5d %5d   ", (int) M, (int) N );
            if ( opts.lapack ) {
                printf( "%7.2f (%7.2f)", cpu_perf, cpu_time );
            }
            else {
                printf("  ---   (  ---  )" );
            }
            printf( "   %7.2f (%7.2f)   ", gpu_perf, gpu_time );
            if ( opts.check ) {
                if ( opts.version == 2 ) {
                    bool okay = (error < tol && error2 < tol);
                    status += ! okay;
                    printf( "%11.2e   %11.2e   %s\n", error, error2, (okay ? "ok" : "failed") );
                }
                else if ( M >= N ) {
                    bool okay = (error < tol);
                    status += ! okay;
                    printf( "%10.2e   %s\n", error, (okay ? "ok" : "failed") );
                }
                else {
                    printf( "(error check only for M >= N)\n" );
                }
            }
            else {
                printf( "    ---\n" );
            }
            
            TESTING_FREE_CPU( tau    );
            TESTING_FREE_CPU( h_A    );
            TESTING_FREE_CPU( h_work );
            
            TESTING_FREE_PIN( h_R );
            
            TESTING_FREE_DEV( d_A );
            
            if ( opts.version != 2 )
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
