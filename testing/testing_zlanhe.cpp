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

#ifdef MAGMA_WITH_MKL
#include <mkl_service.h>  // mkl_version
#endif

// includes, project
#include "magma.h"
#include "magma_lapack.h"
#include "magma_operators.h"
#include "testings.h"

#include "magma_threadsetting.h"  // to work around MKL bug

#define PRECISION_z

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zlanhe
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t   gbytes, gpu_perf, gpu_time, cpu_perf, cpu_time;
    magmaDoubleComplex *h_A;
    double *h_work;
    magmaDoubleComplex_ptr d_A;
    magmaDouble_ptr d_work;
    magma_int_t N, n2, lda, ldda;
    magma_int_t idist    = 3;  // normal distribution (otherwise max norm is always ~ 1)
    magma_int_t ISEED[4] = {0,0,0,1};
    double      error, norm_magma, norm_lapack;
    magma_int_t status = 0;
    magma_int_t lapack_nan_fail = 0;
    magma_int_t lapack_inf_fail = 0;
    bool mkl_warning = false;

    magma_opts opts;
    opts.parse_opts( argc, argv );
    
    double tol = opts.tolerance * lapackf77_dlamch("E");
    
    magma_uplo_t uplo[] = { MagmaLower, MagmaUpper };
    magma_norm_t norm[] = { MagmaInfNorm, MagmaOneNorm, MagmaMaxNorm };
    
    // Double-Complex inf-norm not supported on Tesla (CUDA arch 1.x)
#if defined(PRECISION_z)
    magma_int_t arch = magma_getdevice_arch();
    if ( arch < 200 ) {
        printf("!!!! NOTE: Double-Complex %s and %s norm are not supported\n"
               "!!!! on CUDA architecture %d; requires arch >= 200.\n"
               "!!!! It should report \"parameter number 1 had an illegal value\" below.\n\n",
               MagmaInfNormStr, MagmaOneNormStr, (int) arch );
        for( int inorm = 0; inorm < 2; ++inorm ) {
        for( int iuplo = 0; iuplo < 2; ++iuplo ) {
            printf( "Testing that magmablas_zlanhe( %s, %s, ... ) returns -1 error...\n",
                    lapack_norm_const( norm[inorm] ),
                    lapack_uplo_const( uplo[iuplo] ));
            norm_magma = magmablas_zlanhe( norm[inorm], uplo[iuplo], 1, NULL, 1, NULL );
            if ( norm_magma != -1 ) {
                printf( "expected magmablas_zlanhe to return -1 error, but got %f\n", norm_magma );
                status = 1;
            }
        }}
        printf( "...return values %s\n\n", (status == 0 ? "ok" : "failed") );
    }
#endif

    #ifdef MAGMA_WITH_MKL
    // MKL (11.1.2) has bug in multi-threaded zlanhe; use single thread to work around
    // appears to be corrected in 11.2.3
    MKLVersion mkl_version;
    mkl_get_version( &mkl_version );
    int threads = magma_get_lapack_numthreads();
    bool mkl_single_thread = (mkl_version.MajorVersion <= 11 && mkl_version.MinorVersion < 2);
    if ( mkl_single_thread ) {
        printf( "\nNote: using single thread to work around MKL zlanhe bug.\n\n" );
    }
    #endif
    
    printf("%%   N   norm   uplo   CPU GByte/s (ms)    GPU GByte/s (ms)        error    error      nan      inf\n");
    printf("%%=================================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
      for( int inorm = 0; inorm < 3; ++inorm ) {
      for( int iuplo = 0; iuplo < 2; ++iuplo ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N   = opts.nsize[itest];
            lda = N;
            n2  = lda*N;
            ldda = magma_roundup( N, opts.align );
            // read upper or lower triangle
            gbytes = 0.5*(N+1)*N*sizeof(magmaDoubleComplex) / 1e9;
            
            TESTING_MALLOC_CPU( h_A,    magmaDoubleComplex, n2 );
            TESTING_MALLOC_CPU( h_work, double, N );
            
            TESTING_MALLOC_DEV( d_A,    magmaDoubleComplex, ldda*N );
            TESTING_MALLOC_DEV( d_work, double, N );
            
            /* Initialize the matrix */
            lapackf77_zlarnv( &idist, ISEED, &n2, h_A );
            
            magma_zsetmatrix( N, N, h_A, lda, d_A, ldda );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            norm_magma = magmablas_zlanhe( norm[inorm], uplo[iuplo], N, d_A, ldda, d_work );
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gbytes / gpu_time;
            if (norm_magma == -1) {
                printf( "%5d   %4c   skipped because it isn't supported on this GPU\n",
                        (int) N, lapacke_norm_const( norm[inorm] ));
                continue;
            }
            if (norm_magma < 0)
                printf("magmablas_zlanhe returned error %f: %s.\n",
                       norm_magma, magma_strerror( (int) norm_magma ));
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            #ifdef MAGMA_WITH_MKL
            if ( mkl_single_thread ) {
                // use single thread to work around MKL bug
                magma_set_lapack_numthreads( 1 );
            }
            #endif
            
            cpu_time = magma_wtime();
            norm_lapack = lapackf77_zlanhe(
                lapack_norm_const( norm[inorm] ),
                lapack_uplo_const( uplo[iuplo] ),
                &N, h_A, &lda, h_work );
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gbytes / cpu_time;
            if (norm_lapack < 0)
                printf("lapackf77_zlanhe returned error %f: %s.\n",
                       norm_lapack, magma_strerror( (int) norm_lapack ));
            
            #ifdef MAGMA_WITH_MKL
            if ( mkl_single_thread ) {
                // end single thread to work around MKL bug
                magma_set_lapack_numthreads( threads );
            }
            #endif
            
            /* =====================================================================
               Check the result compared to LAPACK
               Note: MKL (11.1.0) has bug for uplo=Lower with multiple threads.
               Try with $MKL_NUM_THREADS = 1.
               =================================================================== */
            error = fabs( norm_magma - norm_lapack ) / norm_lapack;
            double tol2 = tol;
            if ( norm[inorm] == MagmaMaxNorm ) {
                // max-norm depends on only one element, so for Real precisions,
                // MAGMA and LAPACK should exactly agree (tol2 = 0),
                // while Complex precisions incur roundoff in cuCabs.
                #if defined(PRECISION_s) || defined(PRECISION_d)
                tol2 = 0;
                #endif
            }
            
            bool okay = (error <= tol2);
            status += ! okay;
            mkl_warning |= ! okay;
            
            /* ====================================================================
               Check for NAN and INF propagation
               =================================================================== */
            #define h_A(i_, j_) (h_A + (i_) + (j_)*lda)
            #define d_A(i_, j_) (d_A + (i_) + (j_)*ldda)
            
            magma_int_t i = rand() % N;
            magma_int_t j = rand() % N;
            magma_int_t tmp;
            if ( uplo[iuplo] == MagmaLower && i < j ) {
                tmp = i;
                i = j;
                j = tmp;
            }
            else if ( uplo[iuplo] == MagmaUpper && i > j ) {
                tmp = i;
                i = j;
                j = tmp;
            }
                
            *h_A(i,j) = MAGMA_Z_NAN;
            magma_zsetvector( 1, h_A(i,j), 1, d_A(i,j), 1 );
            norm_magma  = magmablas_zlanhe( norm[inorm], uplo[iuplo], N, d_A, ldda, d_work );
            norm_lapack = lapackf77_zlanhe( lapack_norm_const( norm[inorm] ),
                                            lapack_uplo_const( uplo[iuplo] ),
                                            &N, h_A, &lda, h_work );
            bool nan_okay    = isnan(norm_magma);
            bool la_nan_okay = isnan(norm_lapack);
            lapack_nan_fail += ! la_nan_okay;
            status          += !    nan_okay;
            
            *h_A(i,j) = MAGMA_Z_INF;
            magma_zsetvector( 1, h_A(i,j), 1, d_A(i,j), 1 );
            norm_magma  = magmablas_zlanhe( norm[inorm], uplo[iuplo], N, d_A, ldda, d_work );
            norm_lapack = lapackf77_zlanhe( lapack_norm_const( norm[inorm] ),
                                            lapack_uplo_const( uplo[iuplo] ),
                                            &N, h_A, &lda, h_work );
            bool inf_okay    = isinf(norm_magma);
            bool la_inf_okay = isinf(norm_lapack);
            lapack_inf_fail += ! la_inf_okay;
            status          += !    inf_okay;
            
            printf("%5d   %4c   %4c   %7.2f (%7.2f)   %7.2f (%7.2f)   %#9.3g   %6s   %6s%1s  %6s%1s\n",
                   (int) N,
                   lapacke_norm_const( norm[inorm] ),
                   lapacke_uplo_const( uplo[iuplo] ),
                   cpu_perf, cpu_time*1000., gpu_perf, gpu_time*1000.,
                   error,
                   (okay     ? "ok" : "failed"),
                   (nan_okay ? "ok" : "failed"), (la_nan_okay ? " " : "*"),
                   (inf_okay ? "ok" : "failed"), (la_inf_okay ? " " : "*"));
            
            TESTING_FREE_CPU( h_A    );
            TESTING_FREE_CPU( h_work );
            
            TESTING_FREE_DEV( d_A    );
            TESTING_FREE_DEV( d_work );
            fflush( stdout );
        } // end iter
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
      }} // end iuplo, inorm
      printf( "\n" );
    }
    
    // don't print "failed" here because then run_tests.py thinks MAGMA failed
    if ( lapack_nan_fail ) {
        printf( "* Warning: LAPACK did not pass NAN propagation test; upgrade to LAPACK version >= 3.4.2 (Sep. 2012)\n" );
    }
    if ( lapack_inf_fail ) {
        printf( "* Warning: LAPACK did not pass INF propagation test\n" );
    }
    if ( mkl_warning ) {
        printf("* MKL (e.g., 11.1.0) has a bug in zlanhe with multiple threads; corrected in 11.2.x.\n"
               "  Try again with MKL_NUM_THREADS=1.\n" );
    }
    
    TESTING_FINALIZE();
    return status;
}
