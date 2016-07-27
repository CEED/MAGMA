/*
    -- MAGMA (version 2.0) --
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
#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zhemv
*/
int main(int argc, char **argv)
{
    #ifdef HAVE_clBLAS
    #define dA(i_, j_)  dA, ((i_) + (j_)*ldda)
    #define dX(i_)      dX, ((i_))
    #define dY(i_)      dY, ((i_))
    #define dwork(i_)   dwork, ((i_))
    #else
    #define dA(i_, j_) (dA + (i_) + (j_)*ldda)
    #define dX(i_)     (dX + (i_))
    #define dY(i_)     (dY + (i_))
    #define dwork(i_)  (dwork + (i_))
    #endif
    
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    const magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    const magma_int_t        ione      = 1;
    
    real_Double_t   atomics_perf=0, atomics_time=0;
    real_Double_t   gflops, magma_perf=0, magma_time=0, dev_perf, dev_time, cpu_perf, cpu_time;
    double          magma_error=0, atomics_error=0, dev_error, work[1];
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t N, lda, ldda, sizeA, sizeX, sizeY, blocks, ldwork;
    magma_int_t incx = 1;
    magma_int_t incy = 1;
    magma_int_t nb   = 64;
    magmaDoubleComplex alpha = MAGMA_Z_MAKE(  1.5, -2.3 );
    magmaDoubleComplex beta  = MAGMA_Z_MAKE( -0.6,  0.8 );
    magmaDoubleComplex *A, *X, *Y, *Yatomics, *Ydev, *Ymagma;
    magmaDoubleComplex_ptr dA, dX, dY, dwork;
    int status = 0;
    
    magma_opts opts;
    opts.parse_opts( argc, argv );
    
    double tol = opts.tolerance * lapackf77_dlamch("E");

    printf("%% uplo = %s\n", lapack_uplo_const(opts.uplo) );
    #ifdef HAVE_CUBLAS
        printf("%%   N   MAGMA Gflop/s (ms)    Atomics Gflop/s      %s Gflop/s       CPU Gflop/s   MAGMA error  %s\n",
                g_platform_str, g_platform_str );
        printf("%%==========================================================================================================\n");
    #else
        printf("%%   N   %s Gflop/s       CPU Gflop/s   MAGMA error  %s\n",
                g_platform_str, g_platform_str );
        printf("%%===============================================================\n");
    #endif
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            lda    = N;
            ldda   = magma_roundup( N, opts.align );  // multiple of 32 by default
            sizeA  = N*lda;
            sizeX  = N*incx;
            sizeY  = N*incy;
            gflops = FLOPS_ZHEMV( N ) / 1e9;
            
            TESTING_CHECK( magma_zmalloc_cpu( &A,        sizeA ));
            TESTING_CHECK( magma_zmalloc_cpu( &X,        sizeX ));
            TESTING_CHECK( magma_zmalloc_cpu( &Y,        sizeY ));
            TESTING_CHECK( magma_zmalloc_cpu( &Yatomics, sizeY ));
            TESTING_CHECK( magma_zmalloc_cpu( &Ydev,     sizeY ));
            TESTING_CHECK( magma_zmalloc_cpu( &Ymagma,   sizeY ));
            
            TESTING_CHECK( magma_zmalloc( &dA, ldda*N ));
            TESTING_CHECK( magma_zmalloc( &dX, sizeX ));
            TESTING_CHECK( magma_zmalloc( &dY, sizeY ));
            
            blocks = magma_ceildiv( N, nb );
            ldwork = ldda*blocks;
            TESTING_CHECK( magma_zmalloc( &dwork, ldwork ));
            
            magmablas_zlaset( MagmaFull, ldwork, 1, MAGMA_Z_NAN, MAGMA_Z_NAN, dwork(0), ldwork, opts.queue );
            magmablas_zlaset( MagmaFull, ldda,   N, MAGMA_Z_NAN, MAGMA_Z_NAN, dA(0,0),  ldda,   opts.queue );
            
            /* Initialize the matrix */
            lapackf77_zlarnv( &ione, ISEED, &sizeA, A );
            magma_zmake_hermitian( N, A, lda );
            
            // should not use data from the opposite triangle -- fill with NAN to check
            magma_int_t N1 = N-1;
            if ( opts.uplo == MagmaUpper ) {
                lapackf77_zlaset( "Lower", &N1, &N1, &MAGMA_Z_NAN, &MAGMA_Z_NAN, &A[1], &lda );
            }
            else {
                lapackf77_zlaset( "Upper", &N1, &N1, &MAGMA_Z_NAN, &MAGMA_Z_NAN, &A[lda], &lda );
            }
            
            lapackf77_zlarnv( &ione, ISEED, &sizeX, X );
            lapackf77_zlarnv( &ione, ISEED, &sizeY, Y );
            
            /* =====================================================================
               Performs operation using cuBLAS / clBLAS
               =================================================================== */
            magma_zsetmatrix( N, N, A, lda, dA(0,0), ldda, opts.queue );
            magma_zsetvector( N, X, incx, dX(0), incx, opts.queue );
            magma_zsetvector( N, Y, incy, dY(0), incy, opts.queue );
            
            dev_time = magma_sync_wtime( opts.queue );
            magma_zhemv( opts.uplo, N,
                         alpha, dA(0,0), ldda,
                                dX(0),   incx,
                         beta,  dY(0),   incy, opts.queue );
            dev_time = magma_sync_wtime( opts.queue ) - dev_time;
            dev_perf = gflops / dev_time;
            
            magma_zgetvector( N, dY(0), incy, Ydev, incy, opts.queue );
            
            /* =====================================================================
               Performs operation using cuBLAS - using atomics
               =================================================================== */
            #ifdef HAVE_CUBLAS
                cublasSetAtomicsMode( opts.handle, CUBLAS_ATOMICS_ALLOWED );
                magma_zsetvector( N, Y, incy, dY(0), incy, opts.queue );
                
                // sync on queue doesn't work -- need device sync or use NULL stream -- bug in CUBLAS?
                atomics_time = magma_sync_wtime( NULL /*opts.queue*/ );
                magma_zhemv( opts.uplo,  N,
                             alpha, dA(0,0), ldda,
                                    dX(0),   incx,
                             beta,  dY(0),   incy, opts.queue );
                atomics_time = magma_sync_wtime( NULL /*opts.queue*/ ) - atomics_time;
                atomics_perf = gflops / atomics_time;
                
                magma_zgetvector( N, dY(0), incy, Yatomics, incy, opts.queue );
                cublasSetAtomicsMode( opts.handle, CUBLAS_ATOMICS_NOT_ALLOWED );
            #endif
            
            /* =====================================================================
               Performs operation using MAGMABLAS (only with CUDA)
               =================================================================== */
            #ifdef HAVE_CUBLAS
                magma_zsetvector( N, Y, incy, dY(0), incy, opts.queue );
                
                magma_time = magma_sync_wtime( opts.queue );
                if ( opts.version == 1 ) {
                    magmablas_zhemv_work( opts.uplo, N,
                                          alpha, dA(0,0), ldda,
                                                 dX(0),   incx,
                                          beta,  dY(0),   incy,
                                          dwork(0), ldwork, opts.queue );
                }
                else {
                    // non-work interface (has added overhead)
                    magmablas_zhemv( opts.uplo, N,
                                     alpha, dA(0,0), ldda,
                                            dX(0),   incx,
                                     beta,  dY(0),   incy, opts.queue );
                }
                magma_time = magma_sync_wtime( opts.queue ) - magma_time;
                magma_perf = gflops / magma_time;
                
                magma_zgetvector( N, dY(0), incy, Ymagma, incy, opts.queue );
            #endif
            
            /* =====================================================================
               Performs operation using CPU BLAS
               =================================================================== */
            cpu_time = magma_wtime();
            blasf77_zhemv( lapack_uplo_const(opts.uplo), &N, &alpha, A, &lda, X, &incx, &beta, Y, &incy );
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gflops / cpu_time;
            
            /* =====================================================================
               Check the result
               =================================================================== */
            blasf77_zaxpy( &N, &c_neg_one, Y, &incy, Ydev, &incy );
            dev_error = lapackf77_zlange( "M", &N, &ione, Ydev, &N, work ) / N;
            
            #ifdef HAVE_CUBLAS
                blasf77_zaxpy( &N, &c_neg_one, Y, &incy, Yatomics, &incy );
                atomics_error = lapackf77_zlange( "M", &N, &ione, Yatomics, &N, work ) / N;
                
                blasf77_zaxpy( &N, &c_neg_one, Y, &incy, Ymagma, &incy );
                magma_error = lapackf77_zlange( "M", &N, &ione, Ymagma, &N, work ) / N;
            #endif
            
            bool okay = (magma_error < tol && dev_error < tol && atomics_error < tol);
            status += ! okay;
            printf("%5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %8.2e   %8.2e   %s\n",
                   (long long) N,
                   magma_perf,   1000.*magma_time,
                   atomics_perf, 1000.*atomics_time,
                   dev_perf,     1000.*dev_time,
                   cpu_perf,     1000.*cpu_time,
                   magma_error, dev_error, atomics_error,
                   (okay ? "ok" : "failed"));
            
            magma_free_cpu( A );
            magma_free_cpu( X );
            magma_free_cpu( Y );
            magma_free_cpu( Ydev     );
            magma_free_cpu( Yatomics );
            magma_free_cpu( Ymagma   );
            
            magma_free( dA );
            magma_free( dX );
            magma_free( dY );
            magma_free( dwork );
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
