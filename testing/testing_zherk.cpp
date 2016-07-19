/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
       @author Chongxiao Cao
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
   -- Testing zherk
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, cublas_perf, cublas_time, cpu_perf, cpu_time;
    double          cublas_error, Cnorm, work[1];
    magma_int_t N, K;
    magma_int_t Ak, An;
    magma_int_t sizeA, sizeC;
    magma_int_t lda, ldc, ldda, lddc;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    
    magmaDoubleComplex *h_A, *h_C, *h_Ccublas;
    magmaDoubleComplex_ptr d_A, d_C;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    double alpha = MAGMA_D_MAKE(  0.29, -0.86 );
    double beta  = MAGMA_D_MAKE( -0.48,  0.38 );
    magma_int_t status = 0;
    
    magma_opts opts;
    opts.parse_opts( argc, argv );
    opts.lapack |= opts.check;  // check (-c) implies lapack (-l)
    
    double tol = opts.tolerance * lapackf77_dlamch("E");
    
    #ifdef COMPLEX
    if (opts.transA == MagmaTrans) {
        opts.transA = MagmaConjTrans; 
        printf("%% WARNING: transA = MagmaTrans changed to MagmaConjTrans\n");
    }
    #endif
    
    printf("%% If running lapack (option --lapack), CUBLAS error is computed\n"
           "%% relative to CPU BLAS result.\n\n");
    printf("%% uplo = %s, transA = %s\n",
           lapack_uplo_const(opts.uplo), lapack_trans_const(opts.transA) );
    printf("%%   N     K   CUBLAS Gflop/s (ms)   CPU Gflop/s (ms)  CUBLAS error\n");
    printf("%%=================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            K = opts.ksize[itest];
            gflops = FLOPS_ZHERK(K, N) / 1e9;

            if ( opts.transA == MagmaNoTrans ) {
                lda = An = N;
                Ak = K;
            } else {
                lda = An = K;
                Ak = N;
            }
            
            ldc = N;
            
            ldda = magma_roundup( lda, opts.align );  // multiple of 32 by default
            lddc = magma_roundup( ldc, opts.align );  // multiple of 32 by default
            
            sizeA = lda*Ak;
            sizeC = ldc*N;
            
            TESTING_CHECK( magma_zmalloc_cpu( &h_A,       lda*Ak ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_C,       ldc*N  ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_Ccublas, ldc*N  ));
            
            TESTING_CHECK( magma_zmalloc( &d_A, ldda*Ak ));
            TESTING_CHECK( magma_zmalloc( &d_C, lddc*N  ));
            
            /* Initialize the matrices */
            lapackf77_zlarnv( &ione, ISEED, &sizeA, h_A );
            lapackf77_zlarnv( &ione, ISEED, &sizeC, h_C );
            
            /* =====================================================================
               Performs operation using CUBLAS
               =================================================================== */
            magma_zsetmatrix( An, Ak, h_A, lda, d_A, ldda, opts.queue );
            magma_zsetmatrix( N, N, h_C, ldc, d_C, lddc, opts.queue );

            cublas_time = magma_sync_wtime( opts.queue );
            #ifdef HAVE_CUBLAS
                cublasZherk( opts.handle, cublas_uplo_const(opts.uplo), cublas_trans_const(opts.transA), N, K,
                             &alpha, d_A, ldda,
                             &beta,  d_C, lddc );
            #else
                magma_zherk( opts.uplo, opts.transA, N, K,
                             alpha, d_A, 0, ldda,
                             beta,  d_C, 0, lddc, opts.queue );
            #endif
            cublas_time = magma_sync_wtime( opts.queue ) - cublas_time;
            cublas_perf = gflops / cublas_time;
            
            magma_zgetmatrix( N, N, d_C, lddc, h_Ccublas, ldc, opts.queue );
            
            /* =====================================================================
               Performs operation using CPU BLAS
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                blasf77_zherk( lapack_uplo_const(opts.uplo), lapack_trans_const(opts.transA), &N, &K,
                               &alpha, h_A, &lda,
                               &beta,  h_C, &ldc );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
            }
            
            /* =====================================================================
               Check the result
               =================================================================== */
            if ( opts.lapack ) {
                // compute relative error for both magma & cublas, relative to lapack,
                // |C_magma - C_lapack| / |C_lapack|
                blasf77_zaxpy( &sizeC, &c_neg_one, h_C, &ione, h_Ccublas, &ione );
                Cnorm        = safe_lapackf77_zlanhe( "fro", lapack_uplo_const(opts.uplo), &N, h_C,       &ldc, work );
                cublas_error = safe_lapackf77_zlanhe( "fro", lapack_uplo_const(opts.uplo), &N, h_Ccublas, &ldc, work )
                             / Cnorm;
                
                printf("%5ld %5ld   %7.2f (%7.2f)   %7.2f (%7.2f)    %8.2e   %s\n",
                       long(N), long(K),
                       cublas_perf, 1000.*cublas_time,
                       cpu_perf,    1000.*cpu_time,
                       cublas_error, (cublas_error < tol ? "ok" : "failed"));
                status += ! (cublas_error < tol);
            }
            else {
                printf("%5ld %5ld   %7.2f (%7.2f)    ---   (  ---  )    ---     ---\n",
                       long(N), long(K),
                       cublas_perf, 1000.*cublas_time);
            }
            
            magma_free_cpu( h_A );
            magma_free_cpu( h_C );
            magma_free_cpu( h_Ccublas );
            
            magma_free( d_A );
            magma_free( d_C );
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
