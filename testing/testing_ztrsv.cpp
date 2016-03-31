/*
    -- MAGMA (version 1.1) --
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

#define h_A(i,j) (h_A + (i) + (j)*lda)

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing ztrsm
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, cublas_perf, cublas_time, cpu_perf=0, cpu_time=0;
    double          cublas_error, normA, normx, normr, work[1];
    magma_int_t i, j, N, info;
    magma_int_t sizeA;
    magma_int_t lda, ldda;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t *ipiv;

    magmaDoubleComplex *h_A, *h_b, *h_x, *h_xcublas;
    magmaDoubleComplex_ptr d_A, d_x;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magma_int_t status = 0;
    
    magma_opts opts;
    opts.parse_opts( argc, argv );
    
    double tol = opts.tolerance * lapackf77_dlamch("E");
    
    printf("%% uplo = %s, transA = %s, diag = %s\n",
           lapack_uplo_const(opts.uplo), lapack_trans_const(opts.transA), lapack_diag_const(opts.diag) );
    printf("%%   N  CUBLAS Gflop/s (ms)   CPU Gflop/s (ms)   CUBLAS error\n");
    printf("%%===========================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            gflops = FLOPS_ZTRSM(opts.side, N, 1) / 1e9;
            lda    = N;
            ldda   = magma_roundup( lda, opts.align );  // multiple of 32 by default
            sizeA  = lda*N;
            
            TESTING_CHECK( magma_imalloc_cpu( &ipiv,      N     ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_A,       lda*N ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_b,       N     ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_x,       N     ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_xcublas, N     ));
            
            TESTING_CHECK( magma_zmalloc( &d_A, ldda*N ));
            TESTING_CHECK( magma_zmalloc( &d_x, N      ));
            
            /* Initialize the matrices */
            /* Factor A into LU to get well-conditioned triangular matrix.
             * Copy L to U, since L seems okay when used with non-unit diagonal
             * (i.e., from U), while U fails when used with unit diagonal. */
            lapackf77_zlarnv( &ione, ISEED, &sizeA, h_A );
            lapackf77_zgetrf( &N, &N, h_A, &lda, ipiv, &info );
            for( j = 0; j < N; ++j ) {
                for( i = 0; i < j; ++i ) {
                    *h_A(i,j) = *h_A(j,i);
                }
            }
            
            lapackf77_zlarnv( &ione, ISEED, &N, h_b );
            blasf77_zcopy( &N, h_b, &ione, h_x, &ione );
            
            /* =====================================================================
               Performs operation using CUBLAS
               =================================================================== */
            magma_zsetmatrix( N, N, h_A, lda, d_A, ldda, opts.queue );
            magma_zsetvector( N, h_x, 1, d_x, 1, opts.queue );
            
            cublas_time = magma_sync_wtime( opts.queue );
            #ifdef HAVE_CUBLAS
                cublasZtrsv( opts.handle, cublas_uplo_const(opts.uplo),
                             cublas_trans_const(opts.transA), cublas_diag_const(opts.diag),
                             N,
                             d_A, ldda,
                             d_x, 1 );
            #else
                magma_ztrsv( opts.uplo, opts.transA, opts.diag,
                             N,
                             d_A, 0, ldda,
                             d_x, 0, 1, opts.queue );
            #endif
            cublas_time = magma_sync_wtime( opts.queue ) - cublas_time;
            cublas_perf = gflops / cublas_time;
            
            magma_zgetvector( N, d_x, 1, h_xcublas, 1, opts.queue );
            
            /* =====================================================================
               Performs operation using CPU BLAS
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                blasf77_ztrsv( lapack_uplo_const(opts.uplo), lapack_trans_const(opts.transA), lapack_diag_const(opts.diag),
                               &N,
                               h_A, &lda,
                               h_x, &ione );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
            }
            
            /* =====================================================================
               Check the result
               =================================================================== */
            // ||b - Ax|| / (||A||*||x||)
            // error for CUBLAS
            normA = lapackf77_zlange( "F", &N, &N, h_A, &lda, work );
            
            normx = lapackf77_zlange( "F", &N, &ione, h_xcublas, &ione, work );
            blasf77_ztrmv( lapack_uplo_const(opts.uplo), lapack_trans_const(opts.transA), lapack_diag_const(opts.diag),
                           &N,
                           h_A, &lda,
                           h_xcublas, &ione );
            blasf77_zaxpy( &N, &c_neg_one, h_b, &ione, h_xcublas, &ione );
            normr = lapackf77_zlange( "F", &N, &ione, h_xcublas, &N, work );
            cublas_error = normr / (normA*normx);

            if ( opts.lapack ) {
                printf("%5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %s\n",
                        (int) N,
                        cublas_perf, 1000.*cublas_time,
                        cpu_perf,    1000.*cpu_time,
                        cublas_error, (cublas_error < tol ? "ok" : "failed"));
                status += ! (cublas_error < tol);
            }
            else {
                printf("%5d   %7.2f (%7.2f)     ---  (  ---  )   %8.2e   %s\n",
                        (int) N,
                        cublas_perf, 1000.*cublas_time,
                        cublas_error, (cublas_error < tol ? "ok" : "failed"));
                status += ! (cublas_error < tol);
            }
            
            magma_free_cpu( ipiv );
            magma_free_cpu( h_A  );
            magma_free_cpu( h_b  );
            magma_free_cpu( h_x  );
            magma_free_cpu( h_xcublas );
            
            magma_free( d_A );
            magma_free( d_x );
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
