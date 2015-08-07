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

// includes, project
#include "testings.h"  // before magma.h, to include cublas_v2
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgemm
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t   gflops, magma_perf, magma_time, dev_perf, dev_time, cpu_perf, cpu_time;
    double          magma_error, dev_error, Cnorm, work[1];
    magma_int_t M, N, K;
    magma_int_t Am, An, Bm, Bn;
    magma_int_t sizeA, sizeB, sizeC;
    magma_int_t lda, ldb, ldc, ldda, lddb, lddc;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t status = 0;
    
    magmaDoubleComplex *h_A, *h_B, *h_C, *h_Cmagma, *h_Cdev;
    magmaDoubleComplex_ptr d_A, d_B, d_C;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex alpha = MAGMA_Z_MAKE(  0.29, -0.86 );
    magmaDoubleComplex beta  = MAGMA_Z_MAKE( -0.48,  0.38 );
    
    magma_opts opts;
    opts.parse_opts( argc, argv );
    
    double tol = opts.tolerance * lapackf77_dlamch("E");

    #ifdef HAVE_CUBLAS
        // for CUDA, we can check MAGMA vs. CUBLAS, without running LAPACK
        printf("%% If running lapack (option --lapack), MAGMA and %s error are both computed\n"
               "%% relative to CPU BLAS result. Else, MAGMA error is computed relative to %s result.\n\n",
                g_platform_str, g_platform_str );
        printf("%% transA = %s, transB = %s\n",
               lapack_trans_const(opts.transA),
               lapack_trans_const(opts.transB) );
        printf("%%   M     N     K   MAGMA Gflop/s (ms)  %s Gflop/s (ms)   CPU Gflop/s (ms)  MAGMA error  %s error\n",
                g_platform_str, g_platform_str );
    #else
        // for others, we need LAPACK for check
        opts.lapack |= opts.check;  // check (-c) implies lapack (-l)
        printf("%% transA = %s, transB = %s\n",
               lapack_trans_const(opts.transA),
               lapack_trans_const(opts.transB) );
        printf("%%   M     N     K   %s Gflop/s (ms)   CPU Gflop/s (ms)  %s error\n",
                g_platform_str, g_platform_str );
    #endif
    printf("%%========================================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            K = opts.ksize[itest];
            gflops = FLOPS_ZGEMM( M, N, K ) / 1e9;

            if ( opts.transA == MagmaNoTrans ) {
                lda = Am = M;
                An = K;
            } else {
                lda = Am = K;
                An = M;
            }
            
            if ( opts.transB == MagmaNoTrans ) {
                ldb = Bm = K;
                Bn = N;
            } else {
                ldb = Bm = N;
                Bn = K;
            }
            ldc = M;
            
            ldda = magma_roundup( lda, opts.align );  // multiple of 32 by default
            lddb = magma_roundup( ldb, opts.align );  // multiple of 32 by default
            lddc = magma_roundup( ldc, opts.align );  // multiple of 32 by default
            
            sizeA = lda*An;
            sizeB = ldb*Bn;
            sizeC = ldc*N;
            
            TESTING_MALLOC_CPU( h_A,       magmaDoubleComplex, lda*An );
            TESTING_MALLOC_CPU( h_B,       magmaDoubleComplex, ldb*Bn );
            TESTING_MALLOC_CPU( h_C,       magmaDoubleComplex, ldc*N  );
            TESTING_MALLOC_CPU( h_Cmagma,  magmaDoubleComplex, ldc*N  );
            TESTING_MALLOC_CPU( h_Cdev,    magmaDoubleComplex, ldc*N  );
            
            TESTING_MALLOC_DEV( d_A, magmaDoubleComplex, ldda*An );
            TESTING_MALLOC_DEV( d_B, magmaDoubleComplex, lddb*Bn );
            TESTING_MALLOC_DEV( d_C, magmaDoubleComplex, lddc*N  );
            
            /* Initialize the matrices */
            lapackf77_zlarnv( &ione, ISEED, &sizeA, h_A );
            lapackf77_zlarnv( &ione, ISEED, &sizeB, h_B );
            lapackf77_zlarnv( &ione, ISEED, &sizeC, h_C );
            
            magma_zsetmatrix( Am, An, h_A, lda, d_A, ldda );
            magma_zsetmatrix( Bm, Bn, h_B, ldb, d_B, lddb );
            
            /* =====================================================================
               Performs operation using MAGMABLAS (currently only with CUDA)
               =================================================================== */
            #ifdef HAVE_CUBLAS
                magma_zsetmatrix( M, N, h_C, ldc, d_C, lddc );
                
                magmablasSetKernelStream( opts.queue );
                magma_time = magma_sync_wtime( opts.queue );
                magmablas_zgemm( opts.transA, opts.transB, M, N, K,
                                 alpha, d_A, ldda,
                                        d_B, lddb,
                                 beta,  d_C, lddc );
                magma_time = magma_sync_wtime( opts.queue ) - magma_time;
                magma_perf = gflops / magma_time;
                
                magma_zgetmatrix( M, N, d_C, lddc, h_Cmagma, ldc );
            #endif
            
            /* =====================================================================
               Performs operation using CUBLAS / clBLAS / Xeon Phi MKL
               =================================================================== */
            magma_zsetmatrix( M, N, h_C, ldc, d_C, lddc );
            
            dev_time = magma_sync_wtime( opts.queue );
            #ifdef HAVE_CUBLAS
                // opts.handle also uses opts.queue
                cublasZgemm( opts.handle, cublas_trans_const(opts.transA), cublas_trans_const(opts.transB), M, N, K,
                             &alpha, d_A, ldda,
                                     d_B, lddb,
                             &beta,  d_C, lddc );
            #else
                magma_zgemm( opts.transA, opts.transB, M, N, K,
                             alpha, d_A, 0, ldda,
                                    d_B, 0, lddb,
                             beta,  d_C, 0, lddc, opts.queue );
            #endif
            dev_time = magma_sync_wtime( opts.queue ) - dev_time;
            dev_perf = gflops / dev_time;
            
            magma_zgetmatrix( M, N, d_C, lddc, h_Cdev, ldc );
            
            /* =====================================================================
               Performs operation using CPU BLAS
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                blasf77_zgemm( lapack_trans_const(opts.transA), lapack_trans_const(opts.transB), &M, &N, &K,
                               &alpha, h_A, &lda,
                                       h_B, &ldb,
                               &beta,  h_C, &ldc );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
            }
            
            /* =====================================================================
               Check the result
               =================================================================== */
            if ( opts.lapack ) {
                // compute relative error for both magma & dev, relative to lapack,
                // |C_magma - C_lapack| / |C_lapack|
                Cnorm = lapackf77_zlange( "F", &M, &N, h_C, &ldc, work );
                
                blasf77_zaxpy( &sizeC, &c_neg_one, h_C, &ione, h_Cdev, &ione );
                dev_error = lapackf77_zlange( "F", &M, &N, h_Cdev, &ldc, work ) / Cnorm;
                
                #ifdef HAVE_CUBLAS
                    blasf77_zaxpy( &sizeC, &c_neg_one, h_C, &ione, h_Cmagma, &ione );
                    magma_error = lapackf77_zlange( "F", &M, &N, h_Cmagma, &ldc, work ) / Cnorm;
                    
                    printf("%5d %5d %5d   %7.2f (%7.2f)    %7.2f (%7.2f)   %7.2f (%7.2f)    %8.2e     %8.2e   %s\n",
                           (int) M, (int) N, (int) K,
                           magma_perf,  1000.*magma_time,
                           dev_perf,    1000.*dev_time,
                           cpu_perf,    1000.*cpu_time,
                           magma_error, dev_error,
                           (magma_error < tol && dev_error < tol ? "ok" : "failed"));
                    status += ! (magma_error < tol && dev_error < tol);
                #else
                    printf("%5d %5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)    %8.2e   %s\n",
                           (int) M, (int) N, (int) K,
                           dev_perf,    1000.*dev_time,
                           cpu_perf,    1000.*cpu_time,
                           dev_error,
                           (dev_error < tol ? "ok" : "failed"));
                    status += ! (dev_error < tol);
                #endif
            }
            else {
                #ifdef HAVE_CUBLAS
                    // compute relative error for magma, relative to dev (currently only with CUDA)
                    Cnorm = lapackf77_zlange( "F", &M, &N, h_Cdev, &ldc, work );
                    
                    blasf77_zaxpy( &sizeC, &c_neg_one, h_Cdev, &ione, h_Cmagma, &ione );
                    magma_error = lapackf77_zlange( "F", &M, &N, h_Cmagma, &ldc, work ) / Cnorm;
                    
                    printf("%5d %5d %5d   %7.2f (%7.2f)    %7.2f (%7.2f)     ---   (  ---  )    %8.2e        ---    %s\n",
                           (int) M, (int) N, (int) K,
                           magma_perf,  1000.*magma_time,
                           dev_perf,    1000.*dev_time,
                           magma_error,
                           (magma_error < tol ? "ok" : "failed"));
                    status += ! (magma_error < tol);
                #else
                    printf("%5d %5d %5d   %7.2f (%7.2f)     ---   (  ---  )       ---\n",
                           (int) M, (int) N, (int) K,
                           dev_perf,    1000.*dev_time );
                #endif
            }
            
            TESTING_FREE_CPU( h_A );
            TESTING_FREE_CPU( h_B );
            TESTING_FREE_CPU( h_C );
            TESTING_FREE_CPU( h_Cmagma  );
            TESTING_FREE_CPU( h_Cdev    );
            
            TESTING_FREE_DEV( d_A );
            TESTING_FREE_DEV( d_B );
            TESTING_FREE_DEV( d_C );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    TESTING_FINALIZE();
    return status;
}
