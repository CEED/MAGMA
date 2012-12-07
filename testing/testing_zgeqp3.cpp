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

#define PRECISION_z

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgeqp3
*/
int main( int argc, char** argv)
{
    TESTING_CUDA_INIT();
    
    real_Double_t    gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    cuDoubleComplex *h_A, *h_R, *tau, *h_work;
    magma_int_t *jpvt;
    magma_int_t M, N, n2, lda, lwork, j, info, min_mn, nb;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    
    magma_opts opts;
    parse_opts( argc, argv, &opts );
    
    printf("  M     N     CPU GFlop/s (sec)   GPU GFlop/s (sec)   ||A*P - Q*R||_F\n");
    printf("=====================================================================\n");
    for( int i = 0; i < opts.ntest; ++i ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[i];
            N = opts.nsize[i];
            min_mn = min(M, N);
            lda    = M;
            n2     = lda*N;
            nb     = magma_get_zgeqp3_nb( min_mn );
            gflops = FLOPS_ZGEQRF( M, N ) / 1e9;
            
            lwork = ( N+1 )*nb;
            #if defined(PRECISION_d) || defined(PRECISION_s)
            lwork += 2*N;
            #endif
            if ( opts.check )
                lwork = max( lwork, M*N + N );
            
            #if defined(PRECISION_z) || defined(PRECISION_c)
            double *rwork;
            TESTING_MALLOC(    rwork,  double,          2*N    );
            #endif
            TESTING_MALLOC(    jpvt,   magma_int_t,     N      );
            TESTING_MALLOC(    tau,    cuDoubleComplex, min_mn );
            TESTING_MALLOC(    h_A,    cuDoubleComplex, n2     );
            TESTING_HOSTALLOC( h_R,    cuDoubleComplex, n2     );
            TESTING_HOSTALLOC( h_work, cuDoubleComplex, lwork  );
            
            /* Initialize the matrix */
            lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
            lapackf77_zlacpy( MagmaUpperLowerStr, &M, &N, h_A, &lda, h_R, &lda );
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                for( j = 0; j < N; j++)
                    jpvt[j] = 0;
                
                cpu_time = magma_wtime();
                #if defined(PRECISION_z) || defined(PRECISION_c)
                lapackf77_zgeqp3(&M, &N, h_R, &lda, jpvt, tau, h_work, &lwork, rwork, &info);
                #else
                lapackf77_zgeqp3(&M, &N, h_R, &lda, jpvt, tau, h_work, &lwork, &info);
                #endif
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0)
                    printf("lapack_zgeqp3 returned error %d.\n", info);
            }
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            lapackf77_zlacpy( MagmaUpperLowerStr, &M, &N, h_A, &lda, h_R, &lda );
            for( j = 0; j < N; j++)
                jpvt[j] = 0;
            
            gpu_time = magma_wtime();
            #if defined(PRECISION_z) || defined(PRECISION_c)
            magma_zgeqp3(M, N, h_R, lda, jpvt, tau, h_work, lwork, rwork, &info);
            #else
            magma_zgeqp3(M, N, h_R, lda, jpvt, tau, h_work, lwork, &info);
            #endif
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0)
                printf("magma_zgeqp3 returned error %d.\n", info);
            
            /* =====================================================================
               Check the result
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
                double error, ulp;
                
                magma_int_t minmn = min(M, N);
                ulp = lapackf77_dlamch( "P" );
                
                // Compute norm( A*P - Q*R )
                error = lapackf77_zqpt01( &M, &N, &minmn, h_A, h_R, &lda,
                                          tau, jpvt, h_work, &lwork );
                error *= ulp;
                
                printf("   %8.2e\n", error );
            }
            else {
                printf("     ---  \n");
            }
            
            #if defined(PRECISION_z) || defined(PRECISION_c)
            TESTING_FREE( rwork );
            #endif
            TESTING_FREE( jpvt );
            TESTING_FREE( tau );
            TESTING_FREE( h_A );
            TESTING_HOSTFREE( h_R );
            TESTING_HOSTFREE( h_work );
        }
    }

    TESTING_CUDA_FINALIZE();
    return 0;
}
