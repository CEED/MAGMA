/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @author Raffaele Solca
       @author Stan Tomov
       @author Azzam Haidar

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
   -- Testing zhetrd_gpu
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t    gflops, gpu_perf, cpu_perf, gpu_time, cpu_time;
    double           eps;
    magmaDoubleComplex *h_A, *h_R, *d_R, *h_Q, *h_work, *work, *dwork;
    magmaDoubleComplex *tau;
    double          *diag, *offdiag;
    double           result[2] = {0., 0.};
    magma_int_t N, n2, lda, lwork, info, nb, ldwork;
    magma_int_t ione     = 1;
    magma_int_t itwo     = 2;
    magma_int_t ithree   = 3;
    magma_int_t ISEED[4] = {0,0,0,1};
    
    #if defined(PRECISION_z) || defined(PRECISION_c)
    double *rwork;
    #endif

    eps = lapackf77_dlamch( "E" );

    magma_opts opts;
    parse_opts( argc, argv, &opts );
    
    printf("Running version %d; available are (specified through --version num):\n", 
           opts.version);
    printf("1 - uses ZHEMV from CUBLAS (default)\n");
    printf("2 - uses ZHEMV from MAGMA BLAS that requires extra space\n\n");

    printf("  N     CPU GFlop/s (sec)   GPU GFlop/s (sec)   |A-QHQ'|/N|A|   |I-QQ'|/N\n");
    printf("===========================================================================\n");
    for( int i = 0; i < opts.ntest; ++i ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[i];
            lda    = N;
            n2     = N*lda;
            nb     = magma_get_zhetrd_nb(N);
            lwork  = N*nb;  /* We suppose the magma nb is bigger than lapack nb */
            gflops = FLOPS_ZHETRD( N ) / 1e9;
            ldwork = (N*N+64-1)/64 + 2*N*nb;
            
            TESTING_MALLOC(    h_A,     magmaDoubleComplex, lda*N );
            TESTING_DEVALLOC(  d_R,     magmaDoubleComplex, lda*N );
            TESTING_DEVALLOC(dwork,     magmaDoubleComplex, ldwork);
            TESTING_HOSTALLOC( h_R,     magmaDoubleComplex, lda*N );
            TESTING_HOSTALLOC( h_work,  magmaDoubleComplex, lwork );
            TESTING_MALLOC(    tau,     magmaDoubleComplex, N     );
            TESTING_MALLOC(    diag,    double, N   );
            TESTING_MALLOC(    offdiag, double, N-1 );
            
            if ( opts.check ) {
                TESTING_MALLOC( h_Q,  magmaDoubleComplex, lda*N );
                TESTING_MALLOC( work, magmaDoubleComplex, 2*N*N );
                #if defined(PRECISION_z) || defined(PRECISION_c)
                TESTING_MALLOC( rwork, double, N );
                #endif
            }
            
            /* ====================================================================
               Initialize the matrix
               =================================================================== */
            lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
            magma_zmake_hermitian( N, h_A, lda );
            magma_zsetmatrix( N, N, h_A, lda, d_R, lda );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            if (opts.version == 1)
                magma_zhetrd_gpu( opts.uplo, N, d_R, lda, diag, offdiag,
                                  tau, h_R, lda, h_work, lwork, &info );
            else
                magma_zhetrd2_gpu( opts.uplo, N, d_R, lda, diag, offdiag,
                                   tau, h_R, lda, h_work, lwork, dwork, ldwork, &info );
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0)
                printf("magma_zhetrd_gpu returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            /* =====================================================================
               Check the factorization
               =================================================================== */
            if ( opts.check ) {
                magma_zgetmatrix( N, N, d_R, lda, h_R, lda );
                magma_zgetmatrix( N, N, d_R, lda, h_Q, lda );
                lapackf77_zungtr( &opts.uplo, &N, h_Q, &lda, tau, h_work, &lwork, &info );
                
                #if defined(PRECISION_z) || defined(PRECISION_c)
                lapackf77_zhet21( &itwo, &opts.uplo, &N, &ione,
                                  h_A, &lda, diag, offdiag,
                                  h_Q, &lda, h_R, &lda,
                                  tau, work, rwork, &result[0] );
                
                lapackf77_zhet21( &ithree, &opts.uplo, &N, &ione,
                                  h_A, &lda, diag, offdiag,
                                  h_Q, &lda, h_R, &lda,
                                  tau, work, rwork, &result[1] );
                #else
                lapackf77_zhet21( &itwo, &opts.uplo, &N, &ione,
                                  h_A, &lda, diag, offdiag,
                                  h_Q, &lda, h_R, &lda,
                                  tau, work, &result[0] );
                
                lapackf77_zhet21( &ithree, &opts.uplo, &N, &ione,
                                  h_A, &lda, diag, offdiag,
                                  h_Q, &lda, h_R, &lda,
                                  tau, work, &result[1] );
                #endif
            }
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                lapackf77_zhetrd( &opts.uplo, &N, h_A, &lda, diag, offdiag, tau,
                                  h_work, &lwork, &info );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0)
                    printf("lapackf77_zhetrd returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
            }
            
            /* =====================================================================
               Print performance and error.
               =================================================================== */
            if ( opts.lapack ) {
                printf("%5d   %7.2f (%7.2f)   %7.2f (%7.2f)",
                       (int) N, cpu_perf, cpu_time, gpu_perf, gpu_time );
            } else {
                printf("%5d     ---   (  ---  )   %7.2f (%7.2f)",
                       (int) N, gpu_perf, gpu_time );
            }
            if ( opts.check ) {
                printf("   %8.2e        %8.2e\n", result[0]*eps, result[1]*eps );
            } else {
                printf("     ---             ---\n" );
            }
            
            TESTING_FREE( h_A );
            TESTING_FREE( tau );
            TESTING_FREE( diag );
            TESTING_FREE( offdiag );
            TESTING_HOSTFREE( h_R );
            TESTING_HOSTFREE( h_work );
            TESTING_DEVFREE ( d_R );
            TESTING_DEVFREE ( dwork );
            
            if ( opts.check ) {
                TESTING_FREE( h_Q );
                TESTING_FREE( work );
                #if defined(PRECISION_z) || defined(PRECISION_c)
                TESTING_FREE( rwork );
                #endif
            }
        }
    }

    TESTING_FINALIZE();
    return 0;
}
