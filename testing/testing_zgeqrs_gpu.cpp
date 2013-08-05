/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

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
   -- Testing zgeqrs
*/
int main( int argc, char** argv )
{
    TESTING_INIT();
    
    real_Double_t    gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    double           gpu_error, cpu_error, Anorm, work[1];
    magmaDoubleComplex  c_one     = MAGMA_Z_ONE;
    magmaDoubleComplex  c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex *h_A, *h_A2, *h_B, *h_X, *h_R, *tau, *h_work, tmp[1];
    magmaDoubleComplex *d_A, *d_B;
    magma_int_t M, N, size, nrhs, lda, ldb, ldda, lddb, min_mn, max_mn, nb, info;
    magma_int_t lworkgpu, lhwork;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};

    magma_opts opts;
    parse_opts( argc, argv, &opts );
 
    magma_int_t status = 0;
    double tol = opts.tolerance * lapackf77_dlamch("E");

    nrhs = opts.nrhs;
    
    printf("                                                            ||b-Ax|| / (N||A||)\n");
    printf("    M     N  NRHS   CPU GFlop/s (sec)   GPU GFlop/s (sec)   CPU        GPU     \n");
    printf("===============================================================================\n");
    for( int i = 0; i < opts.ntest; ++i ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[i];
            N = opts.nsize[i];
            if ( M < N ) {
                printf( "skipping M=%d, N=%d because M < N is not yet supported.\n", (int) M, (int) N );
                continue;
            }
            min_mn = min(M, N);
            max_mn = max(M, N);
            lda    = M;
            ldb    = max_mn;
            ldda   = ((M+31)/32)*32;
            lddb   = ((max_mn+31)/32)*32;
            nb     = magma_get_zgeqrf_nb(M);
            gflops = (FLOPS_ZGEQRF( M, N ) + FLOPS_ZGEQRS( M, N, nrhs )) / 1e9;
            
            // query for workspace size
            lworkgpu = (M - N + nb)*(nrhs + nb) + nrhs*nb;
            
            lhwork = -1;
            lapackf77_zgels( MagmaNoTransStr, &M, &N, &nrhs,
                             h_A, &lda, h_X, &ldb, tmp, &lhwork, &info );
            lhwork = (magma_int_t) MAGMA_Z_REAL( tmp[0] );
            lhwork = max( lhwork, lworkgpu );
            
            TESTING_MALLOC(   tau,    magmaDoubleComplex, min_mn    );
            TESTING_MALLOC(   h_A,    magmaDoubleComplex, lda*N     );
            TESTING_MALLOC(   h_A2,   magmaDoubleComplex, lda*N     );
            TESTING_MALLOC(   h_B,    magmaDoubleComplex, ldb*nrhs  );
            TESTING_MALLOC(   h_X,    magmaDoubleComplex, ldb*nrhs  );
            TESTING_MALLOC(   h_R,    magmaDoubleComplex, ldb*nrhs  );
            TESTING_MALLOC(   h_work, magmaDoubleComplex, lhwork    );
            
            TESTING_DEVALLOC( d_A,    magmaDoubleComplex, ldda*N    );
            TESTING_DEVALLOC( d_B,    magmaDoubleComplex, lddb*nrhs );
            
            /* Initialize the matrices */
            size = lda*N;
            lapackf77_zlarnv( &ione, ISEED, &size, h_A );
            lapackf77_zlacpy( MagmaUpperLowerStr, &M, &N, h_A, &lda, h_A2, &lda );
            
            // make random RHS
            size = ldb*nrhs;
            lapackf77_zlarnv( &ione, ISEED, &size, h_B );
            lapackf77_zlacpy( MagmaUpperLowerStr, &M, &nrhs, h_B, &ldb, h_R, &ldb );
            
            // make consistent RHS
            //size = N*nrhs;
            //lapackf77_zlarnv( &ione, ISEED, &size, h_X );
            //blasf77_zgemm( MagmaNoTransStr, MagmaNoTransStr, &M, &nrhs, &N,
            //               &c_one,  h_A, &lda,
            //                        h_X, &ldb,
            //               &c_zero, h_B, &ldb );
            //lapackf77_zlacpy( MagmaUpperLowerStr, &M, &nrhs, h_B, &ldb, h_R, &ldb );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            magma_zsetmatrix( M, N,    h_A, lda, d_A, ldda );
            magma_zsetmatrix( M, nrhs, h_B, ldb, d_B, lddb );
            
            gpu_time = magma_wtime();
            magma_zgels_gpu( MagmaNoTrans, M, N, nrhs, d_A, ldda,
                             d_B, lddb, h_work, lworkgpu, &info );
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0)
                printf("magma_zgels returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            // compute the residual
            magma_zgetmatrix( N, nrhs, d_B, lddb, h_X, ldb );
            blasf77_zgemm( MagmaNoTransStr, MagmaNoTransStr, &M, &nrhs, &N,
                           &c_neg_one, h_A, &lda,
                                       h_X, &ldb,
                           &c_one,     h_R, &ldb );
            Anorm = lapackf77_zlange("f", &M, &N, h_A, &lda, work);
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            lapackf77_zlacpy( MagmaUpperLowerStr, &M, &nrhs, h_B, &ldb, h_X, &ldb );
            
            cpu_time = magma_wtime();
            lapackf77_zgels( MagmaNoTransStr, &M, &N, &nrhs,
                             h_A, &lda, h_X, &ldb, h_work, &lhwork, &info );
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gflops / cpu_time;
            if (info != 0)
                printf("lapackf77_zgels returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            blasf77_zgemm( MagmaNoTransStr, MagmaNoTransStr, &M, &nrhs, &N,
                           &c_neg_one, h_A2, &lda,
                                       h_X,  &ldb,
                           &c_one,     h_B,  &ldb );
            
            cpu_error = lapackf77_zlange("f", &M, &nrhs, h_B, &ldb, work) / (min_mn*Anorm);
            gpu_error = lapackf77_zlange("f", &M, &nrhs, h_R, &ldb, work) / (min_mn*Anorm);
            
            printf("%5d %5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %8.2e",
                   (int) M, (int) N, (int) nrhs,
                   cpu_perf, cpu_time, gpu_perf, gpu_time, cpu_error, gpu_error );
            printf("%s\n", (gpu_error < tol ? "" : "  failed"));
            status |= ! (gpu_error < tol);

            
            TESTING_FREE( tau  );
            TESTING_FREE( h_A  );
            TESTING_FREE( h_A2 );
            TESTING_FREE( h_B  );
            TESTING_FREE( h_X  );
            TESTING_FREE( h_R  );
            TESTING_FREE( h_work );
            TESTING_DEVFREE( d_A );
            TESTING_DEVFREE( d_B );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    TESTING_FINALIZE();
    return status;
}
