/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Mark Gates

*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <cublas.h>

// includes, project
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing ztranspose
   Code is very similar to testing_zsymmetrize.cpp
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t    gbytes, gpu_perf, gpu_time, gpu_perf2=0, gpu_time2=0, cpu_perf, cpu_time;
    double           error, error2, work[1];
    magmaDoubleComplex  c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex *h_A, *h_B, *h_R;
    magmaDoubleComplex *d_A, *d_B;
    magma_int_t M, N, size, lda, ldda, ldb, lddb;
    magma_int_t ione     = 1;
    
    magma_opts opts;
    parse_opts( argc, argv, &opts );
    
    printf("Inplace transpose requires M==N.\n");
    printf("    M     N   CPU GByte/s (sec)   GPU GByte/s (sec) check   Inplace GB/s (sec) check\n");
    printf("====================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            lda    = M;
            ldda   = ((M+31)/32)*32;
            ldb    = N;
            lddb   = ((N+31)/32)*32;
            // load entire matrix, save entire matrix
            gbytes = sizeof(magmaDoubleComplex) * 2.*M*N / 1e9;
            
            TESTING_MALLOC_CPU( h_A, magmaDoubleComplex, lda*N  );  // input:  M x N
            TESTING_MALLOC_CPU( h_B, magmaDoubleComplex, ldb*M  );  // output: N x M
            TESTING_MALLOC_CPU( h_R, magmaDoubleComplex, ldb*M  );  // output: N x M
            
            TESTING_MALLOC_DEV( d_A, magmaDoubleComplex, ldda*N );  // input:  M x N
            TESTING_MALLOC_DEV( d_B, magmaDoubleComplex, lddb*M );  // output: N x M
            
            /* Initialize the matrix */
            for( int j = 0; j < N; ++j ) {
                for( int i = 0; i < M; ++i ) {
                    h_A[i + j*lda] = MAGMA_Z_MAKE( i + j/10000., j );
                }
            }
            for( int j = 0; j < M; ++j ) {
                for( int i = 0; i < N; ++i ) {
                    h_B[i + j*ldb] = MAGMA_Z_MAKE( i + j/10000., j );
                }
            }
            magma_zsetmatrix( N, M, h_B, ldb, d_B, lddb );
            
            /* =====================================================================
               Performs operation using naive out-of-place algorithm
               (LAPACK doesn't implement transpose)
               =================================================================== */
            cpu_time = magma_wtime();
            //for( int j = 1; j < N-1; ++j ) {      // inset by 1 row & col
            //    for( int i = 1; i < M-1; ++i ) {  // inset by 1 row & col
            for( int j = 0; j < N; ++j ) {
                for( int i = 0; i < M; ++i ) {
                    h_B[j + i*ldb] = h_A[i + j*lda];
                }
            }
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gbytes / cpu_time;
            
            /* ====================================================================
               Performs operation using MAGMA, out-of-place
               =================================================================== */
            magma_zsetmatrix( M, N, h_A, lda, d_A, ldda );
            magma_zsetmatrix( N, M, h_B, ldb, d_B, lddb );
            
            gpu_time = magma_sync_wtime( 0 );
            //magmablas_ztranspose2( d_B+1+lddb, lddb, d_A+1+ldda, ldda, M-2, N-2 );  // inset by 1 row & col
            magmablas_ztranspose2( d_B, lddb, d_A, ldda, M, N );
            gpu_time = magma_sync_wtime( 0 ) - gpu_time;
            gpu_perf = gbytes / gpu_time;
            
            /* ====================================================================
               Performs operation using MAGMA, in-place
               =================================================================== */
            if ( M == N ) {
                magma_zsetmatrix( M, N, h_A, lda, d_A, ldda );
                
                gpu_time2 = magma_sync_wtime( 0 );
                //magmablas_ztranspose_inplace( N-2, d_A+1+ldda, ldda );  // inset by 1 row & col
                magmablas_ztranspose_inplace( N, d_A, ldda );
                gpu_time2 = magma_sync_wtime( 0 ) - gpu_time2;
                gpu_perf2 = gbytes / gpu_time2;
            }
            
            /* =====================================================================
               Check the result
               =================================================================== */
            size = ldb*M;
            magma_zgetmatrix( N, M, d_B, lddb, h_R, ldb );
            blasf77_zaxpy( &size, &c_neg_one, h_B, &ione, h_R, &ione );
            error = lapackf77_zlange("f", &N, &M, h_R, &ldb, work );
            
            if ( M == N ) {
                magma_zgetmatrix( N, M, d_A, ldda, h_R, ldb );
                blasf77_zaxpy( &size, &c_neg_one, h_B, &ione, h_R, &ione );
                error2 = lapackf77_zlange("f", &N, &M, h_R, &ldb, work );
    
                printf("%5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)  %4s    %7.2f (%7.2f)  %4s\n",
                       (int) M, (int) N, cpu_perf, cpu_time, gpu_perf, gpu_time,
                       (error  == 0. ? "ok" : "failed"),
                       gpu_perf2, gpu_time2,
                       (error2 == 0. ? "ok" : "failed") );
            }
            else {
                printf("%5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)  %4s      ---   (  ---  )\n",
                       (int) M, (int) N, cpu_perf, cpu_time, gpu_perf, gpu_time,
                       (error  == 0. ? "ok" : "failed") );
            }
            
            TESTING_FREE_CPU( h_A );
            TESTING_FREE_CPU( h_B );
            TESTING_FREE_CPU( h_R );
            
            TESTING_FREE_DEV( d_A );
            TESTING_FREE_DEV( d_B );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    TESTING_FINALIZE();
    return 0;
}
