/*
 *  -- MAGMA (version 1.1) --
 *     Univ. of Tennessee, Knoxville
 *     Univ. of California, Berkeley
 *     Univ. of Colorado, Denver
 *     November 2011
 *
 * @precisions normal z -> c d s
 * @author Mark Gates
 *
 **/
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


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgemm
*/
int main( int argc, char** argv)
{
    TESTING_CUDA_INIT();

    real_Double_t   gflops, magma_perf, magma_time, cublas_perf, cublas_time, cpu_perf, cpu_time;
    double          magma_error, cublas_error, Cnorm, work[1];
    magma_int_t M, N, K;
    magma_int_t Am, An, Bm, Bn;
    magma_int_t sizeA, sizeB, sizeC;
    magma_int_t lda, ldb, ldc, ldda, lddb, lddc;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    
    cuDoubleComplex *h_A, *h_B, *h_C, *h_Cmagma, *h_Ccublas;
    cuDoubleComplex *d_A, *d_B, *d_C;
    cuDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    cuDoubleComplex alpha = MAGMA_Z_MAKE(  0.29, -0.86 );
    cuDoubleComplex beta  = MAGMA_Z_MAKE( -0.48,  0.38 );
    
    magma_opts opts;
    parse_opts( argc, argv, &opts );
    
    printf("If running lapack (option --lapack), MAGMA and CUBLAS error are both computed\n"
           "relative to CPU BLAS result. Else, MAGMA error is computed relative to CUBLAS result.\n\n"
           "transA = %c, transB = %c\n", opts.transA, opts.transB );
    printf("    M     N     K   MAGMA Gflop/s (ms)  CUBLAS Gflop/s (ms)   CPU Gflop/s (ms)  MAGMA error  CUBLAS error\n");
    printf("=========================================================================================================\n");
    for( int i = 0; i < opts.ntest; ++i ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[i];
            N = opts.nsize[i];
            K = opts.ksize[i];
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
            
            ldda = ((lda+31)/32)*32;
            lddb = ((ldb+31)/32)*32;
            lddc = ((ldc+31)/32)*32;
            
            sizeA = lda*An;
            sizeB = ldb*Bn;
            sizeC = ldc*N;
            
            TESTING_MALLOC( h_A,  cuDoubleComplex, lda*An );
            TESTING_MALLOC( h_B,  cuDoubleComplex, ldb*Bn );
            TESTING_MALLOC( h_C,  cuDoubleComplex, ldc*N  );
            TESTING_MALLOC( h_Cmagma,  cuDoubleComplex, ldc*N  );
            TESTING_MALLOC( h_Ccublas, cuDoubleComplex, ldc*N  );
            
            TESTING_DEVALLOC( d_A, cuDoubleComplex, ldda*An );
            TESTING_DEVALLOC( d_B, cuDoubleComplex, lddb*Bn );
            TESTING_DEVALLOC( d_C, cuDoubleComplex, lddc*N  );
            
            /* Initialize the matrices */
            lapackf77_zlarnv( &ione, ISEED, &sizeA, h_A );
            lapackf77_zlarnv( &ione, ISEED, &sizeB, h_B );
            lapackf77_zlarnv( &ione, ISEED, &sizeC, h_C );
            
            /* =====================================================================
               Performs operation using MAGMA-BLAS
               =================================================================== */
            magma_zsetmatrix( Am, An, h_A, lda, d_A, ldda );
            magma_zsetmatrix( Bm, Bn, h_B, ldb, d_B, lddb );
            magma_zsetmatrix( M, N, h_C, ldc, d_C, lddc );
            
            magma_time = magma_sync_wtime( NULL );
            magmablas_zgemm( opts.transA, opts.transB, M, N, K,
                             alpha, d_A, ldda,
                                    d_B, lddb,
                             beta,  d_C, lddc );
            magma_time = magma_sync_wtime( NULL ) - magma_time;
            magma_perf = gflops / magma_time;
            
            magma_zgetmatrix( M, N, d_C, lddc, h_Cmagma, ldc );
            
            /* =====================================================================
               Performs operation using CUDA-BLAS
               =================================================================== */
            magma_zsetmatrix( M, N, h_C, ldc, d_C, lddc );
            
            cublas_time = magma_sync_wtime( NULL );
            cublasZgemm( opts.transA, opts.transB, M, N, K,
                         alpha, d_A, ldda,
                                d_B, lddb,
                         beta,  d_C, lddc );
            cublas_time = magma_sync_wtime( NULL ) - cublas_time;
            cublas_perf = gflops / cublas_time;
            
            magma_zgetmatrix( M, N, d_C, lddc, h_Ccublas, ldc );
            
            /* =====================================================================
               Performs operation using CPU BLAS
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                blasf77_zgemm( &opts.transA, &opts.transB, &M, &N, &K,
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
                // compute relative error for both magma & cublas, relative to lapack,
                // |C_magma - C_lapack| / |C_lapack|
                Cnorm = lapackf77_zlange( "M", &M, &N, h_C, &ldc, work );
                
                blasf77_zaxpy( &sizeC, &c_neg_one, h_C, &ione, h_Cmagma, &ione );
                magma_error = lapackf77_zlange( "M", &M, &N, h_Cmagma, &ldc, work ) / Cnorm;
                
                blasf77_zaxpy( &sizeC, &c_neg_one, h_C, &ione, h_Ccublas, &ione );
                cublas_error = lapackf77_zlange( "M", &M, &N, h_Ccublas, &ldc, work ) / Cnorm;
                
                printf("%5d %5d %5d   %7.2f (%7.2f)    %7.2f (%7.2f)   %7.2f (%7.2f)    %8.2e     %8.2e\n",
                       (int) M, (int) N, (int) K,
                       magma_perf,  1000.*magma_time,
                       cublas_perf, 1000.*cublas_time,
                       cpu_perf,    1000.*cpu_time,
                       magma_error, cublas_error );
            }
            else {
                // compute relative error for magma, relative to cublas
                Cnorm = lapackf77_zlange( "M", &M, &N, h_Ccublas, &ldc, work );
                
                blasf77_zaxpy( &sizeC, &c_neg_one, h_Ccublas, &ione, h_Cmagma, &ione );
                magma_error = lapackf77_zlange( "M", &M, &N, h_Cmagma, &ldc, work ) / Cnorm;
                
                printf("%5d %5d %5d   %7.2f (%7.2f)    %7.2f (%7.2f)     ---   (  ---  )    %8.2e     ---\n",
                       (int) M, (int) N, (int) K,
                       magma_perf,  1000.*magma_time,
                       cublas_perf, 1000.*cublas_time,
                       magma_error );
            }
            
            TESTING_FREE( h_A  );
            TESTING_FREE( h_B  );
            TESTING_FREE( h_C  );
            TESTING_FREE( h_Cmagma  );
            TESTING_FREE( h_Ccublas );
            
            TESTING_DEVFREE( d_A );
            TESTING_DEVFREE( d_B );
            TESTING_DEVFREE( d_C );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    TESTING_CUDA_FINALIZE();
    return 0;
}
