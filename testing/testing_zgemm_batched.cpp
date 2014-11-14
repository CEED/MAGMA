/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
       @author Mark Gates
       @author Azzam Haidar
       @author Tingxing Dong
*/


// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include "common_magma.h"

// includes, project
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgemm_batched
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t   gflops, magma_perf, magma_time, cublas_perf, cublas_time, cpu_perf, cpu_time;
    double          magma_error, cublas_error, magma_err, cublas_err, Cnorm, work[1];
    magma_int_t M, N, K;
    magma_int_t Am, An, Bm, Bn;
    magma_int_t sizeA, sizeB, sizeC;
    magma_int_t lda, ldb, ldc, ldda, lddb, lddc;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t status = 0;
    magma_int_t NN;
    magma_int_t batchCount;

    magmaDoubleComplex *h_A, *h_B, *h_C, *h_Cmagma, *h_Ccublas;
    magmaDoubleComplex *d_A, *d_B, *d_C;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex alpha = MAGMA_Z_MAKE(  0.29, -0.86 );
    magmaDoubleComplex beta  = MAGMA_Z_MAKE( -0.48,  0.38 );
    magmaDoubleComplex **A_array = NULL;
    magmaDoubleComplex **B_array = NULL;
    magmaDoubleComplex **C_array = NULL;


    magma_opts opts;
    parse_opts( argc, argv, &opts );
    batchCount = opts.batchcount;
    cublasHandle_t handle = opts.handle;

    //double tol = opts.tolerance * lapackf77_dlamch("E");
    
    printf("If running lapack (option --lapack), MAGMA and CUBLAS error are both computed\n"
           "relative to CPU BLAS result. Else, MAGMA error is computed relative to CUBLAS result.\n\n"
           "transA = %s, transB = %s\n", 
           lapack_trans_const(opts.transA),
           lapack_trans_const(opts.transB));
    printf("BatchCount    M     N     K   MAGMA Gflop/s (ms)  CUBLAS Gflop/s (ms)  CPU Gflop/s (ms)  MAGMA error  CUBLAS error\n");
    printf("=========================================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            K = opts.ksize[itest];
            gflops = FLOPS_ZGEMM( M, N, K ) / 1e9 * batchCount;

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
            
            NN = N * batchCount;

            lda = ((lda+31)/32)*32;
            ldb = ((ldb+31)/32)*32;
            ldc = ((ldc+31)/32)*32;

            ldda = lda;
            lddb = ldb;
            lddc = ldc;
            

            sizeA = lda*An*batchCount;
            sizeB = ldb*Bn*batchCount;
            sizeC = ldc*N*batchCount;
            
            TESTING_MALLOC_CPU( h_A,  magmaDoubleComplex, sizeA );
            TESTING_MALLOC_CPU( h_B,  magmaDoubleComplex, sizeB );
            TESTING_MALLOC_CPU( h_C,  magmaDoubleComplex, sizeC  );
            TESTING_MALLOC_CPU( h_Cmagma,  magmaDoubleComplex, sizeC  );
            TESTING_MALLOC_CPU( h_Ccublas, magmaDoubleComplex, sizeC  );

            TESTING_MALLOC_DEV( d_A, magmaDoubleComplex, sizeA );
            TESTING_MALLOC_DEV( d_B, magmaDoubleComplex, sizeB );
            TESTING_MALLOC_DEV( d_C, magmaDoubleComplex, sizeC  );

            magma_malloc((void**)&A_array, batchCount * sizeof(*A_array));
            magma_malloc((void**)&B_array, batchCount * sizeof(*B_array));
            magma_malloc((void**)&C_array, batchCount * sizeof(*C_array));

            /* Initialize the matrices */
            lapackf77_zlarnv( &ione, ISEED, &sizeA, h_A );
            lapackf77_zlarnv( &ione, ISEED, &sizeB, h_B );
            lapackf77_zlarnv( &ione, ISEED, &sizeC, h_C );
            
            /* =====================================================================
               Performs operation using MAGMABLAS
               =================================================================== */
            magma_zsetmatrix( Am, An*batchCount, h_A, lda, d_A, ldda );
            magma_zsetmatrix( Bm, Bn*batchCount, h_B, ldb, d_B, lddb );
            magma_zsetmatrix( M, N*batchCount, h_C, ldc, d_C, lddc );
            
            zset_pointer(A_array, d_A, ldda, 0, 0, ldda*An, batchCount);
            zset_pointer(B_array, d_B, lddb, 0, 0, lddb*Bn, batchCount);
            zset_pointer(C_array, d_C, lddc, 0, 0, lddc*N,  batchCount);

            sleep(1);
            magma_time = magma_sync_wtime( NULL );
            magmablas_zgemm_batched(opts.transA, opts.transB, M, N, K,
                             alpha, A_array, ldda,
                                    B_array, lddb,
                             beta,  C_array, lddc, batchCount);
            magma_time = magma_sync_wtime( NULL ) - magma_time;
            magma_perf = gflops / magma_time;            
            magma_zgetmatrix( M, N*batchCount, d_C, lddc, h_Cmagma, ldc );
            
            /* =====================================================================
               Performs operation using CUBLAS
               =================================================================== */

            magma_zsetmatrix( M, N*batchCount, h_C, ldc, d_C, lddc );
            
            cublas_time = magma_sync_wtime( NULL );

            cublasZgemmBatched(handle, cublas_trans_const(opts.transA), cublas_trans_const(opts.transB), M, N, K,
                               &alpha, (const magmaDoubleComplex**) A_array, ldda,
                               (const magmaDoubleComplex**) B_array, lddb,
                               &beta,  C_array, lddc, batchCount );

            cublas_time = magma_sync_wtime( NULL ) - cublas_time;
            cublas_perf = gflops / cublas_time;
            
            magma_zgetmatrix( M, N*batchCount, d_C, lddc, h_Ccublas, ldc );
          
            
            /* =====================================================================
               Performs operation using CPU BLAS
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                for(int i=0; i<batchCount; i++)
                {
                   blasf77_zgemm(
                               lapack_trans_const(opts.transA), lapack_trans_const(opts.transB),
                               &M, &N, &K,
                               &alpha, h_A + i*lda*An, &lda,
                                       h_B + i*ldb*Bn, &ldb,
                               &beta,  h_C + i*ldc*N, &ldc );
                }
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
            }
            
            /* =====================================================================
               Check the result
               =================================================================== */
            if ( opts.lapack ) {
                // compute relative error for both magma & cublas, relative to lapack,
                // |C_magma - C_lapack| / |C_lapack|
                magma_error = 0.0;
                cublas_error = 0.0;

                for(int s=0; s<batchCount; s++)
                {
                    magma_int_t C_batchSize = ldc * N;
 
                    Cnorm = lapackf77_zlange( "M", &M, &N, h_C + s*C_batchSize, &ldc, work );

                    blasf77_zaxpy( &C_batchSize, &c_neg_one, h_C + s*C_batchSize, &ione, h_Cmagma + s*C_batchSize, &ione );
                    magma_err = lapackf77_zlange( "M", &M, &N, h_Cmagma + s*C_batchSize, &ldc, work ) / Cnorm; 

                    if ( isnan(magma_err) || isinf(magma_err) ) {
                      magma_error = magma_err;
                      break;
                    }
                    magma_error = max(fabs(magma_err), magma_error); 

                    blasf77_zaxpy( &C_batchSize, &c_neg_one, h_C + s*C_batchSize, &ione, h_Ccublas + s*C_batchSize, &ione );
                    cublas_err = lapackf77_zlange( "M", &M, &N, h_Ccublas + s*C_batchSize, &ldc, work ) / Cnorm; 
                    
                   if ( isnan(cublas_err) || isinf(cublas_err) ) {
                      cublas_error = cublas_err;
                      break;
                    }
                    cublas_error = max(fabs(cublas_err), cublas_error); 

                }

                    printf("%10d %5d %5d %5d  %7.2f (%7.2f)    %7.2f (%7.2f)   %7.2f (%7.2f)      %8.2e     %8.2e  \n",
                       batchCount, (int) M, (int) N, (int) K, 
                       magma_perf,  1000.*magma_time,
                       cublas_perf, 1000.*cublas_time,
                       cpu_perf,    1000.*cpu_time,
                       magma_error, cublas_error);
            }
            else {
                // compute relative error for magma, relative to cublas

                    Cnorm = lapackf77_zlange( "M", &M, &NN, h_Ccublas, &ldc, work );
                    blasf77_zaxpy( &sizeC, &c_neg_one, h_Ccublas, &ione, h_Cmagma, &ione );
                    magma_error = lapackf77_zlange( "M", &M, &NN, h_Cmagma, &ldc, work ) / Cnorm;

                    printf("%10d %5d %5d %5d  %7.2f (%7.2f)    %7.2f (%7.2f)   ---   (  ---  )    %8.2e     ---\n",
                       batchCount, (int) M, (int) N, (int) K,
                       magma_perf,  1000.*magma_time,
                       cublas_perf, 1000.*cublas_time,
                       magma_error );
            }
            
            TESTING_FREE_CPU( h_A  );
            TESTING_FREE_CPU( h_B  );
            TESTING_FREE_CPU( h_C  );
            TESTING_FREE_CPU( h_Cmagma  );
            TESTING_FREE_CPU( h_Ccublas );

            TESTING_FREE_DEV( d_A );
            TESTING_FREE_DEV( d_B );
            TESTING_FREE_DEV( d_C );
            TESTING_FREE_DEV( A_array );
            TESTING_FREE_DEV( B_array );
            TESTING_FREE_DEV( C_array );

            
            fflush( stdout);

        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    TESTING_FINALIZE();
    return status;
}
