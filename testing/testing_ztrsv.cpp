/*
 *  -- MAGMA (version 1.1) --
 *     Univ. of Tennessee, Knoxville
 *     Univ. of California, Berkeley
 *     Univ. of Colorado, Denver
 *     November 2011
 *
 * @precisions normal z -> c d s
 * @author Chongxiao Cao
 *
 **/

// make sure that asserts are enabled
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>

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
   -- Testing ztrsm
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t   gflops, cublas_perf, cublas_time, cpu_perf, cpu_time;
    double          cublas_error, Cnorm, work[1];
    magma_int_t N, info;
    magma_int_t Ak;
    magma_int_t sizeA;
    magma_int_t lda, ldda;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
   
    magma_int_t *piv;
    magma_err_t err;

    cuDoubleComplex *h_A, *h_x, *h_b1, *h_xcublas, *h_x1, *LU, *LUT;
    cuDoubleComplex *d_A, *d_x;
    cuDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    cuDoubleComplex c_one = MAGMA_Z_ONE;
    
    magma_opts opts;
    parse_opts( argc, argv, &opts );
    
    printf("If running lapack (option --lapack), MAGMA and CUBLAS error are both computed\n"
           "relative to CPU BLAS result. Else, MAGMA error is computed relative to CUBLAS result.\n\n"
           "uplo = %c, transA = %c, diag = %c \n", opts.uplo, opts.transA, opts.diag );
    printf("    N   CUBLAS Gflop/s (ms)   CPU Gflop/s (ms)  CUBLAS error\n");
    printf("=================================================================\n");
    for( int i = 0; i < opts.ntest; ++i ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[i];
            gflops = FLOPS_ZTRSM(opts.side, N, 1) / 1e9;

            lda = N;
            Ak = N;
            
            ldda = ((lda+31)/32)*32;
            
            sizeA = lda*Ak;
            
            TESTING_MALLOC( h_A,  cuDoubleComplex, lda*Ak );
            TESTING_MALLOC( LU,      cuDoubleComplex, lda*Ak );
            TESTING_MALLOC( LUT,  cuDoubleComplex, lda*Ak );
            TESTING_MALLOC( h_x,  cuDoubleComplex, N  );
            TESTING_MALLOC( h_x1,  cuDoubleComplex, N );
            TESTING_MALLOC( h_b1,  cuDoubleComplex, N );
            TESTING_MALLOC( h_xcublas, cuDoubleComplex, N  );
            
            TESTING_DEVALLOC( d_A, cuDoubleComplex, ldda*Ak );
            TESTING_DEVALLOC( d_x, cuDoubleComplex, N  );
            
            /* Initialize the matrices */
            lapackf77_zlarnv( &ione, ISEED, &sizeA, LU );
            err = magma_malloc_cpu( (void**) &piv, N*sizeof(magma_int_t) );  assert( err == 0 );
            lapackf77_zgetrf( &Ak, &Ak, LU, &lda, piv, &info );
            
            int i, j;
            for(i=0;i<Ak;i++){
                for(j=0;j<Ak;j++){
                    LUT[j+i*lda] = LU[i+j*lda];
                }
            }

            lapackf77_zlacpy(MagmaUpperStr, &Ak, &Ak, LUT, &lda, LU, &lda);
            
            if(opts.uplo == MagmaLower){
                lapackf77_zlacpy(MagmaLowerStr, &Ak, &Ak, LU, &lda, h_A, &lda);
            }else{
                lapackf77_zlacpy(MagmaUpperStr, &Ak, &Ak, LU, &lda, h_A, &lda);
            }
            
            lapackf77_zlarnv( &ione, ISEED, &N, h_x );
            memcpy(h_b1, h_x, N*sizeof(cuDoubleComplex));
            /* =====================================================================
               Performs operation using CUDA-BLAS
               =================================================================== */
            magma_zsetmatrix( Ak, Ak, h_A, lda, d_A, ldda );
            magma_zsetvector( N, h_x, 1, d_x, 1 );
            
            cublas_time = magma_sync_wtime( NULL );
            cublasZtrsv( opts.uplo, opts.transA, opts.diag,
                         N, 
                         d_A, ldda,
                         d_x, 1 );
            cublas_time = magma_sync_wtime( NULL ) - cublas_time;
            cublas_perf = gflops / cublas_time;
            
            magma_zgetvector( N, d_x, 1, h_xcublas, 1 );
            
            /* =====================================================================
               Performs operation using CPU BLAS
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                blasf77_ztrsv( &opts.uplo, &opts.transA, &opts.diag, 
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
            memcpy(h_x1, h_xcublas, N*sizeof(cuDoubleComplex));
            blasf77_ztrmv( &opts.uplo, &opts.transA, &opts.diag, 
                            &N,
                            h_A, &lda,
                            h_xcublas, &ione );

            blasf77_zaxpy( &N, &c_neg_one, h_b1, &ione, h_xcublas, &ione );
            double norm1 =  lapackf77_zlange( "M", &N, &ione, h_xcublas, &N, work );
            double normx =  lapackf77_zlange( "M", &N, &ione, h_x1, &ione, work );
            double normA =  lapackf77_zlange( "M", &Ak, &Ak, h_A, &lda, work );


            cublas_error = norm1/(normx*normA);

            printf("%5d   %7.2f (%7.2f)    %7.2f (%7.2f) %8.2e\n",
                    (int) N,
                    cublas_perf, 1000.*cublas_time,
                    cpu_perf,    1000.*cpu_time,
                    cublas_error );
            
            TESTING_FREE( h_A  );
            TESTING_FREE( LU  );
            TESTING_FREE( LUT );
            TESTING_FREE( h_x  );
            TESTING_FREE( h_xcublas );
            TESTING_FREE( h_x1 );
            
            TESTING_DEVFREE( d_A );
            TESTING_DEVFREE( d_x );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    TESTING_FINALIZE();
    return 0;
}
