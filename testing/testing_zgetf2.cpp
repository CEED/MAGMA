/*
 *  -- MAGMA (version 1.1) --
 *     Univ. of Tennessee, Knoxville
 *     Univ. of California, Berkeley
 *     Univ. of Colorado, Denver
 *     November 2011
 *
 * @precisions normal z -> c d s
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

int magma_zgetf2(int m, int n, cuDoubleComplex *A, int lda, int *ipiv, int* info);

double get_LU_error(magma_int_t M, magma_int_t N,
                    cuDoubleComplex *A,  magma_int_t lda,
                    cuDoubleComplex *LU, magma_int_t *IPIV)
{
    magma_int_t min_mn = min(M,N);
    magma_int_t ione   = 1;
    magma_int_t i, j;
    cuDoubleComplex alpha = MAGMA_Z_ONE;
    cuDoubleComplex beta  = MAGMA_Z_ZERO;
    cuDoubleComplex *L, *U;
    double work[1], matnorm, residual;
    
    TESTING_MALLOC( L, cuDoubleComplex, M*min_mn);
    TESTING_MALLOC( U, cuDoubleComplex, min_mn*N);
    memset( L, 0, M*min_mn*sizeof(cuDoubleComplex) );
    memset( U, 0, min_mn*N*sizeof(cuDoubleComplex) );

    lapackf77_zlaswp( &N, A, &lda, &ione, &min_mn, IPIV, &ione);
    lapackf77_zlacpy( MagmaLowerStr, &M, &min_mn, LU, &lda, L, &M      );
    lapackf77_zlacpy( MagmaUpperStr, &min_mn, &N, LU, &lda, U, &min_mn );

    for(j=0; j<min_mn; j++)
        L[j+j*M] = MAGMA_Z_MAKE( 1., 0. );
    
    matnorm = lapackf77_zlange("f", &M, &N, A, &lda, work);

    blasf77_zgemm("N", "N", &M, &N, &min_mn,
                  &alpha, L, &M, U, &min_mn, &beta, LU, &lda);

    for( j = 0; j < N; j++ ) {
        for( i = 0; i < M; i++ ) {
            LU[i+j*lda] = MAGMA_Z_SUB( LU[i+j*lda], A[i+j*lda] );
        }
    }
    residual = lapackf77_zlange("f", &M, &N, LU, &lda, work);

    TESTING_FREE(L);
    TESTING_FREE(U);

    return residual / (matnorm * N);
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgetrf
*/
int main( int argc, char** argv)
{
    TESTING_CUDA_INIT();

    real_Double_t   gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    double          error;
    cuDoubleComplex *h_A, *h_R;
    cuDoubleComplex *d_A;
    magma_int_t     *ipiv;
    magma_int_t M, N, n2, lda, ldda, info, min_mn;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};

    magma_opts opts;
    parse_opts( argc, argv, &opts );
    
    printf("  M     N     CPU GFlop/s (sec)   GPU GFlop/s (sec)   ||PA-LU||/(||A||*N)\n");
    printf("=========================================================================\n");
    for( int i = 0; i < opts.ntest; ++i ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[i];
            N = opts.nsize[i];
            min_mn = min(M, N);
            lda    = M;
            n2     = lda*N;
            ldda   = ((M+31)/32)*32;
            gflops = FLOPS_ZGETRF( M, N ) / 1e9;
            
            TESTING_MALLOC(    ipiv, magma_int_t,     min_mn );
            TESTING_MALLOC(    h_A,  cuDoubleComplex, n2     );
            TESTING_HOSTALLOC( h_R,  cuDoubleComplex, n2     );
            TESTING_DEVALLOC(  d_A,  cuDoubleComplex, ldda*N );
            
            /* Initialize the matrix */
            if ( opts.check ) {
                lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
                lapackf77_zlacpy( MagmaUpperLowerStr, &M, &N, h_A, &lda, h_R, &lda );
            }
            else
                lapackf77_zlarnv( &ione, ISEED, &n2, h_R );
            magma_zsetmatrix( M, N, h_R, lda, d_A, ldda );
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.check ) {
                cpu_time = magma_wtime();
                lapackf77_zgetrf(&M, &N, h_A, &lda, ipiv, &info);
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0)
                    printf("lapackf77_zgetrf returned error %d.\n", (int) info);
            }
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            magma_zgetf2( M, N, d_A, ldda, ipiv, &info);
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0)
                printf("magma_zgetrf_gpu returned error %d.\n", (int) info);
            
            /* =====================================================================
               Check the factorization
               =================================================================== */
            if ( opts.check ) {
                magma_zgetmatrix( M, N, d_A, ldda, h_A, lda );
                error = get_LU_error(M, N, h_R, lda, h_A, ipiv);
                
                printf("%5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e\n",
                       (int) M, (int) N, cpu_perf, cpu_time, gpu_perf, gpu_time, error);
            }
            else {
                printf("%5d %5d     ---   (  ---  )   %7.2f (%7.2f)     ---  \n",
                       (int) M, (int) N, gpu_perf, gpu_time);
            }
            
            TESTING_FREE( ipiv );
            TESTING_FREE( h_A );
            TESTING_HOSTFREE( h_R );
            TESTING_DEVFREE( d_A );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    TESTING_CUDA_FINALIZE();
    return 0;
}
