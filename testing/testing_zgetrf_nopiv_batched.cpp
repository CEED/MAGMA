/*
   -- MAGMA (version 1.5) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date

   @author Azzam Haidar
   @author Tingxing Dong

   @precisions normal z -> s d c
 */
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

double get_LU_error(magma_int_t M, magma_int_t N,
                    magmaDoubleComplex *A,  magma_int_t lda,
                    magmaDoubleComplex *LU, magma_int_t *IPIV)
{
    magma_int_t min_mn = min(M,N);
    magma_int_t ione   = 1;
    magma_int_t i, j;
    magmaDoubleComplex alpha = MAGMA_Z_ONE;
    magmaDoubleComplex beta  = MAGMA_Z_ZERO;
    magmaDoubleComplex *L, *U;
    double work[1], matnorm, residual;
    
    TESTING_MALLOC_CPU( L, magmaDoubleComplex, M*min_mn);
    TESTING_MALLOC_CPU( U, magmaDoubleComplex, min_mn*N);
    memset( L, 0, M*min_mn*sizeof(magmaDoubleComplex) );
    memset( U, 0, min_mn*N*sizeof(magmaDoubleComplex) );

    lapackf77_zlaswp( &N, A, &lda, &ione, &min_mn, IPIV, &ione);
    lapackf77_zlacpy( MagmaLowerStr, &M, &min_mn, LU, &lda, L, &M      );
    lapackf77_zlacpy( MagmaUpperStr, &min_mn, &N, LU, &lda, U, &min_mn );

    for (j=0; j < min_mn; j++)
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

    TESTING_FREE_CPU(L);
    TESTING_FREE_CPU(U);

    return residual / (matnorm * N);
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgetrf_batched
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t   gflops, magma_perf, magma_time, cublas_perf=0., cublas_time=0., cpu_perf=0, cpu_time=0;
    double          error;
    magma_int_t cublas_enable = 0;
    magmaDoubleComplex *h_A, *h_R;
    magmaDoubleComplex *dA_magma;
    magmaDoubleComplex **dA_array = NULL;

    magma_int_t     **dipiv_array = NULL;
    magma_int_t     *ipiv, *cpu_info;
    magma_int_t     *dipiv_magma, *dinfo_magma;
    
    magma_int_t M, N, n2, lda, ldda, min_mn, info;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t batchCount;
    magma_int_t status = 0;

    magma_opts opts( MagmaOptsBatched );
    opts.parse_opts( argc, argv );
    //opts.lapack |= opts.check;

    batchCount = opts.batchcount;
    magma_int_t columns;
    
    double tol = opts.tolerance * lapackf77_dlamch("E");

    printf("%% BatchCount   M     N    CPU GFlop/s (ms)   MAGMA GFlop/s (ms)   CUBLAS GFlop/s (ms)   ||PA-LU||/(||A||*N)\n");
    printf("%%==========================================================================================================\n");
    for( int i = 0; i < opts.ntest; ++i ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[i];
            N = opts.nsize[i];
            min_mn = min(M, N);
            lda    = M;
            n2     = lda*N * batchCount;
            ldda   = magma_roundup( M, opts.align );  // multiple of 32 by default
            gflops = FLOPS_ZGETRF( M, N ) / 1e9 * batchCount;
            
            TESTING_MALLOC_CPU( cpu_info, magma_int_t, batchCount );
            TESTING_MALLOC_CPU( ipiv, magma_int_t,     min_mn * batchCount );
            TESTING_MALLOC_CPU( h_A,  magmaDoubleComplex, n2 );
            TESTING_MALLOC_CPU( h_R,  magmaDoubleComplex, n2 );
            
            TESTING_MALLOC_DEV( dA_magma,  magmaDoubleComplex, ldda*N * batchCount );
            TESTING_MALLOC_DEV( dipiv_magma,  magma_int_t, min_mn * batchCount );
            TESTING_MALLOC_DEV( dinfo_magma,  magma_int_t, batchCount );

            TESTING_MALLOC_DEV( dA_array,    void*, batchCount );
            TESTING_MALLOC_DEV( dipiv_array, void*, batchCount );

            /* Initialize the matrix */
            lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
            // make it diagonally dominant, to not need pivoting
            for( int s=0; s < batchCount; ++s ) {
                for( int i=0; i < min_mn; ++i ) {
                    h_A[ i + i*lda + s*lda*N ] = MAGMA_Z_MAKE(
                        MAGMA_Z_REAL( h_A[ i + i*lda + s*lda*N ] ) + N,
                        MAGMA_Z_IMAG( h_A[ i + i*lda + s*lda*N ] ));
                }
            }
            columns = N * batchCount;
            lapackf77_zlacpy( MagmaFullStr, &M, &columns, h_A, &lda, h_R, &lda );
            magma_zsetmatrix( M, columns, h_R, lda, dA_magma, ldda );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            zset_pointer(dA_array, dA_magma, ldda, 0, 0, ldda*N, batchCount, opts.queue);
            magma_time = magma_sync_wtime( opts.queue );
            info = magma_zgetrf_nopiv_batched( M, N, dA_array, ldda, dinfo_magma, batchCount, opts.queue);
            magma_time = magma_sync_wtime( opts.queue ) - magma_time;
            magma_perf = gflops / magma_time;
            // check correctness of results throught "dinfo_magma" and correctness of argument throught "info"
            magma_getvector( batchCount, sizeof(magma_int_t), dinfo_magma, 1, cpu_info, 1);
            for (int i=0; i < batchCount; i++)
            {
                if (cpu_info[i] != 0 ) {
                    printf("magma_zgetrf_batched matrix %d returned internal error %d\n", i, (int)cpu_info[i] );
                }
            }
            if (info != 0) {
                printf("magma_zgetrf_batched returned argument error %d: %s.\n",
                        (int) info, magma_strerror( info ));
            }

            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                for (int i=0; i < batchCount; i++) {
                    lapackf77_zgetrf(&M, &N, h_A + i*lda*N, &lda, ipiv + i * min_mn, &info);
                    assert( info == 0 );
                }
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0) {
                    printf("lapackf77_zgetrf returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                }
            }
            
            /* =====================================================================
               Check the factorization
               =================================================================== */
            if ( opts.lapack ) {
                printf("%10d %5d %5d   %7.2f (%7.2f)    %7.2f (%7.2f)     %7.2f (%7.2f)",
                       (int) batchCount, (int) M, (int) N, cpu_perf, cpu_time*1000., magma_perf, magma_time*1000., cublas_perf*cublas_enable, cublas_time*1000.*cublas_enable  );
            }
            else {
                printf("%10d %5d %5d     ---   (  ---  )    %7.2f (%7.2f)     %7.2f (%7.2f)",
                       (int) batchCount, (int) M, (int) N, magma_perf, magma_time*1000., cublas_perf*cublas_enable, cublas_time*1000.*cublas_enable );
            }

            if ( opts.check ) {
                // initialize ipiv to 1, 2, 3, ...
                for (int i=0; i < batchCount; i++)
                {
                    for (int k=0; k < min_mn; k++) {
                        ipiv[i*min_mn+k] = k+1;
                    }
                }

                magma_zgetmatrix( M, N*batchCount, dA_magma, ldda, h_A, lda );
                error = 0;
                for (int i=0; i < batchCount; i++)
                {
                    double err;
                    err = get_LU_error( M, N, h_R + i * lda*N, lda, h_A + i * lda*N, ipiv + i * min_mn);
                    if ( isnan(err) || isinf(err) ) {
                        error = err;
                        break;
                    }
                    error = max( err, error );
                }
                bool okay = (error < tol);
                status += ! okay;
                printf("   %8.2e  %s\n", error, (okay ? "ok" : "failed") );
            }
            else {
                printf("     ---  \n");
            }
            
            TESTING_FREE_CPU( cpu_info );
            TESTING_FREE_CPU( ipiv );
            TESTING_FREE_CPU( h_A );
            TESTING_FREE_CPU( h_R );

            TESTING_FREE_DEV( dA_magma );
            TESTING_FREE_DEV( dinfo_magma );
            TESTING_FREE_DEV( dipiv_magma );
            TESTING_FREE_DEV( dipiv_array );
            TESTING_FREE_DEV( dA_array );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }
    TESTING_FINALIZE();
    return status;
}
