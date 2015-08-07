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

#if defined(_OPENMP)
#include <omp.h>
#include "magma_threadsetting.h"
#endif

double get_LU_error(magma_int_t M, magma_int_t N,
                    magmaDoubleComplex *A,  magma_int_t lda,
                    magmaDoubleComplex *LU, magma_int_t *IPIV)
{
    magma_int_t min_mn = min(M, N);
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

    real_Double_t   gflops, magma_perf, magma_time, cublas_perf=0, cublas_time=0, cpu_perf=0, cpu_time=0;
    double          error=0.0;
    magmaDoubleComplex *h_A, *h_R, *h_Amagma;
    magmaDoubleComplex *dA;
    magmaDoubleComplex **dA_array = NULL;

    magma_int_t     **dipiv_array = NULL;
    magma_int_t     *ipiv, *cpu_info;
    magma_int_t     *dipiv_magma, *dinfo_magma;
    int             *dipiv_cublas, *dinfo_cublas;
    
    magma_int_t M, N, n2, lda, ldda, min_mn, info;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t batchCount;

    magma_opts opts( MagmaOptsBatched );
    opts.parse_opts( argc, argv );
    //opts.lapack |= opts.check;
    magma_int_t     status = 0;
    double tol = opts.tolerance * lapackf77_dlamch("E");

    batchCount = opts.batchcount;
    magma_int_t columns;
    
    printf("%% BatchCount    M     N     CPU GFlop/s (ms)    MAGMA GFlop/s (ms)  CUBLAS GFlop/s (ms)  ||PA-LU||/(||A||*N)\n");
    printf("%%========================================================================\n");
    for( int i = 0; i < opts.ntest; ++i ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[i];
            N = opts.nsize[i];
            min_mn = min(M, N);
            lda    = M;
            n2     = lda*N * batchCount;
            ldda   = magma_roundup( M, opts.align );  // multiple of 32 by default
            gflops = FLOPS_ZGETRF( M, N ) / 1e9 * batchCount;
            
            TESTING_MALLOC_CPU( cpu_info, magma_int_t, batchCount);
            TESTING_MALLOC_CPU(    ipiv, magma_int_t,     min_mn * batchCount);
            TESTING_MALLOC_CPU(    h_A,  magmaDoubleComplex, n2     );
            TESTING_MALLOC_CPU(    h_Amagma,  magmaDoubleComplex, n2     );
            TESTING_MALLOC_PIN(    h_R,  magmaDoubleComplex, n2     );
            
            TESTING_MALLOC_DEV(  dA,  magmaDoubleComplex, ldda*N * batchCount);
            TESTING_MALLOC_DEV(  dipiv_magma,  magma_int_t, min_mn * batchCount);
            TESTING_MALLOC_DEV(  dinfo_magma,  magma_int_t, batchCount);
            TESTING_MALLOC_DEV(  dipiv_cublas,  magma_int_t, min_mn * batchCount);
            TESTING_MALLOC_DEV(  dinfo_cublas,  magma_int_t, batchCount);

            magma_malloc((void**)&dA_array, batchCount * sizeof(*dA_array));
            magma_malloc((void**)&dipiv_array, batchCount * sizeof(*dipiv_array));

            /* Initialize the matrix */
            lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
            columns = N * batchCount;
            lapackf77_zlacpy( MagmaUpperLowerStr, &M, &columns, h_A, &lda, h_R, &lda );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            magma_zsetmatrix( M, columns, h_R, lda, dA, ldda );
            zset_pointer(dA_array, dA, ldda, 0, 0, ldda*N, batchCount, opts.queue);
            set_ipointer(dipiv_array, dipiv_magma, 1, 0, 0, min_mn, batchCount, opts.queue);
            
            magma_time = magma_sync_wtime( opts.queue );
            info = magma_zgetrf_batched( M, N, dA_array, ldda, dipiv_array,  dinfo_magma, batchCount, opts.queue);
            magma_time = magma_sync_wtime( opts.queue ) - magma_time;
            magma_perf = gflops / magma_time;
            
            magma_zgetmatrix( M, N*batchCount, dA, ldda, h_Amagma, lda );
            
            // check correctness of results throught "dinfo_magma" and correctness of argument throught "info"
            magma_getvector( batchCount, sizeof(magma_int_t), dinfo_magma, 1, cpu_info, 1);
            
            for (int i=0; i < batchCount; i++)
            {
                if (cpu_info[i] != 0 ) {
                    printf("magma_zgetrf_batched matrix %d returned internal error %d\n",
                            i, int(cpu_info[i]) );
                }
            }
            
            if (info != 0) {
                printf("magma_zgetrf_batched returned argument error %d: %s.\n",
                        int(info), magma_strerror( info ));
            }
            
            /* ====================================================================
               Performs operation using CUBLAS
               =================================================================== */
            magma_zsetmatrix( M, columns, h_R, lda, dA,  ldda );
            zset_pointer(dA_array, dA, ldda, 0, 0, ldda * N, batchCount, opts.queue);

            cublas_time = magma_sync_wtime( opts.queue );
            if (M == N ) {
                cublasZgetrfBatched( opts.handle, N, dA_array, ldda, dipiv_cublas,  dinfo_cublas, batchCount);
            }
            else {
                printf("M != N, CUBLAS required M == N; CUBLAS is disabled\n");
            }
            cublas_time = magma_sync_wtime( opts.queue ) - cublas_time;
            cublas_perf = gflops / cublas_time;
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                // #define BATCHED_DISABLE_PARCPU
                #if !defined (BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
                magma_int_t nthreads = magma_get_lapack_numthreads();
                magma_set_lapack_numthreads(1);
                magma_set_omp_numthreads(nthreads);
                #pragma omp parallel for schedule(dynamic)
                #endif
                for (magma_int_t s=0; s < batchCount; s++)
                {
                    magma_int_t locinfo;
                    lapackf77_zgetrf(&M, &N, h_A + s * lda * N, &lda, ipiv + s * min_mn, &locinfo);
                    if (locinfo != 0)
                        printf("lapackf77_zgetrf matrix %d returned err %d: %s.\n", (int) s, (int) locinfo, magma_strerror( locinfo ));
                }
                #if !defined (BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
                    magma_set_lapack_numthreads(nthreads);
                #endif
                
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
            }
            
            /* =====================================================================
               Check the factorization
               =================================================================== */
            if ( opts.lapack ) {
                printf("%10d   %5d  %5d     %7.2f (%7.2f)   %7.2f (%7.2f)    %7.2f (%7.2f)",
                       int(batchCount), int(M), int(N),
                       cpu_perf, cpu_time*1000.,
                       magma_perf, magma_time*1000.,
                       cublas_perf, cublas_time*1000.  );
            }
            else {
                printf("%10d   %5d  %5d     ---   (  ---  )   %7.2f (%7.2f)    %7.2f (%7.2f)",
                       int(batchCount), int(M), int(N),
                       magma_perf, magma_time*1000.,
                       cublas_perf, cublas_time*1000. );
            }

            double err = 0.0;
            if ( opts.check ) {
                magma_getvector( min_mn * batchCount, sizeof(magma_int_t), dipiv_magma, 1, ipiv, 1 );
                int stop=0;
                for (int i=0; i < batchCount; i++) {
                    for (int k=0; k < min_mn; k++) {
                        if (ipiv[i*min_mn+k] < 1 || ipiv[i*min_mn+k] > M ) {
                            printf("error for matrix %d ipiv @ %d = %d\n", i, k, int(ipiv[i*min_mn+k]));
                            stop = 1;
                        }
                    }
                    if (stop == 1) {
                        err = -1.0;
                        break;
                    }
                    
                    error = get_LU_error( M, N, h_R + i * lda*N, lda, h_Amagma + i * lda*N, ipiv + i * min_mn);
                    if ( isnan(error) || isinf(error) ) {
                        err = error;
                        break;
                    }
                    err = max(fabs(error), err);
                }
                printf("   %8.2e   %s\n", err, (err < tol ? "ok" : "failed") );
                status += ! (error < tol);
            }
            else {
                printf("     ---  \n");
            }
            
            TESTING_FREE_CPU( cpu_info );
            TESTING_FREE_CPU( ipiv );
            TESTING_FREE_CPU( h_A );
            TESTING_FREE_CPU( h_Amagma );
            TESTING_FREE_PIN( h_R );

            TESTING_FREE_DEV( dA );
            TESTING_FREE_DEV( dinfo_magma );
            TESTING_FREE_DEV( dipiv_magma );
            TESTING_FREE_DEV( dipiv_cublas );
            TESTING_FREE_DEV( dinfo_cublas );
            TESTING_FREE_DEV( dipiv_array );
            TESTING_FREE_DEV( dA_array );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }
    TESTING_FINALIZE();
    return status;
}
