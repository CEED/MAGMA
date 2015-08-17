/*
   -- MAGMA (version 1.5) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date

   @author Azzam Haidar

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

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgetrf_batched
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t   gflops, gpu_perf, gpu_time, cpu_perf=0, cpu_time=0;
    magmaDoubleComplex *h_A, *h_R;
    magmaDoubleComplex *d_A, *d_invA;
    magmaDoubleComplex **dA_array = NULL;
    magmaDoubleComplex **dinvA_array = NULL;
    magmaDoubleComplex **C_array = NULL;
    magma_int_t  **dipiv_array = NULL;
    magma_int_t *dinfo_array = NULL;

    magma_int_t     *ipiv, *cpu_info;
    magma_int_t     *d_ipiv, *d_info;
    magma_int_t N, n2, lda, ldda, info, info1, info2;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    
    magma_opts opts( MagmaOptsBatched );
    opts.parse_opts( argc, argv );
    opts.lapack |= opts.check;

    magma_int_t batchCount = opts.batchcount;
    magma_int_t columns;
    double  error  = 0.0, rwork[1];
    double *norm_A = NULL;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magma_int_t     status = 0;
    // need looser bound (3000*eps instead of 30*eps) for tests
    // TODO: should compute ||I - A*A^{-1}|| / (n*||A||*||A^{-1}||)
    opts.tolerance = max( 3000., opts.tolerance );
    double tol = opts.tolerance * lapackf77_dlamch("E");

    printf("%% batchCount    N     N     CPU GFlop/s (ms)    GPU GFlop/s (ms)    ||R||_F / (N*||A||_F)     )\n");
    printf("%%=============================================================================================\n");
    for( magma_int_t i = 0; i < opts.ntest; ++i ) {    
        for( magma_int_t iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[i];
            lda    = N;
            n2     = lda*N * batchCount;
            ldda   = magma_roundup( N, opts.align );  // multiple of 32 by default
            //gflops = (FLOPS_ZGETRF( N, N ) + FLOPS_ZGETRI( N ))/ 1e9 * batchCount; // This is the correct flops but since this getri_batched is based on 2 trsm = getrs and to know the real flops I am using the getrs one
            gflops = (FLOPS_ZGETRF( N, N ) + FLOPS_ZGETRS( N, N ))/ 1e9 * batchCount;

            TESTING_MALLOC_CPU( cpu_info, magma_int_t, batchCount);
            TESTING_MALLOC_CPU(  ipiv, magma_int_t,     N * batchCount);
            TESTING_MALLOC_CPU(  h_A,  magmaDoubleComplex, n2     );
            TESTING_MALLOC_PIN(  h_R,  magmaDoubleComplex, n2     );
            TESTING_MALLOC_DEV(  d_A,  magmaDoubleComplex, ldda*N * batchCount);
            TESTING_MALLOC_DEV(  d_invA,  magmaDoubleComplex, ldda*N * batchCount);
            TESTING_MALLOC_DEV(  d_ipiv,  magma_int_t, N * batchCount);
            TESTING_MALLOC_DEV(  d_info,  magma_int_t, batchCount);

            magma_malloc((void**)&dA_array, batchCount * sizeof(*dA_array));
            magma_malloc((void**)&dinvA_array, batchCount * sizeof(*dinvA_array));
            magma_malloc((void**)&dinfo_array, batchCount * sizeof(magma_int_t));
            magma_malloc((void**)&C_array, batchCount * sizeof(*C_array));
            magma_malloc((void**)&dipiv_array, batchCount * sizeof(*dipiv_array));

            /* Initialize the matrix */
            lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
            columns = N * batchCount;
            lapackf77_zlacpy( MagmaUpperLowerStr, &N, &columns, h_A, &lda, h_R, &lda );
            magma_zsetmatrix( N, columns, h_R, lda, d_A, ldda );

            if ( opts.check ) {
                TESTING_MALLOC_CPU( norm_A, double, batchCount);
                for (magma_int_t i=0; i < batchCount; i++)
                {
                    norm_A[i] = lapackf77_zlange( "f", &N, &N, h_A+ i * lda*N, &lda, rwork );
                }
            }

            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            zset_pointer(dA_array, d_A, ldda, 0, 0, ldda * N, batchCount, opts.queue);
            zset_pointer(dinvA_array, d_invA, ldda, 0, 0, ldda * N, batchCount, opts.queue);
            set_ipointer(dipiv_array, d_ipiv, 1, 0, 0, N, batchCount, opts.queue);

            gpu_time = magma_sync_wtime( opts.queue );
            info1 = magma_zgetrf_batched( N, N, dA_array, ldda, dipiv_array, dinfo_array, batchCount, opts.queue);
            info2 = magma_zgetri_outofplace_batched( N, dA_array, ldda, dipiv_array, dinvA_array, ldda, dinfo_array, batchCount, opts.queue);
            gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;
            gpu_perf = gflops / gpu_time;

            // check correctness of results throught "dinfo_magma" and correctness of argument throught "info"
            magma_getvector( batchCount, sizeof(magma_int_t), dinfo_array, 1, cpu_info, 1);
            for (magma_int_t i=0; i < batchCount; i++)
            {
                if (cpu_info[i] != 0 ) {
                    printf("magma_zgetrf_batched matrix %d returned error %d\n", i, (int)cpu_info[i] );
                }
            }
            if (info1 != 0) printf("magma_zgetrf_batched returned argument error %d: %s.\n", (int) info1, magma_strerror( info1 ));
            if (info2 != 0) printf("magma_zgetri_batched returned argument error %d: %s.\n", (int) info2, magma_strerror( info2 ));
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                // query for workspace size
                magmaDoubleComplex *work;
                magmaDoubleComplex tmp;
                magma_int_t lwork = -1;
                lapackf77_zgetri( &N, NULL, &lda, NULL, &tmp, &lwork, &info );
                if (info != 0)
                    printf("lapackf77_zgetri returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                lwork = magma_int_t( MAGMA_Z_REAL( tmp ));
                TESTING_MALLOC_CPU( work,  magmaDoubleComplex, lwork  );
                lapackf77_zlacpy( MagmaUpperLowerStr, &N, &columns, h_R, &lda, h_A, &lda );
                cpu_time = magma_wtime();
                #if !defined (BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
                magma_int_t nthreads = magma_get_lapack_numthreads();
                magma_set_lapack_numthreads(1);
                magma_set_omp_numthreads(nthreads);
                #pragma omp parallel for schedule(dynamic)
                #endif
                for (int i=0; i < batchCount; i++)
                {
                    magma_int_t locinfo;
                    lapackf77_zgetrf(&N, &N, h_A + i * lda*N, &lda, ipiv + i * N, &locinfo);
                    if (locinfo != 0)
                        printf("lapackf77_zgetrf returned error %d: %s.\n",
                           (int) locinfo, magma_strerror( locinfo ));
                    lapackf77_zgetri(&N, h_A + i * lda*N, &lda, ipiv + i * N, work, &lwork, &locinfo );
                    if (locinfo != 0)
                        printf("lapackf77_zgetri returned error %d: %s.\n",
                           (int) locinfo, magma_strerror( locinfo ));
                }
                #if !defined (BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
                    magma_set_lapack_numthreads(nthreads);
                #endif
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                printf("%10d %6d %6d   %7.2f (%7.2f)   %7.2f (%7.2f)",
                       (int) batchCount, (int) N, (int) N, cpu_perf, cpu_time*1000., gpu_perf, gpu_time*1000. );
                /* =====================================================================
                   Check the factorization
                   =================================================================== */
                double err = 0.0;
                if ( opts.check ) {
                    magma_getvector( N * batchCount, sizeof(magma_int_t), d_ipiv, 1, ipiv, 1 );
                    magma_zgetmatrix( N, N*batchCount, d_invA, ldda, h_R, lda );
                    magma_int_t stop=0;
                    n2     = lda*N;
                    for (magma_int_t i=0; i < batchCount; i++)
                    {
                        for (magma_int_t k=0; k < N; k++) {
                            if (ipiv[i*N+k] < 1 || ipiv[i*N+k] > N )
                            {
                                printf("error for matrix %d ipiv @ %d = %d\n", (int) i, (int) k, (int) ipiv[i*N+k]);
                                stop=1;
                            }
                        }
                        if (stop == 1) {
                            err=-1.0;
                            break;
                        }
                        blasf77_zaxpy( &n2, &c_neg_one, h_A+ i * lda*N, &ione, h_R+ i * lda*N, &ione );
                        error = lapackf77_zlange( "f", &N, &N, h_R+ i * lda*N, &lda, rwork ) / (N*norm_A[i]);
                        if ( isnan(error) || isinf(error) ) {
                            err = error;
                            break;
                        }
                        err = max(fabs(error), err);
                    }
                    printf("   %18.2e   %10.2e   %s\n", err, (err < tol ? "ok" : "failed") );
                    status += ! (error < tol);
                }
                else {
                    printf("     ---  \n");
                }
            }
            else {
                printf("%10d %6d %6d     ---   (  ---  )   %7.2f (%7.2f)",
                       (int) batchCount, (int) N, (int) N, gpu_perf, gpu_time*1000. );
            }


            TESTING_FREE_CPU( cpu_info );
            TESTING_FREE_CPU( ipiv );
            TESTING_FREE_CPU( h_A );
            TESTING_FREE_PIN( h_R );
            TESTING_FREE_DEV( d_A );
            TESTING_FREE_DEV( d_invA );
            TESTING_FREE_DEV( d_ipiv );
            TESTING_FREE_DEV( d_info );
            TESTING_FREE_DEV( dipiv_array );
            TESTING_FREE_DEV( dA_array );
            TESTING_FREE_DEV( dinfo_array );
            TESTING_FREE_DEV( C_array );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }
    TESTING_FINALIZE();
    return status;
}
