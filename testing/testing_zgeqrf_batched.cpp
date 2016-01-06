/*
    -- MAGMA (version 1.6) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Tingxing Dong
       @author Azzam Haidar

       @precisions normal z -> s d c

*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda.h>  // for CUDA_VERSION

// includes, project
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"
#include "batched_kernel_param.h"

#if defined(_OPENMP)
#include <omp.h>
#include "magma_threadsetting.h"
#endif

void get_QR_error(magma_int_t M, magma_int_t N, magma_int_t min_mn,
                    magmaDoubleComplex *h_R,  magmaDoubleComplex *h_A, magma_int_t lda,
                    magmaDoubleComplex *tau,
                    magmaDoubleComplex *Q,  magma_int_t ldq,
                    magmaDoubleComplex *R,  magma_int_t ldr,
                    magmaDoubleComplex *h_work,  magma_int_t lwork,
                    double *work, double *error, double *error2)
{
    /* h_R:input the factorized matrix by lapack QR,
       h_A:input the original matrix copy
       tau: input
    */
    
    const double             d_neg_one = MAGMA_D_NEG_ONE;
    const double             d_one     = MAGMA_D_ONE;
    const magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    const magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    const magmaDoubleComplex c_zero    = MAGMA_Z_ZERO;
    double           Anorm;
    
    magma_int_t info;
    
    // generate M by K matrix Q, where K = min(M,N)
    lapackf77_zlacpy( "Lower", &M, &min_mn, h_R, &lda, Q, &ldq );
    lapackf77_zungqr( &M, &min_mn, &min_mn, Q, &ldq, tau, h_work, &lwork, &info );
    assert( info == 0 );
    
    // copy K by N matrix R
    lapackf77_zlaset( "Lower", &min_mn, &N, &c_zero, &c_zero, R, &ldr );
    lapackf77_zlacpy( "Upper", &min_mn, &N, h_R, &lda,        R, &ldr );
    
    // error = || R - Q^H*A || / (N * ||A||)
    blasf77_zgemm( "Conj", "NoTrans", &min_mn, &N, &M,
    &c_neg_one, Q, &ldq, h_A, &lda, &c_one, R, &ldr );
    
    Anorm = lapackf77_zlange( "1", &M,      &N, h_A, &lda, work );
    *error = lapackf77_zlange( "1", &min_mn, &N, R,   &ldr, work );
    
    if ( N > 0 && Anorm > 0 )
        *error /= (N*Anorm);
    
    // set R = I (K by K identity), then R = I - Q^H*Q
    // error = || I - Q^H*Q || / N
    lapackf77_zlaset( "Upper", &min_mn, &min_mn, &c_zero, &c_one, R, &ldr );
    blasf77_zherk( "Upper", "Conj", &min_mn, &M, &d_neg_one, Q, &ldq, &d_one, R, &ldr );
    *error2 = safe_lapackf77_zlanhe( "1", "Upper", &min_mn, R, &ldr, work );
    if ( N > 0 )
        *error2 /= N;
}


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgeqrf_batched
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t    gflops, magma_perf, magma_time, cublas_perf=0, cublas_time=0, cpu_perf, cpu_time;
    double           magma_error, cublas_error, magma_error2, cublas_error2;

    magmaDoubleComplex *h_A, *h_R, *h_Amagma, *tau, *h_work, tmp[1];
    magmaDoubleComplex *d_A, *dtau_magma, *dtau_cublas;

    magmaDoubleComplex **dA_array = NULL;
    magmaDoubleComplex **dtau_array = NULL;

    magma_int_t   *dinfo_magma, *dinfo_cublas;

    magma_int_t M, N, lda, ldda, lwork, n2, info, min_mn;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t status = 0;

    magma_int_t batchCount;
    magma_int_t column;

    magma_opts opts( MagmaOptsBatched );
    opts.parse_opts( argc, argv );
    batchCount = opts.batchcount;

    double tol = opts.tolerance * lapackf77_dlamch("E");
    
    printf("%% BatchCount   M     N   MAGMA GFlop/s (ms)   CUBLAS GFlop/s (ms)    CPU GFlop/s (ms)   |R - Q^H*A|_mag   |I - Q^H*Q|_mag   |R - Q^H*A|_cub   |I - Q^H*Q|_cub\n");
    printf("%%============================================================================================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M     = opts.msize[itest];
            N     = opts.nsize[itest];
            min_mn = min(M, N);
            lda    = M;
            n2     = lda*N * batchCount;
            ldda = M;
            ldda   = magma_roundup( M, opts.align );  // multiple of 32 by default

            gflops = (FLOPS_ZGEQRF( M, N ) + FLOPS_ZGEQRT( M, N )) / 1e9 * batchCount;

            /* Allocate memory for the matrix */
            TESTING_MALLOC_CPU( tau,   magmaDoubleComplex, min_mn * batchCount );
            TESTING_MALLOC_CPU( h_A,   magmaDoubleComplex, n2     );
            TESTING_MALLOC_CPU( h_Amagma,   magmaDoubleComplex, n2     );
            TESTING_MALLOC_PIN( h_R,   magmaDoubleComplex, n2     );
        
            TESTING_MALLOC_DEV( d_A,   magmaDoubleComplex, ldda*N * batchCount );

            TESTING_MALLOC_DEV( dtau_magma,  magmaDoubleComplex, min_mn * batchCount);
            TESTING_MALLOC_DEV( dtau_cublas,  magmaDoubleComplex, min_mn * batchCount);

            TESTING_MALLOC_DEV(  dinfo_magma,  magma_int_t, batchCount);
            TESTING_MALLOC_DEV(  dinfo_cublas,  magma_int_t, batchCount);

            magma_malloc((void**)&dA_array, batchCount * sizeof(*dA_array));
            magma_malloc((void**)&dtau_array, batchCount * sizeof(*dtau_array));
        
            // to determine the size of lwork
            lwork = -1;
            lapackf77_zgeqrf(&M, &N, NULL, &M, NULL, tmp, &lwork, &info);
            lwork = (magma_int_t)MAGMA_Z_REAL( tmp[0] );
            lwork = max(lwork, N*N);
           
            TESTING_MALLOC_CPU( h_work, magmaDoubleComplex, lwork * batchCount);

            column = N * batchCount;
            /* Initialize the matrix */
            lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
            lapackf77_zlacpy( MagmaUpperLowerStr, &M, &column, h_A, &lda, h_R, &lda );
       
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            magma_zsetmatrix( M, column, h_R, lda,  d_A, ldda );
            magma_zset_pointer( dA_array, d_A, 1, 0, 0, ldda*N, batchCount, opts.queue );
            magma_zset_pointer( dtau_array, dtau_magma, 1, 0, 0, min_mn, batchCount, opts.queue );
    
            magma_time = magma_sync_wtime( opts.queue );
    
            info = magma_zgeqrf_batched(M, N, dA_array, ldda, dtau_array, dinfo_magma, batchCount, opts.queue);

            magma_time = magma_sync_wtime( opts.queue ) - magma_time;
            magma_perf = gflops / magma_time;

            magma_zgetmatrix( M, column, d_A, ldda, h_Amagma, lda);

            if (info != 0)
                printf("magma_zgeqrf_batched returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            /* ====================================================================
               Performs operation using CUBLAS
               =================================================================== */

            /* cublasZgeqrfBatched is only available from CUBLAS v6.5 */
            #if CUDA_VERSION >= 6050
            magma_zsetmatrix( M, column, h_R, lda,  d_A, ldda );
            magma_zset_pointer( dA_array, d_A, 1, 0, 0, ldda*N, batchCount, opts.queue );
            magma_zset_pointer( dtau_array, dtau_cublas, 1, 0, 0, min_mn, batchCount, opts.queue );

            cublas_time = magma_sync_wtime( opts.queue );
    
            int cublas_info;  // int, not magma_int_t
            cublasZgeqrfBatched( opts.handle, M, N, dA_array, ldda, dtau_array, &cublas_info, batchCount);

            cublas_time = magma_sync_wtime( opts.queue ) - cublas_time;
            cublas_perf = gflops / cublas_time;

            if (cublas_info != 0)
                printf("cublasZgeqrfBatched returned error %d: %s.\n",
                       (int) cublas_info, magma_strerror( cublas_info ));
            #endif

            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.check ) {
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
                    lapackf77_zgeqrf(&M, &N, h_A + s * lda * N, &lda, tau + s * min_mn, h_work + s * lwork, &lwork, &locinfo);
                    if (locinfo != 0)
                        printf("lapackf77_zgeqrf matrix %d returned error %d: %s.\n",
                               (int) s, (int) locinfo, magma_strerror( locinfo ));
                }

                #if !defined (BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
                    magma_set_lapack_numthreads(nthreads);
                #endif
                
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0)
                    printf("lapackf77_zgeqrf returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                
                /* =====================================================================
                   Check the MAGMA CUBLAS result compared to LAPACK
                   =================================================================== */
                magma_int_t ldq = M;
                magma_int_t ldr = min_mn;
                magmaDoubleComplex *Q, *R;
                double *work;

                TESTING_MALLOC_CPU( Q,    magmaDoubleComplex, ldq*min_mn );  // M by K
                TESTING_MALLOC_CPU( R,    magmaDoubleComplex, ldr*N );       // K by N
                TESTING_MALLOC_CPU( work, double,             min_mn );

                /* check magma result */
                magma_error  = 0;
                magma_error2 = 0;
                magma_zgetvector(min_mn*batchCount, dtau_magma, 1, tau, 1);
                for (int i=0; i < batchCount; i++)
                {
                    double err, err2;
                    get_QR_error(M, N, min_mn,
                             h_Amagma + i*lda*N, h_R + i*lda*N, lda, tau + i*min_mn,
                             Q, ldq, R, ldr, h_work, lwork,
                             work, &err, &err2);

                    if ( isnan(err) || isinf(err) ) {
                        magma_error = err;
                        break;
                    }
                    magma_error  = max( err,  magma_error  );
                    magma_error2 = max( err2, magma_error2 );
                }

                /* check cublas result */
                cublas_error  = 0;
                cublas_error2 = 0;
                #if CUDA_VERSION >= 6050
                magma_zgetvector(min_mn*batchCount, dtau_magma, 1, tau, 1);
                magma_zgetmatrix( M, column, d_A, ldda, h_A, lda);
                for (int i=0; i < batchCount; i++)
                {
                    double err, err2;
                    get_QR_error(M, N, min_mn,
                             h_A + i*lda*N, h_R + i*lda*N, lda, tau + i*min_mn,
                             Q, ldq, R, ldr, h_work, lwork,
                             work, &err, &err2);

                    if ( isnan(err) || isinf(err) ) {
                        cublas_error = err;
                        break;
                    }
                    cublas_error  = max( err,  cublas_error  );
                    cublas_error2 = max( err2, cublas_error2 );
                }
                #endif

                TESTING_FREE_CPU( Q    );  Q    = NULL;
                TESTING_FREE_CPU( R    );  R    = NULL;
                TESTING_FREE_CPU( work );  work = NULL;

                bool okay = (magma_error < tol && magma_error2 < tol);
                //bool okay_cublas = (cublas_error < tol && cublas_error2 < tol);
                status += ! okay;

                printf("%10d %5d %5d    %7.2f (%7.2f)     %7.2f (%7.2f)   %7.2f (%7.2f)   %15.2e   %15.2e   %15.2e   %15.2e   %s\n",
                       (int)batchCount, (int) M, (int) N,
                       magma_perf,  1000.*magma_time,
                       cublas_perf, 1000.*cublas_time,
                       cpu_perf,    1000.*cpu_time,
                       magma_error, magma_error2,
                       cublas_error, cublas_error2,
                       (okay ? "ok" : "failed") );
            }
            else {
                printf("%10d %5d %5d    %7.2f (%7.2f)     %7.2f (%7.2f)     ---   (  ---  )   ---\n",
                       (int)batchCount, (int) M, (int) N,
                       magma_perf,  1000.*magma_time,
                       cublas_perf, 1000.*cublas_time );
            }
            
            TESTING_FREE_CPU( tau    );
            TESTING_FREE_CPU( h_A    );
            TESTING_FREE_CPU( h_Amagma);
            TESTING_FREE_CPU( h_work );
            TESTING_FREE_PIN( h_R    );
        
            TESTING_FREE_DEV( d_A   );
            TESTING_FREE_DEV( dtau_magma  );
            TESTING_FREE_DEV( dtau_cublas );

            TESTING_FREE_DEV( dinfo_magma );
            TESTING_FREE_DEV( dinfo_cublas );

            TESTING_FREE_DEV( dA_array   );
            TESTING_FREE_DEV( dtau_array  );

            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }
    
    opts.cleanup();
    TESTING_FINALIZE();
    return status;
}
