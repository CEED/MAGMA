/*
 *  -- MAGMA (version 2.0) --
 *     Univ. of Tennessee, Knoxville
 *     Univ. of California, Berkeley
 *     Univ. of Colorado, Denver
 *     @date
 *
 * @precisions normal z -> c d s
 *
 **/
/* includes, system */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <algorithm>

/* includes, project */
#include "flops.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "magma_types.h"
#include "testings.h"


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zpotrf_mgpu
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    /* Constants */
    const magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    const magma_int_t ione = 1;
    
    /* Local variables */
    real_Double_t   gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    double      Anorm, error, work[1];
    magmaDoubleComplex *h_A, *h_R;
    magmaDoubleComplex_ptr d_lA[4] = {NULL, NULL, NULL, NULL};
    magma_int_t N, n2, lda, ldda, info;
    magma_int_t j, k, ngpu0 = 1, ngpu;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t nb, nk, n_local, ldn_local;

    magma_opts opts;
    opts.parse_opts( argc, argv );
    opts.ngpu = std::abs( opts.ngpu );  // always uses multi-GPU code
    ngpu0 = opts.ngpu;

    printf("%% ngpu = %ld, uplo = %s\n", long(opts.ngpu), lapack_uplo_const(opts.uplo) );
    printf("%% N     CPU Gflop/s (sec)   MAGMA Gflop/s (sec)   ||R_magma - R_lapack||_F / ||R_lapack||_F\n");
    printf("%%============================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N   = opts.nsize[itest];
            lda = N;
            n2  = lda*N;
            ldda = magma_roundup( N, opts.align );  // multiple of 32 by default
            gflops = FLOPS_ZPOTRF( N ) / 1e9;

            magma_setdevice(0);
            TESTING_CHECK( magma_zmalloc_cpu( &h_A, n2 ));
            TESTING_CHECK( magma_zmalloc_pinned( &h_R, n2 ));

            nb = magma_get_zpotrf_nb(N);
            if ( ngpu0 > N / nb ) {
                ngpu = N / nb;
                if ( N % nb != 0 ) ngpu++;
                printf( " * too many gpus for the matrix size, using %ld gpus\n", long(ngpu) );
            } else {
                ngpu = ngpu0;
            }

            for (j = 0; j < ngpu; j++) {
                n_local = nb*(N /(nb * ngpu));
                if (j < (N / nb) % ngpu)
                    n_local += nb;
                else if (j == (N / nb) % ngpu)
                    n_local += N % nb;

                ldn_local = magma_roundup( n_local, opts.align );  // multiple of 32 by default
                ldn_local = (ldn_local % 256 == 0) ? ldn_local + 32 : ldn_local;

                magma_setdevice(j);
                TESTING_CHECK( magma_zmalloc( &d_lA[j], ldda * ldn_local ));
            }

            /* Initialize the matrix */
            if (opts.check) {
                lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
                magma_zmake_hpd( N, h_A, lda );
                lapackf77_zlacpy( MagmaFullStr, &N, &N, h_A, &lda, h_R, &lda );
            } else {
                lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
                magma_zmake_hpd( N, h_A, lda );
            }

            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            /* distribute matrix to gpus */
            //magma_zprint( N, N, h_A, lda );
            //if ( opts.uplo == MagmaUpper) {
                for (j = 0; j < N; j += nb) {
                    k = (j / nb) % ngpu;
                    magma_setdevice(k);
                    nk = min(nb, N - j);
                    magma_zsetmatrix( N, nk,
                            h_A + j * lda,                       lda,
                            d_lA[k] + j / (nb * ngpu) * nb * ldda, ldda);
                }
            //} else {
            //}
            
            gpu_time = magma_wtime();
            //magma_zpotrf_mgpu(ngpu, opts.uplo, N, d_lA, ldda, &info);
            magma_zpotrf_mgpu_right(ngpu, opts.uplo, N, d_lA, ldda, &info);
            gpu_time = magma_wtime() - gpu_time;
            if (info != 0) {
                printf("magma_zpotrf_mgpu_right returned error %ld: %s.\n",
                       long(info), magma_strerror( info ));
            }
            gpu_perf = gflops / gpu_time;

            if ( opts.check && info == 0) {
                /* =====================================================================
                   Performs operation using LAPACK
                   =================================================================== */
                cpu_time = magma_wtime();
                lapackf77_zpotrf( lapack_uplo_const(opts.uplo), &N, h_A, &lda, &info );
                //printf( " ==== LAPACK ====\n" );
                //magma_zprint( N, N, h_A, lda );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0) {
                    printf("lapackf77_zpotrf returned error %ld: %s.\n",
                           long(info), magma_strerror( info ));
                }

                /* =====================================================================
                   Check the result compared to LAPACK
                   =================================================================== */
                /* gather matrix from gpus */
                //if ( opts.uplo == MagmaUpper ) {
                    for (j = 0; j < N; j += nb) {
                        k = (j / nb) % ngpu;
                        magma_setdevice(k);
                        nk = min(nb, N - j);
                        magma_zgetmatrix( N, nk,
                                d_lA[k] + j / (nb * ngpu) * nb * ldda, ldda,
                                h_R + j * lda,                        lda );
                    }
                /*} else {
                }*/
                magma_setdevice(0);
                //printf( " ==== MAGMA ====\n" );
                //magma_zprint( N, N, h_R, lda );

                error = safe_lapackf77_zlanhe("f", "L", &N, h_A, &lda, work);
                blasf77_zaxpy(&n2, &c_neg_one, h_A, &ione, h_R, &ione);
                Anorm = safe_lapackf77_zlanhe("f", "L", &N, h_A, &lda, work);
                error = safe_lapackf77_zlanhe("f", "L", &N, h_R, &lda, work)
                      / Anorm;

                printf("%5ld   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e\n",
                        long(N), cpu_perf, cpu_time, gpu_perf, gpu_time, error );
            }
            else {
                printf("%5ld     ---   (  ---  )   %7.2f (%7.2f)     ---  \n",
                        long(N), gpu_perf, gpu_time );
            }

            for (j = 0; j < ngpu; j++) {
                magma_setdevice(j);
                magma_free( d_lA[j] );
            }
            magma_setdevice(0);
            magma_free_cpu( h_A );
            magma_free_pinned( h_R );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    opts.cleanup();
    TESTING_CHECK( magma_finalize() );

    return 0;
}
