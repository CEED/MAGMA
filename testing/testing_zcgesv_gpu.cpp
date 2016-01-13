/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions mixed zc -> ds
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"


int main(int argc, char **argv)
{
    TESTING_INIT();

    real_Double_t   gflopsF, gflopsS, gpu_perf, gpu_time;
    real_Double_t   gpu_perfdf, gpu_perfds;
    real_Double_t   gpu_perfsf, gpu_perfss;
    double          error, Rnorm, Anorm;
    magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex *h_A, *h_B, *h_X;
    magmaDoubleComplex_ptr d_A, d_B, d_X, d_WORKD;
    magmaFloatComplex  *d_As, *d_Bs, *d_WORKS;
    double          *h_workd;
    magma_int_t *h_ipiv, *d_ipiv;
    magma_int_t lda, ldb, ldx;
    magma_int_t ldda, lddb, lddx;
    magma_int_t N, nrhs, gesv_iter, info, size;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};

    printf("%% Epsilon(double): %8.6e\n"
           "%% Epsilon(single): %8.6e\n\n",
           lapackf77_dlamch("Epsilon"), lapackf77_slamch("Epsilon") );
    magma_int_t status = 0;

    magma_opts opts;
    opts.parse_opts( argc, argv );

    double tol = opts.tolerance * lapackf77_dlamch("E");
    
    nrhs = opts.nrhs;
    
    printf("%% trans = %s\n", lapack_trans_const(opts.transA) );
    printf("%%   N  NRHS   DP-Factor  DP-Solve  SP-Factor  SP-Solve  MP-Solve  Iter   |b-Ax|/N|A|\n");
    printf("%%=========================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            ldb  = ldx = lda = N;
            ldda = magma_roundup( N, opts.align );  // multiple of 32 by default
            lddb = lddx = ldda;
            
            gflopsF = FLOPS_ZGETRF( N, N ) / 1e9;
            gflopsS = gflopsF + FLOPS_ZGETRS( N, nrhs ) / 1e9;

            TESTING_MALLOC_CPU( h_A,     magmaDoubleComplex, lda*N    );
            TESTING_MALLOC_CPU( h_B,     magmaDoubleComplex, ldb*nrhs );
            TESTING_MALLOC_CPU( h_X,     magmaDoubleComplex, ldx*nrhs );
            TESTING_MALLOC_CPU( h_ipiv,  magma_int_t,        N        );
            TESTING_MALLOC_CPU( h_workd, double,             N        );
            
            TESTING_MALLOC_DEV( d_A,     magmaDoubleComplex, ldda*N        );
            TESTING_MALLOC_DEV( d_B,     magmaDoubleComplex, lddb*nrhs     );
            TESTING_MALLOC_DEV( d_X,     magmaDoubleComplex, lddx*nrhs     );
            TESTING_MALLOC_DEV( d_ipiv,  magma_int_t,        N             );
            TESTING_MALLOC_DEV( d_WORKS, magmaFloatComplex,  ldda*(N+nrhs) );
            TESTING_MALLOC_DEV( d_WORKD, magmaDoubleComplex, N*nrhs        );
            
            /* Initialize matrices */
            size = lda * N;
            lapackf77_zlarnv( &ione, ISEED, &size, h_A );
            size = ldb * nrhs;
            lapackf77_zlarnv( &ione, ISEED, &size, h_B );
            lapackf77_zlacpy( MagmaFullStr, &N, &nrhs, h_B, &ldb, h_X, &ldx);
            
            magma_zsetmatrix( N, N,    h_A, lda, d_A, ldda );
            magma_zsetmatrix( N, nrhs, h_B, ldb, d_B, lddb );
            
            //=====================================================================
            //              MIXED - GPU
            //=====================================================================
            gpu_time = magma_wtime();
            magma_zcgesv_gpu( opts.transA, N, nrhs,
                              d_A, ldda, h_ipiv, d_ipiv,
                              d_B, lddb, d_X, lddx,
                              d_WORKD, d_WORKS, &gesv_iter, &info);
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflopsS / gpu_time;
            if (info != 0) {
                printf("magma_zcgesv returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            }
            
            //=====================================================================
            //              ERROR DP vs MIXED  - GPU
            //=====================================================================
            magma_zgetmatrix( N, nrhs, d_X, lddx, h_X, ldx );
            
            Anorm = lapackf77_zlange("I", &N, &N, h_A, &lda, h_workd);
            blasf77_zgemm( lapack_trans_const(opts.transA), MagmaNoTransStr,
                           &N, &nrhs, &N,
                           &c_one,     h_A, &lda,
                                       h_X, &ldx,
                           &c_neg_one, h_B, &ldb);
            Rnorm = lapackf77_zlange("I", &N, &nrhs, h_B, &ldb, h_workd);
            error = Rnorm / (N*Anorm);
            
            //=====================================================================
            //                 Double Precision Factor
            //=====================================================================
            magma_zsetmatrix( N, N, h_A, lda, d_A, ldda );
            
            gpu_time = magma_wtime();
            magma_zgetrf_gpu(N, N, d_A, ldda, h_ipiv, &info);
            gpu_time = magma_wtime() - gpu_time;
            gpu_perfdf = gflopsF / gpu_time;
            if (info != 0) {
                printf("magma_zgetrf returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            }
            
            //=====================================================================
            //                 Double Precision Solve
            //=====================================================================
            magma_zsetmatrix( N, N,    h_A, lda, d_A, ldda );
            magma_zsetmatrix( N, nrhs, h_B, ldb, d_B, lddb );
            
            gpu_time = magma_wtime();
            magma_zgetrf_gpu(N, N, d_A, ldda, h_ipiv, &info);
            magma_zgetrs_gpu( opts.transA, N, nrhs, d_A, ldda, h_ipiv, d_B, lddb, &info );
            gpu_time = magma_wtime() - gpu_time;
            gpu_perfds = gflopsS / gpu_time;
            if (info != 0) {
                printf("magma_zgetrs returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            }
            
            //=====================================================================
            //                 Single Precision Factor
            //=====================================================================
            d_As = d_WORKS;
            d_Bs = d_WORKS + ldda*N;
            magma_zsetmatrix( N, N,    h_A, lda,  d_A,  ldda );
            magma_zsetmatrix( N, nrhs, h_B, ldb,  d_B,  lddb );
            magmablas_zlag2c( N, N,    d_A, ldda, d_As, ldda, &info );
            magmablas_zlag2c( N, nrhs, d_B, lddb, d_Bs, lddb, &info );
            
            gpu_time = magma_wtime();
            magma_cgetrf_gpu(N, N, d_As, ldda, h_ipiv, &info);
            gpu_time = magma_wtime() - gpu_time;
            gpu_perfsf = gflopsF / gpu_time;
            if (info != 0) {
                printf("magma_cgetrf returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            }
            
            //=====================================================================
            //                 Single Precision Solve
            //=====================================================================
            magmablas_zlag2c(N, N,    d_A, ldda, d_As, ldda, &info );
            magmablas_zlag2c(N, nrhs, d_B, lddb, d_Bs, lddb, &info );
            
            gpu_time = magma_wtime();
            magma_cgetrf_gpu( N, N,    d_As, ldda, h_ipiv, &info);
            magma_cgetrs_gpu( opts.transA, N, nrhs, d_As, ldda, h_ipiv,
                              d_Bs, lddb, &info);
            gpu_time = magma_wtime() - gpu_time;
            gpu_perfss = gflopsS / gpu_time;
            if (info != 0) {
                printf("magma_cgetrs returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            }
            
            printf("%5d %5d   %7.2f   %7.2f   %7.2f   %7.2f   %7.2f     %4d   %8.2e   %s\n",
                   (int) N, (int) nrhs,
                   gpu_perfdf, gpu_perfds, gpu_perfsf, gpu_perfss, gpu_perf,
                   (int) gesv_iter, error, (error < tol ? "ok" : "failed"));
            status += ! (error < tol);
            
            TESTING_FREE_CPU( h_A     );
            TESTING_FREE_CPU( h_B     );
            TESTING_FREE_CPU( h_X     );
            TESTING_FREE_CPU( h_ipiv  );
            TESTING_FREE_CPU( h_workd );
            
            TESTING_FREE_DEV( d_A     );
            TESTING_FREE_DEV( d_B     );
            TESTING_FREE_DEV( d_X     );
            TESTING_FREE_DEV( d_ipiv  );
            TESTING_FREE_DEV( d_WORKS );
            TESTING_FREE_DEV( d_WORKD );
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
