/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    @author Stan Tomov

    @precisions normal z -> c

*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <cublas.h>

// includes, project
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

#define absv(v1) ((v1)>0? (v1): -(v1))

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zheevd
*/
int main( int argc, char** argv)
{
    TESTING_CUDA_INIT();

    real_Double_t   gpu_time, cpu_time;
    cuDoubleComplex *h_A, *h_R, *h_work, aux_work[1];
    double *rwork, *w1, *w2, result[3], eps, aux_rwork[1];
    magma_int_t *iwork, aux_iwork[1];
    magma_int_t N, n2, info, lwork, lrwork, liwork;
    magma_int_t izero    = 0;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    eps = lapackf77_dlamch( "E" );

    magma_opts opts;
    parse_opts( argc, argv, &opts );
    
    if ( opts.check && opts.jobz == MagmaNoVectors ) {
        fprintf( stderr, "checking results requires vectors; setting jobz=V (option -JV)\n" );
        opts.jobz = MagmaVectors;
    }
    
    printf("    N   CPU Time (sec)   GPU Time (sec)\n");
    printf("=======================================\n");
    for( int i = 0; i < opts.ntest; ++i ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[i];
            n2 = N*N;
            
            /* Query for workspace sizes */
            magma_zheevd( opts.jobz, opts.uplo,
                          N, h_R, N, w1,
                          aux_work,  -1,
                          aux_rwork, -1,
                          aux_iwork, -1,
                          &info );
            lwork  = (magma_int_t) MAGMA_Z_REAL( aux_work[0] );
            lrwork = (magma_int_t) aux_rwork[0];
            liwork = aux_iwork[0];
            
            /* Allocate host memory for the matrix */
            TESTING_MALLOC(    h_A, cuDoubleComplex, N*N );
            TESTING_MALLOC(    w1,  double,          N   );
            TESTING_MALLOC(    w2,  double,          N   );
            TESTING_HOSTALLOC( h_R, cuDoubleComplex, N*N );
            TESTING_HOSTALLOC( h_work, cuDoubleComplex, lwork  );
            TESTING_MALLOC(    rwork,  double,          lrwork );
            TESTING_MALLOC(    iwork,  magma_int_t,     liwork );
            
            /* Initialize the matrix */
            lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
            for( int j=0; j < N; j++) {
                h_A[j*N+j] = MAGMA_Z_MAKE( MAGMA_Z_REAL(h_A[j*N+j]), 0. );
            }
            lapackf77_zlacpy( MagmaUpperLowerStr, &N, &N, h_A, &N, h_R, &N );
            
            /* warm up run */
            magma_zheevd( opts.jobz, opts.uplo,
                          N, h_R, N, w1,
                          h_work, lwork,
                          rwork, lrwork,
                          iwork, liwork,
                          &info );
            
            lapackf77_zlacpy( MagmaUpperLowerStr, &N, &N, h_A, &N, h_R, &N );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            magma_zheevd( opts.jobz, opts.uplo,
                          N, h_R, N, w1,
                          h_work, lwork,
                          rwork, lrwork,
                          iwork, liwork,
                          &info );
            gpu_time = magma_wtime() - gpu_time;
            
            if ( opts.check ) {
                /* =====================================================================
                   Check the results following the LAPACK's [zcds]drvst routine.
                   A is factored as A = U S U' and the following 3 tests computed:
                   (1)    | A - U S U' | / ( |A| N )
                   (2)    | I - U'U | / ( N )
                   (3)    | S(with U) - S(w/o U) | / | S |
                   =================================================================== */
                double temp1, temp2;
                
                // tau=NULL is unused since itype=1
                lapackf77_zhet21( &ione, &opts.uplo, &N, &izero,
                                  h_A, &N,
                                  w1, w1,
                                  h_R, &N,
                                  h_R, &N,
                                  NULL, h_work, rwork, &result[0] );
                
                lapackf77_zlacpy( MagmaUpperLowerStr, &N, &N, h_A, &N, h_R, &N );
                magma_zheevd( MagmaNoVectors, opts.uplo,
                              N, h_R, N, w2,
                              h_work, lwork,
                              rwork, lrwork,
                              iwork, liwork,
                              &info );
                
                temp1 = temp2 = 0;
                for(int j=0; j<N; j++){
                    temp1 = max(temp1, absv(w1[j]));
                    temp1 = max(temp1, absv(w2[j]));
                    temp2 = max(temp2, absv(w1[j]-w2[j]));
                }
                result[2] = temp2 / temp1;
            }
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            cpu_time = magma_wtime();
            lapackf77_zheevd( &opts.jobz, &opts.uplo,
                              &N, h_A, &N, w2,
                              h_work, &lwork,
                              rwork, &lrwork,
                              iwork, &liwork,
                              &info );
            cpu_time = magma_wtime() - cpu_time;
            if (info < 0)
                printf("Argument %d of zheevd had an illegal value.\n", (int) -info);
    
            
            /* =====================================================================
               Print execution time
               =================================================================== */
            printf("%5d   %7.2f          %7.2f\n",
                   (int) N, cpu_time, gpu_time);
            if ( opts.check ){
                printf("Testing the factorization A = U S U' for correctness:\n");
                printf("(1)    | A - U S U' | / (|A| N) = %e\n", result[0]*eps);
                printf("(2)    | I -   U'U  | /  N      = %e\n", result[1]*eps);
                printf("(3)    | S(w/ U)-S(w/o U)|/ |S| = %e\n\n", result[2]);
            }
            
            TESTING_FREE(       h_A);
            TESTING_FREE(        w1);
            TESTING_FREE(        w2);
            TESTING_FREE(     rwork);
            TESTING_FREE(     iwork);
            TESTING_HOSTFREE(h_work);
            TESTING_HOSTFREE(   h_R);
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }
    
    TESTING_CUDA_FINALIZE();
    return 0;
}
