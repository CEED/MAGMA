/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Mark Gates
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zlacpy
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t    gbytes, gpu_perf, gpu_time, cpu_perf, cpu_time;
    double           error, work[1];
    magmaDoubleComplex  c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex *h_A, *h_B, *h_R;
    magmaDoubleComplex_ptr d_A, d_B;
    magma_int_t M, N, size, lda, ldb, ldda, lddb;
    magma_int_t ione     = 1;
    magma_int_t status = 0;
    
    magma_opts opts;
    parse_opts( argc, argv, &opts );

    magma_uplo_t uplo[] = { MagmaLower, MagmaUpper, MagmaFull };
    
    printf("uplo      M     N   CPU GByte/s (ms)    GPU GByte/s (ms)    check\n");
    printf("=================================================================\n");
    for( int iuplo = 0; iuplo < 3; ++iuplo ) {
      for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            lda    = M;
            ldb    = lda;
            ldda   = magma_roundup( M, opts.align );  // multiple of 32 by default
            lddb   = ldda;
            size   = lda*N;
            if ( uplo[iuplo] == MagmaLower || uplo[iuplo] == MagmaUpper ) {
                // load and save triangle (with diagonal)
                gbytes = sizeof(magmaDoubleComplex) * 1.*N*(N+1) / 1e9;
            }
            else {
              // load entire matrix, save entire matrix
              gbytes = sizeof(magmaDoubleComplex) * 2.*M*N / 1e9;
            }
    
            TESTING_MALLOC_CPU( h_A, magmaDoubleComplex, size   );
            TESTING_MALLOC_CPU( h_B, magmaDoubleComplex, size   );
            TESTING_MALLOC_CPU( h_R, magmaDoubleComplex, size   );
            
            TESTING_MALLOC_DEV( d_A, magmaDoubleComplex, ldda*N );
            TESTING_MALLOC_DEV( d_B, magmaDoubleComplex, ldda*N );
            
            /* Initialize the matrix */
            for( int j = 0; j < N; ++j ) {
                for( int i = 0; i < M; ++i ) {
                    h_A[i + j*lda] = MAGMA_Z_MAKE( i + j/10000., j );
                    h_B[i + j*ldb] = MAGMA_Z_MAKE( i - j/10000. + 10000., j );
                }
            }
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            magma_zsetmatrix( M, N, h_A, lda, d_A, ldda );
            magma_zsetmatrix( M, N, h_B, ldb, d_B, lddb );
            
            gpu_time = magma_sync_wtime( 0 );
            //magmablas_zlacpy( uplo[iuplo], M-2, N-2, d_A+1+ldda, ldda, d_B+1+lddb, lddb );  // inset by 1 row & col
            magmablas_zlacpy( uplo[iuplo], M, N, d_A, ldda, d_B, lddb );
            gpu_time = magma_sync_wtime( 0 ) - gpu_time;
            gpu_perf = gbytes / gpu_time;
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            cpu_time = magma_wtime();
            //magma_int_t M2 = M-2;  // inset by 1 row & col
            //magma_int_t N2 = N-2;
            //lapackf77_zlacpy( lapack_uplo_const(uplo[iuplo]), &M2, &N2, h_A+1+lda, &lda, h_B+1+ldb, &ldb );
            lapackf77_zlacpy( lapack_uplo_const(uplo[iuplo]), &M, &N, h_A, &lda, h_B, &ldb );
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gbytes / cpu_time;
            
            if ( opts.verbose ) {
                printf( "A= " );  magma_zprint(     M, N, h_A, lda );
                printf( "B= " );  magma_zprint(     M, N, h_B, lda );
                printf( "dA=" );  magma_zprint_gpu( M, N, d_A, ldda );
                printf( "dB=" );  magma_zprint_gpu( M, N, d_B, ldda );
            }
            
            /* =====================================================================
               Check the result
               =================================================================== */
            magma_zgetmatrix( M, N, d_B, ldda, h_R, lda );
            
            blasf77_zaxpy(&size, &c_neg_one, h_B, &ione, h_R, &ione);
            error = lapackf77_zlange("f", &M, &N, h_R, &lda, work);

            printf("%5s %5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %s\n",
                   lapack_uplo_const(uplo[iuplo]), (int) M, (int) N,
                   cpu_perf, cpu_time*1000., gpu_perf, gpu_time*1000.,
                   (error == 0. ? "ok" : "failed") );
            status += ! (error == 0.);
            
            TESTING_FREE_CPU( h_A );
            TESTING_FREE_CPU( h_B );
            TESTING_FREE_CPU( h_R );
            
            TESTING_FREE_DEV( d_A );
            TESTING_FREE_DEV( d_B );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
      }
      printf( "\n" );
    }

    TESTING_FINALIZE();
    return status;
}
