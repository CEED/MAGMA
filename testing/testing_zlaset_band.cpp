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
#include <cuda_runtime_api.h>
#include <cublas.h>

// includes, project
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zlaset_band
   Code is very similar to testing_zlacpy.cpp
*/
int main( int argc, char** argv)
{
    TESTING_INIT();
    
    #define h_A(i_,j_) (h_A + (i_) + (j_)*lda)
    #define d_A(i_,j_) (d_A + (i_) + (j_)*ldda)

    real_Double_t    gbytes, gpu_perf, gpu_time, cpu_perf, cpu_time;
    double           error, work[1];
    magmaDoubleComplex  c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex *h_A, *h_R;
    magmaDoubleComplex *d_A;
    magmaDoubleComplex offdiag = MAGMA_Z_MAKE( 1.2000, 6.7000 );
    magmaDoubleComplex diag    = MAGMA_Z_MAKE( 3.1415, 2.7183 );
    magma_int_t M, N, nb, cnt, size, lda, ldb, ldda;
    magma_int_t ione     = 1;
    magma_int_t status = 0;
    
    magma_opts opts;
    parse_opts( argc, argv, &opts );
    
    nb = (opts.nb == 0 ? 32 : opts.nb);

    magma_uplo_t uplo[] = { MagmaLower, MagmaUpper, MagmaFull };
    
    printf("K = nb = %d\n", (int) nb );
    printf("uplo       M     N   CPU GByte/s (ms)    GPU GByte/s (ms)    check\n");
    printf("==================================================================\n");
    for( int iuplo = 0; iuplo < 2; ++iuplo ) {
      for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            int inset = 0;
            M = opts.msize[itest] + 2*inset;
            N = opts.nsize[itest] + 2*inset;
            lda    = M;
            ldb    = lda;
            ldda   = ((M+31)/32)*32;
            size   = lda*N;
            
            TESTING_MALLOC_CPU( h_A, magmaDoubleComplex, size   );
            TESTING_MALLOC_CPU( h_R, magmaDoubleComplex, size   );
            
            TESTING_MALLOC_DEV( d_A, magmaDoubleComplex, ldda*N );
            
            /* Initialize the matrix */
            for( int j = 0; j < N; ++j ) {
                for( int i = 0; i < M; ++i ) {
                    h_A[i + j*lda] = MAGMA_Z_MAKE( i + j/10000., j );
                }
            }
            magma_zsetmatrix( M, N, h_A, lda, d_A, ldda );
            
            /* =====================================================================
               Performs operation on CPU
               Also count number of elements touched.
               =================================================================== */
            cpu_time = magma_wtime();
            
            cnt = 0;
            for( int j=inset; j < N-inset; ++j ) {
                for( int k=0; k < nb; ++k ) {  // set k-th sub- or super-diagonal
                    if ( k == 0 && j < M-inset ) {
                        *h_A(j,j)   = diag;
                        cnt += 1;
                    }
                    else if ( uplo[iuplo] == MagmaLower && j+k < M-inset ) {
                        *h_A(j+k,j) = offdiag;
                        cnt += 1;
                    }
                    else if ( uplo[iuplo] == MagmaUpper && j-k >= inset && j-k < M-inset ) {
                        *h_A(j-k,j) = offdiag;
                        cnt += 1;
                    }
                }
            }
            
            gbytes = cnt / 1e9;
            
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gbytes / cpu_time;
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_sync_wtime( 0 );
            
            int mm = M - 2*inset;
            int nn = N - 2*inset;
            magmablas_zlaset_band( uplo[iuplo], mm, nn, nb, offdiag, diag, d_A(inset,inset), ldda );
            
            gpu_time = magma_sync_wtime( 0 ) - gpu_time;
            gpu_perf = gbytes / gpu_time;
            
            /* =====================================================================
               Check the result
               =================================================================== */
            magma_zgetmatrix( M, N, d_A, ldda, h_R, lda );
                        
            //printf( "h_R=" );  magma_zprint( M, N, h_R, lda );
            //printf( "h_A=" );  magma_zprint( M, N, h_A, lda );

            blasf77_zaxpy(&size, &c_neg_one, h_A, &ione, h_R, &ione);
            error = lapackf77_zlange("f", &M, &N, h_R, &lda, work);
            
            printf("%4c   %5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %s\n",
                   lapacke_uplo_const( uplo[iuplo] ), (int) M, (int) N,
                   cpu_perf, cpu_time*1000., gpu_perf, gpu_time*1000.,
                   (error == 0. ? "ok" : "failed") );
            status += ! (error == 0.);
            
            TESTING_FREE_CPU( h_A );
            TESTING_FREE_CPU( h_R );
            
            TESTING_FREE_DEV( d_A );
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
