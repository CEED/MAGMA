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
#include <assert.h>

// includes, project
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

#define PRECISION_z

// if ( A==B ) return 0, else return 1
static int diff_matrix( magma_int_t m, magma_int_t n, cuDoubleComplex *A, magma_int_t lda, cuDoubleComplex *B, magma_int_t ldb )
{
    for( magma_int_t j = 0; j < n; j++ ) {
        for( magma_int_t i = 0; i < m; i++ ) {
            if ( ! MAGMA_Z_EQUAL( A[lda*j+i], B[ldb*j+i] ) )
                return 1;
        }
    }
    return 0;
}

// fill matrix with entries Aij = offset + (i+1) + (j+1)/10000,
// which makes it easy to identify which rows & cols have been swapped.
static void init_matrix( magma_int_t m, magma_int_t n, cuDoubleComplex *A, magma_int_t lda, magma_int_t offset )
{
    assert( lda >= m );
    for( magma_int_t j = 0; j < n; ++j ) {
        for( magma_int_t i=0; i < m; ++i ) {
            A[i + j*lda] = MAGMA_Z_MAKE( offset + (i+1) + (j+1)/10000., 0 );
        }
    }
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zswap, zswapblk, zpermute, zlaswp, zlaswpx
*/
int main( int argc, char** argv)
{
    TESTING_CUDA_INIT();

    cuDoubleComplex *h_A1, *h_A2;
    cuDoubleComplex *d_A1, *d_A2;
    cuDoubleComplex *h_R1, *h_R2;
    
    // row-major and column-major performance
    real_Double_t row_perf0, col_perf0;
    real_Double_t row_perf1, col_perf1;
    real_Double_t row_perf2, col_perf2;
    real_Double_t row_perf3;
    real_Double_t row_perf4;
    real_Double_t row_perf5, col_perf5;
    real_Double_t row_perf6, col_perf6;
    real_Double_t row_perf7;
    real_Double_t cpu_perf;

    real_Double_t time, gbytes;

    magma_int_t N, lda, ldda, nb, j;
    magma_int_t ione = 1;
    magma_int_t *ipiv, *ipiv2;
    magma_int_t *d_ipiv;
    
    magma_opts opts;
    parse_opts( argc, argv, &opts );
    
    magma_queue_t queue = 0;
    
    printf("            cublasZswap       zswap             zswapblk          zlaswp   zpermute zlaswp2  zlaswpx           zcopymatrix      CPU      (all in )\n");
    printf("    N   nb  row-maj/col-maj   row-maj/col-maj   row-maj/col-maj   row-maj  row-maj  row-maj  row-maj/col-maj   row-blk/col-blk  zlaswp   (GByte/s)\n");
    printf("==================================================================================================================================================\n");
    for( int i = 0; i < opts.ntest; ++i ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            // each test is assigned one bit in the check bitmask, bit=1 is failure.
            // shift keeps track of which bit is for current test
            int shift = 1;
            int check = 0;
            N = opts.nsize[i];
            lda    = N;
            ldda   = ((N+31)/32)*32;
            nb     = (opts.nb > 0 ? opts.nb : magma_get_zgetrf_nb( N ));
            // for each swap, does 2N loads and 2N stores
            gbytes = sizeof(cuDoubleComplex) * 4.*N*nb / 1e9;
            
            TESTING_HOSTALLOC( h_A1, cuDoubleComplex, lda*N );
            TESTING_HOSTALLOC( h_A2, cuDoubleComplex, lda*N );
            TESTING_HOSTALLOC( h_R1, cuDoubleComplex, lda*N );
            TESTING_HOSTALLOC( h_R2, cuDoubleComplex, lda*N );
            
            TESTING_MALLOC( ipiv,  magma_int_t, nb );
            TESTING_MALLOC( ipiv2, magma_int_t, nb );
            TESTING_DEVALLOC( d_ipiv, magma_int_t, nb );
            TESTING_DEVALLOC( d_A1, cuDoubleComplex, ldda*N );
            TESTING_DEVALLOC( d_A2, cuDoubleComplex, ldda*N );
            
            for( j=0; j < nb; j++ ) {
                ipiv[j] = (magma_int_t) ((rand()*1.*N) / (RAND_MAX * 1.)) + 1;
            }
            
            /* =====================================================================
             * cublasZswap, row-by-row (2 matrices)
             */
            
            /* Row Major */
            init_matrix( N, N, h_A1, lda, 0 );
            init_matrix( N, N, h_A2, lda, 100 );
            magma_zsetmatrix( N, N, h_A1, lda, d_A1, ldda );
            magma_zsetmatrix( N, N, h_A2, lda, d_A2, ldda );
            
            time = magma_sync_wtime( queue );
            for( j=0; j < nb; j++) {
                if ( j != (ipiv[j]-1)) {
                    cublasZswap( N, d_A1+ldda*j, 1, d_A2+ldda*(ipiv[j]-1), 1);
                }
            }
            time = magma_sync_wtime( queue ) - time;
            row_perf0 = gbytes / time;
            
            for( j=0; j < nb; j++) {
                if ( j != (ipiv[j]-1)) {
                    blasf77_zswap( &N, h_A1+lda*j, &ione, h_A2+lda*(ipiv[j]-1), &ione);
                }
            }
            magma_zgetmatrix( N, N, d_A1, ldda, h_R1, lda );
            magma_zgetmatrix( N, N, d_A2, ldda, h_R2, lda );
            check += (diff_matrix( N, N, h_A1, lda, h_R1, lda ) ||
                      diff_matrix( N, N, h_A2, lda, h_R2, lda ))*shift;
            shift *= 2;
            
            /* Column Major */
            init_matrix( N, N, h_A1, lda, 0 );
            init_matrix( N, N, h_A2, lda, 100 );
            magma_zsetmatrix( N, N, h_A1, lda, d_A1, ldda );
            magma_zsetmatrix( N, N, h_A2, lda, d_A2, ldda );
            
            time = magma_sync_wtime( queue );
            for( j=0; j < nb; j++) {
                if ( j != (ipiv[j]-1)) {
                    cublasZswap( N, d_A1+j, ldda, d_A2+ipiv[j]-1, ldda);
                }
            }
            time = magma_sync_wtime( queue ) - time;
            col_perf0 = gbytes / time;
            
            for( j=0; j < nb; j++) {
                if ( j != (ipiv[j]-1)) {
                    blasf77_zswap( &N, h_A1+j, &lda, h_A2+(ipiv[j]-1), &lda);
                }
            }
            magma_zgetmatrix( N, N, d_A1, ldda, h_R1, lda );
            magma_zgetmatrix( N, N, d_A2, ldda, h_R2, lda );
            check += (diff_matrix( N, N, h_A1, lda, h_R1, lda ) ||
                      diff_matrix( N, N, h_A2, lda, h_R2, lda ))*shift;
            shift *= 2;
            
            /* =====================================================================
             * zswap, row-by-row (2 matrices)
             */
            
            /* Row Major */
            init_matrix( N, N, h_A1, lda, 0 );
            init_matrix( N, N, h_A2, lda, 100 );
            magma_zsetmatrix( N, N, h_A1, lda, d_A1, ldda );
            magma_zsetmatrix( N, N, h_A2, lda, d_A2, ldda );
            
            time = magma_sync_wtime( queue );
            for( j=0; j < nb; j++) {
                if ( j != (ipiv[j]-1)) {
                    magmablas_zswap( N, d_A1+ldda*j, 1, d_A2+ldda*(ipiv[j]-1), 1);
                }
            }
            time = magma_sync_wtime( queue ) - time;
            row_perf1 = gbytes / time;
            
            for( j=0; j < nb; j++) {
                if ( j != (ipiv[j]-1)) {
                    blasf77_zswap( &N, h_A1+lda*j, &ione, h_A2+lda*(ipiv[j]-1), &ione);
                }
            }
            magma_zgetmatrix( N, N, d_A1, ldda, h_R1, lda );
            magma_zgetmatrix( N, N, d_A2, ldda, h_R2, lda );
            check += (diff_matrix( N, N, h_A1, lda, h_R1, lda ) ||
                      diff_matrix( N, N, h_A2, lda, h_R2, lda ))*shift;
            shift *= 2;
            
            /* Column Major */
            init_matrix( N, N, h_A1, lda, 0 );
            init_matrix( N, N, h_A2, lda, 100 );
            magma_zsetmatrix( N, N, h_A1, lda, d_A1, ldda );
            magma_zsetmatrix( N, N, h_A2, lda, d_A2, ldda );
            
            time = magma_sync_wtime( queue );
            for( j=0; j < nb; j++) {
                if ( j != (ipiv[j]-1)) {
                    magmablas_zswap( N, d_A1+j, ldda, d_A2+ipiv[j]-1, ldda );
                }
            }
            time = magma_sync_wtime( queue ) - time;
            col_perf1 = gbytes / time;
            
            for( j=0; j < nb; j++) {
                if ( j != (ipiv[j]-1)) {
                    blasf77_zswap( &N, h_A1+j, &lda, h_A2+(ipiv[j]-1), &lda);
                }
            }
            magma_zgetmatrix( N, N, d_A1, ldda, h_R1, lda );
            magma_zgetmatrix( N, N, d_A2, ldda, h_R2, lda );
            check += (diff_matrix( N, N, h_A1, lda, h_R1, lda ) ||
                      diff_matrix( N, N, h_A2, lda, h_R2, lda ))*shift;
            shift *= 2;
            
            /* =====================================================================
             * zswapblk, blocked version (2 matrices)
             */
            
            /* Row Major */
            init_matrix( N, N, h_A1, lda, 0 );
            init_matrix( N, N, h_A2, lda, 100 );
            magma_zsetmatrix( N, N, h_A1, lda, d_A1, ldda );
            magma_zsetmatrix( N, N, h_A2, lda, d_A2, ldda );
            
            time = magma_sync_wtime( queue );
            magmablas_zswapblk( 'R', N, d_A1, ldda, d_A2, ldda, 1, nb, ipiv, 1, 0);
            time = magma_sync_wtime( queue ) - time;
            row_perf2 = gbytes / time;
            
            for( j=0; j < nb; j++) {
                if ( j != (ipiv[j]-1)) {
                    blasf77_zswap( &N, h_A1+lda*j, &ione, h_A2+lda*(ipiv[j]-1), &ione);
                }
            }
            magma_zgetmatrix( N, N, d_A1, ldda, h_R1, lda );
            magma_zgetmatrix( N, N, d_A2, ldda, h_R2, lda );
            check += (diff_matrix( N, N, h_A1, lda, h_R1, lda ) ||
                      diff_matrix( N, N, h_A2, lda, h_R2, lda ))*shift;
            shift *= 2;
            
            /* Column Major */
            init_matrix( N, N, h_A1, lda, 0 );
            init_matrix( N, N, h_A2, lda, 100 );
            magma_zsetmatrix( N, N, h_A1, lda, d_A1, ldda );
            magma_zsetmatrix( N, N, h_A2, lda, d_A2, ldda );
            
            time = magma_sync_wtime( queue );
            magmablas_zswapblk( 'C', N, d_A1, ldda, d_A2, ldda, 1, nb, ipiv, 1, 0);
            time = magma_sync_wtime( queue ) - time;
            col_perf2 = gbytes / time;
            
            for( j=0; j < nb; j++) {
                if ( j != (ipiv[j]-1)) {
                    blasf77_zswap( &N, h_A1+j, &lda, h_A2+(ipiv[j]-1), &lda);
                }
            }
            magma_zgetmatrix( N, N, d_A1, ldda, h_R1, lda );
            magma_zgetmatrix( N, N, d_A2, ldda, h_R2, lda );
            check += (diff_matrix( N, N, h_A1, lda, h_R1, lda ) ||
                      diff_matrix( N, N, h_A2, lda, h_R2, lda ))*shift;
            shift *= 2;
            
            /* =====================================================================
             * zpermute_long (1 matrix)
             */
            
            /* Row Major */
            memcpy( ipiv2, ipiv, nb*sizeof(magma_int_t) );  // zpermute updates ipiv2
            init_matrix( N, N, h_A1, lda, 0 );
            magma_zsetmatrix( N, N, h_A1, lda, d_A1, ldda );
            
            time = magma_sync_wtime( queue );
            magmablas_zpermute_long2( N, d_A1, ldda, ipiv2, nb, 0 );
            time = magma_sync_wtime( queue ) - time;
            row_perf3 = gbytes / time;
            
            for( j=0; j < nb; j++) {
                if ( j != (ipiv[j]-1)) {
                    blasf77_zswap( &N, h_A1+lda*j, &ione, h_A1+lda*(ipiv[j]-1), &ione);
                }
            }
            magma_zgetmatrix( N, N, d_A1, ldda, h_R1, lda );
            check += diff_matrix( N, N, h_A1, lda, h_R1, lda )*shift;
            shift *= 2;
            
            /* =====================================================================
             * LAPACK-style zlaswp (1 matrix)
             */
            
            /* Row Major */
            init_matrix( N, N, h_A1, lda, 0 );
            magma_zsetmatrix( N, N, h_A1, lda, d_A1, ldda );
            
            time = magma_sync_wtime( queue );
            magmablas_zlaswp( N, d_A1, ldda, 1, nb, ipiv, 1);
            time = magma_sync_wtime( queue ) - time;
            row_perf4 = gbytes / time;
            
            for( j=0; j < nb; j++) {
                if ( j != (ipiv[j]-1)) {
                    blasf77_zswap( &N, h_A1+lda*j, &ione, h_A1+lda*(ipiv[j]-1), &ione);
                }
            }
            magma_zgetmatrix( N, N, d_A1, ldda, h_R1, lda );
            check += diff_matrix( N, N, h_A1, lda, h_R1, lda )*shift;
            shift *= 2;
            
            /* =====================================================================
             * LAPACK-style zlaswp (1 matrix) - d_ipiv on GPU
             */
            
            /* Row Major */
            init_matrix( N, N, h_A1, lda, 0 );
            magma_zsetmatrix( N, N, h_A1, lda, d_A1, ldda );
            
            time = magma_sync_wtime( queue );
            magma_setvector( nb, sizeof(magma_int_t), ipiv, 1, d_ipiv, 1 );
            magmablas_zlaswp2( N, d_A1, ldda, 1, nb, d_ipiv );
            time = magma_sync_wtime( queue ) - time;
            row_perf7 = gbytes / time;
            
            for( j=0; j < nb; j++) {
                if ( j != (ipiv[j]-1)) {
                    blasf77_zswap( &N, h_A1+lda*j, &ione, h_A1+lda*(ipiv[j]-1), &ione);
                }
            }
            magma_zgetmatrix( N, N, d_A1, ldda, h_R1, lda );
            check += diff_matrix( N, N, h_A1, lda, h_R1, lda )*shift;
            shift *= 2;
            
            /* =====================================================================
             * LAPACK-style zlaswpx (extended for row- and col-major) (1 matrix)
             */
            
            /* Row Major */
            init_matrix( N, N, h_A1, lda, 0 );
            magma_zsetmatrix( N, N, h_A1, lda, d_A1, ldda );
            
            time = magma_sync_wtime( queue );
            magmablas_zlaswpx( N, d_A1, ldda, 1, 1, nb, ipiv, 1);
            time = magma_sync_wtime( queue ) - time;
            row_perf5 = gbytes / time;
            
            for( j=0; j < nb; j++) {
                if ( j != (ipiv[j]-1)) {
                    blasf77_zswap( &N, h_A1+lda*j, &ione, h_A1+lda*(ipiv[j]-1), &ione);
                }
            }
            magma_zgetmatrix( N, N, d_A1, ldda, h_R1, lda );
            check += diff_matrix( N, N, h_A1, lda, h_R1, lda )*shift;
            shift *= 2;
            
            /* Col Major */
            init_matrix( N, N, h_A1, lda, 0 );
            magma_zsetmatrix( N, N, h_A1, lda, d_A1, ldda );
            
            time = magma_sync_wtime( queue );
            magmablas_zlaswpx( N, d_A1, 1, ldda, 1, nb, ipiv, 1);
            time = magma_sync_wtime( queue ) - time;
            col_perf5 = gbytes / time;
            
            time = magma_wtime();
            lapackf77_zlaswp( &N, h_A1, &lda, &ione, &nb, ipiv, &ione);
            time = magma_wtime() - time;
            cpu_perf = gbytes / time;
            magma_zgetmatrix( N, N, d_A1, ldda, h_R1, lda );
            check += diff_matrix( N, N, h_A1, lda, h_R1, lda )*shift;
            shift *= 2;
            
            /* =====================================================================
             * Copy matrix.
             */
            
            time = magma_sync_wtime( queue );
            magma_zcopymatrix( N, nb, d_A1, ldda, d_A2, ldda );
            time = magma_sync_wtime( queue ) - time;
            // copy reads 1 matrix and writes 1 matrix, so has half gbytes of swap
            col_perf6 = 0.5 * gbytes / time;
            
            time = magma_sync_wtime( queue );
            magma_zcopymatrix( nb, N, d_A1, ldda, d_A2, ldda );
            time = magma_sync_wtime( queue ) - time;
            // copy reads 1 matrix and writes 1 matrix, so has half gbytes of swap
            row_perf6 = 0.5 * gbytes / time;
            
            printf("%5d  %3d  %6.2f%c/ %6.2f%c  %6.2f%c/ %6.2f%c  %6.2f%c/ %6.2f%c  %6.2f%c  %6.2f%c  %6.2f%c  %6.2f%c/ %6.2f%c  %6.2f / %6.2f  %6.2f  %10s\n",
                   N, nb,
                   row_perf0, ((check & 0x001) != 0 ? '*' : ' '),
                   col_perf0, ((check & 0x002) != 0 ? '*' : ' '),
                   row_perf1, ((check & 0x004) != 0 ? '*' : ' '),
                   col_perf1, ((check & 0x008) != 0 ? '*' : ' '),
                   row_perf2, ((check & 0x010) != 0 ? '*' : ' '),
                   col_perf2, ((check & 0x020) != 0 ? '*' : ' '),
                   row_perf3, ((check & 0x040) != 0 ? '*' : ' '),
                   row_perf4, ((check & 0x080) != 0 ? '*' : ' '),
                   row_perf7, ((check & 0x100) != 0 ? '*' : ' '),
                   row_perf5, ((check & 0x200) != 0 ? '*' : ' '),
                   col_perf5, ((check & 0x400) != 0 ? '*' : ' '),
                   row_perf6,
                   col_perf6,
                   cpu_perf,
                   (check == 0 ? "okay" : "* failures") );
            
            TESTING_HOSTFREE( h_A1 );
            TESTING_HOSTFREE( h_A2 );
            TESTING_HOSTFREE( h_R1 );
            TESTING_HOSTFREE( h_R2 );
            TESTING_DEVFREE(  d_A1 );
            TESTING_DEVFREE(  d_A2 );
            TESTING_FREE( ipiv  );
            TESTING_FREE( ipiv2 );
            TESTING_DEVFREE( d_ipiv );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }
    
    TESTING_CUDA_FINALIZE();
    return 0;
}
