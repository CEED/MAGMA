/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> c d s
       @author Mark Gates
       
       This demonstrates how to use cublas_v2 with magma.
       Simply include cublas_v2.h before magma.h,
       to override its internal include of cublas.h (v1).
*/

#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime_api.h>
#include <cublas_v2.h>  // include before magma.h

#include "magma.h"
#include "magma_lapack.h"

int main( int argc, char** argv )
{
    magma_init();
    cublasHandle_t handle;
    cudaSetDevice( 0 );
    cublasCreate( &handle );
    
    magmaDoubleComplex *A, *B, *C;
    magmaDoubleComplex *dA, *dB, *dC;
    double error, work[1];
    magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magma_int_t ione = 1;
    magma_int_t ISEED[4] = { 1, 2, 3, 4 };
    magma_int_t n = 10;
    magma_int_t lda = n;
    magma_int_t ldda = ((n+31)/32)*32;
    magma_int_t size = lda*n;
    magma_int_t info;
    
    magma_zmalloc_cpu( &A, lda*n );
    magma_zmalloc_cpu( &B, lda*n );
    magma_zmalloc_cpu( &C, lda*n );
    magma_zmalloc( &dA, ldda*n );
    magma_zmalloc( &dB, ldda*n );
    magma_zmalloc( &dC, ldda*n );
    
    // initialize matrices
    lapackf77_zlarnv( &ione, ISEED, &size, A );
    lapackf77_zlarnv( &ione, ISEED, &size, B );
    lapackf77_zlarnv( &ione, ISEED, &size, C );
    // increase diagonal to be SPD
    for( int i=0; i < n; ++i ) {
        C[i+i*lda] = MAGMA_Z_ADD( C[i+i*lda], MAGMA_Z_MAKE( n*n, 0 ));
    }
    
    magma_zsetmatrix( n, n, A, lda, dA, ldda );
    magma_zsetmatrix( n, n, B, lda, dB, ldda );
    magma_zsetmatrix( n, n, C, lda, dC, ldda );
    
    // compute with cublas
    cublasZgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                 &c_neg_one, dA, ldda, dB, ldda, &c_one, dC, ldda );
    
    magma_zpotrf_gpu( MagmaLower, n, dC, ldda, &info );
    if (info != 0)
        printf("magma_zpotrf returned error %d: %s.\n",
               (int) info, magma_strerror( info ));
    
    // compute with LAPACK
    blasf77_zgemm( MagmaNoTransStr, MagmaNoTransStr, &n, &n, &n,
                   &c_neg_one, A, &lda, B, &lda, &c_one, C, &lda );
    
    lapackf77_zpotrf( MagmaLowerStr, &n, C, &lda, &info );
    if (info != 0)
        printf("lapackf77_zpotrf returned error %d: %s.\n",
               (int) info, magma_strerror( info ));
    
    // compute difference
    magma_zgetmatrix( n, n, dC, ldda, A, lda );
    blasf77_zaxpy( &size, &c_neg_one, C, &ione, A, &ione );
    error = lapackf77_zlange( "F", &n, &n, A, &lda, work );
    printf( "n %d, error %8.2e\n", n, error );
    
    magma_free( dA );
    magma_free( dB );
    magma_free( dC );
    magma_free_cpu( A );
    magma_free_cpu( B );
    magma_free_cpu( C );
    
    cublasDestroy( handle );
    magma_finalize();
    return 0;
}
