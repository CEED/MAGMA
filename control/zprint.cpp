/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @author Mark Gates
       @precisions normal z -> s d c

*/
#include "common_magma.h"

#define PRECISION_z

#define A(i,j) (A + i + j*lda)

// -------------------------
// Prints a matrix that is on the CPU host.
extern "C"
void magma_zprint( magma_int_t m, magma_int_t n, cuDoubleComplex *A, magma_int_t lda )
{
    if ( magma_is_devptr( A ) == 1 ) {
        fprintf( stderr, "ERROR: zprint called with device pointer.\n" );
        exit(1);
    }
    
    cuDoubleComplex c_zero = MAGMA_Z_ZERO;
    
    printf( "[\n" );
    for( int i = 0; i < m; ++i ) {
        for( int j = 0; j < n; ++j ) {
            if ( MAGMA_Z_EQUAL( *A(i,j), c_zero )) {
                printf( "   0.    " );
            }
            else {
#if defined(PRECISION_z) || defined(PRECISION_c)
                printf( " %8.4f+%8.4fi", MAGMA_Z_REAL( *A(i,j) ), MAGMA_Z_IMAG( *A(i,j) ));
#else
                printf( " %8.4f", MAGMA_Z_REAL( *A(i,j) ));
#endif
            }
        }
        printf( "\n" );
    }
    printf( "];\n" );
}

// -------------------------
// Prints a matrix that is on the GPU device.
// Internally allocates memory on host, copies it to the host, prints it,
// and de-allocates host memory.
extern "C"
void magma_zprint_gpu( magma_int_t m, magma_int_t n, cuDoubleComplex *dA, magma_int_t ldda )
{
    if ( magma_is_devptr( dA ) == 0 ) {
        fprintf( stderr, "ERROR: zprint_gpu called with host pointer.\n" );
        exit(1);
    }
    
    int lda = m;
    cuDoubleComplex* A = (cuDoubleComplex*) malloc( lda*n*sizeof(cuDoubleComplex) );
    cublasGetMatrix( m, n, sizeof(cuDoubleComplex), dA, ldda, A, lda );
    
    magma_zprint( m, n, A, lda );
    
    free( A );
}
