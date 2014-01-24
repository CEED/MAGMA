/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Mark Gates
*/
#include "common_magma.h"
#include <assert.h>

#define NB 64

/* =====================================================================
    Matrix is m x n, and is divided into block rows, each NB x n.
    Each CUDA block has NB threads to handle one block row.
    Each thread adds one row, iterating across all columns.
    The bottom block of rows may be partially outside the matrix;
    if so, rows outside the matrix (i >= m) are disabled.
    
    TODO. Block in both directions, for large matrices.
    E.g., each block does 64x64 tile, instead of 64xN tile.
*/
__global__ void
zgeadd_kernel(
    int m, int n,
    magmaDoubleComplex alpha,
    const magmaDoubleComplex *dA, int ldda,
    magmaDoubleComplex       *dB, int lddb )
{
    // dA and dB iterate across row i
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if ( i < m ) {
        dA += i;
        dB += i;
        const magmaDoubleComplex *dAend = dA + n*ldda;
        while( dA < dAend ) {
            *dB = alpha*(*dA) + (*dB);
            dA += ldda;
            dB += lddb;
        }
    }
}


/* ===================================================================== */
extern "C" void
magmablas_zgeadd(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    const magmaDoubleComplex *dA, magma_int_t ldda,
    magmaDoubleComplex       *dB, magma_int_t lddb )
{
/*
    Purpose
    =======
    ZGEADD adds two matrices, dB = alpha*dA + dB.
    
    Arguments
    =========
    
    M       (input) INTEGER
            The number of rows of the matrix dA.  M >= 0.
    
    N       (input) INTEGER
            The number of columns of the matrix dA.  N >= 0.
    
    ALPHA   (input) COMPLEX DOUBLE PRECISION
            The scalar alpha.
            
    dA      (input) COMPLEX DOUBLE PRECISION array, dimension (LDDA,N)
            The m by n matrix dA.
    
    LDDA    (input) INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,M).
            
    dB      (input/output) COMPLEX DOUBLE PRECISION array, dimension (LDDB,N)
            The m by n matrix dB.
    
    LDDB    (input) INTEGER
            The leading dimension of the array dB.  LDDB >= max(1,M).
    
    =====================================================================   */

    magma_int_t info = 0;
    if ( m < 0 )
        info = -1;
    else if ( n < 0 )
        info = -2;
    else if ( ldda < max(1,m))
        info = -5;
    else if ( lddb < max(1,m))
        info = -7;
    
    if ( info != 0 ) {
        magma_xerbla( __func__, -(info) );
        return;
    }
    
    if ( m == 0 || n == 0 )
        return;
    
    dim3 threads( NB );
    dim3 grid( (m + NB - 1)/NB );
    
    zgeadd_kernel<<< grid, threads, 0, magma_stream >>>(
        m, n, alpha, dA, ldda, dB, lddb );
}
