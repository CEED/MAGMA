/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> s d c
       @author Mark Gates

*/
#include "common_magma.h"


/* =====================================================================
    Batches zlacpy of multiple arrays;
    y-dimension of grid is different arrays,
    x-dimension of grid is blocks for each array.
    Matrix is divided into block rows of dimension 64 x n.
    Each CUDA block has 64 threads to handle one block row.
    Each thread copies one row, iterating across all columns.
    The bottom block of rows may be partially outside the matrix;
    if so, rows outside the matrix (row >= m) are disabled.
*/
__global__ void
zlacpy_batched_kernel(
    int m, int n,
    const cuDoubleComplex * const *Aarray, int lda,
    cuDoubleComplex              **Barray, int ldb )
{
    const cuDoubleComplex *A = Aarray[ blockIdx.y ];
    cuDoubleComplex       *B = Barray[ blockIdx.y ];
    int row = blockIdx.x*64 + threadIdx.x;
    if ( row < m ) {
        A += row;
        B += row;
        const cuDoubleComplex *Aend = A + lda*n;
        while( A < Aend ) {
            *B = *A;
            A += lda;
            B += ldb;
        }
    }
}


/* ===================================================================== */
extern "C" void
magmablas_zlacpy_batched(
    char uplo, magma_int_t m, magma_int_t n,
    const cuDoubleComplex * const *Aarray, magma_int_t lda,
    cuDoubleComplex              **Barray, magma_int_t ldb,
    magma_int_t batchCount )
{
/*
      Note
    ========
    - UPLO Parameter is disabled
    - Do we want to provide a generic function to the user with all the options?
    
    Purpose
    =======
    ZLACPY copies all or part of a set of two-dimensional matrices A[i] to another
    set of matrices B[i], for i = 0, ..., batchCount-1.
    
    Arguments
    =========
    
    UPLO    (input) CHARACTER*1
            Specifies the part of the matrix A to be copied to B.
            = 'U':      Upper triangular part
            = 'L':      Lower triangular part
            Otherwise:  All of the matrix A
    
    M       (input) INTEGER
            The number of rows of each matrix A.  M >= 0.
    
    N       (input) INTEGER
            The number of columns of the matrix A.  N >= 0.
    
    Aarray  (input) array on GPU of pointers to arrays, with each array a
            COMPLEX DOUBLE PRECISION array, dimension (LDA,N)
            The m by n matrices A[i].  If UPLO = 'U', only the upper triangle
            or trapezoid is accessed; if UPLO = 'L', only the lower
            triangle or trapezoid is accessed.
    
    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).
    
    Barray  (output) array on GPU of pointers to arrays, with each array a
            COMPLEX DOUBLE PRECISION array, dimension (LDB,N)
            On exit, matrix B[i] = matrix A[i] in the locations specified by UPLO.
    
    LDB     (input) INTEGER
            The leading dimension of the array B.  LDB >= max(1,M).
    
    =====================================================================   */

    dim3 threads( 64 );
    dim3 grid( (m+63)/64, batchCount );
    
    //printf( "m %d, n %d, grid %d, threads %d\n", m, n, grid.x, threads.x );
    if ( m == 0 || n == 0 )
        return;
    
    if ( (uplo == 'U') || (uplo == 'u') ) {
        fprintf(stderr, "lacpy upper is not implemented\n");
    }
    else if ( (uplo == 'L') || (uplo == 'l') ) {
        fprintf(stderr, "lacpy lower is not implemented\n");
    }
    else {
        zlacpy_batched_kernel<<< grid, threads, 0, magma_stream >>>( m, n, Aarray, lda, Barray, ldb );
    }
}
