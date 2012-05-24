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

/*
    Matrix is divided into 64 x m block rows.
    Each block has 64 threads.
    Each thread copies one row, iterating across all columns below diagonal.
    The bottom block of rows may be partially outside the matrix;
    if so, rows outside the matrix (i >= m) are disabled.
*/
__global__ void
zsymmetrize_kernel_lower( int m, cuDoubleComplex *A, int lda )
{
    // A iterates across row i and AT iterates down column i.
    int i = blockIdx.x*64 + threadIdx.x;
    cuDoubleComplex *AT = A;
    if ( i < m ) {
        A  += i;
        AT += i*lda;
        cuDoubleComplex *Aend = A + i*lda;
        while( A < Aend ) {
            *AT = cuConj(*A);
            A  += lda;
            AT += 1;
        }
    }
}


// only difference with _lower version is direction A=AT instead of AT=A.
__global__ void
zsymmetrize_kernel_upper( int m, cuDoubleComplex *A, int lda )
{
    // A iterates across row i and AT iterates down column i.
    int i = blockIdx.x*64 + threadIdx.x;
    cuDoubleComplex *AT = A;
    if ( i < m ) {
        A  += i;
        AT += i*lda;
        cuDoubleComplex *Aend = A + i*lda;
        while( A < Aend ) {
            *A = cuConj(*AT);
            A  += lda;
            AT += 1;
        }
    }
}


extern "C" void
magmablas_zsymmetrize( char uplo, int m, cuDoubleComplex *A, int lda )
{
/*
  Purpose
  =======

  ZSYMMETRIZE copies lower triangle to upper triangle, or vice-versa,
  to make A a general representation of a symmetric matrix.

  Arguments
  =========

  UPLO    (input) CHARACTER*1
          Specifies the part of the matrix A that is valid on input.
          = 'U':      Upper triangular part
          = 'L':      Lower triangular part

  M       (input) INTEGER
          The number of rows of the matrix A.  M >= 0.

  A       (input/output) COMPLEX DOUBLE PRECISION array, dimension (LDA,N)
          The m by m matrix A.

  LDA     (input) INTEGER
          The leading dimension of the array A.  LDA >= max(1,M).

  =====================================================================   */

    dim3 threads( 64 );
    dim3 grid( m/64 + (m%64 != 0) );
    
    //printf( "m %d, grid %d, threads %d\n", m, grid.x, threads.x );
    if ( m == 0 )
        return;
    
    if ( (uplo == 'U') || (uplo == 'u') ) {
        zsymmetrize_kernel_upper<<< grid, threads, 0, magma_stream >>>( m, A, lda );
    }
    else if ( (uplo == 'L') || (uplo == 'l') ) {
        zsymmetrize_kernel_lower<<< grid, threads, 0, magma_stream >>>( m, A, lda );
    }
    else {
        printf( "uplo has illegal value\n" );
        exit(1);
    }
}
