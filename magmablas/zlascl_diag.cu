/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
*/
#include "common_magma.h"

#define NB 64


// each thread block does one NB x n block row of A.
// each thread does one row, starting from left edge and moving right to diagonal.
__global__ void
zlascl_diag_lower(int m, int n, magmaDoubleComplex const* D, int ldd, 
                                magmaDoubleComplex*       A, int lda)
{
    int ind = blockIdx.x * NB + threadIdx.x;

    A += ind;
    if (ind < m) {
        for (int j=0; j < n; j++ )
            A[j*lda] /= D[j + j*ldd];
    }
}


// each thread block does one NB x n block row of A.
// each thread does one row, starting from right edge and moving left to diagonal.
__global__ void
zlascl_diag_upper(int m, int n, magmaDoubleComplex const* D, int ldd, 
                                magmaDoubleComplex*       A, int lda)
{
    int ind = blockIdx.x * NB + threadIdx.x;

    A += ind;
    if (ind < m) {
        for (int j=0; j < n; j++ )
            A[j*lda] /= D[ind + ind*ldd];
    }
}


/**
    Purpose
    -------
    ZLASCL_DIAG scales the M by N complex matrix A by the real diagonal matrix dD.
    TYPE specifies that A may be full, upper triangular, lower triangular.

    Arguments
    ---------
    @param[in]
    type    magma_type_t
            TYPE indices the storage type of the input matrix A.
            = MagmaFull:   full matrix.
            = MagmaLower:  lower triangular matrix.
            = MagmaUpper:  upper triangular matrix.
            Other formats that LAPACK supports, MAGMA does not currently support.

    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in]
    dD      DOUBLE PRECISION vector, dimension (LDDD,M)
            The matrix storing the scaling factor on its diagonal. 

    @param[in]
    lddd    INTEGER
            The leading dimension of the array D.  

    @param[in,out]
    dA      COMPLEX*16 array, dimension (LDDA,N)
            The matrix to be scaled by dD.  See TYPE for the
            storage type.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array A.  LDDA >= max(1,M).

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void
magmablas_zlascl_diag_q(
    magma_type_t type, magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr dD, magma_int_t lddd, 
    magmaDoubleComplex_ptr       dA, magma_int_t ldda, 
    magma_int_t *info, magma_queue_t queue )
{
    *info = 0;
    if ( type != MagmaLower && type != MagmaUpper && type != MagmaFull )
        *info = -1;
    else if ( m < 0 )
        *info = -2;
    else if ( n < 0 )
        *info = -3;
    //else if ( ldda < max(1,m) )
    //    *info = -5;
    
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return;  //info;
    }
    
    dim3 grid( magma_ceildiv( m, NB ) );
    dim3 threads( NB );
    
    if (type == MagmaLower) {
        zlascl_diag_lower <<< grid, threads, 0, queue >>> (m, n, dD, lddd, dA, ldda);
    }
    else if (type == MagmaUpper) {
        zlascl_diag_upper <<< grid, threads, 0, queue >>> (m, n, dD, lddd, dA, ldda);
    }
}


/**
    @see magmablas_zlascl2_q
    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void
magmablas_zlascl_diag(
    magma_type_t type, magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr dD, magma_int_t lddd, 
    magmaDoubleComplex_ptr       dA, magma_int_t ldda, 
    magma_int_t *info )
{
    magmablas_zlascl_diag_q( type, m, n, dD, lddd, dA, ldda, info, magma_stream );
}
