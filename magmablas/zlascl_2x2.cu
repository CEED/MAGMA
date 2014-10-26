/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

       @author Ichitaro Yamazaki
*/
#include "common_magma.h"

#define NB 64


// each thread block does one NB x n block row of A.
// each thread does one row, starting from left edge and moving right.
__global__ void
zlascl_2x2_full(int m, const magmaDoubleComplex* W, int ldw, magmaDoubleComplex* A, int lda)
{
    #define A(i,j) (A[(i) + (j)*lda])
    #define W(i,j) (W[(i) + (j)*ldw])
    int ind = blockIdx.x * NB + threadIdx.x;

    magmaDoubleComplex D21 = W( 1, 0 );
    magmaDoubleComplex D11 = MAGMA_Z_DIV( W( 1, 1 ), D21 );
    magmaDoubleComplex D22 = MAGMA_Z_DIV( W( 0, 0 ), MAGMA_Z_CNJG( D21 ) );
    double T = 1.0 / ( MAGMA_Z_REAL( D11*D22 ) - 1.0 );
    D21 = MAGMA_Z_DIV( MAGMA_Z_MAKE(T,0.0), D21 );

    if (ind < m) {
        A( ind, 0 ) = MAGMA_Z_CNJG( D21 )*( D11*W( 2+ind, 0 )-W( 2+ind, 1 ) );
        A( ind, 1 ) = D21*( D22*W( 2+ind, 1 )-W( 2+ind, 0 ) );
    }
}

__global__ void
zlascl_2x2_full_trans(int m, const magmaDoubleComplex* W, int ldw, magmaDoubleComplex* A, int lda)
{
    #define At(i,j) (A[(i)*lda + (j)])
    #define Wt(i,j) (W[(i)*ldw + (j)])
    int ind = blockIdx.x * NB + threadIdx.x;

    magmaDoubleComplex D21 = Wt( 1, 0 );
    magmaDoubleComplex D11 = Wt( 1, 1 ) / D21;
    magmaDoubleComplex D22 = Wt( 0, 0 ) / MAGMA_Z_CNJG( D21 );
    double T = 1.0 / ( MAGMA_Z_REAL( D11*D22 ) - 1.0 );
    D21 = MAGMA_Z_MAKE(T,0.0) / D21;

    if (ind < m) {
        At( ind, 0 ) = MAGMA_Z_CNJG( D21 )*( D11*Wt( 2+ind, 0 )-Wt( 2+ind, 1 ) );
        At( ind, 1 ) = D21*( D22*Wt( 2+ind, 1 )-Wt( 2+ind, 0 ) );
    }
}

// each thread block does one NB x n block row of A.
// each thread does one row, starting from left edge and moving right to diagonal.
__global__ void
zlascl_2x2_lower(int m, const magmaDoubleComplex* W, int ldw, magmaDoubleComplex* A, int lda)
{
/*
    int ind = blockIdx.x * NB + threadIdx.x;

    int break_d = (ind < n) ? ind : n-1;

    double mul = D[ind];
    A += ind;
    if (ind < m) {
        for(int j=0; j <= break_d; j++ )
            A[j*lda] *= mul;
    }
*/
}


// each thread block does one NB x n block row of A.
// each thread does one row, starting from right edge and moving left to diagonal.
__global__ void
zlascl_2x2_upper(int m, const magmaDoubleComplex *W, int ldw, magmaDoubleComplex* A, int lda)
{
/*
    int ind = blockIdx.x * NB + threadIdx.x;

    double mul = D[ind];
    A += ind;
    if (ind < m) {
        for(int j=n-1; j >= ind; j--)
            A[j*lda] *= mul;
    }
*/
}


/**
    Purpose
    -------
    ZLASCL2 scales the M by N complex matrix A by the real diagonal matrix dD.
    TYPE specifies that A may be full, upper triangular, lower triangular.

    Arguments
    ---------
    \param[in]
    type    magma_type_t
            TYPE indices the storage type of the input matrix A.
            = MagmaFull:   full matrix.
            = MagmaLower:  lower triangular matrix.
            = MagmaUpper:  upper triangular matrix.
            Other formats that LAPACK supports, MAGMA does not currently support.

    \param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.

    \param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    \param[in]
    dD      DOUBLE PRECISION vector, dimension (M)
            The diagonal matrix containing the scalar factors. Stored as a vector.

    \param[in,out]
    dA      COMPLEX*16 array, dimension (LDDA,N)
            The matrix to be scaled by dD.  See TYPE for the
            storage type.

    \param[in]
    ldda    INTEGER
            The leading dimension of the array A.  LDDA >= max(1,M).

    \param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value.

    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void
magmablas_zlascl_2x2_q(
    magma_type_t type, magma_trans_t trans, magma_int_t m, 
    const magmaDoubleComplex *dW, magma_int_t lddw, 
    magmaDoubleComplex *dA, magma_int_t ldda, 
    magma_int_t *info, magma_queue_t queue )
{
    *info = 0;
    if ( type != MagmaLower && type != MagmaUpper && type != MagmaFull )
        *info = -1;
    else if ( m < 0 )
        *info = -2;
    else if ( ldda < max(1,m) )
        *info = -4;
    
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return;  //info;
    }
    
    dim3 grid( (m + NB - 1)/NB );
    dim3 threads( NB );
    
    if (type == MagmaLower) {
        zlascl_2x2_lower <<< grid, threads, 0, queue >>> (m, dW, lddw, dA, ldda);
    }
    else if (type == MagmaUpper) {
        zlascl_2x2_upper <<< grid, threads, 0, queue >>> (m, dW, lddw, dA, ldda);
    }
    else if (type == MagmaFull) {
        if (trans == MagmaTrans)
        zlascl_2x2_full_trans  <<< grid, threads, 0, queue >>> (m, dW, lddw, dA, ldda);
        else
        zlascl_2x2_full  <<< grid, threads, 0, queue >>> (m, dW, lddw, dA, ldda);
    }
}


/**
    @see magmablas_zlascl2_q
    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void
magmablas_zlascl_2x2(
    magma_type_t type, magma_trans_t trans, magma_int_t m, 
    magmaDoubleComplex *dW, magma_int_t lddw, 
    magmaDoubleComplex *dA, magma_int_t ldda, 
    magma_int_t *info )
{
    magmablas_zlascl_2x2_q( type, trans, m, dW, lddw, dA, ldda, info, magma_stream );
}
