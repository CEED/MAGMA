/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Mark Gates

       @precisions normal z -> s d c
*/
#include "common_magma.h"
#include "commonblas_z.h"
#include "magma_templates.h"

#define PRECISION_z

#define BLK_X 128
#define BLK_Y 128

/* Compute y = alpha*A*x + beta*y.
 * Each thread block does a BLK_X x N block row of A.
 * Each thread goes across one row, accumulating dot product of row ind and x into res.
 * This simple implementation loads x directly, relying on the cache,
 * without using shared memory.
 */
__global__ void
zgemvn_kernel1_fermi(
    int m, int n, magmaDoubleComplex alpha,
    const magmaDoubleComplex * __restrict__ A, int lda,
    const magmaDoubleComplex * __restrict__ x, int incx, magmaDoubleComplex beta,
    magmaDoubleComplex       * __restrict__ y, int incy)
{
#if (__CUDA_ARCH__ >= 200)
    int ind = blockIdx.x*BLK_X + threadIdx.x;
    if ( ind < m ) {
        A += ind;
        
        magmaDoubleComplex res = MAGMA_Z_ZERO;
        
        #pragma unroll
        for(int j=0; j < n; j++) {
            res += A[j*lda] * x[j*incx];
        }
        
        y[ind*incy] = alpha*res + beta*y[ind*incy];
    }
#endif /* (__CUDA_ARCH__ >= 200) */
}


/* Compute y = alpha*A*x + beta*y.
 * Each thread block does a BLK_X x N block row of A.
 * Each thread goes across one row, accumulating dot product of row ind and x into res.
 * This implementation loads BLK_Y elements into sx, then multiplies
 * BLK_Y columns of A*sx.
 */
__global__ void
zgemvn_kernel2_fermi(
    int m, int n, magmaDoubleComplex alpha,
    const magmaDoubleComplex * __restrict__ A, int lda,
    const magmaDoubleComplex * __restrict__ x, int incx, magmaDoubleComplex beta,
    magmaDoubleComplex       * __restrict__ y, int incy)
{
#if (__CUDA_ARCH__ >= 200)
    int ind = blockIdx.x*BLK_X + threadIdx.x;
    
    // threads past last row redundantly work on last row
    A += min( ind, m-1 );
    x += threadIdx.x*incx;
    
    magmaDoubleComplex res = MAGMA_Z_ZERO;
    
    __shared__ magmaDoubleComplex sx[BLK_Y];
    
    // full block-columns
    int nfull = (n / BLK_Y) * BLK_Y;
    for( int j=0; j < nfull; j += BLK_Y ) {
        // load BLK_Y elements of x into sx
        sx[threadIdx.x] = x[0];
        x += BLK_Y*incx;
        __syncthreads();
        
        // multiply A*sx
        #pragma unroll
        for(int j2=0; j2 < BLK_Y; j2++) {
            res += A[0] * sx[j2];
            A += lda;
        }
        __syncthreads();
    }
    
    // last, partial block-column
    // load remaining npart elements of x into sx
    int npart = n % BLK_Y;
    if ( threadIdx.x < npart ) {
        sx[threadIdx.x] = x[0];
    }
    else {
        sx[threadIdx.x] = MAGMA_Z_ZERO;
    }
    __syncthreads();
        
    // multiply A*sx
    #pragma unroll
    for(int j2=0; j2 < npart; j2++) {
        res += A[0]*sx[j2];
        A += lda;
    }
    
    if ( ind < m ) {
        y[ind*incy] = alpha*res + beta*y[ind*incy];
    }
#endif /* (__CUDA_ARCH__ >= 200) */
}


/* Compute y = alpha * A^T * x + beta*y.
 * Each thread block does one column of A (i.e., one row of A^T).
 * Each thread does a partial sum, then collectively they do a reduction.
 */
__global__ void
zgemvt_kernel_fermi(
    int m, int n, magmaDoubleComplex alpha,
    const magmaDoubleComplex * __restrict__ A, int lda,
    const magmaDoubleComplex * __restrict__ x, int incx, magmaDoubleComplex beta,
    magmaDoubleComplex       * __restrict__ y, int incy)
{
#if (__CUDA_ARCH__ >= 200)
    int tx = threadIdx.x;

    __shared__ magmaDoubleComplex sdata[BLK_X];

    magmaDoubleComplex res = MAGMA_Z_ZERO;
    
    A += blockIdx.y*lda + threadIdx.x;
 
    // partial sums
    int mfull = (m / BLK_X) * BLK_X;
    for(int i=0; i < mfull; i += BLK_X) {
        res += A[i] * x[tx + i];
    }
    if ( tx + mfull < m ) {
        res += A[mfull] * x[tx + mfull];
    }
    sdata[tx] = res;

    // tree reduction of partial sums,
    // from BLK_X sums to ... 128 to 64 to 32 ... to 1 sum in sdata[0]
    magma_sum_reduce< BLK_X >( tx, sdata );

    if ( tx == 0 ) {
        y[blockIdx.y*incy] = alpha*sdata[0] + beta*y[blockIdx.y*incy];
    }
#endif /* (__CUDA_ARCH__ >= 200) */
}


/* Compute y = alpha * A^H * x + beta*y.
 * Same as zgemvt_kernel_fermi but conjugates entries of A.
 */
__global__ void
zgemvc_kernel_fermi(
    int m, int n, magmaDoubleComplex alpha,
    const magmaDoubleComplex * __restrict__ A, int lda,
    const magmaDoubleComplex * __restrict__ x, int incx, magmaDoubleComplex beta,
    magmaDoubleComplex       * __restrict__ y, int incy)
{
#if (__CUDA_ARCH__ >= 200)
    int tx = threadIdx.x;

    __shared__ magmaDoubleComplex sdata[BLK_X];

    magmaDoubleComplex res = MAGMA_Z_ZERO;
    
    A += blockIdx.y*lda + threadIdx.x;
 
    // partial sums
    int mfull = (m / BLK_X) * BLK_X;
    for(int i=0; i < mfull; i += BLK_X) {
        res += conj(A[i]) * x[tx + i];
    }
    if ( tx + mfull < m ) {
        res += conj(A[mfull]) * x[tx + mfull];
    }
    sdata[tx] = res;

    // tree reduction of partial sums,
    // from BLK_X sums to ... 128 to 64 to 32 ... to 1 sum in sdata[0]
    magma_sum_reduce< BLK_X >( tx, sdata );

    if ( tx == 0 ) {
        y[blockIdx.y*incy] = alpha*sdata[0] + beta*y[blockIdx.y*incy];
    }
#endif /* (__CUDA_ARCH__ >= 200) */
}


/**
    Purpose
    -------
    ZGEMV performs one of the matrix-vector operations
    
        y := alpha*A*x    + beta*y,   or
        y := alpha*A**T*x + beta*y,   or
        y := alpha*A**H*x + beta*y,
    
    where alpha and beta are scalars, x and y are vectors and A is an
    m by n matrix.

    Arguments
    ----------
    @param[in]
    trans   magma_trans_t
            On entry, TRANS specifies the operation to be performed as
            follows:
      -     = MagmaNoTrans:    y := alpha*A  *x + beta*y
      -     = MagmaTrans:      y := alpha*A^T*x + beta*y
      -     = MagmaConjTrans:  y := alpha*A^H*x + beta*y

    @param[in]
    m       INTEGER
            On entry, m specifies the number of rows of the matrix A.

    @param[in]
    n       INTEGER
            On entry, n specifies the number of columns of the matrix A
 
    @param[in]
    alpha   COMPLEX_16
            On entry, ALPHA specifies the scalar alpha.

    @param[in]
    dA      COMPLEX_16 array of dimension ( LDDA, n ) on the GPU.
   
    @param[in]
    ldda    INTEGER
            LDDA specifies the leading dimension of A.

    @param[in]
    dx      COMPLEX_16 array of dimension
            n if trans == MagmaNoTrans
            m if trans == MagmaTrans or MagmaConjTrans
     
    @param[in]
    incx    Specifies the increment for the elements of X.
            INCX must not be zero.
  
    @param[in]
    beta    DOUBLE REAL
            On entry, BETA specifies the scalar beta. When BETA is
            supplied as zero then Y need not be set on input.

    @param[out]
    dy      DOUBLE PRECISION array of dimension
            m if trans == MagmaNoTrans
            n if trans == MagmaTrans or MagmaConjTrans

    @param[in]
    incy    Specifies the increment for the elements of Y.
            INCY must not be zero.

    @ingroup magma_dblas2
    ********************************************************************/
extern "C" void
magmablas_zgemv(
    magma_trans_t trans, magma_int_t m, magma_int_t n, magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy, magma_int_t incy)
{
    magma_int_t info = 0;
    if ( trans != MagmaNoTrans && trans != MagmaTrans && trans != MagmaConjTrans )
        info = -1;
    else if ( m < 0 )
        info = -2;
    else if ( n < 0 )
        info = -3;
    else if ( ldda < m )
        info = -6;
    else if ( incx == 0 )
        info = -8;
    else if ( incy == 0 )
        info = -11;
    
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    magma_int_t arch = magma_getdevice_arch();
    if ( arch < 200  ) {
        // --------------------
        // call CUDA ARCH 1.x version
        // magmablas for [sd] precisions, cublas for [zc] precisions.
        #if defined(PRECISION_z) || defined(PRECISION_c)
        magma_zgemv( trans, m, n, alpha, dA, ldda, dx, incx, beta, dy, incy );
        #else
        magmablas_zgemv_tesla( trans, m, n, alpha, dA, ldda, dx, incx, beta, dy, incy );
        #endif
        return;
    }
    
    // --------------------
    // CUDA ARCH 2.x (Fermi) version
    if ( trans == MagmaNoTrans ) {
        dim3 grid( magma_ceildiv( m, BLK_X ) );
        dim3 threads( BLK_X, 1, 1 );
        zgemvn_kernel1_fermi<<< grid, threads, 0, magma_stream >>>
            ( m, n, alpha, dA, ldda, dx, incx, beta, dy, incy );
    }
    else if ( trans == MagmaTrans ) {
        dim3 grid    ( 1, n, 1 );
        dim3 threads ( BLK_X, 1, 1 );
        zgemvt_kernel_fermi<<< grid, threads, 0, magma_stream >>>
            ( m, n, alpha, dA, ldda, dx, incx, beta, dy, incy );
    }
    else if ( trans == MagmaConjTrans ) {
        dim3 grid    ( 1, n, 1 );
        dim3 threads ( BLK_X, 1, 1 );
        zgemvc_kernel_fermi<<< grid, threads, 0, magma_stream >>>
            ( m, n, alpha, dA, ldda, dx, incx, beta, dy, incy );
    }
}
