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
static __device__ void
zgemvn_device(
    int m, int n, magmaDoubleComplex alpha,
    const magmaDoubleComplex * __restrict__ A, int lda,
    const magmaDoubleComplex * __restrict__ x, int incx, magmaDoubleComplex beta,
    magmaDoubleComplex       * __restrict__ y, int incy)
{

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

}

__global__ void
zgemvn_kernel_fermi(
    int m, int n, magmaDoubleComplex alpha,
    const magmaDoubleComplex * __restrict__ A, int lda,
    const magmaDoubleComplex * __restrict__ x, int incx, magmaDoubleComplex beta,
    magmaDoubleComplex       * __restrict__ y, int incy)
{
#if (__CUDA_ARCH__ >= 200)
    zgemvn_device(m, n, alpha, A, lda, x, incx, beta, y, incy);
#endif /* (__CUDA_ARCH__ >= 200) */
}



__global__ void
zgemvn_kernel_batched(
    int m, int n, magmaDoubleComplex alpha,
    magmaDoubleComplex **A_array, int lda,
    magmaDoubleComplex **x_array,  int incx,
    magmaDoubleComplex beta, magmaDoubleComplex  **y_array, int incy)
{
    int batchid = blockIdx.z;

    zgemvn_device(m, n, alpha, A_array[batchid], lda, x_array[batchid], incx, beta, y_array[batchid], incy);
}


//////////////////////////////////////////////////////////////////////////////////////////


/* Compute y = alpha * A^T * x + beta*y.
 * Each thread block does one column of A (i.e., one row of A^T).
 * Each thread does a partial sum, then collectively they do a reduction.
 */
static __device__ void
zgemvt_device(
    int m, int n, magmaDoubleComplex alpha,
    const magmaDoubleComplex * __restrict__ A, int lda,
    const magmaDoubleComplex * __restrict__ x, int incx, magmaDoubleComplex beta,
    magmaDoubleComplex       * __restrict__ y, int incy)
{

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

}



__global__ void
zgemvt_kernel_fermi(
    int m, int n, magmaDoubleComplex alpha,
    const magmaDoubleComplex * __restrict__ A, int lda,
    const magmaDoubleComplex * __restrict__ x, int incx, magmaDoubleComplex beta,
    magmaDoubleComplex       * __restrict__ y, int incy)
{
#if (__CUDA_ARCH__ >= 200)
    zgemvt_device(m, n, alpha, A, lda, x, incx, beta, y, incy);
#endif /* (__CUDA_ARCH__ >= 200) */
}



__global__ void
zgemvt_kernel_batched(
    int m, int n, magmaDoubleComplex alpha,
    magmaDoubleComplex **A_array, int lda,
    magmaDoubleComplex **x_array,  int incx,
    magmaDoubleComplex beta, magmaDoubleComplex  **y_array, int incy)
{
    int batchid = blockIdx.z;

    zgemvt_device(m, n, alpha, A_array[batchid], lda, x_array[batchid], incx, beta, y_array[batchid], incy);
}


//////////////////////////////////////////////////////////////////////////////////////////

/* Compute y = alpha * A^H * x + beta*y.
 * Same as zgemvt_kernel_fermi but conjugates entries of A.
 */
static __device__ void
zgemvc_device(
    int m, int n, magmaDoubleComplex alpha,
    const magmaDoubleComplex * __restrict__ A, int lda,
    const magmaDoubleComplex * __restrict__ x, int incx, magmaDoubleComplex beta,
    magmaDoubleComplex       * __restrict__ y, int incy)
{

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

}

__global__ void
zgemvc_kernel_fermi(
    int m, int n, magmaDoubleComplex alpha,
    const magmaDoubleComplex * __restrict__ A, int lda,
    const magmaDoubleComplex * __restrict__ x, int incx, magmaDoubleComplex beta,
    magmaDoubleComplex       * __restrict__ y, int incy)
{
#if (__CUDA_ARCH__ >= 200)
    zgemvc_device(m, n, alpha, A, lda, x, incx, beta, y, incy);
#endif /* (__CUDA_ARCH__ >= 200) */
}



__global__ void
zgemvc_kernel_batched(
    int m, int n, magmaDoubleComplex alpha,
    magmaDoubleComplex **A_array, int lda,
    magmaDoubleComplex **x_array,  int incx,
    magmaDoubleComplex beta, magmaDoubleComplex  **y_array, int incy)
{
    int batchid = blockIdx.z;

    zgemvc_device(m, n, alpha, A_array[batchid], lda, x_array[batchid], incx, beta, y_array[batchid], incy);
}


//////////////////////////////////////////////////////////////////////////////////////////

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
        dim3 grid( magma_ceildiv(m, BLK_X) );
        dim3 threads( BLK_X, 1, 1 );
        zgemvn_kernel_fermi<<< grid, threads, 0, magma_stream >>>
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



extern "C" void
magmablas_zgemv_batched(
    magma_trans_t trans, magma_int_t m, magma_int_t n, 
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dA_array[], magma_int_t ldda, 
    magmaDoubleComplex_ptr dx_array[], magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy_array[], magma_int_t incy, 
    magma_int_t batchCount, magma_queue_t queue)
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


    // --------------------
    if ( trans == MagmaNoTrans ) {
        dim3 grid( magma_ceildiv(m, BLK_X), 1, batchCount );
        dim3 threads( BLK_X, 1, 1 );
        zgemvn_kernel_batched<<< grid, threads, 0, queue >>>
            ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy );
    }
    else if ( trans == MagmaTrans ) {
        dim3 grid    ( 1, n, batchCount );
        dim3 threads ( BLK_X, 1, 1 );
        zgemvt_kernel_batched<<< grid, threads, 0, queue >>>
            ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy );
    }
    else if ( trans == MagmaConjTrans ) {
        dim3 grid    ( 1, n, batchCount );
        dim3 threads ( BLK_X, 1, 1 );
        zgemvc_kernel_batched<<< grid, threads, 0, queue >>>
            ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy );
    }
}
