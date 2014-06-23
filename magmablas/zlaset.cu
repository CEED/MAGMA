/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

*/
#include "common_magma.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- This is an auxiliary routine called from zgehrd.  The routine is called
      in 16 blocks, 32 thread per block and initializes to zero the 1st
      32x32 block of A.
*/

__global__ void zset_to_zero(magmaDoubleComplex *A, int lda)
{
    int ind = blockIdx.x*lda + threadIdx.x;

    A += ind;
    A[0] = MAGMA_Z_ZERO;
    //A[16*lda] = 0.;
}

__global__ void zset_nbxnb_to_zero(int nb, magmaDoubleComplex *A, int lda)
{
    int ind = blockIdx.x*lda + threadIdx.x, i, j;
    
    A += ind;
    for (i=0; i < nb; i += 32) {
        for (j=0; j < nb; j += 32)
            A[j] = MAGMA_Z_ZERO;
        A += 32*lda;
    }
}

extern "C"
void zzero_32x32_block(magmaDoubleComplex *A, magma_int_t lda)
{
    // zset_to_zero<<< 16, 32, 0, magma_stream >>>(A, lda);
    zset_to_zero<<< 32, 32, 0, magma_stream >>>(A, lda);
}

extern "C"
void zzero_nbxnb_block(magma_int_t nb, magmaDoubleComplex *A, magma_int_t lda)
{
    zset_nbxnb_to_zero<<< 32, 32, 0, magma_stream >>>(nb, A, lda);
}

/* ////////////////////////////////////////////////////////////////////////////
   -- GPU kernel for initializing a matrix by 0
*/
#define zlaset_threads 64

__global__ void zlaset(int m, int n, magmaDoubleComplex *A, int lda)
{
    int ibx = blockIdx.x * zlaset_threads;
    int iby = blockIdx.y * 32;
    
    int ind = ibx + threadIdx.x;
    
    A += ind + __mul24(iby, lda);
    
    #pragma unroll
    for (int i=0; i < 32; i++) {
        if (iby+i < n && ind < m)
            A[i*lda] = MAGMA_Z_ZERO;
    }
}

__global__ void zlaset_identity(int m, int n, magmaDoubleComplex *A, int lda)
{
    int ibx = blockIdx.x * zlaset_threads;
    int iby = blockIdx.y * 32;
    
    int ind = ibx + threadIdx.x;
    
    A += ind + __mul24(iby, lda);
    
    #pragma unroll
    for (int i=0; i < 32; i++) {
        if (iby+i < n && ind < m) {
            if (ind != i+iby)
                A[i*lda] = MAGMA_Z_ZERO;
            else
                A[i*lda] = MAGMA_Z_ONE;
        }
    }
}

__global__ void zlaset_identityonly(int m, int n, magmaDoubleComplex *A, int lda)
{
    int ibx = blockIdx.x * zlaset_threads;
    int iby = blockIdx.y * 32;
    
    int ind = ibx + threadIdx.x;
    
    A += ind + __mul24(iby, lda);
    
    #pragma unroll
    for (int i=0; i < 32; i++) {
        if (iby+i < n && ind < m) {
            if (ind == i+iby)
                A[i*lda] = MAGMA_Z_ONE;
        }
    }
}


__global__ void zlasetlower(int m, int n, magmaDoubleComplex *A, int lda)
{
    int ibx = blockIdx.x * zlaset_threads;
    int iby = blockIdx.y * 32;
    
    int ind = ibx + threadIdx.x;
    
    A += ind + __mul24(iby, lda);
    
    #pragma unroll
    for (int i=0; i < 32; i++) {
        if (iby+i < n && ind < m && ind > i+iby)
            A[i*lda] = MAGMA_Z_ZERO;
    }
}

__global__ void zlasetupper(int m, int n, magmaDoubleComplex *A, int lda)
{
    int ibx = blockIdx.x * zlaset_threads;
    int iby = blockIdx.y * 32;
    
    int ind = ibx + threadIdx.x;
    
    A += ind + __mul24(iby, lda);
    
    #pragma unroll
    for (int i=0; i < 32; i++) {
        if (iby+i < n && ind < m && ind < i+iby)
            A[i*lda] = MAGMA_Z_ZERO;
    }
}


/* ////////////////////////////////////////////////////////////////////////////
   -- Set the m x n matrix pointed by A to 0 on the GPU.
*/
extern "C" void
magmablas_zlaset(magma_uplo_t uplo, magma_int_t m, magma_int_t n,
                 magmaDoubleComplex *A, magma_int_t lda)
{
    dim3 threads(zlaset_threads, 1, 1);
    dim3 grid(m/zlaset_threads+(m % zlaset_threads != 0), n/32+(n%32 != 0));
    
    if (m != 0 && n != 0) {
        if (uplo == MagmaLower)
            zlasetlower<<< grid, threads, 0, magma_stream >>> (m, n, A, lda);
        else if (uplo == MagmaUpper)
            zlasetupper<<< grid, threads, 0, magma_stream >>> (m, n, A, lda);
        else
            zlaset<<< grid, threads, 0, magma_stream >>> (m, n, A, lda);
    }
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Set the m x n matrix pointed by A to I on the GPU.
*/
extern "C" void
magmablas_zlaset_identity(magma_int_t m, magma_int_t n,
                          magmaDoubleComplex *A, magma_int_t lda)
{
    dim3 threads(zlaset_threads, 1, 1);
    dim3 grid(m/zlaset_threads+(m % zlaset_threads != 0), n/32+(n%32 != 0));
    
    if (m != 0 && n != 0)
        zlaset_identity<<< grid, threads, 0, magma_stream >>> (m, n, A, lda);
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Set the m x n matrix pointed by A to I on the diag without touching the offdiag GPU.
*/
extern "C" void
magmablas_zlaset_identityonly(magma_int_t m, magma_int_t n,
                          magmaDoubleComplex *A, magma_int_t lda)
{
    dim3 threads(zlaset_threads, 1, 1);
    dim3 grid(m/zlaset_threads+(m % zlaset_threads != 0), n/32+(n%32 != 0));
    
    if (m != 0 && n != 0)
        zlaset_identityonly<<< grid, threads, 0, magma_stream >>> (m, n, A, lda);
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Given two matrices, 'a' on the CPU and 'da' on the GPU, this function
      returns the Frobenious norm of the difference of the two matrices.
      The function is used for debugging.
*/
extern "C"
double cpu_gpu_zdiff(
    magma_int_t M, magma_int_t N,
    const magmaDoubleComplex *a,  magma_int_t lda,
    const magmaDoubleComplex *da, magma_int_t ldda )
{
    magma_int_t d_one = 1;
    magma_int_t j;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    double  work[1];
    magmaDoubleComplex *ha;
    magma_zmalloc_cpu( &ha, M*N );
    double res;
    
    cublasGetMatrix(M, N, sizeof(magmaDoubleComplex), da, ldda, ha, M);
    for (j=0; j < N; j++)
        blasf77_zaxpy(&M, &c_neg_one, a+j*lda, &d_one, ha+j*M, &d_one);
    res = lapackf77_zlange("f", &M, &N, ha, &M, work);
    
    magma_free_cpu(ha);
    return res;
}



#define LASET_BAND_NB 64

/* ////////////////////////////////////////////////////////////////////////////
 -- GPU kernel for setting the k-1 super-diagonals to OFFDIAG
    and the main diagonal to DIAG.
    Divides matrix into min( ceil((m+k-1)/nb), ceil(n/nb) ) block-columns,
    with k threads in each block.
    Each thread iterates across one diagonal.
    Thread k-1 does the main diagonal, thread k-2 the first super-diagonal, etc.
    
      block 0           block 1
      0                           => skip above matrix
      1 0                         => skip above matrix
      2 1 0                       => skip above matrix
    [ 3 2 1 0         |         ]
    [   3 2 1 0       |         ]
    [     3 2 1 0     |         ]
    [       3 2 1 0   |         ]
    [         3 2 1 0 |         ]
    [           3 2 1 | 0       ]
    [             3 2 | 1 0     ]
    [               3 | 2 1 0   ]
    [                 | 3 2 1 0 ]
    [                 |   3 2 1 ]
                      |     3 2   => skip below matrix
                              3   => skip below matrix
    
    Thread assignment for m=10, n=12, k=4, nb=8. Each column is done in parallel.
    
    @author Raffaele Solca
    @author Mark Gates
 */
__global__
void zlaset_band_upper(
    int m, int n,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    magmaDoubleComplex *A, int lda)
{
    int k   = blockDim.x;
    int ibx = blockIdx.x * LASET_BAND_NB;
    int ind = ibx + threadIdx.x - k + 1;
    
    A += ind + ibx*lda;
    
    magmaDoubleComplex value = offdiag;
    if (threadIdx.x == k-1)
        value = diag;

    #pragma unroll
    for (int j=0; j < LASET_BAND_NB; j++) {
        if (ibx + j < n && ind + j >= 0 && ind + j < m) {
            A[j*(lda+1)] = value;
        }
    }
}

/* ////////////////////////////////////////////////////////////////////////////
 -- GPU kernel for setting the k-1 sub-diagonals to OFFDIAG
    and the main diagonal to DIAG.
    Divides matrix into min( ceil(m/nb), ceil(n/nb) ) block-columns,
    with k threads in each block.
    Each thread iterates across one diagonal.
    Thread 0 does the main diagonal, thread 1 the first sub-diagonal, etc.
    
      block 0           block 1
    [ 0               |         ]
    [ 1 0             |         ]
    [ 2 1 0           |         ]
    [ 3 2 1 0         |         ]
    [   3 2 1 0       |         ]
    [     3 2 1 0     |         ]
    [       3 2 1 0   |         ]
    [         3 2 1 0 |         ]
    [           3 2 1 | 0       ]
    [             3 2 | 1 0     ]
    [               3 | 2 1 0   ]
    [                   3 2 1 0 ]
    [                     3 2 1 ]
                            3 2   => skip below matrix
                              3   => skip below matrix
    
    Thread assignment for m=13, n=12, k=4, nb=8. Each column is done in parallel.
    
    @author Raffaele Solca
    @author Mark Gates
 */

__global__
void zlaset_band_lower(
    int m, int n,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    magmaDoubleComplex *A, int lda)
{
    //int k   = blockDim.x;
    int ibx = blockIdx.x * LASET_BAND_NB;
    int ind = ibx + threadIdx.x;
    
    A += ind + ibx*lda;
    
    magmaDoubleComplex value = offdiag;
    if (threadIdx.x == 0)
        value = diag;
    
    #pragma unroll
    for (int j=0; j < LASET_BAND_NB; j++) {
        if (ibx + j < n && ind + j < m) {
            A[j*(lda+1)] = value;
        }
    }
}


/**
    Purpose
    -------
    ZLASET_BAND_STREAM initializes the main diagonal of dA to DIAG,
    and the K-1 sub- or super-diagonals to OFFDIAG.
    
    This is the same as ZLASET_BAND, but adds stream argument.
    
    Arguments
    ---------
    
    @param[in]
    uplo    magma_uplo_t
            Specifies the part of the matrix dA to be set.
      -     = MagmaUpper:      Upper triangular part
      -     = MagmaLower:      Lower triangular part
    
    @param[in]
    m       INTEGER
            The number of rows of the matrix dA.  M >= 0.
    
    @param[in]
    n       INTEGER
            The number of columns of the matrix dA.  N >= 0.
    
    @param[in]
    k       INTEGER
            The number of diagonals to set, including the main diagonal.  K >= 0.
            Currently, K <= 1024 due to CUDA restrictions (max. number of threads per block).
    
    @param[in]
    OFFDIAG COMPLEX_16
            Off-diagonal elements in the band are set to OFFDIAG.
    
    @param[in]
    DIAG    COMPLEX_16
            All the main diagonal elements are set to DIAG.
    
    @param[in]
    dA      COMPLEX DOUBLE PRECISION array, dimension (LDDA,N)
            The M-by-N matrix dA.
            If UPLO = MagmaUpper, only the upper triangle or trapezoid is accessed;
            if UPLO = MagmaLower, only the lower triangle or trapezoid is accessed.
            On exit, A(i,j) = ALPHA, 1 <= i <= m, 1 <= j <= n where i != j, abs(i-j) < k;
                     A(i,i) = BETA , 1 <= i <= min(m,n)
    
    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,M).
    
    @param[in]
    stream  magma_queue_t
            Stream to execute ZLASET in.
    
    @author Raffaele Solca
    @author Mark Gates
    
    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void
magmablas_zlaset_band_stream(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    magmaDoubleComplex *dA, magma_int_t ldda, magma_queue_t stream)
{
    magma_int_t info = 0;
    if ( uplo != MagmaLower && uplo != MagmaUpper )
        info = -1;
    else if ( m < 0 )
        info = -2;
    else if ( n < 0 )
        info = -3;
    else if ( k < 0 || k > 1024 )
        info = -4;
    else if ( ldda < max(1,m) )
        info = -6;
    
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    if (uplo == MagmaUpper) {
        dim3 threads( min(k,n) );
        dim3 grid( (min(m+k-1,n) - 1)/LASET_BAND_NB + 1 );
        zlaset_band_upper<<< grid, threads, 0, stream >>> (m, n, offdiag, diag, dA, ldda);
    }
    else if (uplo == MagmaLower) {
        dim3 threads( min(k,m) );
        dim3 grid( (min(m,n) - 1)/LASET_BAND_NB + 1 );
        zlaset_band_lower<<< grid, threads, 0, stream >>> (m, n, offdiag, diag, dA, ldda);
    }
}


/**
    @see magmablas_zlaset_band_stream
    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void
magmablas_zlaset_band(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    magmaDoubleComplex *dA, magma_int_t ldda)
{
    magmablas_zlaset_band_stream(uplo, m, n, k, offdiag, diag, dA, ldda, magma_stream);
}
