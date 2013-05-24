/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> s d c
*/

#include <stdio.h>
#include "common_magma.h"
#include "magmablas.h"

#define PRECISION_z

#define A(i, j)  (A + (i) + (j)*lda)   // A(i, j) means at i row, j column

void magma_zswap(
    magma_int_t n, cuDoubleComplex *x, magma_int_t i, magma_int_t j, magma_int_t incx);

void magma_zscal_zgeru(
    magma_int_t m, magma_int_t n, cuDoubleComplex *A, magma_int_t lda);


extern "C" magma_int_t
magma_zgetf2_gpu(
    magma_int_t m, magma_int_t n,
    cuDoubleComplex *A, magma_int_t lda,
    magma_int_t *ipiv,
    magma_int_t* info )
{
/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    ZGETF2 computes an LU factorization of a general m-by-n matrix A
    using partial pivoting with row interchanges.

    The factorization has the form
        A = P * L * U
    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 2 BLAS version of the algorithm.

    Arguments
    =========

    M       (input) INTEGER
            The number of rows of the matrix A.  M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix A.  N >= 0 and N <= 1024.

    A       (input/output) COMPLEX_16 array, dimension (LDA,N)
            On entry, the m by n matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    IPIV    (output) INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).

    INFO    (output) INTEGER
            = 0: successful exit
            < 0: if INFO = -k, the k-th argument had an illegal value
            > 0: if INFO = k, U(k,k) is exactly zero. The factorization
                 has been completed, but the factor U is exactly
                 singular, and division by zero will occur if it is used
                 to solve a system of equations.

    ===================================================================== */

    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if (n < 0 || n > 1024) {
        *info = -2;
    } else if (lda < max(1,m)) {
        *info = -4;
    }

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    // Quick return if possible
    if (m == 0 || n == 0) {
        return *info;
    }

    magma_int_t minmn = min(m, n);
    magma_int_t j;

    for (j = 0; j < minmn; j++) {
        cudaDeviceSetCacheConfig( cudaFuncCachePreferShared );

        // Find pivot and test for singularity.
        int jp;
        jp = cublasIzamax(m-j, A(j,j), 1);
        jp = jp - 1 + j;
        ipiv[j] = jp + 1;
        // Can't check value of A since it is on GPU
        //if ( A(jp, j) != 0.0) {
            cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );
            
            // Apply the interchange to columns 1:N.
            if (jp != j){
                magma_zswap(n, A, j, jp, lda);
            }
            
            // Compute elements J+1:M of J-th column.
            if (j < m) {
                magma_zscal_zgeru(m-j, n-j, A(j, j), lda);
            }
        //}
        //else if (*info == 0) {
        //    *info = j;
        //}
    }

    return *info;
}


#define zswap_bs 64
#define zgeru_bs 1024


__global__ void kernel_zswap(int n, cuDoubleComplex *x, int i, int j, int incx)
{
    int id = blockIdx.x * zswap_bs + threadIdx.x;

    if (id < n) {
        cuDoubleComplex res = x[i + incx*id];
        x[i + incx*id] = x[j + incx*id];
        x[j + incx*id] = res;
    }
}


void magma_zswap(int n, cuDoubleComplex *x, int i, int j, int incx)
{
/*
    zswap two row vectors: ith and jth
*/
    dim3 threads(zswap_bs, 1, 1);
    int num_blocks = (n - 1)/zswap_bs + 1;
    dim3 grid(num_blocks,1);
    kernel_zswap<<< grid, threads, 0, magma_stream >>>(n, x, i, j, incx);
}


extern __shared__ cuDoubleComplex shared_data[];

__global__ void
kernel_zscal_zgeru(int m, int n, cuDoubleComplex *A, int lda)
{
    cuDoubleComplex *shared_y = (cuDoubleComplex *)shared_data;

    int tid = blockIdx.x * zgeru_bs + threadIdx.x;

    cuDoubleComplex reg = MAGMA_Z_ZERO;

    if (threadIdx.x < n) {
        shared_y[threadIdx.x] = A[lda * threadIdx.x];
    }

    __syncthreads();

    if (tid < m && tid > 0) {
        reg = A[tid];

        reg *= MAGMA_Z_DIV(MAGMA_Z_ONE, shared_y[0]);

        A[tid] = reg;

        #pragma unroll
        for(int i=1; i < n; i++) {
            A[tid + i*lda] += (MAGMA_Z_NEG_ONE) * shared_y[i] * reg;
        }
    }
}


void magma_zscal_zgeru(int m, int n, cuDoubleComplex *A, int lda)
{
/*

    Specialized kernel which merged zscal and zgeru the two kernels
    1) zscale a column vector
    2) Performe a zgeru Operation A := alpha*x*y**T + A,

*/
    dim3 threads(zgeru_bs, 1, 1);
    int num_blocks = (m - 1)/zgeru_bs + 1;
    dim3 grid(num_blocks,1);
    size_t shared_size = sizeof(cuDoubleComplex)*(n);
    kernel_zscal_zgeru<<< grid, threads, shared_size, magma_stream>>>(m, n, A, lda);
}
