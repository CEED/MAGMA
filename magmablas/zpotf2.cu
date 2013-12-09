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

//#if (GPUSHMEM < 200)
#define zdotc_max_bs 512  // 512 is max threads for 1.x cards
//#else
//#define zdotc_max_bs 1024
//#endif

#define A(i, j)  (A + (i) + (j)*lda)   // A(i, j) means at i row, j column

void zpotf2_zdscal(magma_int_t n, magmaDoubleComplex *x, magma_int_t incx);
void zpotf2_zdotc(magma_int_t n, magmaDoubleComplex *x, magma_int_t incx);

#if defined(PRECISION_z) || defined(PRECISION_c)
void zlacgv(magma_int_t n, magmaDoubleComplex *x, magma_int_t incx);
#endif

extern "C" magma_int_t
magma_zpotf2_gpu(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    magma_int_t *info )
{
/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======

    zpotf2 computes the Cholesky factorization of a real symmetric
    positive definite matrix A.

    The factorization has the form
        A = U' * U , if UPLO = 'U', or
        A = L  * L', if UPLO = 'L',
    where U is an upper triangular matrix and L is lower triangular.

    This is the unblocked version of the algorithm, calling Level 2 BLAS.

    Arguments
    =========

    UPLO    (input) CHARACTER*1
            Specifies whether the upper or lower triangular part of the
            symmetric matrix A is stored.
            = 'U':  Upper triangular
            = 'L':  Lower triangular

    N       (input) INTEGER
            The order of the matrix A.  N >= 0 and N <= 1024.
            On CUDA architecture 1.x cards, N <= 512.

    A       (input/output) COMPLEX_16 array, dimension (LDA,N)
            On entry, the symmetric matrix A.  If UPLO = 'U', the leading
            n by n upper triangular part of A contains the upper
            triangular part of the matrix A, and the strictly lower
            triangular part of A is not referenced.  If UPLO = 'L', the
            leading n by n lower triangular part of A contains the lower
            triangular part of the matrix A, and the strictly upper
            triangular part of A is not referenced.

            On exit, if INFO = 0, the factor U or L from the Cholesky
            factorization A = U'*U  or A = L*L'.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    INFO    (output) INTEGER
            = 0: successful exit
            < 0: if INFO = -k, the k-th argument had an illegal value
            > 0: if INFO = k, the leading minor of order k is not
                 positive definite, and the factorization could not be
                 completed.

    ===================================================================== */

    magma_int_t j;

    *info = 0;
    if ( uplo != 'U' && uplo != 'u' && uplo != 'L' && uplo != 'l') {
        *info = -1;
    } else if (n < 0 || n > zdotc_max_bs) {
        *info = -2;
    } else if (lda < max(1,n)) {
        *info = -4;
    }

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    // Quick return if possible
    if (n == 0) {
        return *info;
    }

    magmaDoubleComplex alpha = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex beta  = MAGMA_Z_ONE;

    if (uplo == 'U' || uplo == 'u') {
        for(j = 0; j < n; j++) {
            zpotf2_zdotc(j, A(0,j), 1); // including zdotc product and update a(j,j)
            if (j < n) {
                #if defined(PRECISION_z) || defined(PRECISION_c)
                zlacgv(j, A(0, j), 1);
                #endif
                cublasZgemv( MagmaTrans, j, n-j-1,
                             alpha, A(0, j+1), lda,
                                    A(0, j),   1,
                             beta,  A(j, j+1), lda); // cublas is better in upper case

                #if defined(PRECISION_z) || defined(PRECISION_c)
                zlacgv(j, A(0, j), 1);
                #endif
                zpotf2_zdscal(n-j, A(j,j), lda);
            }
        }
    }
    else {
        for(j = 0; j < n; j++) {
            zpotf2_zdotc(j, A(j,0), lda); // including zdotc product and update a(j,j)
            if (j < n) {
                #if defined(PRECISION_z) || defined(PRECISION_c)
                zlacgv(j, A(j, 0), lda);
                #endif
                magmablas_zgemv( MagmaNoTrans, n-j-1, j,
                                 alpha, A(j+1, 0), lda,
                                        A(j,0),    lda,
                                 beta,  A(j+1, j), 1 );// magmablas is better in lower case

                #if defined(PRECISION_z) || defined(PRECISION_c)
                zlacgv(j, A(j, 0), lda);
                #endif
                zpotf2_zdscal(n-j, A(j,j), 1);
            }
        }
    }

    return *info;
}

#define zdscal_bs  32
#define zdotc_bs  512
#define zlacgv_bs 512

// dynamically allocated shared memory, set to size number of threads when the kernel is launched.
// See CUDA Guide B.2.3
extern __shared__ double shared_data[];

__global__ void kernel_zdotc(int n, magmaDoubleComplex *x, int incx, int threadSize)
{
    int tx = threadIdx.x;

    double *sdata = shared_data;

    magmaDoubleComplex res = MAGMA_Z_ZERO;

    if (tx < n) {
       res = x[tx*incx];
    }

    sdata[tx] = MAGMA_Z_REAL(res * MAGMA_Z_CNJG(res));

    __syncthreads();

    for(int s = blockDim.x/2; s > 32; s >>= 1 ) {
        if (tx < s) {
            sdata[tx] += sdata[tx+s];
        }
        __syncthreads();
    }

    if (tx < 32) {
        volatile double* smem = sdata;
        smem[tx] += smem[tx+32];
        smem[tx] += smem[tx+16];
        smem[tx] += smem[tx+8];
        smem[tx] += smem[tx+4];
        smem[tx] += smem[tx+2];
        smem[tx] += smem[tx+1];
    }

    if (tx == 0) {
        double xreal = MAGMA_Z_REAL(x[n*incx]);
        x[n*incx] = MAGMA_Z_MAKE( sqrt(xreal - sdata[0]), 0 );
    }
}

void zpotf2_zdotc(magma_int_t n, magmaDoubleComplex *x, magma_int_t incx)
{
/*
    Specialized Zdotc
    1) performs zdotc sum = x[0:n-1]*conj(x[0:n-1])
    2) updates x[n] = sqrt(x[n]-sum);

*/
    if (n > zdotc_max_bs) {
        printf("n = %d > %d is not supported in zpotf2_zdotc\n", (int) n, (int) zdotc_max_bs);
        exit(1);
    }
    int threadSize;

    if (n <= 1024 && n > 512) {
        threadSize = 1024;
    }
    else if (n <= 512 && n > 256 ) {
        threadSize = 512;
    }
    else if (n <= 256 && n > 128) {
        threadSize = 256;
    }
    else if (n <= 128 && n > 64) {
        threadSize = 128;
    }
    else {
        threadSize = 64;
    }

    kernel_zdotc<<< 1, threadSize, threadSize * sizeof(double), magma_stream>>> (n, x, incx, threadSize);
}

__global__ void kernel_zdscal(int n, magmaDoubleComplex *x, int incx)
{
    int id = blockIdx.x * zdscal_bs + threadIdx.x;

    __shared__ magmaDoubleComplex factor;

    if (threadIdx.x == 0) {
        factor = MAGMA_Z_MAKE(1.0/MAGMA_Z_REAL(x[0]), 0.0);
    }

    __syncthreads();

    if ( id < n && id >0) {
        x[id*incx] = x[id*incx] * factor;
    }
}


void zpotf2_zdscal(magma_int_t n, magmaDoubleComplex *x, magma_int_t incx)
{
/*
    Specialized Zdscal perform x[1:n-1]/x[0]

*/
    dim3 threads(zdscal_bs, 1, 1);
    int num_blocks = (n - 1)/zdscal_bs + 1;
    dim3 grid(num_blocks,1);
    kernel_zdscal<<< grid, threads, 0, magma_stream >>> (n, x, incx);
}


#if defined(PRECISION_z) || defined(PRECISION_c)

__global__ void kernel_zlacgv(int n, magmaDoubleComplex *x, int incx)
{
    int id = blockIdx.x * zlacgv_bs + threadIdx.x;

    if ( id < n ) {
        x[id*incx] = MAGMA_Z_CNJG(x[id*incx]);
    }
}


void zlacgv(magma_int_t n, magmaDoubleComplex *x, magma_int_t incx)
{
/*
    Purpose
    =======

    ZLACGV conjugates a complex vector of length N.

    Arguments
    =========

    N       (input) INTEGER
            The length of the vector X.  N >= 0.

    X       (input/output) COMPLEX*16 array, dimension
                           (1+(N-1)*abs(INCX))
            On entry, the vector of length N to be conjugated.
            On exit, X is overwritten with conjg(X).

    INCX    (input) INTEGER
            The spacing between successive elements of X.

    ===================================================================== */

    dim3 threads(zlacgv_bs, 1, 1);
    int num_blocks = (n - 1)/zlacgv_bs + 1;
    dim3 grid(num_blocks,1);
    kernel_zlacgv<<< grid, threads, 0, magma_stream >>> (n, x, incx);
}

#endif // defined(PRECISION_z) || defined(PRECISION_c)
