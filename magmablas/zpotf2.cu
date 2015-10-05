/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @precisions normal z -> s d c
*/
#include "common_magma.h"

#define PRECISION_z

#define zdotc_max_bs 512  // 512 is max threads for 1.x cards

void zpotf2_zdscal(magma_int_t n, magmaDoubleComplex *x, magma_int_t incx);
void zpotf2_zdotc(magma_int_t n, magmaDoubleComplex *x, magma_int_t incx);

#if defined(PRECISION_z) || defined(PRECISION_c)
void magmablas_zlacgv(magma_int_t n, magmaDoubleComplex *x, magma_int_t incx);
#endif

/**
    Purpose
    -------

    zpotf2 computes the Cholesky factorization of a real symmetric
    positive definite matrix A.

    The factorization has the form
        A = U**H * U,  if UPLO = MagmaUpper, or
        A = L  * L**H, if UPLO = MagmaLower,
    where U is an upper triangular matrix and L is lower triangular.

    This is the unblocked version of the algorithm, calling Level 2 BLAS.

    Arguments
    ---------

    @param[in]
    uplo    magma_uplo_t
            Specifies whether the upper or lower triangular part of the
            symmetric matrix A is stored.
      -     = MagmaUpper:  Upper triangular
      -     = MagmaLower:  Lower triangular

    @param[in]
    n       INTEGER
            The order of the matrix A.  N >= 0 and N <= 512.

    @param[in,out]
    dA      COMPLEX_16 array, dimension (LDDA,N)
            On entry, the symmetric matrix A.  If UPLO = MagmaUpper, the leading
            n by n upper triangular part of A contains the upper
            triangular part of the matrix A, and the strictly lower
            triangular part of A is not referenced.  If UPLO = MagmaLower, the
            leading n by n lower triangular part of A contains the lower
            triangular part of the matrix A, and the strictly upper
            triangular part of A is not referenced.
    \n
            On exit, if INFO = 0, the factor U or L from the Cholesky
            factorization A = U**H * U  or A = L * L**H.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array A.  LDDA >= max(1,N).

    @param[out]
    info    INTEGER
      -     = 0: successful exit
      -     < 0: if INFO = -k, the k-th argument had an illegal value
      -     > 0: if INFO = k, the leading minor of order k is not
                 positive definite, and the factorization could not be
                 completed.

    @ingroup magma_zposv_aux
    ********************************************************************/
extern "C" magma_int_t
magma_zpotf2_gpu(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t *info )
{
#define dA(i_, j_)  (dA + (i_) + (j_)*ldda)

    magma_int_t j;

    *info = 0;
    if ( uplo != MagmaUpper && uplo != MagmaLower) {
        *info = -1;
    } else if (n < 0 || n > zdotc_max_bs) {
        *info = -2;
    } else if (ldda < max(1,n)) {
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

    if (uplo == MagmaUpper) {
        for (j = 0; j < n; j++) {
            zpotf2_zdotc(j, dA(0,j), 1); // including zdotc product and update a(j,j)
            if (j < n) {
                #if defined(PRECISION_z) || defined(PRECISION_c)
                magmablas_zlacgv(j, dA(0, j), 1);
                #endif
                magma_zgemv( MagmaTrans, j, n-j-1,
                             alpha, dA(0, j+1), ldda,
                                    dA(0, j),   1,
                             beta,  dA(j, j+1), ldda);

                #if defined(PRECISION_z) || defined(PRECISION_c)
                magmablas_zlacgv(j, dA(0, j), 1);
                #endif
                zpotf2_zdscal(n-j, dA(j,j), ldda);
            }
        }
    }
    else {
        for (j = 0; j < n; j++) {
            zpotf2_zdotc(j, dA(j,0), ldda); // including zdotc product and update a(j,j)
            if (j < n) {
                #if defined(PRECISION_z) || defined(PRECISION_c)
                magmablas_zlacgv(j, dA(j, 0), ldda);
                #endif
                magma_zgemv( MagmaNoTrans, n-j-1, j,
                             alpha, dA(j+1, 0), ldda,
                                    dA(j,0),    ldda,
                             beta,  dA(j+1, j), 1 );

                #if defined(PRECISION_z) || defined(PRECISION_c)
                magmablas_zlacgv(j, dA(j, 0), ldda);
                #endif
                zpotf2_zdscal(n-j, dA(j,j), 1);
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

    for (int s = blockDim.x/2; s > 32; s >>= 1 ) {
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
        fprintf( stderr, "n = %d > %d is not supported in zpotf2_zdotc\n", (int) n, (int) zdotc_max_bs);
        return;
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

    if ( id < n && id > 0) {
        x[id*incx] = x[id*incx] * factor;
    }
}


void zpotf2_zdscal(magma_int_t n, magmaDoubleComplex *x, magma_int_t incx)
{
    /*
    Specialized Zdscal perform x[1:n-1]/x[0]
    */
    dim3 threads(zdscal_bs, 1, 1);
    int num_blocks = magma_ceildiv( n, zdscal_bs );
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


/**
    Purpose
    -------

    ZLACGV conjugates a complex vector of length N.

    Arguments
    ---------

    @param[in]
    n       INTEGER
            The length of the vector X.  N >= 0.

    @param[in,out]
    x       COMPLEX*16 array, dimension (1+(N-1)*abs(INCX))
            On entry, the vector of length N to be conjugated.
            On exit, X is overwritten with conjg(X).

    @param[in]
    incx    INTEGER
            The spacing between successive elements of X.

    @ingroup magma_zaux1
    ********************************************************************/
void magmablas_zlacgv(magma_int_t n, magmaDoubleComplex *x, magma_int_t incx)
{
    dim3 threads(zlacgv_bs, 1, 1);
    int num_blocks = magma_ceildiv( n, zlacgv_bs );
    dim3 grid(num_blocks,1);
    kernel_zlacgv<<< grid, threads, 0, magma_stream >>> (n, x, incx);
}

#endif // defined(PRECISION_z) || defined(PRECISION_c)
