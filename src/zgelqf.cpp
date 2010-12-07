/*
    -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

       @precisions normal z -> s d c

*/

#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include "magma.h"
#include "magmablas.h"

extern "C" magma_int_t
magma_zgelqf( magma_int_t m, magma_int_t n, 
              cuDoubleComplex *a,    magma_int_t lda,   cuDoubleComplex *tau, 
              cuDoubleComplex *work, magma_int_t lwork, magma_int_t *info)
{
/*  -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

    Purpose
    =======

    ZGELQF computes an LQ factorization of a real M-by-N matrix A:
    A = L * Q.

    Arguments
    =========

    M       (input) INTEGER
            The number of rows of the matrix A.  M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix A.  N >= 0.

    A       (input/output) COMPLEX_16 array, dimension (LDA,N)
            On entry, the M-by-N matrix A.
            On exit, the elements on and below the diagonal of the array
            contain the m-by-min(m,n) lower trapezoidal matrix L (L is
            lower triangular if m <= n); the elements above the diagonal,
            with the array TAU, represent the orthogonal matrix Q as a
            product of elementary reflectors (see Further Details).

            Higher performance is achieved if A is in pinned memory, e.g.
            allocated using cudaMallocHost.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    TAU     (output) COMPLEX_16 array, dimension (min(M,N))
            The scalar factors of the elementary reflectors (see Further
            Details).

    WORK    (workspace/output) COMPLEX_16 array, dimension (MAX(1,LWORK))
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.

            Higher performance is achieved if WORK is in pinned memory, e.g.
            allocated using cudaMallocHost.

    LWORK   (input) INTEGER
            The dimension of the array WORK.  LWORK >= max(1,M).
            For optimum performance LWORK >= M*NB, where NB is the
            optimal blocksize.

            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued.

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
                  if INFO = -10 internal GPU memory allocation failed.

    Further Details
    ===============

    The matrix Q is represented as a product of elementary reflectors

       Q = H(k) . . . H(2) H(1), where k = min(m,n).

    Each H(i) has the form

       H(i) = I - tau * v * v'

    where tau is a real scalar, and v is a real vector with
    v(1:i-1) = 0 and v(i) = 1; v(i+1:n) is stored on exit in A(i,i+1:n),
    and tau in TAU(i).

    =====================================================================    */

    #define  a_ref(a_1,a_2) ( a+(a_2)*(lda) + (a_1))
    #define min(a,b)  (((a)<(b))?(a):(b))
    #define max(a,b)  (((a)>(b))?(a):(b))

    cuDoubleComplex c_one = MAGMA_Z_ONE;

    int iinfo, ldda;
    long int lquery;

    /* Function Body */
    *info = 0;
    int nb = magma_get_zgelqf_nb(m);

    work[0] = MAGMA_Z_MAKE( (double)(m*nb), 0 );
    lquery = (lwork == -1);
    if (m < 0) {
	*info = -1;
    } else if (n < 0) {
	*info = -2;
    } else if (lda < max(1,m)) {
	*info = -4;
    } else if (lwork < max(1,m) && ! lquery) {
	*info = -7;
    }
    if (*info != 0) {
	return 0;
    } else if (lquery) {
	return 0;
    }

    /*  Quick return if possible */
    if (min(m, n) == 0) {
	work[0] = c_one;
	return 0;
    }

    int maxm, maxn, maxdim;
    cuDoubleComplex *dA, *dAT;
    cublasStatus status;

    maxm = ((m + 31)/32)*32;
    maxn = ((n + 31)/32)*32;
    maxdim = max(maxm, maxn);

    if (maxdim*maxdim < 2*maxm*maxn)
        {
            ldda = maxdim;

            status = cublasAlloc(maxdim*maxdim, sizeof(cuDoubleComplex), (void**)&dA);
            if (status != CUBLAS_STATUS_SUCCESS) {
                *info = -10;
                return 0;
            }

            cublasSetMatrix( m, n, sizeof(cuDoubleComplex), a, lda, dA, ldda);
            dAT = dA;
            magmablas_zinplace_transpose( dAT, ldda, ldda );
        }
    else
        {
            ldda = maxn;

            status = cublasAlloc(2*maxn*maxm, sizeof(cuDoubleComplex), (void**)&dA);
            if (status != CUBLAS_STATUS_SUCCESS) {
                *info = -10;
                return 0;
            }

            cublasSetMatrix( m, n, sizeof(cuDoubleComplex), a, lda, dA, maxm);

            dAT = dA + maxn * maxm;
            magmablas_ztranspose2( dAT, ldda, dA, maxm, m, n );
        }

    magma_zgeqrf2_gpu(n, m, dAT, ldda, tau, &iinfo);

    if (maxdim*maxdim< 2*maxm*maxn){
        magmablas_zinplace_transpose( dAT, ldda, ldda );
        cublasGetMatrix( m, n, sizeof(cuDoubleComplex), dA, ldda, a, lda);
    } else {
        magmablas_ztranspose2( dA, maxm, dAT, ldda, n, m );
        cublasGetMatrix( m, n, sizeof(cuDoubleComplex), dA, maxm, a, lda);
    }

    cublasFree(dA);

    return 0;

    /*     End of MAGMA_ZGELQF */

} /* magma_zgelqf */

#undef  a_ref
#undef min
#undef max
