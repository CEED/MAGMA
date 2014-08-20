/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

*/
#include "common_magma.h"

/**
    Purpose
    -------
    ZGEQRF4 computes a QR factorization of a COMPLEX_16 M-by-N matrix A:
    A = Q * R using multiple GPUs. This version does not require work space on the GPU
    passed as input. GPU memory is allocated in the routine.

    Arguments
    ---------
    @param[in]
    num_gpus INTEGER
            The number of GPUs to be used for the factorization.

    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in,out]
    A       COMPLEX_16 array, dimension (LDA,N)
            On entry, the M-by-N matrix A.
            On exit, the elements on and above the diagonal of the array
            contain the min(M,N)-by-N upper trapezoidal matrix R (R is
            upper triangular if m >= n); the elements below the diagonal,
            with the array TAU, represent the orthogonal matrix Q as a
            product of min(m,n) elementary reflectors (see Further
            Details).
    \n
            Higher performance is achieved if A is in pinned memory, e.g.
            allocated using magma_malloc_pinned.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    @param[out]
    tau     COMPLEX_16 array, dimension (min(M,N))
            The scalar factors of the elementary reflectors (see Further
            Details).

    @param[out]
    work    (workspace) COMPLEX_16 array, dimension (MAX(1,LWORK))
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
    \n
            Higher performance is achieved if WORK is in pinned memory, e.g.
            allocated using magma_malloc_pinned.

    @param[in]
    lwork   INTEGER
            The dimension of the array WORK.  LWORK >= N*NB,
            where NB can be obtained through magma_get_zgeqrf_nb(M).
    \n
            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.

    Further Details
    ---------------
    The matrix Q is represented as a product of elementary reflectors

       Q = H(1) H(2) . . . H(k), where k = min(m,n).

    Each H(i) has the form

       H(i) = I - tau * v * v'

    where tau is a complex scalar, and v is a complex vector with
    v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in A(i+1:m,i),
    and tau in TAU(i).

    @ingroup magma_zgeqrf_comp
    ********************************************************************/
extern "C" magma_int_t
magma_zgeqrf4(magma_int_t num_gpus, magma_int_t m, magma_int_t n,
              magmaDoubleComplex *A,    magma_int_t lda, magmaDoubleComplex *tau,
              magmaDoubleComplex *work, magma_int_t lwork,
              magma_int_t *info )
{
    magmaDoubleComplex *da[MagmaMaxGPUs];
    magmaDoubleComplex c_one = MAGMA_Z_ONE;

    int i, k, ldda;

    *info = 0;
    int nb = magma_get_zgeqrf_nb(min(m, n));

    int lwkopt = n * nb;
    work[0] = MAGMA_Z_MAKE( (double)lwkopt, 0 );
    int lquery = (lwork == -1);
    if (num_gpus < 0 || num_gpus > MagmaMaxGPUs) {
        *info = -1;
    } else if (m < 0) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (lda < max(1,m)) {
        *info = -5;
    } else if (lwork < max(1,n) && ! lquery) {
        *info = -8;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }
    else if (lquery)
        return *info;

    k = min(m,n);
    if (k == 0) {
        work[0] = c_one;
        return *info;
    }

    magma_device_t orig_dev;
    magma_getdevice( &orig_dev );
    
    ldda    = ((m+31)/32)*32;

    magma_int_t  n_local[MagmaMaxGPUs];
    for (i=0; i < num_gpus; i++) {
        n_local[i] = ((n/nb)/num_gpus)*nb;
        if (i < (n/nb)%num_gpus)
            n_local[i] += nb;
        else if (i == (n/nb)%num_gpus)
            n_local[i] += n%nb;

        magma_setdevice(i);
        
        // TODO on failure, free previously allocated memory
        if (MAGMA_SUCCESS != magma_zmalloc( &da[i], ldda*n_local[i] )) {
            *info = MAGMA_ERR_DEVICE_ALLOC;
            return *info;
        }
    }

    if (m > nb && n > nb) {
        /* Copy the matrix to the GPUs in 1D block cyclic distribution */
        magma_zsetmatrix_1D_col_bcyclic(m, n, A, lda, da, ldda, num_gpus, nb);

        /* Factor using the GPU interface */
        magma_zgeqrf2_mgpu( num_gpus, m, n, da, ldda, tau, info);

        /* Copy the matrix back from the GPUs to the CPU */
        magma_zgetmatrix_1D_col_bcyclic(m, n, da, ldda, A, lda, num_gpus, nb);
    }
    else {
        lapackf77_zgeqrf(&m, &n, A, &lda, tau, work, &lwork, info);
    }


    /* Free the allocated GPU memory */
    for (i=0; i < num_gpus; i++) {
        magma_setdevice(i);
        magma_free( da[i] );
    }
    magma_setdevice( orig_dev );

    return *info;
} /* magma_zgeqrf4 */
