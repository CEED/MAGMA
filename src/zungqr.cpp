/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> s d c

       @author Stan Tomov
       @author Mark Gates
*/
#include "common_magma.h"

extern "C" magma_int_t
magma_zungqr(magma_int_t m, magma_int_t n, magma_int_t k,
             cuDoubleComplex *A, magma_int_t lda,
             cuDoubleComplex *tau,
             cuDoubleComplex *dT, magma_int_t nb,
             magma_int_t *info)
{
/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======
    ZUNGQR generates an M-by-N COMPLEX_16 matrix Q with orthonormal columns,
    which is defined as the first N columns of a product of K elementary
    reflectors of order M

          Q  =  H(1) H(2) . . . H(k)

    as returned by ZGEQRF.

    Arguments
    =========
    M       (input) INTEGER
            The number of rows of the matrix Q. M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix Q. M >= N >= 0.

    K       (input) INTEGER
            The number of elementary reflectors whose product defines the
            matrix Q. N >= K >= 0.

    A       (input/output) COMPLEX_16 array A, dimension (LDDA,N).
            On entry, the i-th column must contain the vector
            which defines the elementary reflector H(i), for
            i = 1,2,...,k, as returned by ZGEQRF_GPU in the
            first k columns of its array argument A.
            On exit, the M-by-N matrix Q.

    LDA     (input) INTEGER
            The first dimension of the array A. LDA >= max(1,M).

    TAU     (input) COMPLEX_16 array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by ZGEQRF_GPU.

    DT      (input) COMPLEX_16 array on the GPU device.
            DT contains the T matrices used in blocking the elementary
            reflectors H(i), e.g., this can be the 6th argument of
            magma_zgeqrf_gpu.

    NB      (input) INTEGER
            This is the block size used in ZGEQRF_GPU, and correspondingly
            the size of the T matrices, used in the factorization, and
            stored in DT.

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument has an illegal value
    =====================================================================    */

#define  A(i,j) ( A + (i) + (j)*lda )
#define dA(i,j) (dA + (i) + (j)*ldda)
#define dT(j)   (dT + (j)*nb)

    cuDoubleComplex c_zero = MAGMA_Z_ZERO;
    cuDoubleComplex c_one  = MAGMA_Z_ONE;

    magma_int_t  m_kk, n_kk, k_kk, mi;
    magma_int_t lwork, ldda;
    magma_int_t i, ib, ki, kk, iinfo;
    magma_int_t lddwork;
    cuDoubleComplex *dA, *dV, *dW;
    cuDoubleComplex *work;

    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if ((n < 0) || (n > m)) {
        *info = -2;
    } else if ((k < 0) || (k > n)) {
        *info = -3;
    } else if (lda < max(1,m)) {
        *info = -5;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    if (n <= 0) {
        return *info;
    }

    // first kk columns are handled by blocked method.
    if ((nb > 1) && (nb < k)) {
        ki = (k - nb - 1) / nb * nb;
        kk = min(k, ki + nb);
    } else {
        kk = 0;
    }

    // Allocate GPU work space
    // ldda*n     for matrix dA
    // ldda*nb    for dV
    // lddwork*nb for dW larfb workspace
    ldda    = ((m + 31) / 32) * 32;
    lddwork = ((n + 31) / 32) * 32;
    if (MAGMA_SUCCESS != magma_zmalloc( &dA, ldda*n + ldda*nb + lddwork*nb )) {
        *info = MAGMA_ERR_DEVICE_ALLOC;
        return *info;
    }
    dV = dA + ldda*n;
    dW = dA + ldda*n + ldda*nb;

    // Allocate CPU work space
    lwork = n * nb;
    work = (cuDoubleComplex*) malloc( lwork * sizeof(cuDoubleComplex));
    if (work == NULL) {
        magma_free( dA );
        *info = MAGMA_ERR_HOST_ALLOC;
        return *info;
    }

    cudaStream_t stream;
    magma_queue_create( &stream );

    // Use unblocked code for the last or only block.
    if (kk < n) {
        m_kk = m - kk;
        n_kk = n - kk;
        k_kk = k - kk;
        lapackf77_zungqr( &m_kk, &n_kk, &k_kk,
                          A(kk, kk), &lda,
                          &tau[kk], work, &lwork, &iinfo );
        
        magma_zsetmatrix( m_kk, n_kk,
                          A(kk, kk),  lda,
                          dA(kk, kk), ldda );
        
        // Set A(1:kk,kk+1:n) to zero.
        magmablas_zlaset( MagmaUpperLower, kk, n - kk, dA(0, kk), ldda );
    }

    if (kk > 0) {
        // Use blocked code
        // stream: set Aii (V) --> laset --> laset --> larfb --> [next]
        // CPU has no computation
        magmablasSetKernelStream( stream );
        
        for (i = ki; i >= 0; i -= nb) {
            ib = min(nb, k - i);

            // Send current panel to the GPU
            mi = m - i;
            lapackf77_zlaset( "Upper", &ib, &ib, &c_zero, &c_one, A(i, i), &lda );
            magma_zsetmatrix_async( mi, ib,
                                    A(i, i), lda,
                                    dV,      ldda, stream );

            // set panel to identity
            magmablas_zlaset( MagmaUpperLower, i, ib, dA(0, i), ldda );
            magmablas_zlaset_identity( mi, ib, dA(i, i), ldda );
            
            if (i < n) {
                // Apply H to A(i:m,i:n) from the left
                magma_zlarfb_gpu( MagmaLeft, MagmaNoTrans, MagmaForward, MagmaColumnwise,
                                  mi, n-i, ib,
                                  dV,       ldda, dT(i), nb,
                                  dA(i, i), ldda, dW, lddwork );
            }
        }
    }

    // copy result back to CPU
    magma_zgetmatrix( m, n,
                      dA(0, 0), ldda, A(0, 0), lda);

    magmablasSetKernelStream( NULL );
    magma_queue_destroy( stream );
    magma_free( dA );
    free( work );

    return *info;
} /* magma_zungqr */
