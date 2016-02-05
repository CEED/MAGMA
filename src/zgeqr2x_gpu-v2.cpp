/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Stan Tomov

       @precisions normal z -> s d c

*/
#include "magma_internal.h"
    
/**
    Purpose
    -------
    ZGEQR2 computes a QR factorization of a complex m by n matrix A:
    A = Q * R.

    This expert routine requires two more arguments than the standard
    zgeqr2, namely, dT and ddA, explained below. The storage for A is
    also not as in the LAPACK's zgeqr2 routine (see below).

    The first is used to output the triangular
    n x n factor T of the block reflector used in the factorization.
    The second holds the diagonal nxn blocks of A, i.e., the diagonal
    submatrices of R. This routine implements the left looking QR.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in,out]
    dA      COMPLEX_16 array, dimension (LDDA,N)
            On entry, the m by n matrix A.
            On exit, the unitary matrix Q as a
            product of elementary reflectors (see Further Details).
    \n
            the elements on and above the diagonal of the array
            contain the min(m,n) by n upper trapezoidal matrix R (R is
            upper triangular if m >= n); the elements below the diagonal,
            with the array TAU, represent the unitary matrix Q as a
            product of elementary reflectors (see Further Details).

    @param[in]
    ldda    INTEGER
            The leading dimension of the array A.  LDDA >= max(1,M).

    @param[out]
    dtau    COMPLEX_16 array, dimension (min(M,N))
            The scalar factors of the elementary reflectors (see Further
            Details).

    @param[out]
    dT      COMPLEX_16 array, dimension N x N.
            Stores the triangular N x N factor T of the block reflector
            used in the factorization. The lower triangular part is 0.

    @param[out]
    ddA     COMPLEX_16 array, dimension N x N.
            Stores the elements of the upper N x N diagonal block of A.
            LAPACK stores this array in A. There are 0s below the diagonal.

    @param
    dwork   (workspace) DOUBLE PRECISION array, dimension (3 N)

    @param[out]
    info    INTEGER
      -     = 0: successful exit
      -     < 0: if INFO = -i, the i-th argument had an illegal value

    Further Details
    ---------------
    The matrix Q is represented as a product of elementary reflectors

       Q = H(1) H(2) . . . H(k), where k = min(m,n).

    Each H(i) has the form

       H(i) = I - tau * v * v'

    where tau is a complex scalar, and v is a complex vector with
    v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in A(i+1:m,i),
    and tau in TAU(i).

    @ingroup magma_zgeqrf_aux
    ********************************************************************/
extern "C" magma_int_t
magma_zgeqr2x2_gpu(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr dtau,
    magmaDoubleComplex_ptr dT,
    magmaDoubleComplex_ptr ddA,
    magmaDouble_ptr dwork,
    magma_int_t *info)
{
    #define dA(i_,j_) (dA + (i_) + (j_)*ldda)
    
    magma_int_t i, min_mn;
    
    magmaDoubleComplex_ptr dwork2 = (magmaDoubleComplex *)dwork;
    magmaDouble_ptr dnorm = dwork + 4*n;

    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (ldda < max(1,m)) {
        *info = -4;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    magma_queue_t queue;
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queue );

    /* Compute the norms of the trailing columns */
    min_mn = min(m,n);
    magmablas_dznrm2_cols( m, min_mn, dA(0,0), ldda, dnorm, queue );

    for (i = 0; i < min_mn; ++i) {
        /*   1. Apply H' to A(:,i) from the left
             2. Adjust the dnorm[i] to hold the norm of A(i:m,i) */
        if (i > 0) {
            magma_zlarfbx_gpu( m, i, dA(0, 0), ldda,
                              dT, min_mn, dA(0, i), dwork2, queue );
            magmablas_dznrm2_adjust( i, dnorm+i, dA(0, i), queue );
        }

        /*  Generate elementary reflector H(i) to annihilate A(i+1:m,i)
            1. 1 is not yet put on the diagonal of A
            2. Elements above the diagonal are copied in ddA and the ones
               in A are set to zero
            3. update T */
        magma_zlarfgtx_gpu( m-i, dA(i, i), dA(min(i+1,m), i), dtau+i,
                            dnorm+i, ddA + i + i*(n), i,
                            dA(i,0), ldda,  dT, min_mn, dwork2, queue );
    }

    magma_queue_destroy( queue );

    return *info;
} /* magma_zgeqr2 */
