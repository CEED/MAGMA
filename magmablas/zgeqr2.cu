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
    ZGEQR2 computes a QR factorization of a complex m by n matrix A:
    A = Q * R.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in,out]
    A       COMPLEX*16 array, dimension (LDA,N)
            On entry, the m by n matrix A.
            On exit, the elements on and above the diagonal of the array
            contain the min(m,n) by n upper trapezoidal matrix R (R is
            upper triangular if m >= n); the elements below the diagonal,
            with the array TAU, represent the unitary matrix Q as a
            product of elementary reflectors (see Further Details).

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    @param[out]
    tau     COMPLEX*16 array, dimension (min(M,N))
            The scalar factors of the elementary reflectors (see Further
            Details).

    @param
    work    (workspace) DOUBLE_PRECISION array, dimension (N)

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
magma_zgeqr2_gpu(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex *dA, magma_int_t ldda,
    magmaDoubleComplex *dtau, double *dwork,
    magma_int_t *info)
{
    #define  da_ref(a_1,a_2) ( dA+(a_2)*(ldda) + (a_1))
    
    magma_int_t i, k;

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

    /* Compute the norms of the trailing columns */
    k = min(m,n);

    magmaDoubleComplex *Aks;
    magma_zmalloc( &Aks, k );    

    for (i = 0; i < k; ++i) {

        /*  Generate elementary reflector H(i) to annihilate A(i+1:m,i) */
        magma_zlarfg_gpu(m-i, da_ref(i, i), da_ref(min(i+1,m), i), dtau+i, dwork, &Aks[i]);

        if (n-i-1>0)
           /* Apply H(i)' to A(i:m,i+1:n) from the left */
           magma_zlarf_gpu(m-i, n-i-1, da_ref(i, i), dtau+i, da_ref(i, i+1), ldda,
                           dwork);
    }

    if (n >= m)
       magma_zcopymatrix( 1, k-1, Aks, 1, da_ref(0, 0), ldda+1 );
    else
       magma_zcopymatrix( 1, k, Aks, 1, da_ref(0, 0), ldda+1 );

    magma_free(Aks);
    return *info;
} /* magma_zgeqr2 */
