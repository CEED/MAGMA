/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Stan Tomov
       @author Mark Gates
       
       @precisions normal z -> s d c
*/
#include "magma_internal.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Auxiliary function: 'a' is pointer to the current panel holding the
      Householder vectors for the QR factorization of the panel. This routine
      puts ones on the diagonal and zeros in the upper triangular part of 'a'.
      The upper triangular values are stored in work.
      
      Then, the inverse is calculated in place in work, so as a final result,
      work holds the inverse of the upper triangular diagonal block.
*/
void zsplit_diag_block(magma_int_t ib, magmaDoubleComplex *a, magma_int_t lda, magmaDoubleComplex *work)
{
    magma_int_t i, j, info;
    magmaDoubleComplex *cola, *colw;
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO;
    magmaDoubleComplex c_one  = MAGMA_Z_ONE;

    for (i=0; i < ib; i++) {
        cola = a    + i*lda;
        colw = work + i*ib;
        for (j=0; j < i; j++) {
            colw[j] = cola[j];
            cola[j] = c_zero;
        }
        colw[i] = cola[i];
        cola[i] = c_one;
    }
    lapackf77_ztrtri( MagmaUpperStr, MagmaNonUnitStr, &ib, work, &ib, &info );
}

/**
    Purpose
    -------
    ZGEQRF computes a QR factorization of a complex M-by-N matrix A:
    A = Q * R.
    
    This version stores the triangular dT matrices used in
    the block QR factorization so that they can be applied directly (i.e.,
    without being recomputed) later. As a result, the application
    of Q is much faster. Also, the upper triangular matrices for V have 0s
    in them. The corresponding parts of the upper triangular R are inverted
    and stored separately in dT.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in,out]
    dA      COMPLEX_16 array on the GPU, dimension (LDDA,N)
            On entry, the M-by-N matrix A.
            On exit, the elements on and above the diagonal of the array
            contain the min(M,N)-by-N upper trapezoidal matrix R (R is
            upper triangular if m >= n); the elements below the diagonal,
            with the array TAU, represent the orthogonal matrix Q as a
            product of min(m,n) elementary reflectors (see Further
            Details).

    @param[in]
    ldda     INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,M).
            To benefit from coalescent memory accesses LDDA must be
            divisible by 16.

    @param[out]
    tau     COMPLEX_16 array, dimension (min(M,N))
            The scalar factors of the elementary reflectors (see Further
            Details).

    @param[out]
    dT      (workspace) COMPLEX_16 array on the GPU,
            dimension (2*MIN(M, N) + ceil(N/32)*32 )*NB,
            where NB can be obtained through magma_get_zgeqrf_nb(M).
            It starts with MIN(M,N)*NB block that store the triangular T
            matrices, followed by the MIN(M,N)*NB block of the diagonal
            inverses for the R matrix. The rest of the array is used as workspace.

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
magma_zgeqrf_gpu(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex *tau,
    magmaDoubleComplex_ptr dT,
    magma_int_t *info )
{
    #ifdef HAVE_clBLAS
    #define dA(i_, j_)  dA, (dA_offset + (i_) + (j_)*(ldda))
    #define dT(i_)      dT, (dT_offset + (i_)*nb)
    #define dUT(i_)     dT, (dT_offset + (  minmn + (i_))*nb)
    #define dwork(i_)   dT, (dT_offset + (2*minmn + (i_))*nb)
    #else
    #define dA(i_, j_) (dA + (i_) + (j_)*(ldda))
    #define dT(i_)     (dT + (i_)*nb)
    #define dUT(i_)    (dT + (  minmn + (i_))*nb)
    #define dwork(i_)  (dT + (2*minmn + (i_))*nb)
    #endif
    
    magma_int_t i, k, minmn, old_i, old_ib, rows, cols;
    magma_int_t ib, nb;
    magma_int_t ldwork, lddwork, lwork, lhwork;
    magmaDoubleComplex *work, *hwork, *ut;
    
    /* check arguments */
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
    
    k = minmn = min(m,n);
    if (k == 0)
        return *info;
    
    nb = magma_get_zgeqrf_nb( m );
    
    // work  is m*nb for panel
    // hwork is n*nb
    // ut    is nb*nb
    lwork  = (m + n + nb)*nb;
    lhwork = n*nb;
    
    if (MAGMA_SUCCESS != magma_zmalloc_pinned( &work, lwork )) {
        *info = MAGMA_ERR_HOST_ALLOC;
        return *info;
    }
    hwork = work + m*nb;
    ut    = work + m*nb + n*nb;
    memset( ut, 0, nb*nb*sizeof(magmaDoubleComplex) );
    
    magma_queue_t queues[2];
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queues[0] );
    magma_queue_create( cdev, &queues[1] );
    
    ldwork  = m;
    lddwork = n;
    
    if ( nb > 1 && nb < k ) {
        /* Use blocked code initially */
        old_i = 0; old_ib = nb;
        for (i = 0; i < k-nb; i += nb) {
            ib = min( k-i, nb );
            rows = m - i;
            magma_zgetmatrix_async( rows, ib,
                                    dA(i,i),  ldda,
                                    work, ldwork, queues[1] );
            if (i > 0) {
                /* Apply H' to A(i:m,i+2*ib:n) from the left */
                cols = n - old_i - 2*old_ib;
                magma_zlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                                  m-old_i, cols, old_ib,
                                  dA(old_i, old_i         ), ldda, dT(old_i), nb,
                                  dA(old_i, old_i+2*old_ib), ldda, dwork(0), lddwork, queues[0] );
                
                /* store the diagonal block */
                magma_zsetmatrix_async( old_ib, old_ib,
                                        ut,         old_ib,
                                        dUT(old_i), old_ib, queues[0] );
            }
            
            magma_queue_sync( queues[1] );
            lapackf77_zgeqrf( &rows, &ib, work, &ldwork, tau+i, hwork, &lhwork, info );
            /* Form the triangular factor of the block reflector in hwork
               H = H(i) H(i+1) . . . H(i+ib-1) */
            lapackf77_zlarft( MagmaForwardStr, MagmaColumnwiseStr,
                              &rows, &ib,
                              work, &ldwork, tau+i, hwork, &ib );
            
            /* Put 0s in the upper triangular part of a panel (and 1s on the
               diagonal); copy the upper triangular to ut and invert it. */
            magma_queue_sync( queues[0] );
            zsplit_diag_block( ib, work, ldwork, ut );
            magma_zsetmatrix( rows, ib,
                              work, ldwork,
                              dA(i,i), ldda, queues[0] );
            
            if (i + ib < n) {
                /* Send the triangular factor T to the GPU */
                magma_zsetmatrix( ib, ib, hwork, ib, dT(i), nb, queues[0] );
                
                if (i+nb < k-nb) {
                    /* Apply H' to A(i:m,i+ib:i+2*ib) from the left */
                    magma_zlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                                      rows, ib, ib,
                                      dA(i, i   ), ldda, dT(i),  nb,
                                      dA(i, i+ib), ldda, dwork(0), lddwork, queues[0] );
                }
                else {
                    cols = n-i-ib;
                    magma_zlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                                      rows, cols, ib,
                                      dA(i, i   ), ldda, dT(i),  nb,
                                      dA(i, i+ib), ldda, dwork(0), lddwork, queues[0] );
                    /* store the diagonal block */
                    magma_zsetmatrix( ib, ib,
                                      ut, ib,
                                      dUT(i), ib, queues[0] );
                }
                magma_queue_sync( queues[0] );
                old_i  = i;
                old_ib = ib;
            }
        }
    } else {
        i = 0;
    }
    
    /* Use unblocked code to factor the last or only block. */
    if (i < k) {
        ib   = n-i;
        rows = m-i;
        magma_zgetmatrix( rows, ib, dA(i, i), ldda, work, rows, queues[0] );
        lapackf77_zgeqrf( &rows, &ib, work, &rows, tau+i, hwork, &lhwork, info );
        magma_zsetmatrix( rows, ib, work, rows, dA(i, i), ldda, queues[0] );
    }
    
    magma_queue_destroy( queues[0] );
    magma_queue_destroy( queues[1] );
    
    magma_free_pinned( work );
    
    return *info;
} /* magma_zgeqrf_gpu */
