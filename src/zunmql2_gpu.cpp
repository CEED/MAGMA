/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Raffaele Solca

       @precisions normal z -> s d c

*/
#include "common_magma.h"

/**
    Purpose
    -------
    ZUNMQL overwrites the general complex M-by-N matrix C with

    @verbatim
                    SIDE = 'L'     SIDE = 'R'
    TRANS = 'N':      Q * C          C * Q
    TRANS = 'C':      Q**H * C       C * Q**H
    @endverbatim

    where Q is a complex unitary matrix defined as the product of k
    elementary reflectors

          Q = H(k) . . . H(2) H(1)

    as returned by ZGEQLF. Q is of order M if SIDE = 'L' and of order N
    if SIDE = 'R'.

    Arguments
    ---------
    @param[in]
    side    CHARACTER*1
      -     = 'L': apply Q or Q**H from the Left;
      -     = 'R': apply Q or Q**H from the Right.

    @param[in]
    trans   CHARACTER*1
      -     = 'N':  No transpose, apply Q;
      -     = 'C':  Transpose, apply Q**H.

    @param[in]
    m       INTEGER
            The number of rows of the matrix C. M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix C. N >= 0.

    @param[in]
    k       INTEGER
            The number of elementary reflectors whose product defines
            the matrix Q.
            If SIDE = 'L', M >= K >= 0;
            if SIDE = 'R', N >= K >= 0.

    @param[in]
    dA      COMPLEX_16 array, dimension (LDA,K)
            The i-th column must contain the vector which defines the
            elementary reflector H(i), for i = 1,2,...,k, as returned by
            ZGEQLF in the last k columns of its array argument A.
            The diagonal and the lower part
            are destroyed, the reflectors are not modified.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array DA.
            LDDA >= max(1,M) if SIDE = 'L'; LDDA >= max(1,N) if SIDE = 'R'.

    @param[in]
    tau     COMPLEX_16 array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by ZGEQLF.

    @param[in,out]
    dC      COMPLEX_16 array, dimension (LDDC,N)
            On entry, the M-by-N matrix C.
            On exit, C is overwritten by Q*C or Q**H*C or C*Q**H or C*Q.

    @param[in]
    lddc    INTEGER
            The leading dimension of the array C. LDDC >= max(1,M).

    @param[in]
    wA      (workspace) COMPLEX_16 array, dimension
                                 (LDWA,M) if SIDE = 'L'
                                 (LDWA,N) if SIDE = 'R'
            The vectors which define the elementary reflectors, as
            returned by ZHETRD_GPU.

    @param[in]
    ldwa    INTEGER
            The leading dimension of the array wA.
            LDWA >= max(1,M) if SIDE = 'L'; LDWA >= max(1,N) if SIDE = 'R'.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value

    @ingroup magma_zgeqlf_comp
    ********************************************************************/
extern "C" magma_int_t
magma_zunmql2_gpu(magma_side_t side, magma_trans_t trans,
                  magma_int_t m, magma_int_t n, magma_int_t k,
                  magmaDoubleComplex *dA, magma_int_t ldda,
                  magmaDoubleComplex *tau,
                  magmaDoubleComplex *dC, magma_int_t lddc,
                  magmaDoubleComplex *wA, magma_int_t ldwa,
                  magma_int_t *info)
{
    /* Allocate work space on the GPU */
    magmaDoubleComplex *dwork;
    magma_zmalloc( &dwork, 2*(m + 64)*64 );

    magma_int_t wa_offset, dc_offset, i__4;
    
    magma_int_t i__;
    magmaDoubleComplex t[2*4160]        /* was [65][64] */;
    magma_int_t i1, i2, i3, ib, nb, mi, ni, nq, nw;
    magma_int_t ldwork;
    int left, notran;

    wa_offset = 1 + ldwa;
    wA -= wa_offset;
    --tau;
    dc_offset = 1 + lddc;
    dC -= dc_offset;

    *info  = 0;
    left   = (side == MagmaLeft);
    notran = (trans == MagmaNoTrans);

    /* NQ is the order of Q and NW is the minimum dimension of WORK */
    if (left) {
        nq = m;
        nw = max(1,n);
    } else {
        nq = n;
        nw = max(1,m);
    }
    if (! left && side != MagmaRight) {
        *info = -1;
    } else if (! notran && trans != MagmaConjTrans) {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (k < 0 || k > nq) {
        *info = -5;
    } else if (ldda < max(1,nq)) {
        *info = -7;
    } else if (lddc < max(1,m)) {
        *info = -10;
    } else if (ldwa < max(1,nq)) {
        *info = -12;
    }
    
    // size of the block
    nb = 64;

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0) {
        return *info;
    }

    ldwork = nw;
        
    /* Use hybrid CPU-GPU code */
    if ((left && notran) || (! left && ! notran)) {
        i1 = 1;
        i2 = k;
        i3 = nb;
    } else {
        i1 = (k - 1) / nb * nb + 1;
        i2 = 1;
        i3 = -nb;
    }
    
    // silence "uninitialized" warnings
    mi = 0;
    ni = 0;
    
    if (left) {
        ni = n;
    } else {
        mi = m;
    }
    
    magmablas_zsetdiag1subdiag0( MagmaUpper, k, nb, dA, ldda);
    
    for (i__ = i1; (i3 < 0 ? i__ >= i2 : i__ <= i2); i__ += i3) {
        ib = min(nb, k - i__ + 1);
        
        /* Form the triangular factor of the block reflector
           H = H(i+ib-1) . . . H(i+1) H(i) */
        i__4 = nq - k + i__ + ib - 1;
        lapackf77_zlarft("Backward", "Columnwise", &i__4, &ib,
                         &wA[i__ * ldwa + 1], &ldwa, &tau[i__], t, &ib);
    
        if (left) {
            /* H or H' is applied to C(1:m-k+i+ib-1,1:n) */
            mi = m - k + i__ + ib - 1;
        }
        else {
            /* H or H' is applied to C(1:m,1:n-k+i+ib-1) */
            ni = n - k + i__ + ib - 1;
        }
        
        /* Apply H or H'; First copy T to the GPU */
        magma_zsetmatrix( ib, ib, t, ib, dwork+i__4*ib, ib );
        magma_zlarfb_gpu(side, trans, MagmaBackward, MagmaColumnwise,
                         mi, ni, ib,
                         &dA[(i__-1) * ldda], ldda, dwork+i__4*ib, ib,
                         &dC[1+lddc], lddc,
                         dwork+i__4*ib + ib*ib, ldwork);
    }

    magma_free( dwork );

    return *info;
} /* magma_zunmql */
