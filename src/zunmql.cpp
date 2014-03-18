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
    A       COMPLEX_16 array, dimension (LDA,K)
            The i-th column must contain the vector which defines the
            elementary reflector H(i), for i = 1,2,...,k, as returned by
            ZGEQLF in the last k columns of its array argument A.
            A is modified by the routine but restored on exit.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.
            If SIDE = 'L', LDA >= max(1,M);
            if SIDE = 'R', LDA >= max(1,N).

    @param[in]
    tau     COMPLEX_16 array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by ZGEQLF.

    @param[in,out]
    C       COMPLEX_16 array, dimension (LDC,N)
            On entry, the M-by-N matrix C.
            On exit, C is overwritten by Q*C or Q**H*C or C*Q**H or C*Q.

    @param[in]
    ldc     INTEGER
            The leading dimension of the array C. LDC >= max(1,M).

    @param[out]
    work    (workspace) COMPLEX_16 array, dimension (MAX(1,LWORK))
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.

    @param[in]
    lwork   INTEGER
            The dimension of the array WORK.
            If SIDE = 'L', LWORK >= max(1,N);
            if SIDE = 'R', LWORK >= max(1,M).
            For optimum performance LWORK >= N*NB if SIDE = 'L', and
            LWORK >= M*NB if SIDE = 'R', where NB is the optimal
            blocksize.
    \n
            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued by XERBLA.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value

    @ingroup magma_zgeqlf_comp
    ********************************************************************/
extern "C" magma_int_t
magma_zunmql(magma_side_t side, magma_trans_t trans,
             magma_int_t m, magma_int_t n, magma_int_t k,
             magmaDoubleComplex *a, magma_int_t lda,
             magmaDoubleComplex *tau,
             magmaDoubleComplex *c, magma_int_t ldc,
             magmaDoubleComplex *work, magma_int_t lwork,
             magma_int_t *info)
{
    const char* side_  = lapack_side_const( side  );
    const char* trans_ = lapack_trans_const( trans );

    magma_int_t i__4, i__;
    magmaDoubleComplex *T;
    magma_int_t i1, i2, i3, ib, nb, mi, ni, nq, nw;
    magma_int_t iinfo, ldwork, lwkopt=0;
    int lquery, left, notran;

    *info  = 0;
    left   = (side == MagmaLeft);
    notran = (trans == MagmaNoTrans);
    lquery = (lwork == -1);

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
    } else if (lda < max(1,nq)) {
        *info = -7;
    } else if (ldc < max(1,m)) {
        *info = -10;
    }

    if (*info == 0) {
        if (m == 0 || n == 0) {
            lwkopt = 1;
        } else {
            /* Determine the block size.  NB may be at most NBMAX, where
               NBMAX is used to define the local array T.                 */
            nb = 64;
            lwkopt = nw * nb;
        }
        work[0] = MAGMA_Z_MAKE( lwkopt, 0 );
        
        if (lwork < nw && ! lquery) {
            *info = -12;
        }
    }

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }
    else if (lquery) {
        return *info;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0) {
        return *info;
    }

    /* Allocate work space on the GPU */
    magmaDoubleComplex *dwork, *dc;
    magma_zmalloc( &dc, (m)*(n) );
    magma_zmalloc( &dwork, 2*(m + 64)*64 );

    /* Copy matrix C from the CPU to the GPU */
    magma_zsetmatrix( m, n, c, ldc, dc, m );

    /* work space on CPU */
    if (  MAGMA_SUCCESS != magma_zmalloc_pinned( &T, 2*nb*nb ) ) {
        magma_free( dc );
        magma_free( dwork );
        *info = MAGMA_ERR_HOST_ALLOC;
        return *info;
    }
    ldwork = nw;

    if ( nb >= k ) {
        /* Use CPU code */
        lapackf77_zunmql(side_, trans_, &m, &n, &k, a, &lda, tau,
                         c, &ldc, work, &lwork, &iinfo);
    }
    else {
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

        for (i__ = i1; (i3 < 0 ? i__ >= i2 : i__ <= i2); i__ += i3) {
            ib = min(nb, k - i__ + 1);
            
            /* Form the triangular factor of the block reflector
               H = H(i+ib-1) . . . H(i+1) H(i) */
            i__4 = nq - k + i__ + ib - 1;
            lapackf77_zlarft("Backward", "Columnwise", &i__4, &ib,
                             &a[(i__-1) * lda], &lda, &tau[i__-1], T, &ib);
            
            /* 1) Put 0s in the lower triangular part of A;
               2) copy the panel from A to the GPU, and
               3) restore A                                      */
            zpanel_to_q( MagmaLower, ib, &a[i__-1 + (i__-1) * lda], lda, T+ib*ib);
            magma_zsetmatrix( i__4, ib, &a[(i__-1) * lda], lda, dwork, i__4 );
            zq_to_panel( MagmaLower, ib, &a[i__-1 + (i__-1) * lda], lda, T+ib*ib);
            
            if (left) {
                /* H or H' is applied to C(1:m-k+i+ib-1,1:n) */
                mi = m - k + i__ + ib - 1;
            }
            else {
                /* H or H' is applied to C(1:m,1:n-k+i+ib-1) */
                ni = n - k + i__ + ib - 1;
            }
            
            /* Apply H or H'; First copy T to the GPU */
            magma_zsetmatrix( ib, ib, T, ib, dwork+i__4*ib, ib );
            magma_zlarfb_gpu(side, trans, MagmaBackward, MagmaColumnwise,
                             mi, ni, ib,
                             dwork, i__4, dwork+i__4*ib, ib,
                             dc, m,
                             dwork+i__4*ib + ib*ib, ldwork);
        }

        magma_zgetmatrix( m, n, dc, m, c, ldc );
    }
    work[0] = MAGMA_Z_MAKE( lwkopt, 0 );

    magma_free( dc );
    magma_free( dwork );

    magma_free_pinned( T);

    return *info;
} /* magma_zunmql */
