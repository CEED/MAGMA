/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Raffaele Solca
       @author Stan Tomov

       @precisions normal z -> s d c

*/
#include "common_magma.h"

/**
    Purpose
    -------
    ZUNMTR overwrites the general complex M-by-N matrix C with

                    SIDE = 'L'     SIDE = 'R'
    TRANS = 'N':      Q * C          C * Q
    TRANS = 'T':      Q**H * C       C * Q**H

    where Q is a complex orthogonal matrix of order nq, with nq = m if
    SIDE = 'L' and nq = n if SIDE = 'R'. Q is defined as the product of
    nq-1 elementary reflectors, as returned by ZHETRD:

    if UPLO = 'U', Q = H(nq-1) . . . H(2) H(1);

    if UPLO = 'L', Q = H(1) H(2) . . . H(nq-1).

    Arguments
    ---------
    @param[in]
    side    CHARACTER*1
      -     = 'L': apply Q or Q**H from the Left;
      -     = 'R': apply Q or Q**H from the Right.

    @param[in]
    uplo    CHARACTER*1
      -     = 'U': Upper triangle of A contains elementary reflectors
                   from ZHETRD;
      -     = 'L': Lower triangle of A contains elementary reflectors
                   from ZHETRD.

    @param[in]
    trans   CHARACTER*1
      -     = 'N':  No transpose, apply Q;
      -     = 'T':  Transpose, apply Q**H.

    @param[in]
    m       INTEGER
            The number of rows of the matrix C. M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix C. N >= 0.

    @param[in]
    dA      COMPLEX_16 array, dimension
                                 (LDDA,M) if SIDE = 'L'
                                 (LDDA,N) if SIDE = 'R'
            The vectors which define the elementary reflectors, as
            returned by ZHETRD_GPU. On output the diagonal, the subdiagonal and the
            upper part (UPLO='L') or lower part (UPLO='U') are destroyed.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array DA.
            LDDA >= max(1,M) if SIDE = 'L'; LDDA >= max(1,N) if SIDE = 'R'.

    @param[in]
    tau     COMPLEX_16 array, dimension
                                 (M-1) if SIDE = 'L'
                                 (N-1) if SIDE = 'R'
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by ZHETRD.

    @param[in,out]
    dC      COMPLEX_16 array, dimension (LDDC,N)
            On entry, the M-by-N matrix C.
            On exit, C is overwritten by (Q*C) or (Q**H * C) or (C * Q**H) or (C*Q).

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

    @ingroup magma_zheev_comp
    ********************************************************************/
extern "C" magma_int_t
magma_zunmtr_gpu(magma_side_t side, magma_uplo_t uplo, magma_trans_t trans,
                 magma_int_t m, magma_int_t n,
                 magmaDoubleComplex *dA,    magma_int_t ldda,
                 magmaDoubleComplex *tau,
                 magmaDoubleComplex *dC,    magma_int_t lddc,
                 magmaDoubleComplex *wA,    magma_int_t ldwa,
                 magma_int_t *info)
{
    magma_int_t i1, i2, mi, ni, nq, nw;
    int left, upper;
    magma_int_t iinfo;

    *info = 0;
    left   = (side == MagmaLeft);
    upper  = (uplo == MagmaUpper);

    /* NQ is the order of Q and NW is the minimum dimension of WORK */
    if (left) {
        nq = m;
        nw = n;
    } else {
        nq = n;
        nw = m;
    }
    if (! left && side != MagmaRight) {
        *info = -1;
    } else if (! upper && uplo != MagmaLower) {
        *info = -2;
    } else if (trans != MagmaNoTrans &&
               trans != MagmaConjTrans) {
        *info = -3;
    } else if (m < 0) {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else if (ldda < max(1,nq)) {
        *info = -7;
    } else if (lddc < max(1,m)) {
        *info = -10;
    } else if (ldwa < max(1,nq)) {
        *info = -12;
    }

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0 || nq == 1) {
        return *info;
    }

    if (left) {
        mi = m - 1;
        ni = n;
    } else {
        mi = m;
        ni = n - 1;
    }

    if (upper) {
        magma_zunmql2_gpu(side, trans, mi, ni, nq-1, &dA[ldda], ldda, tau,
                          dC, lddc, &wA[ldwa], ldwa, &iinfo);
    }
    else {
        /* Q was determined by a call to ZHETRD with UPLO = 'L' */
        if (left) {
            i1 = 1;
            i2 = 0;
        } else {
            i1 = 0;
            i2 = 1;
        }
        magma_zunmqr2_gpu(side, trans, mi, ni, nq-1, &dA[1], ldda, tau,
                          &dC[i1 + i2*lddc], lddc, &wA[1], ldwa, &iinfo);
    }

    return *info;
} /* magma_zunmtr */
