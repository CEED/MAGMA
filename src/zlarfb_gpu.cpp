/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Stan Tomov
       @author Mark Gates
       @precisions normal z -> s d c
*/
#include "common_magma.h"

/**
    Purpose
    =======
    ZLARFB applies a complex block reflector H or its transpose H^H to a
    COMPLEX_16 m by n matrix C, from the left.

    Arguments
    =========
    @param[in]
    side    CHARACTER
      -     = 'L': apply H or H^H from the Left
      -     = 'R': apply H or H^H from the Right

    @param[in]
    trans   CHARACTER
      -     = 'N': apply H   (No transpose)
      -     = 'C': apply H^H (Conjugate transpose)

    @param[in]
    direct  CHARACTER
            Indicates how H is formed from a product of elementary
            reflectors
            = 'F': H = H(1) H(2) . . . H(k) (Forward)
            = 'B': H = H(k) . . . H(2) H(1) (Backward)

    @param[in]
    storev  CHARACTER
            Indicates how the vectors which define the elementary
            reflectors are stored:
            = 'C': Columnwise
            = 'R': Rowwise

    @param[in]
    m       INTEGER
            The number of rows of the matrix C.

    @param[in]
    n       INTEGER
            The number of columns of the matrix C.

    @param[in]
    k       INTEGER
            The order of the matrix T (= the number of elementary
            reflectors whose product defines the block reflector).

    @param[in]
    DV      COMPLEX_16 array on the GPU, dimension
                (LDV,K) if STOREV = 'C'
                (LDV,M) if STOREV = 'R' and SIDE = 'L'
                (LDV,N) if STOREV = 'R' and SIDE = 'R'
            The matrix V. See further details.

    @param[in]
    ldv     INTEGER
            The leading dimension of the array V.
            If STOREV = 'C' and SIDE = 'L', LDV >= max(1,M);
            if STOREV = 'C' and SIDE = 'R', LDV >= max(1,N);
            if STOREV = 'R', LDV >= K.

    @param[in]
    DT      COMPLEX_16 array on the GPU, dimension (LDT,K)
            The triangular k by k matrix T in the representation of the
            block reflector.

    @param[in]
    ldt     INTEGER
            The leading dimension of the array T. LDT >= K.

    @param[in,out]
    DC      COMPLEX_16 array on the GPU, dimension (LDC,N)
            On entry, the m by n matrix C.
            On exit, C is overwritten by H*C, or H^H*C, or C*H, or C*H^H.

    @param[in]
    ldc     INTEGER
            The leading dimension of the array C. LDA >= max(1,M).

    WORK    (workspace) COMPLEX_16 array, dimension (LDWORK,K)

    @param[in]
    ldwork  INTEGER
            The leading dimension of the array WORK.
            If SIDE = 'L', LDWORK >= max(1,N);
            if SIDE = 'R', LDWORK >= max(1,M);

    Further Details
    ===============
    The shape of the matrix V and the storage of the vectors which define
    the H(i) is best illustrated by the following example with n = 5 and
    k = 3.
    All elements including 0's and 1's are stored, unlike LAPACK.

    DIRECT = 'F' and STOREV = 'C':         DIRECT = 'F' and STOREV = 'R':

                 V = (  1  0  0 )                 V = (  1 v1 v1 v1 v1 )
                     ( v1  1  0 )                     (  0  1 v2 v2 v2 )
                     ( v1 v2  1 )                     (  0  0  1 v3 v3 )
                     ( v1 v2 v3 )
                     ( v1 v2 v3 )

    DIRECT = 'B' and STOREV = 'C':         DIRECT = 'B' and STOREV = 'R':

                 V = ( v1 v2 v3 )                 V = ( v1 v1  1  0  0 )
                     ( v1 v2 v3 )                     ( v2 v2 v2  1  0 )
                     (  1 v2 v3 )                     ( v3 v3 v3 v3  1 )
                     (  0  1 v3 )
                     (  0  0  1 )

    @ingroup magma_zaux3
    =================================================================== */
extern "C" magma_int_t
magma_zlarfb_gpu( magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
                  magma_int_t m, magma_int_t n, magma_int_t k,
                  const magmaDoubleComplex *dV,    magma_int_t ldv,
                  const magmaDoubleComplex *dT,    magma_int_t ldt,
                  magmaDoubleComplex *dC,          magma_int_t ldc,
                  magmaDoubleComplex *dwork,       magma_int_t ldwork )
{
    magmaDoubleComplex c_zero    = MAGMA_Z_ZERO;
    magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;

    /* Check input arguments */
    magma_int_t info = 0;
    if (m < 0) {
        info = -5;
    } else if (n < 0) {
        info = -6;
    } else if (k < 0) {
        info = -7;
    } else if ( ((storev == MagmaColumnwise) && (side == MagmaLeft) && ldv < max(1,m)) ||
                ((storev == MagmaColumnwise) && (side == MagmaRight) && ldv < max(1,n)) ||
                ((storev == MagmaRowwise) && ldv < k) ) {
        info = -9;
    } else if (ldt < k) {
        info = -11;
    } else if (ldc < max(1,m)) {
        info = -13;
    } else if ( ((side == MagmaLeft) && ldwork < max(1,n)) ||
                ((side == MagmaRight) && ldwork < max(1,m)) ) {
        info = -15;
    }
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return info;
    }
    
    /* Function Body */
    if (m <= 0 || n <= 0) {
        return MAGMA_SUCCESS;
    }

    // opposite of trans
    magma_trans_t transt;
    if (trans == MagmaNoTrans)
        transt = MagmaConjTrans;
    else
        transt = MagmaNoTrans;
    
    // whether T is upper or lower triangular
    magma_uplo_t uplo;
    if (direct == MagmaForward)
        uplo = MagmaUpper;
    else
        uplo = MagmaLower;
    
    // whether V is stored transposed or not
    magma_trans_t notransV, transV;
    if (storev == MagmaColumnwise) {
        notransV = MagmaNoTrans;
        transV   = MagmaConjTrans;
    }
    else {
        notransV = MagmaConjTrans;
        transV   = MagmaNoTrans;
    }

    if ( side == MagmaLeft ) {
        // Form H C or H^H C
        // Comments assume H C. When forming H^H C, T gets transposed via transt.
        
        // W = C^H V
        magma_zgemm( MagmaConjTrans, notransV,
                     n, k, m,
                     c_one,  dC,    ldc,
                             dV,    ldv,
                     c_zero, dwork, ldwork);

        // W = W T^H = C^H V T^H
        magma_ztrmm( MagmaRight, uplo, transt, MagmaNonUnit,
                     n, k,
                     c_one, dT,    ldt,
                            dwork, ldwork);

        // C = C - V W^H = C - V T V^H C = (I - V T V^H) C = H C
        magma_zgemm( notransV, MagmaConjTrans,
                     m, n, k,
                     c_neg_one, dV,    ldv,
                                dwork, ldwork,
                     c_one,     dC,    ldc);
    }
    else {
        // Form C H or C H^H
        // Comments assume C H. When forming C H^H, T gets transposed via trans.
        
        // W = C V
        magma_zgemm( MagmaNoTrans, notransV,
                     m, k, n,
                     c_one,  dC,    ldc,
                             dV,    ldv,
                     c_zero, dwork, ldwork);

        // W = W T = C V T
        magma_ztrmm( MagmaRight, uplo, trans, MagmaNonUnit,
                     m, k,
                     c_one, dT,    ldt,
                            dwork, ldwork);

        // C = C - W V^H = C - C V T V^H = C (I - V T V^H) = C H
        magma_zgemm( MagmaNoTrans, transV,
                     m, n, k,
                     c_neg_one, dwork, ldwork,
                                dV,    ldv,
                     c_one,     dC,    ldc);
    }

    return MAGMA_SUCCESS;
} /* magma_zlarfb */
