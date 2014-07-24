/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
       @author Azzam Haidar
       @precisions normal z -> s d c
*/
#include "common_magma.h"

/**
    Purpose
    -------
    ZLARFB applies a complex block reflector H or its transpose H' to a
    COMPLEX_16 m by n matrix C, from the left.
    
    __Note that this function assumes__ that the upper part of dV is 0
    because it is referenced. Same for upper/lower part of dT.

    Arguments
    ---------
    @param[in]
    side    magma_side_t
      -     = MagmaLeft:      apply H or H' from the Left
      -     = MagmaRight:     apply H or H' from the Right

    @param[in]
    trans   magma_trans_t
      -     = MagmaNoTrans:   apply H   (No transpose)
      -     = MagmaConjTrans: apply H' (Conjugate transpose)

    @param[in]
    direct  magma_direct_t
            Indicates how H is formed from a product of elementary
            reflectors
      -     = MagmaForward:  H = H(1) H(2) . . . H(k) (Forward)
      -     = MagmaBackward: H = H(k) . . . H(2) H(1) (Backward)

    @param[in]
    storev  magma_storev_t
            Indicates how the vectors which define the elementary
            reflectors are stored:
      -     = MagmaColumnwise: Columnwise
      -     = MagmaRowwise:    Rowwise

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
    dV      COMPLEX_16 array on the GPU, dimension
                (LDV,K) if STOREV = MagmaColumnwise
                (LDV,M) if STOREV = MagmaRowwise and SIDE = MagmaLeft
                (LDV,N) if STOREV = MagmaRowwise and SIDE = MagmaRight
            The matrix V. See further details.

    @param[in]
    ldv     INTEGER
            The leading dimension of the array V.
            If STOREV = MagmaColumnwise and SIDE = MagmaLeft, LDV >= max(1,M);
            if STOREV = MagmaColumnwise and SIDE = MagmaRight, LDV >= max(1,N);
            if STOREV = MagmaRowwise, LDV >= K.

    @param[in]
    dT      COMPLEX_16 array on the GPU, dimension (LDT,K)
            The triangular k by k matrix T in the representation of the
            block reflector.

    @param[in]
    ldt     INTEGER
            The leading dimension of the array T. LDT >= K.

    @param[in,out]
    dC      COMPLEX_16 array on the GPU, dimension (LDC,N)
            On entry, the m by n matrix C.
            On exit, C is overwritten by H*C, or H'*C, or C*H, or C*H'.

    @param[in]
    ldc     INTEGER
            The leading dimension of the array C. LDA >= max(1,M).

    @param
    dwork   (workspace) COMPLEX_16 array, dimension (LDWORK,K)

    @param[in]
    ldwork  INTEGER
            The leading dimension of the array WORK.
            If SIDE = MagmaLeft,  LDWORK >= max(1,N);
            if SIDE = MagmaRight, LDWORK >= max(1,M);

    @param
    dworkvt (workspace) COMPLEX_16 array, dimension (LDWORKT,K)

    @param[in]
    ldworkvt INTEGER
            The leading dimension of the array WORKVT.
            LDWORKVT >= max(1,min(M,N));

    Further Details
    ---------------
    The shape of the matrix V and the storage of the vectors which define
    the H(i) is best illustrated by the following example with n = 5 and
    k = 3.
    All elements including 0's and 1's are stored, unlike LAPACK.

        DIRECT = MagmaForward and         DIRECT = MagmaForward and
        STOREV = MagmaColumnwise:         STOREV = MagmaRowwise:

                 V = (  1  0  0 )                 V = (  1 v1 v1 v1 v1 )
                     ( v1  1  0 )                     (  0  1 v2 v2 v2 )
                     ( v1 v2  1 )                     (  0  0  1 v3 v3 )
                     ( v1 v2 v3 )
                     ( v1 v2 v3 )

        DIRECT = MagmaBackward and        DIRECT = MagmaBackward and 
        STOREV = MagmaColumnwise:         STOREV = MagmaRowwise:

                 V = ( v1 v2 v3 )                 V = ( v1 v1  1  0  0 )
                     ( v1 v2 v3 )                     ( v2 v2 v2  1  0 )
                     (  1 v2 v3 )                     ( v3 v3 v3 v3  1 )
                     (  0  1 v3 )
                     (  0  0  1 )

    @ingroup magma_zaux3
    ********************************************************************/
extern "C" magma_int_t
magma_zlarfb_gpu_gemm( magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
                  magma_int_t m, magma_int_t n, magma_int_t k,
                  const magmaDoubleComplex *dV,    magma_int_t ldv,
                  const magmaDoubleComplex *dT,    magma_int_t ldt,
                  magmaDoubleComplex *dC,          magma_int_t ldc,
                  magmaDoubleComplex *dwork,       magma_int_t ldwork,
                  magmaDoubleComplex *dworkvt,     magma_int_t ldworkvt)
{
    magmaDoubleComplex c_zero    = MAGMA_Z_ZERO;
    magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;

    magmaDoubleComplex *dwVT = dwork;           // size = m*k
    magmaDoubleComplex *dwQ  = dworkvt;         // size = m*m
    magmaDoubleComplex *dwC  = dwQ + m*m;       // size = m*n


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
        printf("check that workspace dworkvt is of good size \n");

    if ( side == MagmaLeft ) {
        // Form H C or H' C
        // Comments assume H C.
        // When forming H' C, T gets transposed via transt for m >= n or by trans for m < n.
        
        // dwVT = V T
        magma_zgemm( notransV, trans,
                     m, k, k,
                     c_one,  dV, ldv,
                             dT, ldt,
                     c_zero, dwVT, m);
        // dwQ = dwVT * V' = V T V'
        magma_zgemm( MagmaNoTrans, transV,
                     m, m, k,
                     c_one,  dwVT,   m,
                             dV,    ldv,
                     c_zero, dwQ,   m);
        // copy C to Wc then do a gemm C = (I-VTV')*C = C - dwQ * dwC
         magma_zcopymatrix( m, n, dC, ldc, dwC, m );

        // C = C - dwQ*dwC = C - V T V'C
        magma_zgemm( MagmaNoTrans, MagmaNoTrans,
                     m, n, m,
                     c_neg_one, dwQ,   m,
                                dwC,   m,
                     c_one,     dC, ldc);
    }
    else {
        // Form C H or C H'
        // Comments assume C H.
        // When forming C H', T gets transposed via trans.
        printf("not implemented\n");
    }

    return MAGMA_SUCCESS;
} /* magma_zlarfb */
