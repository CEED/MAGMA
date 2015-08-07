/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
       @author Azzam Haidar
       @author Tingxing Dong
       @precisions normal z -> s d c
*/

#include "common_magma.h"
#include "cublas_v2.h"

/**
    Purpose
    -------
    ZLARFB applies a complex block reflector H or its transpose H^H to a
    COMPLEX_16 m by n matrix C, from the left.
    
    __Note that this function assumes__ that the upper part of dV_array is 0
    because it is referenced. Same for upper/lower part of dT_array.

    Arguments
    ---------
    @param[in]
    side    magma_side_t
      -     = MagmaLeft:      apply H or H^H from the Left
      -     = MagmaRight:     apply H or H^H from the Right

    @param[in]
    trans   magma_trans_t
      -     = MagmaNoTrans:    apply H   (No transpose)
      -     = Magma_ConjTrans: apply H^H (Conjugate transpose)

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
    dV_array      COMPLEX_16 array on the GPU, dimension
                (LDDV,K) if STOREV = MagmaColumnwise
                (LDDV,M) if STOREV = MagmaRowwise and SIDE = MagmaLeft
                (LDDV,N) if STOREV = MagmaRowwise and SIDE = MagmaRight
            The matrix V. See further details.

    @param[in]
    lddv    INTEGER
            The leading dimension of the array V.
            If STOREV = MagmaColumnwise and SIDE = MagmaLeft, LDDV >= max(1,M);
            if STOREV = MagmaColumnwise and SIDE = MagmaRight, LDDV >= max(1,N);
            if STOREV = MagmaRowwise, LDDV >= K.

    @param[in]
    dT_array      COMPLEX_16 array on the GPU, dimension (LDDT,K)
            The triangular k by k matrix T in the representation of the
            block reflector.

    @param[in]
    lddt    INTEGER
            The leading dimension of the array T. LDDT >= K.

    @param[in,out]
    dC_array      COMPLEX_16 array on the GPU, dimension (LDDC,N)
            On entry, the m by n matrix C.
            On exit, C is overwritten by H*C, or H^H*C, or C*H, or C*H^H.

    @param[in]
    lddc    INTEGER
            The leading dimension of the array C. LDA >= max(1,M).

    @param
    dwork_array   (workspace) COMPLEX_16 array, dimension (LDWORK,K)

    @param[in]
    ldwork  INTEGER
            The leading dimension of the array WORK.
            If SIDE = MagmaLeft,  LDWORK >= max(1,N);
            if SIDE = MagmaRight, LDWORK >= max(1,M);

    @param
    dworkvt_array (workspace) COMPLEX_16 array, dimension (LDWORKT,K)

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
magma_zlarfb_gemm_batched(
    magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex_const_ptr dV_array[],    magma_int_t lddv,
    magmaDoubleComplex_const_ptr dT_array[],    magma_int_t lddt,
    magmaDoubleComplex_ptr dC_array[],          magma_int_t lddc,
    magmaDoubleComplex_ptr dwork_array[],       magma_int_t ldwork,
    magmaDoubleComplex_ptr dworkvt_array[],     magma_int_t ldworkvt,
    magma_int_t batchCount, magma_queue_t queue, cublasHandle_t myhandle)
{
    magmaDoubleComplex c_zero    = MAGMA_Z_ZERO;
    magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;

    /* Function Body */
    magma_int_t info = 0;
    if (m <= 0 || n <= 0) {
        return info;
    }
    // internal variable
    magma_int_t ldwvt = (m > n ?  k : m);
    magma_int_t ldw;
    if ( side == MagmaLeft ) {
        ldw = k;
    } else {
        ldw = m;
    }
    // opposite of trans
    magma_trans_t transt;
    if (trans == MagmaNoTrans)
        transt = Magma_ConjTrans;
    else
        transt = MagmaNoTrans;
    
    MAGMA_UNUSED( transt );  // TODO: is this a bug that it isn't used?
    
    
    // whether V is stored transposed or not
    magma_trans_t notransV, transV;
    if (storev == MagmaColumnwise) {
        notransV = MagmaNoTrans;
        transV   = Magma_ConjTrans;
    }
    else {
        notransV = Magma_ConjTrans;
        transV   = MagmaNoTrans;
    }

    if ( side == MagmaLeft ) {
        // Form H C or H^H C
        // Comments assume H C.
        // When forming H^H C, T gets transposed via transt for m >= n or by trans for m < n.
        
        // W = V' C                              
        magma_zgemm_batched( Magma_ConjTrans,notransV, /*NontransLeft*/
                     k, n, m,
                     c_one,  dV_array,    lddv,
                             dC_array,    lddc,
                     c_zero, dwork_array, ldw,
                     batchCount, queue, myhandle);

        if (m <= n) {
            // W2 = V T
            magma_zgemm_batched( notransV, trans, /* (NoTrans), trans(ConjTrans),*/
                         m, k, k,
                         c_one,  dV_array, lddv,
                                 dT_array, lddt,
                         c_zero, dworkvt_array, ldwvt,
                         batchCount, queue, myhandle);


            // C = C - W2 W = C - V T V' C = (I - V T V') C = H C
            magma_zgemm_batched( MagmaNoTrans, MagmaNoTrans,
                         m, n, k,
                         c_neg_one, dworkvt_array,  ldwvt,
                                    dwork_array,    ldw,
                         c_one,     dC_array,       lddc,
                         batchCount, queue, myhandle);
        }
        else {
            // W2 = T W  = T  V' C
            magma_zgemm_batched( trans, MagmaNoTrans,
                         k, n, k,
                         c_one,  dT_array, lddt,
                                 dwork_array, ldw,
                         c_zero, dworkvt_array, ldwvt,
                         batchCount, queue, myhandle);

            // C = C - V W2 = C - V T V' C = (I - V T V') C = H C
            magma_zgemm_batched( notransV, MagmaNoTrans,
                         m, n, k,
                         c_neg_one, dV_array,  lddv,
                                    dworkvt_array,  ldwvt,
                         c_one,     dC_array,       lddc,
                         batchCount, queue, myhandle);
        }
    }
    else {
        // Form C H or C H^H
        // Comments assume C H.
        // When forming C H^H, T gets transposed via trans.
        
        // W = C V
        magma_zgemm_batched( MagmaNoTrans, notransV,
                     m, k, n,
                     c_one,  dC_array,    lddc,
                             dV_array,    lddv,
                     c_zero, dwork_array, ldw,
                     batchCount, queue, myhandle);
        if (m <= n) {
            // W2 = W T = C V T
            magma_zgemm_batched( MagmaNoTrans, trans,
                         m, k, k,
                         c_one,  dwork_array, ldw,
                                 dT_array, lddt,
                         c_zero, dworkvt_array, ldwvt,
                         batchCount, queue, myhandle);

            // C = C - W2 V' = C - C V T V' = C (I - V T V') = C H
            magma_zgemm_batched( MagmaNoTrans, transV,
                         m, n, k,
                         c_neg_one, dworkvt_array, ldwvt,
                                    dV_array,    lddv,
                         c_one,     dC_array,    lddc,
                         batchCount, queue, myhandle);
        }
        else {
            // W2 = T V'
            magma_zgemm_batched( trans, transV,
                         k, n, k,
                         c_one,  dT_array, lddt,
                                 dV_array, lddv,
                         c_zero, dworkvt_array, ldwvt,
                         batchCount, queue, myhandle);
            // C = C - W W2 = C - C V T V' = C (I - V T V') = C H
            magma_zgemm_batched( MagmaNoTrans, MagmaNoTrans,
                         m, n, k,
                         c_neg_one, dwork_array,   ldw,
                                    dworkvt_array, ldwvt,
                         c_one,     dC_array,      lddc,
                         batchCount, queue, myhandle);
        }
    }
    return MAGMA_SUCCESS;
} /* magma_zlarfb */



