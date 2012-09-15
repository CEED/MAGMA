/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @author Stan Tomov
       @author Mark Gates

       @precisions normal z -> s d c

*/
#include "common_magma.h"

extern "C" magma_int_t
magma_zunmqr_gpu(char side, char trans,
                 magma_int_t m, magma_int_t n, magma_int_t k,
                 cuDoubleComplex *dA,    magma_int_t ldda,
                 cuDoubleComplex *tau,
                 cuDoubleComplex *dC,    magma_int_t lddc,
                 cuDoubleComplex *hwork, magma_int_t lwork,
                 cuDoubleComplex *dT,    magma_int_t nb,
                 magma_int_t *info)
{
/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======
    ZUNMQR_GPU overwrites the general complex M-by-N matrix C with

                    SIDE = 'L'     SIDE = 'R'
    TRANS = 'N':      Q * C          C * Q
    TRANS = 'T':      Q**H * C       C * Q**H

    where Q is a complex orthogonal matrix defined as the product of k
    elementary reflectors

          Q = H(1) H(2) . . . H(k)

    as returned by ZGEQRF. Q is of order M if SIDE = 'L' and of order N
    if SIDE = 'R'.

    Arguments
    =========
    SIDE    (input) CHARACTER*1
            = 'L': apply Q or Q**H from the Left;
            = 'R': apply Q or Q**H from the Right.

    TRANS   (input) CHARACTER*1
            = 'N':  No transpose, apply Q;
            = 'T':  Transpose, apply Q**H.

    M       (input) INTEGER
            The number of rows of the matrix C. M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix C. N >= 0.

    K       (input) INTEGER
            The number of elementary reflectors whose product defines
            the matrix Q.
            If SIDE = 'L', M >= K >= 0;
            if SIDE = 'R', N >= K >= 0.

    DA      (input) COMPLEX_16 array on the GPU, dimension (LDDA,K)
            The i-th column must contain the vector which defines the
            elementary reflector H(i), for i = 1,2,...,k, as returned by
            ZGEQRF in the first k columns of its array argument DA.
            DA is modified by the routine but restored on exit.

    LDDA    (input) INTEGER
            The leading dimension of the array DA.
            If SIDE = 'L', LDDA >= max(1,M);
            if SIDE = 'R', LDDA >= max(1,N).

    TAU     (input) COMPLEX_16 array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by ZGEQRF.

    DC      (input/output) COMPLEX_16 array on the GPU, dimension (LDDC,N)
            On entry, the M-by-N matrix C.
            On exit, C is overwritten by Q*C or Q**H * C or C * Q**H or C*Q.

    LDDC    (input) INTEGER
            The leading dimension of the array DC. LDDC >= max(1,M).

    HWORK   (workspace/output) COMPLEX_16 array, dimension (MAX(1,LWORK))
            On exit, if INFO = 0, HWORK(1) returns the optimal LWORK.

    LWORK   (input) INTEGER
            The dimension of the array HWORK.
            LWORK >= (M-K+NB)*(N+NB) + N*NB if SIDE = 'L', and
            LWORK >= (N-K+NB)*(M+NB) + M*NB if SIDE = 'R',
            where NB is the optimal blocksize.

            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the HWORK array, returns
            this value as the first entry of the HWORK array, and no error
            message related to LWORK is issued by XERBLA.

    DT      (input) COMPLEX_16 array on the GPU that is the output
            (the 9th argument) of magma_zgeqrf_gpu.

    NB      (input) INTEGER
            This is the blocking size that was used in pre-computing DT, e.g.,
            the blocking size used in magma_zgeqrf_gpu.

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
    =====================================================================   */

    #define dA(a_1,a_2) (dA + (a_1) + (a_2)*ldda)
    #define dC(a_1,a_2) (dC + (a_1) + (a_2)*lddc)
    #define dT(a_1)     (dT + (a_1)*nb)

    cuDoubleComplex c_one = MAGMA_Z_ONE;

    char side_[2]  = {side,  0};
    char trans_[2] = {trans, 0};

    cuDoubleComplex *dwork;
    magma_int_t i, lddwork;
    magma_int_t i1, i2, step, ib, ic, jc, ma, mi, ni, nq, nw;
    int left, notran, lquery;
    magma_int_t lwkopt;

    *info = 0;
    left   = lapackf77_lsame(side_,  "L");
    notran = lapackf77_lsame(trans_, "N");
    lquery = (lwork == -1);

    /* NQ is the order of Q and NW is the minimum dimension of WORK */
    if (left) {
        nq = m;
        nw = n;
    } else {
        nq = n;
        nw = m;
    }
    lwkopt = (nq - k + nb)*(nw + nb) + nw*nb;
    hwork[0] = MAGMA_Z_MAKE( lwkopt, 0 );
    
    if ( (!left) && (!lapackf77_lsame(side_, "R")) ) {
        *info = -1;
    } else if ( (!notran) && (!lapackf77_lsame(trans_, MagmaConjTransStr)) ) {
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
    } else if (lwork < lwkopt && ! lquery) {
        *info = -12;
    }

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }
    else if (lquery) {
        return *info;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0 || k == 0) {
        hwork[0] = c_one;
        return *info;
    }

    lddwork = k;
    dwork = dT(2*lddwork);

    if ( (left && (! notran)) || ((! left) && notran) ) {
        // left  trans:    Q^T C
        // right notrans:  C Q
        // multiply from first block, i = 0, to next-to-last block, i < k-nb
        i1 = 0;
        i2 = k-nb;
        step = nb;
    } else {
        // left  notrans:  Q C
        // right trans:    C Q^T
        // multiply from next-to-last block, i = floor((k-1-nb)/nb)*nb, to first block, i = 0
        i1 = ((k - 1 - nb) / nb) * nb;
        i2 = 0;
        step = -nb;
    }

    if (left) {
        ni = n;
        jc = 0;
    } else {
        mi = m;
        ic = 0;
    }
    
    /* Use unblocked code to multiply last or only block (cases Q*C or C*Q^T). */
    // workspace left:  A(mi*nb) + C(mi*ni) + work(ni*nb_la) = (m-k-nb)*nb + (m-k-nb)*n + n*nb
    // workspace right: A(ni*nb) + C(mi*ni) + work(mi*nb_la) = (n-k-nb)*nb + m*(n-k-nb) + m*nb
    if ( step < 0 ) {
        // i is beginning of last block
        i = i1 - step;
        if ( i >= k ) {
            i = i1;
        }
        ib = k - i;
        if (left) {
            // ni=n, jc=0, H or H^T is applied to C(i:m-1,0:n-1)
            mi = m - i;
            ma = mi;
            ic = i;
        }
        else {
            // mi=m, ic=0, H or H^T is applied to C(0:m-1,i:n-1)
            ni = n - i;
            ma = ni;
            jc = i;
        }
        
        cuDoubleComplex* hA = hwork;
        cuDoubleComplex* hC = hwork + ma*ib;
        cuDoubleComplex* hW = hwork + ma*ib + mi*ni;
        magma_int_t lhwork = lwork - (ma*ib + mi*ni);
        
        magma_zgetmatrix( ma, ib, dA(i,  i ), ldda, hA, ma );
        magma_zgetmatrix( mi, ni, dC(ic, jc), lddc, hC, mi );

        lapackf77_zunmqr( side_, trans_,
                          &mi, &ni, &ib,
                          hA, &ma, tau+i,
                          hC, &mi,
                          hW, &lhwork, info );

        // send the updated part of C back to the GPU
        magma_zsetmatrix( mi, ni, hC, mi, dC(ic, jc), lddc );
    }

    /* Use blocked code to multiply blocks */
    if (nb < k) {
        for( i=i1; (step<0 ? i>=i2 : i<i2); i+=step ) {
            ib = min(nb, k - i);
            if (left) {
                // ni=n, jc=0, H or H^T is applied to C(i:m-1,0:n-1)
                mi = m - i;
                ic = i;
            }
            else {
                // mi=m, ic=0, H or H^T is applied to C(0:m-1,i:n-1)
                ni = n - i;
                jc = i;
            }
            
            magma_zlarfb_gpu( side, trans, MagmaForward, MagmaColumnwise,
                              mi, ni, ib,
                              dA(i,  i ), ldda, dT(i), nb,
                              dC(ic, jc), lddc, dwork, nw );
        }
    }
    else {
        i = i1;
    }

    /* Use unblocked code to multiply the last or only block (cases Q^T*C or C*Q). */
    if ( step > 0 ) {
        ib = k-i;
        if (left) {
            // ni=n, jc=0, H or H^T is applied to C(i:m-1,0:n-1)
            mi = m - i;
            ma = mi;
            ic = i;
        }
        else {
            // mi=m, ic=0, H or H^T is applied to C(0:m-1,i:n-1)
            ni = n - i;
            ma = ni;
            jc = i;
        }
        
        cuDoubleComplex* hA = hwork;
        cuDoubleComplex* hC = hwork + ma*ib;
        cuDoubleComplex* hW = hwork + ma*ib + mi*ni;
        magma_int_t lhwork = lwork - (ma*ib + mi*ni);
        
        magma_zgetmatrix( ma, ib, dA(i,  i ), ldda, hA, ma );
        magma_zgetmatrix( mi, ni, dC(ic, jc), lddc, hC, mi );

        lapackf77_zunmqr( side_, trans_,
                          &mi, &ni, &ib,
                          hA, &ma, tau+i,
                          hC, &mi,
                          hW, &lhwork, info );
        
        // send the updated part of C back to the GPU
        magma_zsetmatrix( mi, ni, hC, mi, dC(ic, jc), lddc );
    }
    
    // TODO sync. For cases Q*C and C*Q^T, last call is magma_zlarfb_gpu,
    // which is async magma_gemm calls, so zunmqr can be unfinished.

    hwork[0] = MAGMA_Z_MAKE( lwkopt, 0 );
    return *info;
}   /* end of magma_zunmqr_gpu */
