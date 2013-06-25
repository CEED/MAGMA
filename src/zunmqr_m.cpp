/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @author Raffaele Solca
       @author Azzam Haidar
       @author Stan Tomov

       @precisions normal z -> s d c

*/
#include "common_magma.h"

#define  A(i, j) ( a+(j)*lda  + (i))
#define  C(i, j) ( c+(j)*ldc  + (i))

#define dC(gpui, i, j) (dw[gpui]+(j)*lddc + (i))
#define dA_c(gpui, ind, i, j) (dw[gpui] + maxnlocal*lddc + (ind)*lddar*lddac + (i) + (j)*lddac)
#define dA_r(gpui, ind, i, j) (dw[gpui] + maxnlocal*lddc + (ind)*lddar*lddac + (i) + (j)*lddar)
#define dt(gpui, ind)    (dw[gpui] + maxnlocal*lddc + 2*lddac*lddar + (ind)*((nb+1)*nb))
#define dwork(gpui, ind) (dw[gpui] + maxnlocal*lddc + 2*lddac*lddar + 2*((nb+1)*nb) + (ind)*(lddwork*nb))

extern"C"{
    void magmablas_zsetdiag1subdiag0_stream(char uplo, int k, int nb, magmaDoubleComplex *A, int lda, magma_queue_t stream);
}

extern "C" magma_int_t
magma_zunmqr_m(magma_int_t nrgpu, char side, char trans,
               magma_int_t m, magma_int_t n, magma_int_t k,
               magmaDoubleComplex *a,    magma_int_t lda,
               magmaDoubleComplex *tau,
               magmaDoubleComplex *c,    magma_int_t ldc,
               magmaDoubleComplex *work, magma_int_t lwork,
               magma_int_t *info)
{
/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======
    ZUNMQR overwrites the general complex M-by-N matrix C with

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

    A       (input) COMPLEX_16 array, dimension (LDA,K)
            The i-th column must contain the vector which defines the
            elementary reflector H(i), for i = 1,2,...,k, as returned by
            ZGEQRF in the first k columns of its array argument A.

    LDA     (input) INTEGER
            The leading dimension of the array A.
            If SIDE = 'L', LDA >= max(1,M);
            if SIDE = 'R', LDA >= max(1,N).

    TAU     (input) COMPLEX_16 array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by ZGEQRF.

    C       (input/output) COMPLEX_16 array, dimension (LDC,N)
            On entry, the M-by-N matrix C.
            On exit, C is overwritten by Q*C or Q**H*C or C*Q**H or C*Q.

    LDC     (input) INTEGER
            The leading dimension of the array C. LDC >= max(1,M).

    WORK    (workspace/output) COMPLEX_16 array, dimension (MAX(1,LWORK))
            On exit, if INFO = 0, WORK(0) returns the optimal LWORK.

    LWORK   (input) INTEGER
            The dimension of the array WORK.
            If SIDE = 'L', LWORK >= max(1,N);
            if SIDE = 'R', LWORK >= max(1,M).
            For optimum performance LWORK >= N*NB if SIDE = 'L', and
            LWORK >= M*NB if SIDE = 'R', where NB is the optimal
            blocksize.

            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued by XERBLA.

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
    =====================================================================   */
    magmaDoubleComplex c_one = MAGMA_Z_ONE;

    char side_[2] = {side, 0};
    char trans_[2] = {trans, 0};

    magma_int_t nb = 128;
    magmaDoubleComplex *t ;
    magma_zmalloc_pinned (&t, nb*nb);
    printf("calling zunmqr_m with nb=%d\n",nb);

    magmaDoubleComplex* dw[MagmaMaxGPUs];
    magma_queue_t stream [MagmaMaxGPUs][2];
    magma_event_t  event [MagmaMaxGPUs][2];

    magma_int_t ind_c;

    magma_int_t igpu = 0;
    int gpu_b;
    magma_getdevice(&gpu_b);

    *info = 0;

    magma_int_t left = lapackf77_lsame(side_, "L");
    magma_int_t notran = lapackf77_lsame(trans_, "N");
    magma_int_t lquery = (lwork == -1);

    /* NQ is the order of Q and NW is the minimum dimension of WORK */
    magma_int_t nq, nw;
    if (left) {
        nq = m;
        nw = n;
    } else {
        nq = n;
        nw = m;
    }


    if (! left && ! lapackf77_lsame(side_, "R")) {
        *info = -1;
    } else if (! notran && ! lapackf77_lsame(trans_, "T")) {
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
    } else if (lwork < max(1,nw) && ! lquery) {
        *info = -12;
    }

    magma_int_t lwkopt = max(1,nw) * nb;
    if (*info == 0)
    {
        MAGMA_Z_SET2REAL( work[0], lwkopt );
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
        work[0] = c_one;
        return *info;
    }

    if (nb >= k)
    {
        /* Use CPU code */
        lapackf77_zunmqr(side_, trans_, &m, &n, &k, a, &lda, tau,
                         c, &ldc, work, &lwork, info);
        return *info;
    }

    magma_int_t lddc = (m+63)/64*64;
    magma_int_t lddac = nq;
    magma_int_t lddar = nb;
    magma_int_t lddwork = nw;

    magma_int_t nlocal[ MagmaMaxGPUs ] = { 0 };

    magma_int_t nb_l=256;
    magma_int_t nbl = (n-1)/nb_l+1; // number of blocks
    magma_int_t maxnlocal = (nbl+nrgpu-1)/nrgpu*nb_l;

    nrgpu = min(nrgpu, (n+nb_l-1)/nb_l); // Don't use GPU that will not have data.

    magma_int_t ldw = maxnlocal*lddc // dC
    + 2*lddac*lddar // 2*dA
    + 2*(nb + 1 + lddwork)*nb; // 2*(dT and dwork)

    for (igpu = 0; igpu < nrgpu; ++igpu){
        magma_setdevice(igpu);
        if (MAGMA_SUCCESS != magma_zmalloc( &dw[igpu], ldw)) {

            magma_xerbla( __func__, -(*info) );
            *info = MAGMA_ERR_DEVICE_ALLOC;

            return *info;
        }
        magma_queue_create( &stream[igpu][0] );
        magma_queue_create( &stream[igpu][1] );
        magma_event_create( &event[igpu][0] );
        magma_event_create( &event[igpu][1] );
    }

    /* Use hybrid CPU-MGPU code */
    if (left) {

        //copy C to mgpus
        for (magma_int_t i = 0; i < nbl; ++i){
            magma_int_t igpu = i%nrgpu;
            magma_setdevice(igpu);
            magma_int_t kb = min(nb_l, n-i*nb_l);
            magma_zsetmatrix_async( m, kb,
                                   C(0, i*nb_l), ldc,
                                   dC(igpu, 0, i/nrgpu*nb_l), lddc, stream[igpu][0] );
            nlocal[igpu] += kb;
        }

        magma_int_t i1, i2, i3;
        if ( !notran ) {
            i1 = 0;
            i2 = k;
            i3 = nb;
        } else {
            i1 = (k - 1) / nb * nb;
            i2 = 0;
            i3 = -nb;
        }

        ind_c = 0;

        for (magma_int_t i = i1; (i3 < 0 ? i >= i2 : i < i2); i += i3)
        {
            // start the copy of A panel
            magma_int_t kb = min(nb, k - i);
            for (igpu = 0; igpu < nrgpu; ++igpu){
                magma_setdevice(igpu);
                magma_event_sync(event[igpu][ind_c]); // check if the new data can be copied
                magma_zsetmatrix_async(nq-i, kb,
                                       A(i, i),                 lda,
                                       dA_c(igpu, ind_c, i, 0), lddac, stream[igpu][0] );
                // Put 0s in the upper triangular part of dA;
                magmablas_zsetdiag1subdiag0_stream('L', kb, kb, dA_c(igpu, ind_c, i, 0), lddac, stream[igpu][0]);
            }

            /* Form the triangular factor of the block reflector
             H = H(i) H(i+1) . . . H(i+ib-1) */
            magma_int_t nqi = nq - i;
            lapackf77_zlarft("F", "C", &nqi, &kb, A(i, i), &lda,
                             &tau[i], t, &kb);

            /* H or H' is applied to C(1:m,i:n) */

            /* Apply H or H'; First copy T to the GPU */
            for (igpu = 0; igpu < nrgpu; ++igpu){
                magma_setdevice(igpu);
                magma_zsetmatrix_async(kb, kb,
                                       t,               kb,
                                       dt(igpu, ind_c), kb, stream[igpu][0] );
            }

            for (igpu = 0; igpu < nrgpu; ++igpu){
                magma_setdevice(igpu);
                magma_queue_sync( stream[igpu][0] ); // check if the data was copied
                magmablasSetKernelStream(stream[igpu][1]);
                magma_zlarfb_gpu( side, trans, MagmaForward, MagmaColumnwise,
                                 m-i, nlocal[igpu], kb,
                                 dA_c(igpu, ind_c, i, 0), lddac, dt(igpu, ind_c), kb,
                                 dC(igpu, i, 0), lddc,
                                 dwork(igpu, ind_c), lddwork);
                magma_event_record(event[igpu][ind_c], stream[igpu][1] );
            }

            ind_c = (ind_c+1)%2;
        }

        for (igpu = 0; igpu < nrgpu; ++igpu){
            magma_setdevice(igpu);
            magma_queue_sync( stream[igpu][1] );
        }

        //copy C from mgpus
        for (magma_int_t i = 0; i < nbl; ++i){
            magma_int_t igpu = i%nrgpu;
            magma_setdevice(igpu);
            magma_int_t kb = min(nb_l, n-i*nb_l);
            magma_zgetmatrix( m, kb,
                              dC(igpu, 0, i/nrgpu*nb_l), lddc,
                              C(0, i*nb_l), ldc );
//            magma_zgetmatrix_async( m, kb,
//                                   dC(igpu, 0, i/nrgpu*nb_l), lddc,
//                                   C(0, i*nb_l), ldc, stream[igpu][0] );
        }

    } else {

        fprintf(stderr, "The case (side == right) is not implemented\n");
        magma_xerbla( __func__, 1 );
        return *info;
        /*
         if ( notran ) {
         i1 = 0;
         i2 = k;
         i3 = nb;
         } else {
         i1 = (k - 1) / nb * nb;
         i2 = 0;
         i3 = -nb;
         }

         mi = m;
         ic = 0;

         for (i = i1; (i3 < 0 ? i >= i2 : i < i2); i += i3)
         {
         ib = min(nb, k - i);

         // Form the triangular factor of the block reflector
         // H = H(i) H(i+1) . . . H(i+ib-1)
         i__4 = nq - i;
         lapackf77_zlarft("F", "C", &i__4, &ib, A(i, i), &lda,
         &tau[i], t, &ib);

         // 1) copy the panel from A to the GPU, and
         // 2) Put 0s in the upper triangular part of dA;
         magma_zsetmatrix( i__4, ib, A(i, i), lda, dA(i, 0), ldda );
         magmablas_zsetdiag1subdiag0('L', ib, ib, dA(i, 0), ldda);


         // H or H' is applied to C(1:m,i:n)
         ni = n - i;
         jc = i;

         // Apply H or H'; First copy T to the GPU
         magma_zsetmatrix( ib, ib, t, ib, dt, ib );
         magma_zlarfb_gpu( side, trans, MagmaForward, MagmaColumnwise,
         mi, ni, ib,
         dA(i, 0), ldda, dt, ib,
         dC(ic, jc), lddc,
         dwork, lddwork);
         }
         */
    }

    MAGMA_Z_SET2REAL( work[0], lwkopt );

    for (igpu = 0; igpu < nrgpu; ++igpu){
        magma_setdevice(igpu);
        magmablasSetKernelStream(NULL);
        magma_event_destroy( event[igpu][0] );
        magma_event_destroy( event[igpu][1] );
        magma_queue_destroy( stream[igpu][0] );
        magma_queue_destroy( stream[igpu][1] );
        magma_free( dw[igpu] );
    }

    magma_setdevice(gpu_b);

    return *info;
} /* magma_zunmqr */
