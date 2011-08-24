/*
    -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

       @author Raffaele Solca

       @precisions normal z -> s d c

*/
#include "common_magma.h"

extern "C" magma_int_t
magma_zunmql(const char side, const char trans, 
	     magma_int_t m, magma_int_t n, magma_int_t k, 
	     cuDoubleComplex *a, magma_int_t lda, 
	     cuDoubleComplex *tau, 
	     cuDoubleComplex *c, magma_int_t ldc, 
	     cuDoubleComplex *work, magma_int_t lwork,
	     magma_int_t *info)
{
/*  -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

    Purpose   
    =======   
    ZUNMQL overwrites the general complex M-by-N matrix C with   

                    SIDE = 'L'     SIDE = 'R'   
    TRANS = 'N':      Q * C          C * Q   
    TRANS = 'C':      Q**H * C       C * Q**H   

    where Q is a complex unitary matrix defined as the product of k   
    elementary reflectors   

          Q = H(k) . . . H(2) H(1)   

    as returned by ZGEQLF. Q is of order M if SIDE = 'L' and of order N   
    if SIDE = 'R'.   

    Arguments   
    =========   
    SIDE    (input) CHARACTER*1   
            = 'L': apply Q or Q**H from the Left;   
            = 'R': apply Q or Q**H from the Right.   

    TRANS   (input) CHARACTER*1   
            = 'N':  No transpose, apply Q;   
            = 'C':  Transpose, apply Q**H.   

    M       (input) INTEGER   
            The number of rows of the matrix C. M >= 0.   

    N       (input) INTEGER   
            The number of columns of the matrix C. N >= 0.   

    K       (input) INTEGER   
            The number of elementary reflectors whose product defines   
            the matrix Q.   
            If SIDE = 'L', M >= K >= 0;   
            if SIDE = 'R', N >= K >= 0.   

    A       (input) COMPLEX*16 array, dimension (LDA,K)   
            The i-th column must contain the vector which defines the   
            elementary reflector H(i), for i = 1,2,...,k, as returned by   
            ZGEQLF in the last k columns of its array argument A.   
            A is modified by the routine but restored on exit.   

    LDA     (input) INTEGER   
            The leading dimension of the array A.   
            If SIDE = 'L', LDA >= max(1,M);   
            if SIDE = 'R', LDA >= max(1,N).   

    TAU     (input) COMPLEX*16 array, dimension (K)   
            TAU(i) must contain the scalar factor of the elementary   
            reflector H(i), as returned by ZGEQLF.   

    C       (input/output) COMPLEX*16 array, dimension (LDC,N)   
            On entry, the M-by-N matrix C.   
            On exit, C is overwritten by Q*C or Q**H*C or C*Q**H or C*Q.   

    LDC     (input) INTEGER   
            The leading dimension of the array C. LDC >= max(1,M).   

    WORK    (workspace/output) COMPLEX*16 array, dimension (MAX(1,LWORK))   
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.   

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
    =====================================================================    */
    
    char side_[2] = {side, 0};
    char trans_[2] = {trans, 0};

    /* Allocate work space on the GPU */
    cuDoubleComplex *dwork, *dc;
    cublasAlloc((m)*(n), sizeof(cuDoubleComplex), (void**)&dc);
    cublasAlloc(2*(m+64)*64, sizeof(cuDoubleComplex), (void**)&dwork);

    /* Copy matrix C from the CPU to the GPU */
    cublasSetMatrix( m, n, sizeof(cuDoubleComplex), c, ldc, dc, m);
    dc -= (1 + m);

    magma_int_t a_offset, c_dim1, c_offset, i__4;
    
    static magma_int_t i__;
    static cuDoubleComplex t[2*4160]	/* was [65][64] */;
    static magma_int_t i1, i2, i3, ib, nb, mi, ni, nq, nw;
    static magma_int_t iinfo, ldwork, lwkopt;
    long int lquery, left, notran;

    a_offset = 1 + lda;
    a -= a_offset;
    --tau;
    c_dim1 = ldc;
    c_offset = 1 + c_dim1;
    c -= c_offset;

    *info  = 0;
    left   = lapackf77_lsame(side_, "L");
    notran = lapackf77_lsame(trans_, "N");
    lquery = (lwork == -1);

    /* NQ is the order of Q and NW is the minimum dimension of WORK */
    if (left) {
	nq = m;
	nw = max(1,n);
    } else {
	nq = n;
	nw = max(1,m);
    }
    if (! left && ! lapackf77_lsame(side_, "R")) {
	*info = -1;
    } else if (! notran && ! lapackf77_lsame(trans_, "C")) {
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
      MAGMA_Z_SET2REAL( work[0], lwkopt );

      if (lwork < nw && ! lquery) {
	*info = -12;
      }
    }

    if (*info != 0) {
      return MAGMA_SUCCESS;
    } else if (lquery) {
      return MAGMA_SUCCESS;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0) {
      return MAGMA_SUCCESS;
    }

    ldwork = nw;

    if ( nb >= k ) 
      {
	/* Use CPU code */
	lapackf77_zunmql(side_, trans_, &m, &n, &k, &a[a_offset], &lda, &tau[1], 
			 &c[c_offset], &ldc, work, &lwork, &iinfo);
      } 
    else 
      {
	/* Use hybrid CPU-GPU code */
	if (left && notran || ! left && ! notran) {
	    i1 = 1;
	    i2 = k;
	    i3 = nb;
	} else {
	    i1 = (k - 1) / nb * nb + 1;
	    i2 = 1;
	    i3 = -nb;
	}

	if (left) {
	    ni = n;
	} else {
	    mi = m;
	}

	for (i__ = i1; i3 < 0 ? i__ >= i2 : i__ <= i2; i__ += i3) {
	  ib = min(nb, k - i__ + 1);
	  
	  /* Form the triangular factor of the block reflector   
	     H = H(i+ib-1) . . . H(i+1) H(i) */
	  i__4 = nq - k + i__ + ib - 1;
	  lapackf77_zlarft("Backward", "Columnwise", &i__4, &ib, 
			   &a[i__ * lda + 1], &lda, &tau[i__], t, &ib);

	  /* 1) Put 0s in the lower triangular part of A;
	     2) copy the panel from A to the GPU, and
	     3) restore A                                      */
	  zpanel_to_q('L', ib, &a[i__ + i__ * lda], lda, t+ib*ib);
	  cublasSetMatrix(i__4, ib, sizeof(cuDoubleComplex),
			  &a[1 + i__ * lda], lda,
			  dwork, i__4);
	  zq_to_panel('L', ib, &a[i__ + i__ * lda], lda, t+ib*ib);

	  if (left) 
	    {	    
	      /* H or H' is applied to C(1:m-k+i+ib-1,1:n) */
	      mi = m - k + i__ + ib - 1;
	    } 
	  else 
	    {
	      /* H or H' is applied to C(1:m,1:n-k+i+ib-1) */
	      ni = n - k + i__ + ib - 1;
	    }
	  
	  /* Apply H or H'; First copy T to the GPU */
	  cublasSetMatrix(ib, ib, sizeof(cuDoubleComplex), t, ib, dwork+i__4*ib, ib);
	  magma_zlarfb_gpu(side, trans, MagmaBackward, MagmaColumnwise,
			   mi, ni, ib, 
			   dwork, i__4, dwork+i__4*ib, ib, 
			   &dc[1+m], m,
			   dwork+i__4*ib + ib*ib, ldwork);
	}

	cublasGetMatrix(m, n, sizeof(cuDoubleComplex), &dc[1+m], m, &c[c_offset], ldc);
    }
    MAGMA_Z_SET2REAL( work[0], lwkopt );

    dc += (1 + m);
    cublasFree(dc);
    cublasFree(dwork);

    return MAGMA_SUCCESS;
} /* magma_zunmql */
