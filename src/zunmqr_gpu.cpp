/*
    -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

       @precisions normal z -> s d c

*/

#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include "magma.h"

extern "C" magma_int_t
magma_zunmqr_gpu(char side, char trans, magma_int_t m, magma_int_t n, magma_int_t k, 
		 double2 *a, magma_int_t lda, double2 *tau, double2 *c, magma_int_t ldc,
		 double2 *work, magma_int_t *lwork, double2 *td, magma_int_t nb, magma_int_t *info)
{
/*  -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

    Purpose   
    =======   
    ZUNMQR overwrites the general real M-by-N matrix C with   

                    SIDE = 'L'     SIDE = 'R'   
    TRANS = 'N':      Q * C          C * Q   
    TRANS = 'T':      Q\*\*H * C       C * Q\*\*H   

    where Q is a real orthogonal matrix defined as the product of k   
    elementary reflectors   

          Q = H(1) H(2) . . . H(k)   

    as returned by ZGEQRF. Q is of order M if SIDE = 'L' and of order N   
    if SIDE = 'R'.   

    Arguments   
    =========   

    SIDE    (input) CHARACTER*1   
            = 'L': apply Q or Q\*\*H from the Left;   
            = 'R': apply Q or Q\*\*H from the Right.   

    TRANS   (input) CHARACTER*1   
            = 'N':  No transpose, apply Q;   
            = 'T':  Transpose, apply Q\*\*H.   

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
            A is modified by the routine but restored on exit.   

    LDA     (input) INTEGER   
            The leading dimension of the array A.   
            If SIDE = 'L', LDA >= max(1,M);   
            if SIDE = 'R', LDA >= max(1,N).   

    TAU     (input) COMPLEX_16 array, dimension (K)   
            TAU(i) must contain the scalar factor of the elementary   
            reflector H(i), as returned by ZGEQRF.   

    C       (input/output) COMPLEX_16 array, dimension (LDC,N)   
            On entry, the M-by-N matrix C.   
            On exit, C is overwritten by Q*C or Q\*\*H*C or C*Q\*\*H or C*Q.   

    LDC     (input) INTEGER   
            The leading dimension of the array C. LDC >= max(1,M).   

    WORK    (workspace/output) COMPLEX_16 array, dimension (MAX(1,LWORK))   
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

    TD      (input) COMPLEX_16 array that is the output (the 9th argument)
            of magma_zgeqrf_gpu2.

    NB      (input) INTEGER
            This is the blocking size that was used in pre-computing TD, e.g.,
            the blocking size used in magma_zgeqrf_gpu2.

    INFO    (output) INTEGER   
            = 0:  successful exit   
            < 0:  if INFO = -i, the i-th argument had an illegal value   

    =====================================================================   */

    #define a_ref(a_1,a_2) ( a+(a_2)*(lda) + (a_1))
    #define c_ref(a_1,a_2) ( c+(a_2)*(ldc) + (a_1))
    #define t_ref(a_1)     (td+(a_1))
    #define min(a,b)  (((a)<(b))?(a):(b))
    #define max(a,b)  (((a)>(b))?(a):(b))
    
    double2 c_one = MAGMA_Z_ONE;

    char side_[2] = {side, 0};
    char trans_[2] = {trans, 0};

    double2 *dwork;
    int i, lddwork;

    int i1, i2, i3, ib, ic, jc, mi, ni, nq, nw;
    long int left, notran, lquery;
    static int lwkopt;

    /* Function Body */
    *info = 0;
    left = lapackf77_lsame(side_, "L");
    notran = lapackf77_lsame(trans_, "N");
    lquery = *lwork == -1;

    /* NQ is the order of Q and NW is the minimum dimension of WORK */

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
    } else if (*lwork < max(1,nw) && ! lquery) {
	*info = -12;
    }

    lwkopt = (abs(m-k) + nb + 2*(n))*nb;
    MAGMA_Z_SET2REAL( work[0], lwkopt );

    if (*info != 0) {
	return 0;
    } else if (lquery) {
	return 0;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0 || k == 0) {
	work[0] = c_one;
	return 0;
    }

    lddwork= k;
    dwork  = td+2*lddwork*nb; 

    if ( (left && (! notran)) || ( (!left) && notran ) ) {
      i1 = 0;
      i2 = k-nb;
      i3 = nb;
    } else {
      i1 = (k - 1 - nb) / nb * nb;
      i2 = 0;
      i3 = -nb;
    }
    
    if (left) {
      ni = n;
      jc = 0;
    } else {
      mi = m;
      ic = 0;
    }
    
    if (nb < k)
      {
	for (i = i1; i3 < 0 ? i > i2 : i < i2; i += i3) 
	  {
	    ib = min(nb, k - i);
	    if (left){
	      mi = m - i;
	      ic = i;
	    }
	    else {
	      ni = n - i;
	      jc = i;
	    }
	    magma_zlarfb(MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise, 
                         mi, ni, ib, a_ref(i, i), lda, 
			 t_ref(i), lddwork, c_ref(ic, jc), ldc, dwork, nw);
	  }
      } 
    else 
      {
	i = i1;
      }
    
    /* Use unblocked code to multiply the last or only block. */
    if (i < k) {
      ib   = k-i;
      if (left){
	mi = m - i;
	ic = i;
      }
      else {
	ni = n - i;
	jc = i;
      }

      cublasGetMatrix(mi, ib, sizeof(double2), a_ref(i,i), lda, work, mi);
      cublasGetMatrix(mi, ni, sizeof(double2), c_ref(ic, jc), ldc, 
		      work+mi*ib, mi);

      int lhwork = *lwork - mi*(ib + ni);
      lapackf77_zunmqr("l", "t", &mi, &ni, &ib, work, &mi,
	      tau+i, work+mi*ib, &mi, work+mi*(ib+ni), &lhwork, info);
      
      // send the updated part of c back to the GPU
      cublasSetMatrix(mi, ni, sizeof(double2),
                      work+mi*ib, mi, c_ref(ic, jc), ldc);
    }

    return 0;
    
    /* End of MAGMA_ZUNMQR_GPU */
}


