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

extern "C" int zung2r_(int*, int*, int*, double2*, int*, double2*, double2*, int*);


extern "C" int
magma_zungqr(int *m, int *n, int *k, double2 *a, 
	     int *lda, double2 *tau, double2 *work, int *lwork, int *info)
{
/*  -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

    Purpose   
    =======   

    ZUNGQR generates an M-by-N real matrix Q with orthonormal columns,   
    which is defined as the first N columns of a product of K elementary   
    reflectors of order M   

          Q  =  H(1) H(2) . . . H(k)   

    as returned by ZGEQRF.   

    Arguments   
    =========   

    M       (input) INTEGER   
            The number of rows of the matrix Q. M >= 0.   

    N       (input) INTEGER   
            The number of columns of the matrix Q. M >= N >= 0.   

    K       (input) INTEGER   
            The number of elementary reflectors whose product defines the   
            matrix Q. N >= K >= 0.   

    A       (input/output) COMPLEX_16 array, dimension (LDA,N)   
            On entry, the i-th column must contain the vector which   
            defines the elementary reflector H(i), for i = 1,2,...,k, as   
            returned by ZGEQRF in the first k columns of its array   
            argument A.   
            On exit, the M-by-N matrix Q.   

    LDA     (input) INTEGER   
            The first dimension of the array A. LDA >= max(1,M).   

    TAU     (input) COMPLEX_16 array, dimension (K)   
            TAU(i) must contain the scalar factor of the elementary   
            reflector H(i), as returned by ZGEQRF.   

    WORK    (workspace/output) COMPLEX_16 array, dimension (MAX(1,LWORK))   
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.   

    LWORK   (input) INTEGER   
            The dimension of the array WORK. LWORK >= max(1,N).   
            For optimum performance LWORK >= N*NB, where NB is the   
            optimal blocksize.   

            If LWORK = -1, then a workspace query is assumed; the routine   
            only calculates the optimal size of the WORK array, returns   
            this value as the first entry of the WORK array, and no error   
            message related to LWORK is issued by XERBLA.   

    INFO    (output) INTEGER   
            = 0:  successful exit   
            < 0:  if INFO = -i, the i-th argument has an illegal value   

    =====================================================================    */

    #define min(a,b)       (((a)<(b))?(a):(b))
    #define max(a,b)       (((a)>(b))?(a):(b))

    double2 c_one = MAGMA_Z_ONE;
    double2 c_zero = MAGMA_Z_ZERO;

    int a_dim1, a_offset, i__1, i__2, i__3;
    static int i__, j, l, ib, nb, ki, kk, nx, nbmin, iinfo;
    static int ldwork, lwkopt;
    long int lquery;

    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --tau;
    --work;

    /* Function Body */
    *info = 0;
    nb = magma_get_zgeqrf_nb(*m);
    lwkopt = (*m + *n) * nb;
    MAGMA_Z_SET2REAL( work[1], lwkopt );
    lquery = *lwork == -1;
    if (*m < 0) {
	*info = -1;
    } else if (*n < 0 || *n > *m) {
	*info = -2;
    } else if (*k < 0 || *k > *n) {
	*info = -3;
    } else if (*lda < max(1,*m)) {
	*info = -5;
    } else if (*lwork < max(1,*n) && ! lquery) {
	*info = -8;
    }
    if (*info != 0)
      return 0;
    else if (lquery)
	return 0;

    /*  Quick return if possible */
    if (*n <= 0) {
      work[1] = c_one;
      return 0;
    }

    nbmin = 2;
    nx = nb;
    
    if (nb >= nbmin && nb < *k && nx < *k) 
      {
	/*  Use blocked code after the last block.   
	    The first kk columns are handled by the block method. */
	ki = (*k - nx - 1) / nb * nb;
	kk = min(*k, ki + nb);
	
	/* Set A(1:kk,kk+1:n) to zero. */
	for (j = kk + 1; j <= *n; ++j)
	  for (i__ = 1; i__ <= kk; ++i__)
	    a[i__ + j * a_dim1] = c_zero;
      }
    else 
      {
	kk = 0;
      }
    
    /* Use unblocked code for the last or only block. */
    if (kk < *n) 
      {
	i__1 = *m - kk;
	i__2 = *n - kk;
	i__3 = *k - kk;
	zung2r_(&i__1, &i__2, &i__3, &a[kk + 1 + (kk + 1) * a_dim1], lda, &
		tau[kk + 1], &work[1], &iinfo);
      }

    if (kk > 0) 
      {
	/* Use blocked code */
	for (i__ = ki + 1; i__ >= 1; i__-=nb) 
	  {
	    ib = min(nb, *k - i__ + 1);
	    if (i__ + ib <= *n) 
	      {
		/* Form the triangular factor of the block reflector   
		   H = H(i) H(i+1) . . . H(i+ib-1) */
		i__2 = *m - i__ + 1;
		zlarft_("Forward", "Columnwise", &i__2, &ib, 
			&a[i__+i__*a_dim1], lda, &tau[i__], &work[1], &ldwork);

		/* Apply H to A(i:m,i+ib:n) from the left */
		i__2 = *m - i__ + 1;
		i__3 = *n - i__ - ib + 1;
		zlarfb_("Left", "No transpose", "Forward", "Columnwise", &
			i__2, &i__3, &ib, &a[i__ + i__ * a_dim1], lda, &work[
			1], &ldwork, &a[i__ + (i__ + ib) * a_dim1], lda, &
			work[ib + 1], &ldwork);
	      }
	    
	    /* Apply H to rows i:m of current block */
	    i__2 = *m - i__ + 1;
	    zung2r_(&i__2, &ib, &ib, &a[i__ + i__ * a_dim1], lda, &tau[i__], &
		    work[1], &iinfo);

	    /* Set rows 1:i-1 of current block to zero */
	    i__2 = i__ + ib - 1;
	    for (j = i__; j <= i__2; ++j)
	      for (l = 1; l <= i__-1; ++l)
		a[l + j * a_dim1] = c_zero;
	  }
      }

    return 0;

/*     End of MAGMA_ZUNGQR */

} /* magma_zungqr */

#undef min
#undef max

