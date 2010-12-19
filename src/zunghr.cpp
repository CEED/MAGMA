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
magma_zunghr(magma_int_t *n, magma_int_t *ilo, magma_int_t *ihi, 
	     cuDoubleComplex *a, magma_int_t *lda, 
	     cuDoubleComplex *tau,
	     cuDoubleComplex *work, magma_int_t *lwork,
	     magma_int_t *info)
{
/*  -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

    Purpose   
    =======   
    ZUNGHR generates a complex unitary matrix Q which is defined as the   
    product of IHI-ILO elementary reflectors of order N, as returned by   
    ZGEHRD:   

    Q = H(ilo) H(ilo+1) . . . H(ihi-1).   

    Arguments   
    =========   
    N       (input) INTEGER   
            The order of the matrix Q. N >= 0.   

    ILO     (input) INTEGER   
    IHI     (input) INTEGER   
            ILO and IHI must have the same values as in the previous call   
            of ZGEHRD. Q is equal to the unit matrix except in the   
            submatrix Q(ilo+1:ihi,ilo+1:ihi).   
            1 <= ILO <= IHI <= N, if N > 0; ILO=1 and IHI=0, if N=0.   

    A       (input/output) COMPLEX*16 array, dimension (LDA,N)   
            On entry, the vectors which define the elementary reflectors,   
            as returned by ZGEHRD.   
            On exit, the N-by-N unitary matrix Q.   

    LDA     (input) INTEGER   
            The leading dimension of the array A. LDA >= max(1,N).   

    TAU     (input) COMPLEX*16 array, dimension (N-1)   
            TAU(i) must contain the scalar factor of the elementary   
            reflector H(i), as returned by ZGEHRD.   

    WORK    (workspace/output) COMPLEX*16 array, dimension (MAX(1,LWORK))   
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.   

    LWORK   (input) INTEGER   
            The dimension of the array WORK. LWORK >= IHI-ILO.   
            For optimum performance LWORK >= (IHI-ILO)*NB, where NB is   
            the optimal blocksize.   

            If LWORK = -1, then a workspace query is assumed; the routine   
            only calculates the optimal size of the WORK array, returns   
            this value as the first entry of the WORK array, and no error   
            message related to LWORK is issued by XERBLA.   

    INFO    (output) INTEGER   
            = 0:  successful exit   
            < 0:  if INFO = -i, the i-th argument had an illegal value   
    ===================================================================== */

    #define min(a,b)  (((a)<(b))?(a):(b))
    #define max(a,b)  (((a)>(b))?(a):(b))

  extern magma_int_t ilaenv_(magma_int_t *, char *, char *, magma_int_t *,
			     magma_int_t *, magma_int_t *, magma_int_t *,
			     magma_int_t, magma_int_t);

    static magma_int_t c__1 = 1;
    static magma_int_t c_n1 = -1;
    
    magma_int_t a_dim1, a_offset, i__1, i__2, i__3, i__4;

    magma_int_t i__, j, nb, nh, iinfo;
    magma_int_t lwkopt, lquery;

    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --tau;
    --work;

    *info = 0;
    nh = *ihi - *ilo;
    lquery = *lwork == -1;
    if (*n < 0) {
	*info = -1;
    } else if (*ilo < 1 || *ilo > max(1,*n)) {
	*info = -2;
    } else if (*ihi < min(*ilo,*n) || *ihi > *n) {
	*info = -3;
    } else if (*lda < max(1,*n)) {
	*info = -5;
    } else if (*lwork < max(1,nh) && ! lquery) {
	*info = -8;
    }

    if (*info == 0) {
	nb = ilaenv_(&c__1, "ZUNGQR", " ", &nh, &nh, &nh, &c_n1, 6, 1);
	lwkopt = max(1,nh) * nb;
	MAGMA_Z_SET2REAL(work[1], (double) lwkopt);
    }

    if (*info != 0) {
	i__1 = -(*info);
	magma_xerbla("magma_zunghr", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

    /* Quick return if possible */
    if (*n == 0) 
      {
	MAGMA_Z_SET2REAL(work[1], 1.);
	return 0;
      }

    /* Shift the vectors which define the elementary reflectors one   
       column to the right, and set the first ilo and the last n-ihi   
       rows and columns to those of the unit matrix */
    i__1 = *ilo + 1;
    for (j = *ihi; j >= i__1; --j) {
	i__2 = j - 1;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    i__3 = i__ + j * a_dim1;
	    a[i__3] = MAGMA_Z_ZERO;
	}
	i__2 = *ihi;
	for (i__ = j + 1; i__ <= i__2; ++i__) {
	    i__3 = i__ + j * a_dim1;
	    i__4 = i__ + (j - 1) * a_dim1;
	    a[i__3] = a[i__4];
	}
	i__2 = *n;
	for (i__ = *ihi + 1; i__ <= i__2; ++i__) {
	    i__3 = i__ + j * a_dim1;
	    a[i__3] = MAGMA_Z_ZERO;
	}
    }
    i__1 = *ilo;
    for (j = 1; j <= i__1; ++j) {
	i__2 = *n;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    i__3 = i__ + j * a_dim1;
	    a[i__3] = MAGMA_Z_ZERO;
	}
	i__2 = j + j * a_dim1;
	a[i__2] = MAGMA_Z_ONE;
    }
    i__1 = *n;
    for (j = *ihi + 1; j <= i__1; ++j) {
	i__2 = *n;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    i__3 = i__ + j * a_dim1;
	    a[i__3] = MAGMA_Z_ZERO; 
	}
	i__2 = j + j * a_dim1;
	a[i__2] = MAGMA_Z_ONE;
    }

    if (nh > 0) 
      {
	/* Generate Q(ilo+1:ihi,ilo+1:ihi) */
	zungqr_(&nh, &nh, &nh, &a[*ilo + 1 + (*ilo + 1) * a_dim1], lda, 
		&tau[*ilo], &work[1], lwork, &iinfo);
    }
    MAGMA_Z_SET2REAL(work[1], (double) lwkopt);
    return 0;
} /* zunghr_ */
