/*
    -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

       @precisions normal z -> s d c

*/

#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include "magma.h"
#include "magmablas.h"

extern "C" int ssytd2_(char *, int *, double2 *, int *,
		       double2 *, double2 *, double2 *, int *);
extern "C" int ssyr2k_(char *, char *, int *, int *, double2 *, double2 *, 
		       int *, double2 *, int *, double2 *, double2 *, int *);
extern "C" int slatrd_(char *, int *, int *, double2 *,
		       int *, double2 *, double2 *, double2 *, int *);

extern "C" void magmablas_zsyr2k(char, char, int, int, double2, const double2 *, int,
				const double2 *, int, double2, double2 *, int);



extern "C" int  magma_zlatrd(char *, int *, int *, double2 *,
			     int *, double2 *, double2 *, double2 *, int *,
			     double2 *, int *, double2 *, int *);


double2 cpu_gpu_sdiff(int M, int N, double2 * a, int lda, double2 *da, int ldda)
{
  int one = 1, j;
  double2 mone = -1.f, work[1];
  double2 *ha = (double2*)malloc( M * N * sizeof(double2));

  cublasGetMatrix(M, N, sizeof(double2), da, ldda, ha, M);
  for(j=0; j<N; j++)
    zaxpy_(&M, &mone, a+j*lda, &one, ha+j*M, &one);
  double2 res = zlange_("f", &M, &N, ha, &M, work);

  free(ha);
  return res;
}


extern "C" magma_int_t
magma_zsytrd(char uplo_, magma_int_t n_, double2 *a, magma_int_t lda_, double2 *d__, double2 *e, 
	     double2 *tau, double2 *work, magma_int_t *lwork, double2 *da, magma_int_t *info)
{
/*  -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

    Purpose   
    =======   
    SSYTRD reduces a real hemmetric matrix A to real hemmetric   
    tridiagonal form T by an orthogonal similarity transformation:   
    Q\*\*H * A * Q = T.   

    Arguments   
    =========   
    UPLO    (input) CHARACTER*1   
            = 'U':  Upper triangle of A is stored;   
            = 'L':  Lower triangle of A is stored.   

    N       (input) INTEGER   
            The order of the matrix A.  N >= 0.   

    A       (input/output) COMPLEX_16 array, dimension (LDA,N)   
            On entry, the hemmetric matrix A.  If UPLO = 'U', the leading   
            N-by-N upper triangular part of A contains the upper   
            triangular part of the matrix A, and the strictly lower   
            triangular part of A is not referenced.  If UPLO = 'L', the   
            leading N-by-N lower triangular part of A contains the lower   
            triangular part of the matrix A, and the strictly upper   
            triangular part of A is not referenced.   
            On exit, if UPLO = 'U', the diagonal and first superdiagonal   
            of A are overwritten by the corresponding elements of the   
            tridiagonal matrix T, and the elements above the first   
            superdiagonal, with the array TAU, represent the orthogonal   
            matrix Q as a product of elementary reflectors; if UPLO   
            = 'L', the diagonal and first subdiagonal of A are over-   
            written by the corresponding elements of the tridiagonal   
            matrix T, and the elements below the first subdiagonal, with   
            the array TAU, represent the orthogonal matrix Q as a product   
            of elementary reflectors. See Further Details.   

    LDA     (input) INTEGER   
            The leading dimension of the array A.  LDA >= max(1,N).   

    D       (output) COMPLEX_16 array, dimension (N)   
            The diagonal elements of the tridiagonal matrix T:   
            D(i) = A(i,i).   

    E       (output) COMPLEX_16 array, dimension (N-1)   
            The off-diagonal elements of the tridiagonal matrix T:   
            E(i) = A(i,i+1) if UPLO = 'U', E(i) = A(i+1,i) if UPLO = 'L'.   

    TAU     (output) COMPLEX_16 array, dimension (N-1)   
            The scalar factors of the elementary reflectors (see Further   
            Details).   

    WORK    (workspace/output) COMPLEX_16 array, dimension (MAX(1,LWORK))   
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.   

    LWORK   (input) INTEGER   
            The dimension of the array WORK.  LWORK >= 1.   
            For optimum performance LWORK >= N*NB, where NB is the   
            optimal blocksize.   

            If LWORK = -1, then a workspace query is assumed; the routine   
            only calculates the optimal size of the WORK array, returns   
            this value as the first entry of the WORK array, and no error   
            message related to LWORK is issued by XERBLA.   

    DA      (workspace)  SINGLE array on the GPU, dimension
            N*N + 2*N*NB + NB*NB,
            where NB can be obtained through magma_get_zsytrd_nb(N).

    INFO    (output) INTEGER   
            = 0:  successful exit   
            < 0:  if INFO = -i, the i-th argument had an illegal value   

    Further Details   
    ===============   

    If UPLO = 'U', the matrix Q is represented as a product of elementary   
    reflectors   

       Q = H(n-1) . . . H(2) H(1).   

    Each H(i) has the form   

       H(i) = I - tau * v * v'

    where tau is a real scalar, and v is a real vector with   
    v(i+1:n) = 0 and v(i) = 1; v(1:i-1) is stored on exit in   
    A(1:i-1,i+1), and tau in TAU(i).   

    If UPLO = 'L', the matrix Q is represented as a product of elementary   
    reflectors   

       Q = H(1) H(2) . . . H(n-1).   

    Each H(i) has the form   

       H(i) = I - tau * v * v'   

    where tau is a real scalar, and v is a real vector with   
    v(1:i) = 0 and v(i+1) = 1; v(i+2:n) is stored on exit in A(i+2:n,i),   
    and tau in TAU(i).

    The contents of A on exit are illustrated by the following examples   
    with n = 5:   

    if UPLO = 'U':                       if UPLO = 'L':   

      (  d   e   v2  v3  v4 )              (  d                  )   
      (      d   e   v3  v4 )              (  e   d              )   
      (          d   e   v4 )              (  v1  e   d          )   
      (              d   e  )              (  v1  v2  e   d      )   
      (                  d  )              (  v1  v2  v3  e   d  )   

    where d and e denote diagonal and off-diagonal elements of T, and vi   
    denotes an element of the vector defining H(i).   

    =====================================================================    */

    #define max(a,b) ((a) >= (b) ? (a) : (b))
  
    char uplo[2] = {uplo_, 0};
    int *n = &n_;
    int *lda = &lda_;

    int N = *n, ldda = *lda;
    int nb = magma_get_zsytrd_nb(*n); 
    double2 *dwork = da + (*n)*ldda - 1;

    static double2 c_b22 = -1.f;
    static double2 c_b23 = 1.f;
    
    /* System generated locals */
    int a_dim1, a_offset, i__1, i__3;
    /* Local variables */
    static int i__, j, kk, nx;
    static int iinfo;
    static int ldwork, lddwork, lwkopt;
    static long int lquery;

    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --d__;
    --e;
    --tau;
    --work;

    /* Function Body */
    *info = 0;
    long int upper = lsame_(uplo, "U");
    lquery = *lwork == -1;
    if (! upper && ! lsame_(uplo, "L")) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < max(1,*n)) {
	*info = -4;
    } else if (*lwork < 1 && ! lquery) {
	*info = -9;
    }

    if (*info == 0) {
      /* Determine the block size. */
      ldwork = lddwork = *n;
      lwkopt = *n * nb;
      work[1] = (double2) lwkopt;
    }

    if (*info != 0)
      return 0;
    else if (lquery)
      return 0;

    /* Quick return if possible */
    if (*n == 0) {
	work[1] = 1.f;
	return 0;
    }

    nx = 128;

    if (upper) {

        /* Copy the matrix to the GPU */ 
        cublasSetMatrix(N, N, sizeof(double2), a+a_offset, *lda, da, ldda);

        /*  Reduce the upper triangle of A.   
	    Columns 1:kk are handled by the unblocked method. */
        kk = *n - (*n - nx + nb - 1) / nb * nb;
	i__1 = kk + 1;
	for (i__ = *n - nb + 1; i__ >= i__1; i__ -= nb) 
	  {
	    /* Reduce columns i:i+nb-1 to tridiagonal form and form the   
	       matrix W which is needed to update the unreduced part of   
	       the matrix */
	    i__3 = i__ + nb - 1;
	    magma_zlatrd(uplo, &i__3, &nb, &a[a_offset], lda, &e[1], &tau[1], 
			 &work[1], &ldwork, da, &ldda, dwork+1, &lddwork);

	    /* Update the unreduced submatrix A(1:i-1,1:i-1), using an   
	       update of the form:  A := A - V*W' - W*V' */
	    i__3 = i__ - 1;
	    ssyr2k_(uplo, "No transpose", &i__3, &nb, &c_b22, 
		    &a[i__ * a_dim1 + 1], lda, &work[1], 
		    &ldwork, &c_b23, &a[a_offset], lda);

	    /* Copy superdiagonal elements back into A, and diagonal   
	       elements into D */
	    i__3 = i__ + nb - 1;
	    for (j = i__; j <= i__3; ++j) {
		a[j - 1 + j * a_dim1] = e[j - 1];
		d__[j] = a[j + j * a_dim1];
	    }
	  }

	/*  Use unblocked code to reduce the last or only block */
	ssytd2_(uplo, &kk, &a[a_offset], lda, &d__[1], &e[1], &tau[1], &iinfo);
    } 
    else 
      {
	/* Copy the matrix to the GPU */
	if (1<=*n-nx)
	  cublasSetMatrix(N, N, sizeof(double2), a+a_offset, *lda, da, ldda);

	/* Reduce the lower triangle of A */
	for (i__ = 1; i__ <= *n-nx; i__ += nb) 
	  {
	    /* Reduce columns i:i+nb-1 to tridiagonal form and form the
               matrix W which is needed to update the unreduced part of
               the matrix */
            i__3 = *n - i__ + 1;

	    /*   Get the current panel (no need for the 1st iteration) */
	    if (i__!=1)
	      cublasGetMatrix(i__3, nb, sizeof(double2),
			      da + (i__-1)*ldda  + (i__-1), ldda,
			      a  +  i__   *a_dim1+  i__   , *lda);
	    
	    magma_zlatrd(uplo, &i__3, &nb, &a[i__+i__ * a_dim1], lda, &e[i__], 
			 &tau[i__], &work[1], &ldwork, 
			 da + (i__-1)+(i__-1) * a_dim1, &ldda,
			 dwork+1, &lddwork);
	    
	    /* Update the unreduced submatrix A(i+ib:n,i+ib:n), using   
	       an update of the form:  A := A - V*W' - W*V' */
	    i__3 = *n - i__ - nb + 1;
	    cublasSetMatrix(*n - i__ + 1, nb, sizeof(double2),
                            work  + 1, ldwork,
                            dwork + 1, lddwork);

	    cublasSsyr2k('L', 'N', i__3, nb, c_b22, 
			 &da[(i__-1) + nb + (i__-1) * a_dim1], ldda, 
			 &dwork[nb + 1], lddwork, c_b23, 
			 &da[(i__-1) + nb + ((i__-1) + nb) * a_dim1], ldda);
	    
	    /* Copy subdiagonal elements back into A, and diagonal   
	       elements into D */
	    i__3 = i__ + nb - 1;
	    for (j = i__; j <= i__3; ++j) {
	      a[j + 1 + j * a_dim1] = e[j];
	      d__[j] = a[j + j * a_dim1];
	    }
	  }

	/* Use unblocked code to reduce the last or only block */
	i__1 = *n - i__ + 1;

	if (1<=*n-nx)
	  cublasGetMatrix(i__1, i__1, sizeof(double2),
			  da + (i__-1) + (i__-1) * a_dim1, ldda,
			  a  +  i__    +  i__    * a_dim1, *lda);
	
	zsytrd_(uplo, &i__1, &a[i__ + i__ * a_dim1], lda, &d__[i__], &e[i__],
                &tau[i__], &work[1], lwork, &iinfo);
	
      }

    work[1] = (double2) lwkopt;
    return 0;
    
    /* End of SSYTRD */
    
} /* zsytrd_ */
