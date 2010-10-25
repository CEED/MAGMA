/*
    -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include "magma.h"
#include "magmablas.h"

extern "C" int sgebd2_(int *, int *, float *, int *, float *, float *, float *,
		       float *, float *, int *);
float cpu_gpu_sdiff(int M, int N, float * a, int lda, float *da, int ldda);

extern "C" magma_int_t
magma_sgebrd(magma_int_t m_, magma_int_t n_, float *a, magma_int_t lda_, 
	     float *d__, float *e, float *tauq, float *taup, float *work, 
	     magma_int_t *lwork, float *da, magma_int_t *info)
{
/*  -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

    Purpose   
    =======   
    SGEBRD reduces a general real M-by-N matrix A to upper or lower   
    bidiagonal form B by an orthogonal transformation: Q**T * A * P = B.   

    If m >= n, B is upper bidiagonal; if m < n, B is lower bidiagonal.   

    Arguments   
    =========   
    M       (input) INTEGER   
            The number of rows in the matrix A.  M >= 0.   

    N       (input) INTEGER   
            The number of columns in the matrix A.  N >= 0.   

    A       (input/output) REAL array, dimension (LDA,N)   
            On entry, the M-by-N general matrix to be reduced.   
            On exit,   
            if m >= n, the diagonal and the first superdiagonal are   
              overwritten with the upper bidiagonal matrix B; the   
              elements below the diagonal, with the array TAUQ, represent   
              the orthogonal matrix Q as a product of elementary   
              reflectors, and the elements above the first superdiagonal,   
              with the array TAUP, represent the orthogonal matrix P as   
              a product of elementary reflectors;   
            if m < n, the diagonal and the first subdiagonal are   
              overwritten with the lower bidiagonal matrix B; the   
              elements below the first subdiagonal, with the array TAUQ,   
              represent the orthogonal matrix Q as a product of   
              elementary reflectors, and the elements above the diagonal,   
              with the array TAUP, represent the orthogonal matrix P as   
              a product of elementary reflectors.   
            See Further Details.   

    LDA     (input) INTEGER   
            The leading dimension of the array A.  LDA >= max(1,M).   

    D       (output) REAL array, dimension (min(M,N))   
            The diagonal elements of the bidiagonal matrix B:   
            D(i) = A(i,i).   

    E       (output) REAL array, dimension (min(M,N)-1)   
            The off-diagonal elements of the bidiagonal matrix B:   
            if m >= n, E(i) = A(i,i+1) for i = 1,2,...,n-1;   
            if m < n, E(i) = A(i+1,i) for i = 1,2,...,m-1.   

    TAUQ    (output) REAL array dimension (min(M,N))   
            The scalar factors of the elementary reflectors which   
            represent the orthogonal matrix Q. See Further Details.   

    TAUP    (output) REAL array, dimension (min(M,N))   
            The scalar factors of the elementary reflectors which   
            represent the orthogonal matrix P. See Further Details.   

    WORK    (workspace/output) REAL array, dimension (MAX(1,LWORK))   
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.   

    LWORK   (input) INTEGER   
            The length of the array WORK.  LWORK >= max(1,M,N).   
            For optimum performance LWORK >= (M+N)*NB, where NB   
            is the optimal blocksize.   

            If LWORK = -1, then a workspace query is assumed; the routine   
            only calculates the optimal size of the WORK array, returns   
            this value as the first entry of the WORK array, and no error   
            message related to LWORK is issued by XERBLA.   

    INFO    (output) INTEGER   
            = 0:  successful exit   
            < 0:  if INFO = -i, the i-th argument had an illegal value.   

    Further Details   
    ===============   
    The matrices Q and P are represented as products of elementary   
    reflectors:   

    If m >= n,   
       Q = H(1) H(2) . . . H(n)  and  P = G(1) G(2) . . . G(n-1)   
    Each H(i) and G(i) has the form:   
       H(i) = I - tauq * v * v'  and G(i) = I - taup * u * u'   
    where tauq and taup are real scalars, and v and u are real vectors;   
    v(1:i-1) = 0, v(i) = 1, and v(i+1:m) is stored on exit in A(i+1:m,i);   
    u(1:i) = 0, u(i+1) = 1, and u(i+2:n) is stored on exit in A(i,i+2:n);   
    tauq is stored in TAUQ(i) and taup in TAUP(i).   

    If m < n,   
       Q = H(1) H(2) . . . H(m-1)  and  P = G(1) G(2) . . . G(m)   
    Each H(i) and G(i) has the form:   
       H(i) = I - tauq * v * v'  and G(i) = I - taup * u * u'   
    where tauq and taup are real scalars, and v and u are real vectors;   
    v(1:i) = 0, v(i+1) = 1, and v(i+2:m) is stored on exit in A(i+2:m,i);   
    u(1:i-1) = 0, u(i) = 1, and u(i+1:n) is stored on exit in A(i,i+1:n);   
    tauq is stored in TAUQ(i) and taup in TAUP(i).   

    The contents of A on exit are illustrated by the following examples:   

    m = 6 and n = 5 (m > n):          m = 5 and n = 6 (m < n):   

      (  d   e   u1  u1  u1 )           (  d   u1  u1  u1  u1  u1 )   
      (  v1  d   e   u2  u2 )           (  e   d   u2  u2  u2  u2 )   
      (  v1  v2  d   e   u3 )           (  v1  e   d   u3  u3  u3 )   
      (  v1  v2  v3  d   e  )           (  v1  v2  e   d   u4  u4 )   
      (  v1  v2  v3  v4  d  )           (  v1  v2  v3  e   d   u5 )   
      (  v1  v2  v3  v4  v5 )   

    where d and e denote diagonal and off-diagonal elements of B, vi   
    denotes an element of the vector defining H(i), and ui an element of   
    the vector defining G(i).   

    =====================================================================    */

    #define max(a,b) ((a) >= (b) ? (a) : (b)) 
    #define min(a,b)  (((a)<(b))?(a):(b))

    int *m = &m_;
    int *n = &n_;
    int *lda = &lda_;

    static float c_b21 = -1.f;
    static float c_b22 = 1.f;
    
    /* System generated locals */
    int a_dim1, a_offset, i__1, i__2, i__3, i__4;
    /* Local variables */
    static int i__, j, nx;
    static float ws;
    static int iinfo;
    
    static int minmn;
    static int ldwrkx, ldwrky, lwkopt;
    static long int lquery;

    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --d__;
    --e;
    --tauq;
    --taup;
    --work;

    /* Function Body */
    *info = 0;

    //TimeStruct start, end;

    int nb = magma_get_sgebrd_nb(*n), ldda = *m; 
    float *dwork = da + (*n)*ldda - 1;

    lwkopt = (*m + *n) * nb;
    work[1] = (float) lwkopt;
    lquery = *lwork == -1;
    if (*m < 0) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < max(1,*m)) {
	*info = -4;
    } else /* if(complicated condition) */ {
      /* Computing MAX */
      i__1 = max(1,*m);
      if (*lwork < max(i__1,*n) && ! lquery) {
	*info = -10;
      }
    }
    if (*info < 0)
	return 0;
    else if (lquery)
	return 0;

    /* Quick return if possible */
    minmn = min(*m,*n);
    if (minmn == 0) {
      work[1] = 1.f;
      return 0;
    }

    ws = (float)max(*m,*n);
    ldwrkx = *m;
    ldwrky = *n;

    // float nflops = 0.f;
    
    /* Set the block/unblock crossover point NX. */
    nx = 128;

    /* Copy the matrix to the GPU */
    if (minmn-nx>=1)
      cublasSetMatrix(*m, *n, sizeof(float), a+a_offset, *lda, da, ldda);

    for (i__ = 1; i__ <= minmn - nx; i__ += nb) {

      /*  Reduce rows and columns i:i+nb-1 to bidiagonal form and return   
          the matrices X and Y which are needed to update the unreduced   
          part of the matrix */
      i__3 = *m - i__ + 1;
      i__4 = *n - i__ + 1;

      /*   Get the current panel (no need for the 1st iteration) */
      // TTT
      if (i__!=1) {
	cublasGetMatrix(i__3, nb, sizeof(float),
			da + (i__-1)*ldda  + (i__-1), ldda,
			a  +  i__   *a_dim1+  i__   , *lda);
	cublasGetMatrix(nb, i__4 - nb, sizeof(float),
                        da + (i__-1+nb)*ldda  + (i__-1), ldda,
                        a  + (i__  +nb)*a_dim1+  i__   , *lda);
      }
      
      // if (i__== 1+nb)
      /*
      printf("Difference(%4d, %4d) L/U = %e, %e\n", i__3, nb,
	     cpu_gpu_sdiff(i__3, nb,
			   a  +  i__   *a_dim1+  i__   , *lda,
			   da + (i__-1)*ldda  + (i__-1), ldda),
	     cpu_gpu_sdiff(nb, i__4 - nb,
			   a  + (i__  +nb)*a_dim1+  i__   , *lda,
			   da + (i__-1+nb)*ldda  + (i__-1), ldda));
      */
      magma_slabrd(i__3, i__4, nb, 
		   &a[i__ + i__ * a_dim1], *lda, &d__[i__],
		   &e[i__], &tauq[i__], &taup[i__], 
		   &work[1], ldwrkx,                   //  x
		   &work[ldwrkx * nb + 1], ldwrky,     //  y
		   da + (i__-1)+(i__-1) * ldda, ldda,
		   &dwork[1], ldwrkx,                  // dx
		   &dwork[ldwrkx * nb+1], ldwrky);     // dy

      /*  Update the trailing submatrix A(i+nb:m,i+nb:n), using an update   
          of the form  A := A - V*Y' - X*U' */
      i__3 = *m - i__ - nb + 1;
      i__4 = *n - i__ - nb + 1;
      /* TTT
      sgemm_("No transpose", "Transpose", &i__3, &i__4, &nb, &c_b21, 
	     &a[i__ + nb + i__ * a_dim1], lda, &work[ldwrkx * nb + nb + 1],
	     &ldwrky, &c_b22, &a[i__ + nb + (i__ + nb) * a_dim1], lda);
      */
      
      // Send Y back to the GPU
      cublasSetMatrix(max(i__3,i__4), 2*nb, sizeof(float),
                      work  + nb + 1 , ldwrky,
                      dwork + nb + 1 , ldwrky);

      //start = get_current_time();
      cublasSgemm('N', 'T', i__3, i__4, nb, c_b21,
		  &da[(i__-1) + nb + (i__-1) * a_dim1], ldda, 
		  &dwork[ldwrkx * nb + nb + 1], ldwrky, c_b22, 
		  &da[(i__-1) + nb + ((i__-1) + nb) * a_dim1], ldda);
      //end = get_current_time();
      //printf("N T %5d %5d GFlop/s: %f \n",  i__3, i__4,
      //       (2. * i__3 * i__4 * nb)/(1000000.*GetTimerValue(start,end)));
      // nflops += 2. * i__3 * i__4 * nb;

      /*
      printf("  Difference after 1st gemm = %e\n",
             cpu_gpu_sdiff(i__3, i__4,
			   &a[i__ + nb + (i__ + nb) * a_dim1], *lda,
			   &da[(i__-1) + nb + ((i__-1) + nb) * a_dim1],ldda));
      */
      i__3 = *m - i__ - nb + 1;
      i__4 = *n - i__ - nb + 1;
      /* TTT
      sgemm_("No transpose", "No transpose", &i__3, &i__4, &nb, &c_b21,
	     &work[nb + 1], &ldwrkx, &a[i__ + (i__ + nb) * a_dim1], lda,
	     &c_b22, &a[i__ + nb + (i__ + nb) * a_dim1], lda);
      */

      //start = get_current_time();
      cublasSgemm('N', 'N', i__3, i__4, nb, c_b21,
		  &dwork[nb + 1], ldwrkx, 
		  &da[(i__-1) + ((i__-1) + nb) * a_dim1], ldda,
		  c_b22, &da[(i__-1) + nb + ((i__-1) + nb) * a_dim1], ldda);
      //end = get_current_time();
      //printf("N N %5d %5d GFlop/s: %f \n",  i__3, i__4,
      //       (2. * i__3 * i__4 * nb)/(1000000.*GetTimerValue(start,end)));
      // nflops += 2. * i__3 * i__4 * nb;

      /*
      printf("  Difference after 2nd gemm = %e\n",
             cpu_gpu_sdiff(i__3, i__4,
                           &a[i__ + nb + (i__ + nb) * a_dim1], *lda,
                           &da[(i__-1) + nb + ((i__-1) + nb) * a_dim1],ldda));
      */
      /* Copy diagonal and off-diagonal elements of B back into A */
      if (*m >= *n) {
	i__3 = i__ + nb - 1;
	for (j = i__; j <= i__3; ++j) {
	  a[j + j * a_dim1] = d__[j];
	  a[j + (j + 1) * a_dim1] = e[j];
	  /* L10: */
	}
      } else {
	i__3 = i__ + nb - 1;
	for (j = i__; j <= i__3; ++j) {
	  a[j + j * a_dim1] = d__[j];
	  a[j + 1 + j * a_dim1] = e[j];
	  /* L20: */
	}
      }

      /* L30: */
    }
    
    /* Use unblocked code to reduce the remainder of the matrix */
    i__2 = *m - i__ + 1;
    i__1 = *n - i__ + 1;
    // TTT
    if (1<=*n-nx)
      cublasGetMatrix(i__2, i__1, sizeof(float),
		      da + (i__-1) + (i__-1) * a_dim1, ldda,
		      a  +  i__    +  i__    * a_dim1, *lda);
     
    sgebd2_(&i__2, &i__1, &a[i__ + i__ * a_dim1], lda, &d__[i__], &e[i__],
	    &tauq[i__], &taup[i__], &work[1], &iinfo);
    work[1] = ws;

    //printf("sgemm \% = %f\n", 100.*3.*nflops/(8.*(*n)*(*n)*(*n)));

    return 0;

    /* End of SGEBRD */
} /* sgebrd_ */

#undef max
#undef min

