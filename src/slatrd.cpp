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

extern "C" void mssymv2(int m, int k, float *A, int lda, float *X, float *Y);
extern "C" void test_mssymv_v2(char, int, float, float *, int, float *,
			       int, float, float *, int, float *);
extern "C" void test_mssymv_v3(char, int, float, float *, int, float *,
                               int, float, float *, int, float *);

extern "C"
int magma_slatrd(char *uplo, int *n, int *nb, float *a, 
		 int *lda, float *e, float *tau, float *w, int *ldw,
		 float *da, int *ldda, float *dw, int *lddw)
{
/*  -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

    Purpose   
    =======   
    SLATRD reduces NB rows and columns of a real symmetric matrix A to   
    symmetric tridiagonal form by an orthogonal similarity   
    transformation Q' * A * Q, and returns the matrices V and W which are   
    needed to apply the transformation to the unreduced part of A.   

    If UPLO = 'U', SLATRD reduces the last NB rows and columns of a   
    matrix, of which the upper triangle is supplied;   
    if UPLO = 'L', SLATRD reduces the first NB rows and columns of a   
    matrix, of which the lower triangle is supplied.   

    This is an auxiliary routine called by SSYTRD.   

    Arguments   
    =========   
    UPLO    (input) CHARACTER*1   
            Specifies whether the upper or lower triangular part of the   
            symmetric matrix A is stored:   
            = 'U': Upper triangular   
            = 'L': Lower triangular   

    N       (input) INTEGER   
            The order of the matrix A.   

    NB      (input) INTEGER   
            The number of rows and columns to be reduced.   

    A       (input/output) REAL array, dimension (LDA,N)   
            On entry, the symmetric matrix A.  If UPLO = 'U', the leading   
            n-by-n upper triangular part of A contains the upper   
            triangular part of the matrix A, and the strictly lower   
            triangular part of A is not referenced.  If UPLO = 'L', the   
            leading n-by-n lower triangular part of A contains the lower   
            triangular part of the matrix A, and the strictly upper   
            triangular part of A is not referenced.   
            On exit:   
            if UPLO = 'U', the last NB columns have been reduced to   
              tridiagonal form, with the diagonal elements overwriting   
              the diagonal elements of A; the elements above the diagonal   
              with the array TAU, represent the orthogonal matrix Q as a   
              product of elementary reflectors;   
            if UPLO = 'L', the first NB columns have been reduced to   
              tridiagonal form, with the diagonal elements overwriting   
              the diagonal elements of A; the elements below the diagonal   
              with the array TAU, represent the  orthogonal matrix Q as a   
              product of elementary reflectors.   
            See Further Details.   

    LDA     (input) INTEGER   
            The leading dimension of the array A.  LDA >= (1,N).   

    E       (output) REAL array, dimension (N-1)   
            If UPLO = 'U', E(n-nb:n-1) contains the superdiagonal   
            elements of the last NB columns of the reduced matrix;   
            if UPLO = 'L', E(1:nb) contains the subdiagonal elements of   
            the first NB columns of the reduced matrix.   

    TAU     (output) REAL array, dimension (N-1)   
            The scalar factors of the elementary reflectors, stored in   
            TAU(n-nb:n-1) if UPLO = 'U', and in TAU(1:nb) if UPLO = 'L'.   
            See Further Details.   

    W       (output) REAL array, dimension (LDW,NB)   
            The n-by-nb matrix W required to update the unreduced part   
            of A.   

    LDW     (input) INTEGER   
            The leading dimension of the array W. LDW >= max(1,N).   

    Further Details   
    ===============   
    If UPLO = 'U', the matrix Q is represented as a product of elementary   
    reflectors   

       Q = H(n) H(n-1) . . . H(n-nb+1).   

    Each H(i) has the form   

       H(i) = I - tau * v * v'   

    where tau is a real scalar, and v is a real vector with   
    v(i:n) = 0 and v(i-1) = 1; v(1:i-1) is stored on exit in A(1:i-1,i),   
    and tau in TAU(i-1).   

    If UPLO = 'L', the matrix Q is represented as a product of elementary   
    reflectors   

       Q = H(1) H(2) . . . H(nb).   

    Each H(i) has the form   

       H(i) = I - tau * v * v'   

    where tau is a real scalar, and v is a real vector with   
    v(1:i) = 0 and v(i+1) = 1; v(i+1:n) is stored on exit in A(i+1:n,i),   
    and tau in TAU(i).   

    The elements of the vectors v together form the n-by-nb matrix V   
    which is needed, with W, to apply the transformation to the unreduced   
    part of the matrix, using a symmetric rank-2k update of the form:   
    A := A - V*W' - W*V'.   

    The contents of A on exit are illustrated by the following examples   
    with n = 5 and nb = 2:   

    if UPLO = 'U':                       if UPLO = 'L':   

      (  a   a   a   v4  v5 )              (  d                  )   
      (      a   a   v4  v5 )              (  1   d              )   
      (          a   1   v5 )              (  v1  1   a          )   
      (              d   1  )              (  v1  v2  a   a      )   
      (                  d  )              (  v1  v2  a   a   a  )   

    where d denotes a diagonal element of the reduced matrix, a denotes   
    an element of the original matrix that is unchanged, and vi denotes   
    an element of the vector defining H(i).   

    =====================================================================    */
 
#define min(a,b)  (((a)<(b))?(a):(b))

    //TimeStruct start, end;

    static float c_b5 = -1.f;
    static float c_b6 = 1.f;
    static int c__1 = 1;
    static float c_b16 = 0.f;
    
    /* System generated locals */
    int a_dim1, a_offset, w_dim1, w_offset, i__2, i__3;
    /* Local variables */
    static int i__, iw;
  
    static float alpha;

    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --e;
    --tau;
    w_dim1 = *ldw;
    w_offset = 1 + w_dim1;
    w -= w_offset;
    dw-= 1 + *lddw;

    float *f = (float *)malloc((*n)*sizeof(float ));
    static cudaStream_t stream[2];
    cudaStreamCreate(&stream[0]);
    cudaStreamCreate(&stream[1]);

    /* Function Body */
    if (*n <= 0) {
      return 0;
    }

    // TTT
    float *dwork;
    int bsz = 64 ;
    int blocks   = *n / bsz  + ( *n % bsz != 0 )  ;
    int workspace = 2* bsz * blocks * (  blocks +  1) /2 ;
    workspace = (*ldda) * (blocks +  1);
    cublasAlloc(workspace, sizeof(float), (void **)&dwork); 

    if (lsame_(uplo, "U")) {
      /* Reduce last NB columns of upper triangle */
      for (i__ = *n; i__ >= *n - *nb + 1; --i__) {
	iw = i__ - *n + *nb;
	if (i__ < *n) {
	  /* Update A(1:i,i) */
	  i__2 = *n - i__;
	  sgemv_("No transpose", &i__, &i__2, &c_b5, 
		 &a[(i__+1)*a_dim1 + 1], lda, &w[i__ + (iw + 1)*w_dim1], ldw, 
		 &c_b6, &a[i__ * a_dim1 + 1], &c__1);
	  i__2 = *n - i__;
	  sgemv_("No transpose", &i__, &i__2, &c_b5, 
		 &w[(iw+1)*w_dim1 + 1], ldw, &a[i__ + (i__+1) * a_dim1], lda, 
		 &c_b6, &a[i__ * a_dim1 + 1], &c__1);
	}
	if (i__ > 1) {
	  /* Generate elementary reflector H(i) to annihilate A(1:i-2,i) */
	  i__2 = i__ - 1;
	  slarfg_(&i__2, &a[i__ - 1 + i__ * a_dim1], &a[i__ * a_dim1 + 1], 
		  &c__1, &tau[i__ - 1]);
	  e[i__ - 1] = a[i__ - 1 + i__ * a_dim1];
	  a[i__ - 1 + i__ * a_dim1] = 1.f;
  
	  /* Compute W(1:i-1,i) */
	  i__2 = i__ - 1;
	  ssymv_("Upper", &i__2, &c_b6, &a[a_offset], lda, 
		 &a[i__*a_dim1 +1], &c__1, &c_b16, &w[iw* w_dim1+1], &c__1);
	  if (i__ < *n) {
	    i__2 = i__ - 1;
	    i__3 = *n - i__;
	    sgemv_("Transpose", &i__2, &i__3, &c_b6, 
		   &w[(iw+1)*w_dim1 + 1], ldw, &a[i__ * a_dim1 + 1], &c__1, 
		   &c_b16, &w[i__ + 1 + iw * w_dim1], &c__1);
	    i__2 = i__ - 1;
	    i__3 = *n - i__;
	    sgemv_("No transpose", &i__2, &i__3, &c_b5, 
		   &a[(i__+1)*a_dim1 + 1], lda, &w[i__ + 1 + iw * w_dim1], &
		   c__1, &c_b6, &w[iw * w_dim1 + 1], &c__1);
	    i__2 = i__ - 1;
	    i__3 = *n - i__;
	    sgemv_("Transpose", &i__2, &i__3, &c_b6, 
		   &a[(i__ + 1) * a_dim1 + 1], lda, &a[i__ * a_dim1 + 1], 
		   &c__1, &c_b16, &w[i__ + 1 + iw * w_dim1], &c__1);
	    i__2 = i__ - 1;
	    i__3 = *n - i__;
	    sgemv_("No transpose", &i__2, &i__3, &c_b5, 
		   &w[(iw + 1) *  w_dim1 + 1], ldw, &w[i__ + 1 + iw * w_dim1],
		   &c__1, &c_b6, &w[iw * w_dim1 + 1], &c__1);
	  }
	  i__2 = i__ - 1;
	  sscal_(&i__2, &tau[i__ - 1], &w[iw * w_dim1 + 1], &c__1);
	  i__2 = i__ - 1;
	  alpha = tau[i__ - 1] * -.5f * 
	    sdot_(&i__2, &w[iw*w_dim1+1], &c__1, &a[i__ * a_dim1 + 1], &c__1);
	  i__2 = i__ - 1;
	  saxpy_(&i__2, &alpha, &a[i__ * a_dim1 + 1], &c__1, 
		 &w[iw * w_dim1 + 1], &c__1);
	}
	
	/* L10: */
      }
    } else {

      /* TTT: only the L case  is done for now */

      /*  Reduce first NB columns of lower triangle */
      for (i__ = 1; i__ <= *nb; ++i__) {
	/* Update A(i:n,i) */
	i__2 = *n - i__ + 1;
	i__3 = i__ - 1;
	//fprintf(stderr,"i= %d i__2 = %d\n", i__, i__2); 
	sgemv_("No transpose", &i__2, &i__3, &c_b5, &a[i__ + a_dim1], lda, 
	       &w[i__ + w_dim1], ldw, &c_b6, &a[i__ + i__ * a_dim1], &c__1);
	sgemv_("No transpose", &i__2, &i__3, &c_b5, &w[i__ + w_dim1], ldw, 
	       &a[i__ + a_dim1], lda, &c_b6, &a[i__ + i__ * a_dim1], &c__1);
	if (i__ < *n) {
	  /* Generate elementary reflector H(i) to annihilate A(i+2:n,i) */
	  i__2 = *n - i__;
	  i__3 = i__ + 2;
	  slarfg_(&i__2, &a[i__ + 1 + i__ * a_dim1], 
		  &a[min(i__3,*n) + i__ * a_dim1], &c__1, &tau[i__]);
	  e[i__] = a[i__ + 1 + i__ * a_dim1];
	  a[i__ + 1 + i__ * a_dim1] = 1.f;

	  /* Compute W(i+1:n,i) */ 

	  // TTT : this is the time consuming operation
	  // 1. Send the block reflector  A(i+1:n,i) to the GPU
	  cublasSetVector(i__2, sizeof(float),
			  a + i__   + 1 + i__   * a_dim1, 1,
                          da+(i__-1)+ 1 +(i__-1)* (*ldda), 1);
	  
	  
           cublasSsymv('L', i__2, c_b6, da+ (i__-1)+1 + ((i__-1)+1) * (*ldda),
		      *ldda, da+ (i__-1)+1 + (i__-1)* a_dim1, c__1, c_b16,
		      dw+ i__ + 1 + i__ * w_dim1, c__1);
	  
	  /*
	  magmablas_ssymv6('L', *n, c_b6, da,
		           *ldda, da + (i__-1)* a_dim1, c__1, c_b16,
			              dw + 1 + i__ * w_dim1, c__1, dwork, i__-1);
	  */
	  /*
	  magmablas_ssymv6('L', i__2, c_b6, da+ (i__-1)+1 + ((i__-1)+1) * (*ldda),
			   *ldda, da+ (i__-1)+1 + (i__-1)* a_dim1, c__1, c_b16,
			   dw+ i__ + 1 + i__ * w_dim1, c__1, dwork, -1);
	  */

	  //start = get_current_time();
	  // the symmetric matrix-vector (works with the syrks) TTT
	   // This works when matrix is divisible by the blocking size
	  /*
            mssymv2(*n, i__,  da, *ldda, 
		  da + (i__-1)* a_dim1, dw + 1 +  i__ *w_dim1);
	  */

	  /*
	  if (*n<4500)
	    test_mssymv_v2('L', *n, c_b6, da, *ldda, 
			   da+ (i__-1)* a_dim1, c__1, c_b16,
			   dw+ 1 + i__ * w_dim1, c__1, dw);
	  else
	    if (*n%64==0)
	      test_mssymv_v3('L', *n, c_b6, da, *ldda,
			     da+ (i__-1)* a_dim1, c__1, c_b16,
			     dw+ 1 + i__ * w_dim1, c__1, dw);
	    else
	      test_mssymv_v3('L', *n+32, c_b6, da, *ldda,
                             da+ (i__-1)* a_dim1, c__1, c_b16,
                             dw+ 1 + i__ * w_dim1, c__1, dw);
	  */
	  /*
	  magmablas_sgemv(*n, *n,
			  da, *ldda,
			  da + (i__-1)* a_dim1, dw + 1 +  i__ *w_dim1);
	  */
	  /*
	  magmablas_ssymv(
		      'L', i__2, c_b6, da+ (i__-1)+1 + ((i__-1)+1) * (*ldda),
		      *ldda, da+ (i__-1)+1 + (i__-1)* a_dim1, c__1, c_b16,
		      dw+ i__ + 1 + i__ * w_dim1, c__1);
	  */
	  //end = get_current_time();
	  //printf("%4d, ssyrk %4d GFlop/s: %5.2f \n", *n, i__2, 
	  //	 2.*i__2*i__2/(1000000.*GetTimerValue(start,end)));

	  cudaMemcpy2DAsync(w + i__ + 1 + i__ * w_dim1, w_dim1*sizeof(float),
			    dw+ i__ + 1 + i__ * w_dim1, w_dim1*sizeof(float),
			    sizeof(float)*i__2, 1,
			    cudaMemcpyDeviceToHost,stream[1]);

	  /*
	  ssymv_("Lower", &i__2, &c_b6, &a[i__ + 1 + (i__ + 1) * a_dim1],
		 lda, &a[i__ + 1 + i__ * a_dim1], &c__1, &c_b16, 
		 &w[i__ + 1 + i__ * w_dim1], &c__1);
	  */
	 
	  i__3 = i__ - 1;
	  sgemv_("Transpose", &i__2, &i__3, &c_b6, &w[i__ + 1 + w_dim1], 
		 ldw, &a[i__ + 1 + i__ * a_dim1], &c__1, &c_b16, 
		 &w[i__ * w_dim1 + 1], &c__1);

	  // put the result back
	  /*
	  cublasGetVector(i__2, sizeof(float),
                          dw+ i__ + 1 + i__ * w_dim1, c__1,
                          w + i__ + 1 + i__ * w_dim1, c__1);
	  */

	  /*
	  sgemv_("No transpose", &i__2, &i__3, &c_b5, 
		 &a[i__ + 1 + a_dim1], lda, &w[i__ * w_dim1 + 1], &c__1, 
		 &c_b6, &w[i__ + 1 + i__ * w_dim1], &c__1);
	  sgemv_("Transpose", &i__2, &i__3, &c_b6, &a[i__ + 1 + a_dim1], 
		 lda, &a[i__ + 1 + i__ * a_dim1], &c__1, &c_b16, 
		 &w[i__ * w_dim1 + 1], &c__1);
	  */

	  sgemv_("No transpose", &i__2, &i__3, &c_b5,
                 &a[i__ + 1 + a_dim1], lda, &w[i__ * w_dim1 + 1], &c__1,
                 &c_b16, f, &c__1);
	  sgemv_("Transpose", &i__2, &i__3, &c_b6, &a[i__ + 1 + a_dim1],
                 lda, &a[i__ + 1 + i__ * a_dim1], &c__1, &c_b16,
                 &w[i__ * w_dim1 + 1], &c__1);

	  /*
	  cublasGetVector(i__2, sizeof(float),
                          dw+ i__ + 1 + i__ * w_dim1, c__1,
                          w + i__ + 1 + i__ * w_dim1, c__1);
	  */
	  cudaStreamSynchronize(stream[1]);

	  if (i__3!=0)
	  saxpy_(&i__2, &c_b6, f, &c__1, &w[i__ + 1 + i__ * w_dim1], &c__1);
	  //==========


	  sgemv_("No transpose", &i__2, &i__3, &c_b5, &w[i__ + 1 + w_dim1], 
		 ldw, &w[i__ * w_dim1 + 1], &c__1, &c_b6, 
		 &w[i__ + 1 + i__ * w_dim1], &c__1);
	  sscal_(&i__2, &tau[i__], &w[i__ + 1 + i__ * w_dim1], &c__1);
	  alpha = tau[i__]* -.5f*sdot_(&i__2, &w[i__ +1+ i__ * w_dim1], 
				       &c__1, &a[i__ +1+ i__ * a_dim1], &c__1);
	  saxpy_(&i__2, &alpha, &a[i__ + 1 + i__ * a_dim1], &c__1, 
		 &w[i__ + 1 + i__ * w_dim1], &c__1);
	}

	/* L20: */
      }
    }

    free(f);
    cublasFree(dwork);

    return 0;

    /* End of SLATRD */
} /* slatrd_ */

#undef min
