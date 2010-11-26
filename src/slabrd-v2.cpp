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

#define PRECISION_s

#define A(i, j)  (a +(j)*lda  + (i))
#define X(i, j)  (x +(j)*ldx  + (i))
#define Y(i, j)  (y +(j)*ldy  + (i))
#define dA(i, j) (da+(j)*ldda + (i))
#define dX(i, j) (dx+(j)*lddx + (i))
#define dY(i, j) (dy+(j)*lddy + (i))

extern "C" magma_int_t
magma_slabrd2(magma_int_t m, magma_int_t n, magma_int_t nb,
             float *a, magma_int_t lda,
             float *d, float *e,
	     float *tauq, float *taup,
             float *x,  magma_int_t ldx,
             float *y,  magma_int_t ldy,
	     float *da, magma_int_t ldda,
	     float *dx, magma_int_t lddx,
             float *dy, magma_int_t lddy)
{
/*  -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

    Purpose
    =======
    SLABRD reduces the first NB rows and columns of a real general
    m by n matrix A to upper or lower bidiagonal form by an orthogonal
    transformation Q' * A * P, and returns the matrices X and Y which
    are needed to apply the transformation to the unreduced part of A.

    If m >= n, A is reduced to upper bidiagonal form; if m < n, to lower
    bidiagonal form.

    This is an auxiliary routine called by SGEBRD

    Arguments
    =========
    M       (input) INTEGER
            The number of rows in the matrix A.

    N       (input) INTEGER
            The number of columns in the matrix A.

    NB      (input) INTEGER
            The number of leading rows and columns of A to be reduced.

    A       (input/output) REAL array, dimension (LDA,N)
            On entry, the m by n general matrix to be reduced.
            On exit, the first NB rows and columns of the matrix are
            overwritten; the rest of the array is unchanged.
            If m >= n, elements on and below the diagonal in the first NB
              columns, with the array TAUQ, represent the orthogonal
              matrix Q as a product of elementary reflectors; and
              elements above the diagonal in the first NB rows, with the
              array TAUP, represent the orthogonal matrix P as a product
              of elementary reflectors.
            If m < n, elements below the diagonal in the first NB
              columns, with the array TAUQ, represent the orthogonal
              matrix Q as a product of elementary reflectors, and
              elements on and above the diagonal in the first NB rows,
              with the array TAUP, represent the orthogonal matrix P as
              a product of elementary reflectors.
            See Further Details.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    D       (output) REAL array, dimension (NB)
            The diagonal elements of the first NB rows and columns of
            the reduced matrix.  D(i) = A(i,i).

    E       (output) REAL array, dimension (NB)
            The off-diagonal elements of the first NB rows and columns of
            the reduced matrix.

    TAUQ    (output) REAL array dimension (NB)
            The scalar factors of the elementary reflectors which
            represent the orthogonal matrix Q. See Further Details.

    TAUP    (output) REAL array, dimension (NB)
            The scalar factors of the elementary reflectors which
            represent the orthogonal matrix P. See Further Details.

    X       (output) REAL array, dimension (LDX,NB)
            The m-by-nb matrix X required to update the unreduced part
            of A.

    LDX     (input) INTEGER
            The leading dimension of the array X. LDX >= M.

    Y       (output) REAL array, dimension (LDY,NB)
            The n-by-nb matrix Y required to update the unreduced part
            of A.

    LDY     (input) INTEGER
            The leading dimension of the array Y. LDY >= N.

    Further Details
    ===============

    The matrices Q and P are represented as products of elementary
    reflectors:

       Q = H(1) H(2) . . . H(nb)  and  P = G(1) G(2) . . . G(nb)

    Each H(i) and G(i) has the form:

       H(i) = I - tauq * v * v'  and G(i) = I - taup * u * u'

    where tauq and taup are real scalars, and v and u are real vectors.

    If m >= n, v(1:i-1) = 0, v(i) = 1, and v(i:m) is stored on exit in
    A(i:m,i); u(1:i) = 0, u(i+1) = 1, and u(i+1:n) is stored on exit in
    A(i,i+1:n); tauq is stored in TAUQ(i) and taup in TAUP(i).

    If m < n, v(1:i) = 0, v(i+1) = 1, and v(i+1:m) is stored on exit in
    A(i+2:m,i); u(1:i-1) = 0, u(i) = 1, and u(i:n) is stored on exit in
    A(i,i+1:n); tauq is stored in TAUQ(i) and taup in TAUP(i).

    The elements of the vectors v and u together form the m-by-nb matrix
    V and the nb-by-n matrix U' which are needed, with X and Y, to apply
    the transformation to the unreduced part of the matrix, using a block
    update of the form:  A := A - V*Y' - X*U'.

    The contents of A on exit are illustrated by the following examples
    with nb = 2:

    m = 6 and n = 5 (m > n):          m = 5 and n = 6 (m < n):

      (  1   1   u1  u1  u1 )           (  1   u1  u1  u1  u1  u1 )
      (  v1  1   1   u2  u2 )           (  1   1   u2  u2  u2  u2 )
      (  v1  v2  a   a   a  )           (  v1  1   a   a   a   a  )
      (  v1  v2  a   a   a  )           (  v1  v2  a   a   a   a  )
      (  v1  v2  a   a   a  )           (  v1  v2  a   a   a   a  )
      (  v1  v2  a   a   a  )

    where a denotes an element of the original matrix which is unchanged,
    vi denotes an element of the vector defining H(i), and ui an element
    of the vector defining G(i).

    =====================================================================    */

    #define max(a,b) ((a) >= (b) ? (a) : (b))
    #define min(a,b) (((a)<(b))?(a):(b))

    /* Table of constant values */
    static float c_b4 = -1.f;
    static float c_b5 = 1.f;
    static int c__1 = 1;
    static float c_b16 = 0.f;
    
    /* System generated locals */
    int a_dim1, a_offset, x_dim1, x_offset, y_dim1, y_offset, i__1, i__2, 
	    i__3;
    /* Local variables */
    static int i__;

    a_dim1 = lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --d;
    --e;
    --tauq;
    --taup;

    x_dim1 = ldx;
    x_offset = 1 + x_dim1;
    x -= x_offset;
    //dy-=1 + lddx;
    dx-=1 + lddx;

    y_dim1 = ldy;
    y_offset = 1 + y_dim1;
    y -= y_offset;
    dy-= 1 + lddy;

    /* Function Body */
    if (m <= 0 || n <= 0) {
	return 0;
    }

    float *f = (float *)malloc(max(n,m)*sizeof(float ));
    static cudaStream_t stream[2];
    cudaStreamCreate(&stream[0]);
    cudaStreamCreate(&stream[1]);

    if (m >= n) {

        /* Reduce to upper bidiagonal form */

	i__1 = nb;
	for (i__ = 1; i__ <= i__1; ++i__) {

	    /*  Update A(i:m,i) */
	    i__2 = m - i__ + 1;
	    i__3 = i__ - 1;
	    sgemv_("No transpose", &i__2, &i__3, &c_b4, A(i__ , 1), &lda, 
		   Y(i__ , 1), &ldy, &c_b5, A(i__ , i__), &c__1);
	    sgemv_("No transpose", &i__2, &i__3, &c_b4, X(i__ , 1), &ldx, 
		   A(1, i__), &c__1, &c_b5, A(i__,i__), &c__1);
	    
	    /* Generate reflection Q(i) to annihilate A(i+1:m,i) */
	    i__2 = m - i__ + 1;
	    i__3 = i__ + 1;
	    slarfg_(&i__2, A(i__ , i__), 
		    A(min(i__3,m), i__), &c__1, &tauq[i__]);
	    d[i__] = *A(i__ , i__);
	    if (i__ < n) {
	        *A(i__ , i__) = 1.f;

		/* Compute Y(i+1:n,i) */
		i__2 = m - i__ + 1;
		i__3 = n - i__;

		// 1. Send the block reflector  A(i+1:m,i) to the GPU ------
		cublasSetVector(i__2, sizeof(float),
				A( i__ , i__ ), 1,
				dA(i__-1 , i__-1), 1);
		// 2. Multiply ---------------------------------------------
		cublasSgemv('T', i__2, i__3, c_b5, 
			    dA(i__-1 , i__-1 + 1), ldda, 
			    dA(i__-1 , i__-1), c__1, c_b16, 
			    dY(i__+1 , i__  ), c__1);
		
		// 3. Put the result back ----------------------------------
		cudaMemcpy2DAsync( Y( i__+1,i__), y_dim1*sizeof(float),
				   dY(i__+1,i__), y_dim1*sizeof(float),
				  sizeof(float)*i__3, 1,
				  cudaMemcpyDeviceToHost,stream[1]);

		i__2 = m - i__ + 1;
		i__3 = i__ - 1;
		sgemv_("Transpose", &i__2, &i__3, &c_b5, A(i__, 1), 
		       &lda, A(i__ , i__ ), &c__1, &c_b16, 
		       Y(1, i__ ), &c__1);

		i__2 = n - i__;
                i__3 = i__ - 1;
                sgemv_("N", &i__2, &i__3, &c_b4, Y(i__ + 1, 1), &ldy, 
		       Y(1, i__ ), &c__1, 
		       &c_b16, f, &c__1);
                i__2 = m - i__ + 1;
                i__3 = i__ - 1;
                sgemv_("Transpose", &i__2, &i__3, &c_b5, X(i__ , 1),
		       &ldx, A(i__ , i__), &c__1, &c_b16, 
		       Y(1, i__), &c__1);
		
		// 4. Synch to make sure the result is back ----------------
		cudaStreamSynchronize(stream[1]);

		if (i__3!=0){
		  i__2 = n - i__;
		  saxpy_(&i__2, &c_b5, f,&c__1, Y(i__+1,i__),&c__1);
		}

		i__2 = i__ - 1;
		i__3 = n - i__;
		sgemv_("Transpose", &i__2, &i__3, &c_b4, A(1,i__ + 1), 
		       &lda, Y(1,i__), &c__1, &c_b5, 
		       Y(i__ + 1 , i__ ), &c__1);
		i__2 = n - i__;
		sscal_(&i__2, &tauq[i__], Y(i__ + 1 , i__), &c__1);

		/* Update A(i,i+1:n) */
		i__2 = n - i__;
		sgemv_("No transpose", &i__2, &i__, &c_b4, Y(i__ + 1 , 1), 
		       &ldy, A(i__ , 1), &lda, &c_b5, A(i__ , i__ + 1), &lda);
		i__2 = i__ - 1;
		i__3 = n - i__;
		sgemv_("Transpose", &i__2, &i__3, &c_b4, A(1,i__ + 1), 
		       &lda, X(i__ , 1), &ldx, &c_b5, A(i__ , i__ + 1), &lda);

		/* Generate reflection P(i) to annihilate A(i,i+2:n) */
		i__2 = n - i__;
		/* Computing MIN */
		i__3 = i__ + 2;
		slarfg_(&i__2, A(i__ , i__ + 1), A(i__ , min(i__3,n)), 
			&lda, &taup[i__]);
		e[i__] = *A(i__ , i__ + 1);
		*A(i__ , i__ + 1) = 1.f;

		/* Compute X(i+1:m,i) */
		i__2 = m - i__;
		i__3 = n - i__;
                // 1. Send the block reflector  A(i+1:m,i) to the GPU ------
                cublasSetVector(i__3, sizeof(float),
                                A(i__ , i__  +1), lda,
                                dA(i__-1, (i__-1)+1), ldda);
                // 2. Multiply ---------------------------------------------
                cublasSgemv('N', i__2, i__3, c_b5,
                            dA(i__-1+1 ,i__-1+1), ldda,
                            dA(i__-1 , i__-1 + 1), ldda, 
			    c_b16, dX(i__ + 1 , i__), c__1);

		// 3. Put the result back ----------------------------------
		cudaMemcpy2DAsync( X(i__+1,i__), x_dim1*sizeof(float),
				  dX(i__+1,i__), x_dim1*sizeof(float),
				  sizeof(float)*i__2, 1,
                                  cudaMemcpyDeviceToHost,stream[1]);

		i__2 = n - i__;
		sgemv_("Transpose", &i__2, &i__, &c_b5, Y(i__ + 1 , 1), 
		       &ldy, A(i__ , i__ + 1), &lda, &c_b16, X(1,i__), &c__1);

		i__2 = m - i__;
                sgemv_("N", &i__2, &i__, &c_b4, A(i__ + 1 ,1), &lda, 
		       X(1, i__), &c__1, &c_b16, f, &c__1);
                i__2 = i__ - 1;
                i__3 = n - i__;
		sgemv_("N", &i__2, &i__3, &c_b5, A(1, i__+1), 
		       &lda, A(i__ , i__ + 1), &lda,
		       &c_b16, X(1,i__), &c__1);

		// 4. Synch to make sure the result is back ----------------
                cudaStreamSynchronize(stream[1]);
		if (i__!=0){
                  i__2 = m - i__;
                  saxpy_(&i__2, &c_b5, f,&c__1, X(i__+1,i__),&c__1);
		}


		i__2 = m - i__;
		i__3 = i__ - 1;
		sgemv_("No transpose", &i__2, &i__3, &c_b4, X(i__ + 1 , 1), 
		       &ldx, X(1, i__), &c__1, &c_b5, X(i__ + 1 , i__), &c__1);
		i__2 = m - i__;
		sscal_(&i__2, &taup[i__], X(i__ + 1 , i__), &c__1);
	    }
	}
    } else {

      /* Reduce to lower bidiagonal form */

      i__1 = nb;
      for (i__ = 1; i__ <= i__1; ++i__) {

	/* Update A(i,i:n) */
	i__2 = n - i__ + 1;
	i__3 = i__ - 1;
	sgemv_("No transpose", &i__2, &i__3, &c_b4, Y(i__ ,1), &ldy, 
	       A(i__ ,1), &lda, &c_b5, A(i__ , i__), &lda);
	i__2 = i__ - 1;
	i__3 = n - i__ + 1;
	sgemv_("Transpose", &i__2, &i__3, &c_b4, A(1,i__), 
	       &lda, X(i__,1), &ldx, &c_b5, A(i__ , i__), &lda);

	/* Generate reflection P(i) to annihilate A(i,i+1:n) */
	i__2 = n - i__ + 1;
	/* Computing MIN */
	i__3 = i__ + 1;
	slarfg_(&i__2, A(i__ , i__), 
		A(i__ , min(i__3,n)), &lda, &taup[i__]);
	d[i__] = *A(i__ , i__ );
	if (i__ < m) {
	  *A(i__ , i__) = 1.f;
	  
	  /* Compute X(i+1:m,i) */
	  i__2 = m - i__;
	  i__3 = n - i__ + 1;
	  sgemv_("No transpose", &i__2, &i__3, &c_b5, 
		 A(i__ + 1 , i__), &lda, A(i__ , i__), &lda, 
		 &c_b16, X(i__ + 1 , i__), &c__1);
	  i__2 = n - i__ + 1;
	  i__3 = i__ - 1;
	  sgemv_("Transpose", &i__2, &i__3, &c_b5, Y(i__ ,1), 
		 &ldy, A(i__ , i__), &lda, &c_b16, X(1,i__), &c__1);
	  i__2 = m - i__;
	  i__3 = i__ - 1;
	  sgemv_("No transpose", &i__2, &i__3, &c_b4, 
		 A(i__ + 1 ,1), &lda, X(1,i__), &c__1, &c_b5, 
		 X(i__ + 1 , i__), &c__1);
	  i__2 = i__ - 1;
	  i__3 = n - i__ + 1;
	  sgemv_("No transpose", &i__2, &i__3, &c_b5, 
		 A(1, i__), &lda, A(i__ , i__), &lda, &c_b16, 
		 X(1, i__), &c__1);
	  i__2 = m - i__;
	  i__3 = i__ - 1;
	  sgemv_("No transpose", &i__2, &i__3, &c_b4, 
		 X(i__ + 1 ,1), &ldx, X(1,i__), &c__1, &c_b5,
		 X(i__ + 1,i__), &c__1);
	  i__2 = m - i__;
	  sscal_(&i__2, &taup[i__], X(i__ + 1 , i__), &c__1);
	  
	  /* Update A(i+1:m,i) */
	  i__2 = m - i__;
	  i__3 = i__ - 1;
	  sgemv_("No transpose", &i__2, &i__3, &c_b4, 
		 A(i__ + 1,1), &lda, Y(i__ ,1), &ldy, &c_b5, 
		 A(i__ + 1 , i__), &c__1);
	  i__2 = m - i__;
	  sgemv_("No transpose", &i__2, &i__, &c_b4, 
		 X(i__ + 1 ,1), &ldx, A(1,i__), &c__1, &c_b5,
		 A(i__ + 1 , i__), &c__1);
	  
	  /* Generate reflection Q(i) to annihilate A(i+2:m,i) */
	  i__2 = m - i__;
	  /* Computing MIN */
	  i__3 = i__ + 2;
	  slarfg_(&i__2, A(i__ + 1 , i__),
		  A(min(i__3,m) , i__), &c__1, &tauq[i__]);
	  e[i__] = *A(i__ + 1 , i__);
	  *A(i__ + 1 , i__) = 1.f;
	  
	  /* Compute Y(i+1:n,i) */
	  i__2 = m - i__;
	  i__3 = n - i__;
	  sgemv_("Transpose", &i__2, &i__3, &c_b5, 
		 A(i__ + 1 , i__ + 1), &lda, 
		 A(i__ + 1 , i__ ), &c__1, 
		 &c_b16, Y(i__ + 1 , i__), &c__1);
	  i__2 = m - i__;
	  i__3 = i__ - 1;
	  sgemv_("Transpose", &i__2, &i__3, &c_b5, A(i__ + 1 , 1), 
		 &lda, A(i__ + 1 , i__), &c__1, &c_b16, Y(1,i__), &c__1);
	  i__2 = n - i__;
	  i__3 = i__ - 1;
	  sgemv_("No transpose", &i__2, &i__3, &c_b4, 
		 Y(i__ + 1,1), &ldy, Y(1,i__), &c__1, 
		 &c_b5, Y(i__ + 1 , i__), &c__1);
	  i__2 = m - i__;
	  sgemv_("Transpose", &i__2, &i__, &c_b5, X(i__ + 1 ,1), 
		 &ldx, A(i__ + 1 , i__), &c__1, &c_b16, 
		 Y(1,i__), &c__1);
	  i__2 = n - i__;
	  sgemv_("Transpose", &i__, &i__2, &c_b4, 
		 A(1,i__ + 1), &lda, Y(1,i__), 
		 &c__1, &c_b5, Y(i__ + 1 , i__), &c__1);
	  i__2 = n - i__;
	  sscal_(&i__2, &tauq[i__], Y(i__ + 1 , i__), &c__1);
	}
      }
    }
    
    free(f);
    
    return 0;

    /* End of MAGMA_SLABRD */

} /* slabrd_ */

#undef min
#undef max
