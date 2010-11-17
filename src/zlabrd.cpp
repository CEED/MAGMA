/*
    -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

       @precisions normal z -> s d c

*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include "magma.h"
#include "magmablas.h"

#define PRECISION_z

#define A(i, j)  (a +(j)*lda  + (i))
#define X(i, j)  (x +(j)*ldx  + (i))
#define Y(i, j)  (y +(j)*ldy  + (i))
#define dA(i, j) (da+(j)*ldda + (i))
#define dX(i, j) (dx+(j)*lddx + (i))
#define dY(i, j) (dy+(j)*lddy + (i))

extern "C" magma_int_t
magma_zlabrd(magma_int_t m, magma_int_t n, magma_int_t nb,
             cuDoubleComplex *a, magma_int_t lda,
             double *d, double *e,
	     cuDoubleComplex *tauq, cuDoubleComplex *taup,
             cuDoubleComplex *x,  magma_int_t ldx,
             cuDoubleComplex *y,  magma_int_t ldy,
	     cuDoubleComplex *da, magma_int_t ldda,
	     cuDoubleComplex *dx, magma_int_t lddx,
             cuDoubleComplex *dy, magma_int_t lddy)
{
/*  -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

    Purpose
    =======
    ZLABRD reduces the first NB rows and columns of a real general
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

    A       (input/output) COMPLEX_16 array, dimension (LDA,N)
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

    D       (output) COMPLEX_16 array, dimension (NB)
            The diagonal elements of the first NB rows and columns of
            the reduced matrix.  D(i) = A(i,i).

    E       (output) COMPLEX_16 array, dimension (NB)
            The off-diagonal elements of the first NB rows and columns of
            the reduced matrix.

    TAUQ    (output) COMPLEX_16 array dimension (NB)
            The scalar factors of the elementary reflectors which
            represent the orthogonal matrix Q. See Further Details.

    TAUP    (output) COMPLEX_16 array, dimension (NB)
            The scalar factors of the elementary reflectors which
            represent the orthogonal matrix P. See Further Details.

    X       (output) COMPLEX_16 array, dimension (LDX,NB)
            The m-by-nb matrix X required to update the unreduced part
            of A.

    LDX     (input) INTEGER
            The leading dimension of the array X. LDX >= M.

    Y       (output) COMPLEX_16 array, dimension (LDY,NB)
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
    cuDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    cuDoubleComplex c_one     = MAGMA_Z_ONE;
    cuDoubleComplex c_zero    = MAGMA_Z_ZERO;
    cuDoubleComplex alpha;
    static int      ione      = 1;
    int nrow, ncol, nrow2, ncol2, ip1;
    
    /* Local variables */
    static int i;

    /* Function Body */
    if (m <= 0 || n <= 0) {
	return 0;
    }

    cuDoubleComplex *f = (cuDoubleComplex *)malloc(max(n,m)*sizeof(cuDoubleComplex ));
    static cudaStream_t stream[2];
    cudaStreamCreate(&stream[0]);
    cudaStreamCreate(&stream[1]);

    if (m >= n) {

        /* Reduce to upper bidiagonal form */
	for (i=0; i <nb; ++i) {
            
	    /*  Update A(i:m,i) */
	    nrow = m - i;
#if defined(PRECISION_z) || defined(PRECISION_c)
            lapackf77_zlacgv( &i, Y( i, 0 ), &ldy );
#endif
	    blasf77_zgemv( MagmaNoTransStr, &nrow, &i, 
                           &c_neg_one, A(i, 0), &lda,
                                       Y(i, 0), &ldy, 
                           &c_one,     A(i, i), &ione);
#if defined(PRECISION_z) || defined(PRECISION_c)
	    lapackf77_zlacgv( &i, Y( i, 0 ), &ldy );
#endif
            blasf77_zgemv( MagmaNoTransStr, &nrow, &i, 
                           &c_neg_one, X(i, 0), &ldx,
                                       A(0, i), &ione, 
                           &c_one,     A(i, i), &ione);

	    /* Generate reflection Q(i) to annihilate A(i+1:m,i) */
            alpha = *A(i, i);
	    lapackf77_zlarfg( &nrow, &alpha,
                              A( min(i+1, m), i), &ione, tauq+i );
	    d[i] = MAGMA_Z_GET_X( alpha );

	    if ( (i+1) < n ) {
		*A(i, i) = c_one;

		/* Compute Y(i+1:n,i) */
		ncol = n - i - 1;
                ip1 = i+1;

		// 1. Send the block reflector  A(i+1:m,i) to the GPU ------
		cublasSetVector( nrow, sizeof(cuDoubleComplex),
                                 A(i, i), 1, dA(i, i), 1);

		// 2. Multiply ---------------------------------------------
		cublasZgemv( MagmaConjTrans, nrow, ncol, 
                             c_one,  dA( i,   i+1), ldda,
                                     dA( i,   i  ), 1,
                             c_zero, dY( i+1, i  ), 1);

		// 3. Put the result back ----------------------------------
		cudaMemcpy2DAsync( Y(  i+1, i), ldy *sizeof(cuDoubleComplex),
                                   dY( i+1, i), lddy*sizeof(cuDoubleComplex),
                                   sizeof(cuDoubleComplex)*ncol, 1,
                                   cudaMemcpyDeviceToHost, stream[1]);

		blasf77_zgemv( MagmaConjTransStr, &nrow, &i, 
                               &c_one,  A(i, 0), &lda,
                                        A(i, i), &ione, 
                               &c_zero, Y(0, i), &ione);

                blasf77_zgemv( MagmaNoTransStr, &ncol, &i, 
                               &c_neg_one, Y(i+1, 0), &ldy,
                                           Y(0,   i), &ione, 
                               &c_zero,    f,         &ione);

                blasf77_zgemv( MagmaConjTransStr, &nrow, &i, 
                               &c_one,  X(i, 0), &ldx, 
                                        A(i, i), &ione, 
                               &c_zero, Y(0, i), &ione);

		// 4. Synch to make sure the result is back ----------------
		cudaStreamSynchronize(stream[1]);

		if (i != 0){
                    blasf77_zaxpy(&ncol, &c_one, f, &ione, Y( i+1, i), &ione);
		}

		blasf77_zgemv( MagmaConjTransStr, &i, &ncol, 
                               &c_neg_one, A(0, i+1), &lda,
                                           Y(0, i  ), &ione, 
                               &c_one,     Y(i+1, i), &ione);

		blasf77_zscal( &ncol, tauq+i, Y(i+1, i), &ione);

		/* Update A(i,i+1:n) */
#if defined(PRECISION_z) || defined(PRECISION_c)
                lapackf77_zlacgv( &ncol, A( i, i+1 ), &lda );
                lapackf77_zlacgv( &ip1,  A( i, 0   ), &lda );
#endif

		blasf77_zgemv( MagmaNoTransStr, &ncol, &ip1, 
                               &c_neg_one, Y(i+1, 0), &ldy, 
                                           A(i,   0), &lda, 
                               &c_one,     A(i, i+1), &lda);

#if defined(PRECISION_z) || defined(PRECISION_c)
                lapackf77_zlacgv( &ip1, A( i, 0 ), &lda );
                lapackf77_zlacgv( &i,   X( i, 0 ), &ldx );
#endif

		blasf77_zgemv( MagmaConjTransStr, &i, &ncol, 
                               &c_neg_one, A(0, i+1), &lda, 
                                           X(i, 0  ), &ldx,
                               &c_one,     A(i, i+1), &lda);

#if defined(PRECISION_z) || defined(PRECISION_c)
                lapackf77_zlacgv( &i, X( i, 0 ), &ldx );
#endif

		/* Generate reflection P(i) to annihilate A(i,i+2:n) */
                alpha = *A(i, i+1);
		lapackf77_zlarfg(&ncol, &alpha, A( i, min(i+2, n)), &lda, taup+i);
		e[i] = MAGMA_Z_GET_X( alpha );
		*A(i, i+1) = c_one;

		/* Compute X(i+1:m,i) */
		nrow2 = m - i - 1;
                // 1. Send the block reflector  A(i+1:m,i) to the GPU ------
                cublasSetVector( ncol, sizeof(cuDoubleComplex),
                                 A( i, i+1), lda,
                                 dA(i, i+1), ldda);
                // 2. Multiply ---------------------------------------------
                cublasZgemv( MagmaNoTrans, nrow2, ncol, 
                             c_one,  dA(i+1, i+1), ldda,
                                     dA(i,   i+1), ldda,
                             c_zero, dX(i+1, i  ), ione);
                
		// 3. Put the result back ----------------------------------
		cudaMemcpy2DAsync( X(i+1, i), ldx *sizeof(cuDoubleComplex),
				  dX(i+1, i), lddx*sizeof(cuDoubleComplex),
				  sizeof(cuDoubleComplex)*nrow2, 1,
                                  cudaMemcpyDeviceToHost, stream[1]);

		blasf77_zgemv( MagmaConjTransStr, &ncol, &ip1, 
                               &c_one,  Y(i+1, 0), &ldy,
                                        A(i, i+1), &lda, 
                               &c_zero, X(0, i  ), &ione);

                blasf77_zgemv( MagmaNoTransStr, &nrow2, &ip1, 
                               &c_neg_one, A(i+1, 0), &lda,
                                           X(0,   i), &ione, 
                               &c_zero,    f,         &ione);

		blasf77_zgemv( MagmaNoTransStr, &i, &ncol, 
                               &c_one,  A(0, i+1), &lda, 
                                        A(i, i+1), &lda,
                               &c_zero, X(0, i  ), &ione);

		// 4. Synch to make sure the result is back ----------------
                cudaStreamSynchronize(stream[1]);
		if ( i != 0){
                    blasf77_zaxpy(&nrow2, &c_one, f, &ione, X(i+1, i), &ione);
		}

		blasf77_zgemv(MagmaNoTransStr, &nrow2, &i, 
                              &c_neg_one, X(i+1, 0), &ldx, 
                                          X(0,   i), &ione, 
                              &c_one,     X(i+1, i), &ione);

		blasf77_zscal(&nrow2, taup+i, X(i+1, i), &ione);

#if defined(PRECISION_z) || defined(PRECISION_c)
                lapackf77_zlacgv( &ncol, A( i, i+1 ), &lda );
#endif
	    }
	}
    } else {

        /* Reduce to lower bidiagonal form */
        for (i=0; i <nb; ++i) {

            /* Update A(i,i:n) */
            ncol = n - i;

#if defined(PRECISION_z) || defined(PRECISION_c)
            lapackf77_zlacgv( &ncol, A( i, i ), &lda );
            lapackf77_zlacgv( &i,    A( i, 0 ), &lda );
#endif

            blasf77_zgemv( MagmaNoTransStr, &ncol, &i, 
                           &c_neg_one, Y(i, 0), &ldy,
                                       A(i, 0), &lda, 
                           &c_one,     A(i, i), &lda);

#if defined(PRECISION_z) || defined(PRECISION_c)
            lapackf77_zlacgv( &i, A( i, 0 ), &lda );
            lapackf77_zlacgv( &i, X( i, 0 ), &ldx );
#endif

            blasf77_zgemv( MagmaConjTransStr, &i, &ncol, 
                           &c_neg_one, A(0, i), &lda, 
                                       X(i, 0), &ldx, 
                           &c_one,     A(i, i), &lda);

#if defined(PRECISION_z) || defined(PRECISION_c)
            lapackf77_zlacgv( &i, X( i, 0 ), &ldx );
#endif

            /* Generate reflection P(i) to annihilate A(i,i+1:n) */
            alpha = *A(i, i);
            lapackf77_zlarfg(&ncol, &alpha, A(i, min(i+1, n)), &lda, taup+i);
            d[i] = MAGMA_Z_GET_X( alpha );

            if ( (i+1) < m) {
                *A(i, i) = c_one;

                /* Compute X(i+1:m,i) */
                nrow = m - i - 1;
                ip1 = i+1;
                blasf77_zgemv( MagmaNoTransStr, &nrow, &ncol, 
                               &c_one,  A(i+1, i), &lda, 
                                        A(i,   i), &lda,
                               &c_zero, X(i+1, i), &ione);

                blasf77_zgemv( MagmaConjTransStr, &ncol, &i, 
                               &c_one,  Y(i, 0), &ldy, 
                                        A(i, i), &lda, 
                               &c_zero, X(0, i), &ione);

                blasf77_zgemv( MagmaNoTransStr, &nrow, &i, 
                               &c_neg_one, A(i+1, 0), &lda, 
                                           X(0,   i), &ione, 
                               &c_one,     X(i+1, i), &ione);

                blasf77_zgemv( MagmaNoTransStr, &i, &ncol, 
                               &c_one,  A(0, i), &lda, 
                                        A(i, i), &lda, 
                               &c_zero, X(0, i), &ione);

                blasf77_zgemv( MagmaNoTransStr, &nrow, &i, 
                               &c_neg_one, X(i+1, 0), &ldx, 
                                           X(0,   i), &ione, 
                               &c_one,     X(i+1, i), &ione);

                blasf77_zscal(&nrow, taup+i, X(i+1, i), &ione);

#if defined(PRECISION_z) || defined(PRECISION_c)
                lapackf77_zlacgv( &ncol, A( i, i ), &lda );
                
                /* Update A(i+1:m,i) */
                lapackf77_zlacgv( &i, Y( i, 0 ), &ldy );
#endif

                blasf77_zgemv( MagmaNoTransStr, &nrow, &i, 
                               &c_neg_one, A(i+1, 0), &lda, 
                                           Y(i,   0), &ldy, 
                               &c_one,     A(i+1, i), &ione);

#if defined(PRECISION_z) || defined(PRECISION_c)
                lapackf77_zlacgv( &i, Y( i, 0 ), &ldy );
#endif

                blasf77_zgemv( MagmaNoTransStr, &nrow, &ip1, 
                               &c_neg_one, X(i+1, 0), &ldx, 
                                           A(0,   i), &ione,
                               &c_one,     A(i+1, i), &ione);

                /* Generate reflection Q(i) to annihilate A(i+2:m,i) */
                alpha = *A(i+1, i);
                lapackf77_zlarfg( &nrow, &alpha, 
                                  A( min(i+2, m), i), &ione, tauq+i);
                e[i] = MAGMA_Z_GET_X( alpha );
                *A(i+1, i) = c_one;

                /* Compute Y(i+1:n,i) */
                ncol2 = n - i - 1;
                blasf77_zgemv( MagmaConjTransStr, &nrow, &ncol2, 
                               &c_one,  A(i+1, i+1), &lda,
                                        A(i+1, i  ), &ione,
                               &c_zero, Y(i+1, i  ), &ione);

                blasf77_zgemv( MagmaConjTransStr, &nrow, &i, 
                               &c_one,  A(i+1, 0), &lda, 
                                        A(i+1, i), &ione, 
                               &c_zero, Y(0,   i), &ione);

                blasf77_zgemv(MagmaNoTransStr, &ncol2, &i, 
                              &c_neg_one, Y(i+1, 0), &ldy, 
                                          Y(0,   i), &ione,
                              &c_one,     Y(i+1, i), &ione);

                blasf77_zgemv(MagmaConjTransStr, &nrow, &ip1, 
                              &c_one,  X(i+1, 0), &ldx, 
                                       A(i+1, i), &ione,
                              &c_zero, Y(0,   i), &ione);

                blasf77_zgemv(MagmaConjTransStr, &ip1, &ncol2, 
                              &c_neg_one, A(0, i+1), &lda, 
                                          Y(0, i  ), &ione, 
                              &c_one,     Y(i+1, i), &ione);

                blasf77_zscal(&ncol2, tauq+i, Y( i+1, i), &ione);
            }
#if defined(PRECISION_z) || defined(PRECISION_c)
            else {
                lapackf77_zlacgv( &ncol, A( i, i ), &lda );
            }
#endif
        }
    }

    free(f);

    return 0;

    /* End of MAGMA_ZLABRD */

} /* zlabrd_ */

#undef min
#undef max
