/*
    -- MAGMA (version 0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2009
*/

#include "cuda_runtime_api.h"
#include "cublas.h"
#include "magma.h"
#include "magmablas.h"
#include <stdio.h>

extern "C" int 
magma_slahr2(int *n, int *k, int *nb, 
	     float *d_a, float *d_v, float *a, int *lda, 
	     float *tau, float *t, int *ldt, float *y, int *ldy)
{
/*  -- MAGMA auxiliary routine (version 0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2009

    Purpose   
    =======   

    SLAHR2 reduces the first NB columns of a real general n-BY-(n-k+1)   
    matrix A so that elements below the k-th subdiagonal are zero. The   
    reduction is performed by an orthogonal similarity transformation   
    Q' * A * Q. The routine returns the matrices V and T which determine   
    Q as a block reflector I - V*T*V', and also the matrix Y = A * V.   

    This is an auxiliary routine called by DGEHRD.   

    Arguments   
    =========   

    N       (input) INTEGER   
            The order of the matrix A.   

    K       (input) INTEGER   
            The offset for the reduction. Elements below the k-th   
            subdiagonal in the first NB columns are reduced to zero.   
            K < N.   

    NB      (input) INTEGER   
            The number of columns to be reduced.   

    A       (input/output) DOUBLE PRECISION array, dimension (LDA,N-K+1)   
            On entry, the n-by-(n-k+1) general matrix A.   
            On exit, the elements on and above the k-th subdiagonal in   
            the first NB columns are overwritten with the corresponding   
            elements of the reduced matrix; the elements below the k-th   
            subdiagonal, with the array TAU, represent the matrix Q as a   
            product of elementary reflectors. The other columns of A are   
            unchanged. See Further Details.   

    LDA     (input) INTEGER   
            The leading dimension of the array A.  LDA >= max(1,N).   

    TAU     (output) DOUBLE PRECISION array, dimension (NB)   
            The scalar factors of the elementary reflectors. See Further   
            Details.   

    T       (output) DOUBLE PRECISION array, dimension (LDT,NB)   
            The upper triangular matrix T.   

    LDT     (input) INTEGER   
            The leading dimension of the array T.  LDT >= NB.   

    Y       (output) DOUBLE PRECISION array, dimension (LDY,NB)   
            The n-by-nb matrix Y.   

    LDY     (input) INTEGER   
            The leading dimension of the array Y. LDY >= N.   

    Further Details   
    ===============   

    The matrix Q is represented as a product of nb elementary reflectors   

       Q = H(1) H(2) . . . H(nb).   

    Each H(i) has the form   

       H(i) = I - tau * v * v'   

    where tau is a real scalar, and v is a real vector with   
    v(1:i+k-1) = 0, v(i+k) = 1; v(i+k+1:n) is stored on exit in   
    A(i+k+1:n,i), and tau in TAU(i).   

    The elements of the vectors v together form the (n-k+1)-by-nb matrix   
    V which is needed, with T and Y, to apply the transformation to the   
    unreduced part of the matrix, using an update of the form:   
    A := (I - V*T*V') * (A - Y*T*V').   

    The contents of A on exit are illustrated by the following example   
    with n = 7, k = 3 and nb = 2:   

       ( a   a   a   a   a )   
       ( a   a   a   a   a )   
       ( a   a   a   a   a )   
       ( h   h   a   a   a )   
       ( v1  h   a   a   a )   
       ( v1  v2  a   a   a )   
       ( v1  v2  a   a   a )   

    where a denotes an element of the original matrix A, h denotes a   
    modified element of the upper Hessenberg matrix H, and vi denotes an   
    element of the vector defining H(i).

    This implementation follows the hybrid algorithm and notations described in

    S. Tomov and J. Dongarra, "Accelerating the reduction to upper Hessenberg
    form through hybrid GPU-based computing," University of Tennessee Computer
    Science Technical Report, UT-CS-09-642 (also LAPACK Working Note 219),
    May 24, 2009.

    =====================================================================    */

    #define min(a,b) ((a) <= (b) ? (a) : (b))

    int N = *n, ldda = *n;

    /* Table of constant values */
    static float c_b4 = -1.;
    static float c_b5 = 1.;
    static int c__1 = 1;
    static float c_b38 = 0.;
    
    /* System generated locals */
    int a_dim1, a_offset, t_dim1, t_offset, y_dim1, y_offset, i__2, i__3;
    float d__1;
    /* Local variables */
    static int i__;
    static float ei;

    --tau;
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    t_dim1 = *ldt;
    t_offset = 1 + t_dim1;
    t -= t_offset;
    y_dim1 = *ldy;
    y_offset = 1 + y_dim1;
    y -= y_offset;

    /* Function Body */
    if (*n <= 1)
      return 0;
    

    for (i__ = 1; i__ <= *nb; ++i__) {
	if (i__ > 1) {

	  /* Update A(K+1:N,I); Update I-th column of A - Y * V' */
	  i__2 = *n - *k + 1;
	  i__3 = i__ - 1;
	  scopy_(&i__3, &a[*k+i__-1+a_dim1], lda, &t[*nb*t_dim1+1], &c__1);
	  strmv_("u","n","n",&i__3,&t[t_offset], ldt, &t[*nb*t_dim1+1], &c__1);
	  sgemv_("NO TRANSPOSE", &i__2, &i__3, &c_b4, &y[*k + y_dim1],
		 ldy, &t[*nb*t_dim1+1], &c__1, &c_b5, &a[*k+i__*a_dim1],&c__1);

	  /* Apply I - V * T' * V' to this column (call it b) from the   
             left, using the last column of T as workspace   

             Let  V = ( V1 )   and   b = ( b1 )   (first I-1 rows)   
                      ( V2 )             ( b2 )   
             where V1 is unit lower triangular   
             w := V1' * b1                                                 */
	  
	  i__2 = i__ - 1;
	  scopy_(&i__2, &a[*k+1+i__*a_dim1], &c__1, &t[*nb*t_dim1+1], &c__1);
	  strmv_("Lower", "Transpose", "UNIT", &i__2, &a[*k + 1 + a_dim1], 
		 lda, &t[*nb * t_dim1 + 1], &c__1);

	  /* w := w + V2'*b2 */
	  i__2 = *n - *k - i__ + 1;
	  i__3 = i__ - 1;
	  sgemv_("T", &i__2, &i__3, &c_b5, &a[*k + i__ + a_dim1], lda, 
		 &a[*k+i__+i__*a_dim1], &c__1, &c_b5, &t[*nb*t_dim1+1], &c__1);

	  /* w := T'*w */
	  i__2 = i__ - 1;
	  strmv_("U","T","N",&i__2, &t[t_offset], ldt, &t[*nb*t_dim1+1],&c__1);
	  
	  /* b2 := b2 - V2*w */
	  i__2 = *n - *k - i__ + 1;
	  i__3 = i__ - 1;
	  sgemv_("N", &i__2, &i__3, &c_b4, &a[*k + i__ + a_dim1], lda, 
		 &t[*nb*t_dim1+1], &c__1, &c_b5, &a[*k+i__+i__*a_dim1], &c__1);

	  /* b1 := b1 - V1*w */
	  i__2 = i__ - 1;
	  strmv_("L","N","U",&i__2,&a[*k+1+a_dim1],lda,&t[*nb*t_dim1+1],&c__1);
	  saxpy_(&i__2, &c_b4, &t[*nb * t_dim1 + 1], &c__1, 
		 &a[*k + 1 + i__ * a_dim1], &c__1);
	  
	  a[*k + i__ - 1 + (i__ - 1) * a_dim1] = ei;
	}
	
	/* Generate the elementary reflector H(I) to annihilate A(K+I+1:N,I) */
	i__2 = *n - *k - i__ + 1;
	i__3 = *k + i__ + 1;
	slarfg_(&i__2, &a[*k + i__ + i__ * a_dim1], 
		&a[min(i__3,*n) + i__ * a_dim1], &c__1, &tau[i__]);
	ei = a[*k + i__ + i__ * a_dim1];
	a[*k + i__ + i__ * a_dim1] = 1.;

	/* Compute  Y(K+1:N,I) */
        i__2 = *n - *k;
	i__3 = *n - *k - i__ + 1;
        cublasSetVector(i__3, sizeof(float), 
                        &a[*k + i__ + i__*a_dim1], 1, d_v+(i__-1)*(ldda+1), 1);
	/*
	 cublasSgemv('N', i__2+1, i__3, c_b5, 
	               d_a -1 + *k + i__ *ldda, ldda, 
	               d_v+(i__-1)*(ldda+1), c__1, c_b38, 
                       d_a-1 + *k + (i__-1)*ldda, c__1);     
	*/
        magmablas_sgemv(i__2+1, i__3, 
			d_a -1 + *k + i__ *ldda, ldda, 
			d_v+(i__-1)*(ldda+1), 
			d_a -1 + *k + (i__-1)*ldda);
	 
	i__2 = *n - *k - i__ + 1;
	i__3 = i__ - 1;
	sgemv_("T", &i__2, &i__3, &c_b5, &a[*k + i__ + a_dim1], lda,
	       &a[*k+i__+i__*a_dim1], &c__1, &c_b38, &t[i__*t_dim1+1], &c__1);

	/* Compute T(1:I,I) */
	i__2 = i__ - 1;
	d__1 = -tau[i__];
	sscal_(&i__2, &d__1, &t[i__ * t_dim1 + 1], &c__1);
	strmv_("U","N","N", &i__2, &t[t_offset], ldt, &t[i__*t_dim1+1], &c__1);
	t[i__ + i__ * t_dim1] = tau[i__];

        cublasGetVector(*n - *k + 1, sizeof(float),
	                d_a-1+ *k+(i__-1)*ldda, 1, y+ *k + i__*y_dim1, 1);
    }
    a[*k + *nb + *nb * a_dim1] = ei;

    return 0;

/*     End of MAGMA_SLAHR2 */

} /* magma_slahr2 */

#undef min
