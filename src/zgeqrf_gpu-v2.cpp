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

void zsplit_diag_block(int ib, double2 *a, int lda, double2 *work){
  int i, j, info;
  double2 *cola, *colw;
  double2 c_zero = MAGMA_Z_ZERO;
  double2 c_one = MAGMA_Z_ONE;

  for(i=0; i<ib; i++){
    cola = a    + i*lda;
    colw = work + i*ib;
    for(j=0; j<i; j++){
      colw[j] = cola[j];
      cola[j] = c_zero;
    }
    colw[i] = cola[i];
    cola[i] = c_one;
  }
  ztrtri_("u", "n", &ib, work, &ib, &info);
}

extern "C" magma_int_t 
magma_zgeqrf_gpu2(magma_int_t m_, magma_int_t n_, double2 *a, magma_int_t  lda_,  
                  double2  *tau, double2 *dwork, magma_int_t *info )
{
/*  -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

    Purpose   
    =======   

    ZGEQRF computes a QR factorization of a real M-by-N matrix A:   
    A = Q * R. This version stores the triangular matrices used in 
    the factorization so that they can be applied directly (i.e.,
    without being recomputed) later. As a result, the application 
    of Q is much faster.

    Arguments   
    =========   

    M       (input) INTEGER   
            The number of rows of the matrix A.  M >= 0.   

    N       (input) INTEGER   
            The number of columns of the matrix A.  N >= 0.   

    A       (input/output) COMPLEX_16 array on the GPU, dimension (LDA,N)   
            On entry, the M-by-N matrix A.   
            On exit, the elements on and above the diagonal of the array   
            contain the min(M,N)-by-N upper trapezoidal matrix R (R is   
            upper triangular if m >= n); the elements below the diagonal,   
            with the array TAU, represent the orthogonal matrix Q as a   
            product of min(m,n) elementary reflectors (see Further   
            Details).

    LDA     (input) INTEGER   
            The leading dimension of the array A.  LDA >= max(1,M).   
            To benefit from coalescent memory accesses LDA must be
            dividable by 16.

    TAU     (output) COMPLEX_16 array, dimension (min(M,N))   
            The scalar factors of the elementary reflectors (see Further   
            Details).   

    DWORK   (workspace/output)  COMPLEX_16 array on the GPU, dimension 3*N*NB,
            where NB can be obtained through magma_get_zgeqrf_nb(M).
            It starts with NB*NB blocks that store the triangular T 
            matrices, followed by the NB*NB blocks of the diagonal 
            inverses for the R matrix.

    INFO    (output) INTEGER   
            = 0:  successful exit   
            < 0:  if INFO = -i, the i-th argument had an illegal value   

    Further Details   
    ===============   

    The matrix Q is represented as a product of elementary reflectors   

       Q = H(1) H(2) . . . H(k), where k = min(m,n).   

    Each H(i) has the form   

       H(i) = I - tau * v * v'   

    where tau is a real scalar, and v is a real vector with   
    v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in A(i+1:m,i),   
    and tau in TAU(i).   

    =====================================================================    */

   #define a_ref(a_1,a_2) ( a+(a_2)*(*lda) + (a_1))
   #define t_ref(a_1)     (dwork+(a_1))
   #define d_ref(a_1)     (dwork+(lddwork+(a_1))*nb)
   #define dd_ref(a_1)    (dwork+(2*lddwork+(a_1))*nb)
   #define work_ref(a_1)  ( work + (a_1)) 
   #define hwork          ( work + (nb)*(*m))
   #define min(a,b)       (((a)<(b))?(a):(b))
   #define max(a,b)       (((a)>(b))?(a):(b))

   int *m = &m_;
   int *n = &n_;
   int *lda = &lda_;

   int i, k, ldwork, lddwork, old_i, old_ib, rows, cols;
   int nbmin, ib, ldda;

   /* Function Body */
   *info = 0;
   int nb = magma_get_zgeqrf_nb(*m);

   double2 c_zero = MAGMA_Z_ZERO;

   if (*m < 0) {
     *info = -1;
   } else if (*n < 0) {
     *info = -2;
   } else if (*lda < max(1,*m)) {
     *info = -4;
   }
   if (*info != 0)
     return 0;

   k = min(*m,*n);
   if (k == 0)
     return 0;

   int lwork  = (*m + *n +nb)*nb; 
   int lhwork = lwork -  (*m)*nb;

   static cudaStream_t stream[2];
   cudaStreamCreate(&stream[0]);
   cudaStreamCreate(&stream[1]);

   double2 *work;
   cudaMallocHost((void**)&work, lwork*sizeof(double2));

   double2 *ut = hwork+nb*(*n);
   for(i=0; i<nb*nb; i++)
     ut[i] = c_zero;

   ldda = *m;
   nbmin = 2;
   ldwork = *m;
   lddwork= k;

   if (nb >= nbmin && nb < k) {
      /* Use blocked code initially */
      old_i = 0; old_ib = nb;
      for (i = 0; i < k-nb; i += nb) {
	ib = min(k-i, nb);
	rows = *m -i;
	cudaMemcpy2DAsync(  work_ref(i), ldwork*sizeof(double2),
			    a_ref(i,i), (*lda)*sizeof(double2),
			    sizeof(double2)*rows, ib,
			    cudaMemcpyDeviceToHost,stream[1]);
	if (i>0){
	  /* Apply H' to A(i:m,i+2*ib:n) from the left */
	  cols = *n-old_i-2*old_ib;
	  magma_zlarfb('F', 'C', *m-old_i, cols, old_ib, 
		       a_ref(old_i, old_i), *lda, t_ref(old_i), lddwork, 
		       a_ref(old_i, old_i+2*old_ib), *lda, 
		       dd_ref(0), lddwork);
	  
	  /* store the diagonal */
	  cudaMemcpy2DAsync(d_ref(old_i), old_ib * sizeof(double2),
                            ut, old_ib * sizeof(double2),
                            sizeof(double2)*old_ib, old_ib,
                            cudaMemcpyHostToDevice,stream[0]);
	}

	cudaStreamSynchronize(stream[1]);
	zgeqrf_(&rows, &ib, work_ref(i), &ldwork, tau+i, hwork, &lhwork, info);
	/* Form the triangular factor of the block reflector
	   H = H(i) H(i+1) . . . H(i+ib-1) */
	zlarft_("F", "C", &rows, &ib, work_ref(i), &ldwork, tau+i, hwork, &ib);
	
	/* Put 0s in the upper triangular part of a panel (and 1s on the 
	   diagonal); copy the upper triangular in ut and invert it     */
	cudaStreamSynchronize(stream[0]);
	zsplit_diag_block(ib, work_ref(i), ldwork, ut); 
	cublasSetMatrix(rows, ib, sizeof(double2), 
			work_ref(i), ldwork, a_ref(i,i), *lda);

	if (i + ib < *n) {
	  /* Send the triangular factor T to the GPU */
	  cublasSetMatrix(ib, ib, sizeof(double2), hwork, ib, t_ref(i), lddwork);

	  if (i+nb < k-nb){
	    /* Apply H' to A(i:m,i+ib:i+2*ib) from the left */
	    magma_zlarfb('F', 'C', rows, ib, ib, a_ref(i,i), *lda, t_ref(i),
			 lddwork, a_ref(i,i+ib), *lda, dd_ref(0), lddwork);
	  }
	  else {
	    cols = *n-i-ib;
	    magma_zlarfb('F','C',rows, cols, ib, a_ref(i,i), *lda, t_ref(i),
			 lddwork, a_ref(i,i+ib), *lda, dd_ref(0), lddwork);
	    /* Fix the diagonal block */
	    cublasSetMatrix(ib, ib, sizeof(double2), ut, ib, d_ref(i), ib);
	  }
	  old_i = i;
	  old_ib = ib;
	}
      }  
   } else {
     i = 0;
   }

   /* Use unblocked code to factor the last or only block. */
   if (i < k) {
      ib   = *n-i;
      rows = *m-i;
      cublasGetMatrix(rows, ib, sizeof(double2),
		      a_ref(i,i), *lda, work, rows);
      lhwork = lwork - rows*ib;
      zgeqrf_(&rows, &ib, work, &rows, tau+i, work+ib*rows, &lhwork, info);
      cublasSetMatrix(rows, ib, sizeof(double2),
		      work, rows, a_ref(i,i), *lda);
   }
   cublasFree(work);
   return 0; 
  
/*     End of MAGMA_ZGEQRF */

} /* magma_zgeqrf */

#undef a_ref
#undef t_ref
#undef d_ref
#undef work_ref
#undef min
#undef max
