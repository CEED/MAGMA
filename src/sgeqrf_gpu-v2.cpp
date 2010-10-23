/*
    -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010
*/

#include "cuda_runtime_api.h"
#include "cublas.h"
#include "magma.h"
#include <stdio.h>
#include <stdlib.h>

void ssplit_diag_block(int ib, float *a, int lda, float *work){
  int i, j, info;
  float *cola, *colw;

  for(i=0; i<ib; i++){
    cola = a    + i*lda;
    colw = work + i*ib;
    for(j=0; j<i; j++){
      colw[j] = cola[j];
      cola[j] = 0.;
    }
    colw[i] = cola[i];
    cola[i] = 1.;
  }
  strtri_("u", "n", &ib, work, &ib, &info);
}

extern "C" magma_int_t 
magma_sgeqrf_gpu2(magma_int_t m_, magma_int_t n_, float *a, magma_int_t  lda_,  float  *tau,
		  float *work, magma_int_t *lwork, float *dwork, magma_int_t *info )
{
/*  -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

    Purpose   
    =======   

    SGEQRF computes a QR factorization of a real M-by-N matrix A:   
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

    A       (input/output) REAL array on the GPU, dimension (LDA,N)   
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

    TAU     (output) REAL array, dimension (min(M,N))   
            The scalar factors of the elementary reflectors (see Further   
            Details).   

    WORK    (workspace/output) REAL array, dimension (MAX(1,LWORK))   
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.   

            Higher performance is achieved if WORK is in pinned memory, e.g.
            allocated using cudaMallocHost.

    LWORK   (input) INTEGER   
            The dimension of the array WORK.  LWORK >= (M+N+NB)*NB,   
            where NB can be obtained through magma_get_sgeqrf_nb(M).

            If LWORK = -1, then a workspace query is assumed; the routine   
            only calculates the optimal size of the WORK array, returns   
            this value as the first entry of the WORK array, and no error   
            message related to LWORK is issued.   

    DWORK   (workspace/output)  REAL array on the GPU, dimension 3*N*NB,
            where NB can be obtained through magma_get_sgeqrf_nb(M).
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
   int nb = magma_get_sgeqrf_nb(*m);

   int lwkopt = (*n+*m) * nb;
   work[0] = (float) lwkopt;
   long int lquery = *lwork == -1;
   if (*m < 0) {
     *info = -1;
   } else if (*n < 0) {
     *info = -2;
   } else if (*lda < max(1,*m)) {
     *info = -4;
   } else if (*lwork < max(1,*n) && ! lquery) {
     *info = -7;
   }
   if (*info != 0)
     return 0;
   else if (lquery)
     return 0;

   k = min(*m,*n);
   if (k == 0) {
     work[0] = 1.f;
     return 0;
   }

   int lhwork = *lwork - (*m)*nb;

   static cudaStream_t stream[2];
   cudaStreamCreate(&stream[0]);
   cudaStreamCreate(&stream[1]);

   float *ut = hwork+nb*(*n);
   for(i=0; i<nb*nb; i++)
     ut[i] = 0.;

   ldda = *m;
   nbmin = 2;
   ldwork = *m;
   lddwork= k;

   if (nb >= nbmin && nb < k) {
      /* Use blocked code initially */
      for (i = 0; i < k-nb; i += nb) {
	ib = min(k-i, nb);
	rows = *m -i;
	cudaMemcpy2DAsync(  work_ref(i), ldwork*sizeof(float),
			    a_ref(i,i), (*lda)*sizeof(float),
			    sizeof(float)*rows, ib,
			    cudaMemcpyDeviceToHost,stream[1]);
	if (i>0){
	  /* Apply H' to A(i:m,i+2*ib:n) from the left */
	  cols = *n-old_i-2*old_ib;
	  magma_slarfb('F', 'C', *m-old_i, cols, old_ib, 
		       a_ref(old_i, old_i), *lda, t_ref(old_i), lddwork, 
		       a_ref(old_i, old_i+2*old_ib), *lda, 
		       dd_ref(0), lddwork);
	  
	  /* store the diagonal */
	  cudaMemcpy2DAsync(d_ref(old_i), old_ib * sizeof(float),
                            ut, old_ib * sizeof(float),
                            sizeof(float)*old_ib, old_ib,
                            cudaMemcpyHostToDevice,stream[0]);
	}

	cudaStreamSynchronize(stream[1]);
	sgeqrf_(&rows, &ib, work_ref(i), &ldwork, tau+i, hwork, &lhwork, info);
	/* Form the triangular factor of the block reflector
	   H = H(i) H(i+1) . . . H(i+ib-1) */
	slarft_("F", "C", &rows, &ib, work_ref(i), &ldwork, tau+i, hwork, &ib);
	
	/* Put 0s in the upper triangular part of a panel (and 1s on the 
	   diagonal); copy the upper triangular in ut and invert it     */
	cudaStreamSynchronize(stream[0]);
	ssplit_diag_block(ib, work_ref(i), ldwork, ut); 
	cublasSetMatrix(rows, ib, sizeof(float), 
			work_ref(i), ldwork, a_ref(i,i), *lda);

	if (i + ib < *n) {
	  /* Send the triangular factor T to the GPU */
	  cublasSetMatrix(ib, ib, sizeof(float), hwork, ib, t_ref(i), lddwork);

	  if (i+nb < k-nb){
	    /* Apply H' to A(i:m,i+ib:i+2*ib) from the left */
	    magma_slarfb('F', 'C', rows, ib, ib, a_ref(i,i), *lda, t_ref(i),
			 lddwork, a_ref(i,i+ib), *lda, dd_ref(0), lddwork);
	  }
	  else {
	    cols = *n-i-ib;
	    magma_slarfb('F','C',rows, cols, ib, a_ref(i,i), *lda, t_ref(i),
			 lddwork, a_ref(i,i+ib), *lda, dd_ref(0), lddwork);
	    /* Fix the diagonal block */
	    cublasSetMatrix(ib, ib, sizeof(float), ut, ib, d_ref(i), ib);
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
      cublasGetMatrix(rows, ib, sizeof(float),
		      a_ref(i,i), *lda, work, rows);
      lhwork = *lwork - rows*ib;
      sgeqrf_(&rows, &ib, work, &rows, tau+i, work+ib*rows, &lhwork, info);
      cublasSetMatrix(rows, ib, sizeof(float),
		      work, rows, a_ref(i,i), *lda);
   }
   
   return 0; 
  
/*     End of MAGMA_SGEQRF */

} /* magma_sgeqrf */

#undef a_ref
#undef t_ref
#undef d_ref
#undef work_ref
#undef min
#undef max
