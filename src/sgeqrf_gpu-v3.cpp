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
#include <stdio.h>

extern "C" int 
magma_sgeqrf_gpu3(int *m, int *n, float *a, int  *lda,  float  *tau,
		  float *work, int *lwork, float *dwork, int *info )
{
/*  -- MAGMA (version 0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2009

    Purpose   
    =======   

    SGEQRF computes a QR factorization of a real M-by-N matrix A:   
    A = Q * R.   

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

    TAU     (output) REAL array, dimension (min(M,N))   
            The scalar factors of the elementary reflectors (see Further   
            Details).   

    WORK    (workspace/output) REAL array, dimension (MAX(1,LWORK))   
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.   

            Higher performance is achieved if A is in pinned memory, e.g.
            allocated using cudaMallocHost.

    LWORK   (input) INTEGER   
            The dimension of the array WORK.  LWORK >= (M+N)*NB,   
            where NB can be obtained through magma_get_sgeqrf_nb(M).

            If LWORK = -1, then a workspace query is assumed; the routine   
            only calculates the optimal size of the WORK array, returns   
            this value as the first entry of the WORK array, and no error   
            message related to LWORK is issued.   

    DWORK   (workspace)  REAL array on the GPU, dimension N*NB,
            where NB can be obtained through magma_get_sgeqrf_nb(M).

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
   #define work_ref(a_1)  ( work + (a_1)) 
   #define hwork          ( work + (NB)*(*m))
   #define min(a,b)       (((a)<(b))?(a):(b))
   #define max(a,b)       (((a)>(b))?(a):(b))

   int i, k, ldwork, lddwork, old_i, old_ib, rows;
   int nbmin, nx, ib, ldda;

   /* Function Body */
   *info = 0;
   int nb = magma_get_sgeqrf_nb(*m);
   int NB = nb;
   
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

   ldda = *m;
   nbmin = 2;
   // (TTT) nx = 192;
   ib = nx = nb;
   ldwork = *m;
   lddwork= *n;

   if (nb >= nbmin && nb < k && nx < k) {
      /* Use blocked code initially */
      for (i = 0; i < k-nx; i += ib) {
	ib = min(k-i, nb);
	rows = *m -i;
	// send the next panel to the CPU
	cudaMemcpy2DAsync(  work_ref(i), ldwork*sizeof(float),
			    a_ref(i,i), (*lda)*sizeof(float),
			    sizeof(float)*rows, ib,
			    cudaMemcpyDeviceToHost,stream[1]);
	if (i>0){
	  /* Apply H' to A(i:m,i+2*ib:n) from the left */
	  // update T2
	  magma_slarfb('F', 'C', *m-old_i, *n-old_i-old_ib-ib, &old_ib, 
		       a_ref(old_i, old_i), lda, dwork, &lddwork, 
		       a_ref(old_i, old_i+old_ib+ib), lda, 
		       dwork+old_ib, &lddwork);

	  // fix the old panel
	  cudaMemcpy2DAsync(a_ref(old_i, old_i), (*lda) * sizeof(float), 
			    work_ref(old_i), ldwork * sizeof(float),
			    sizeof(float)*old_ib, old_ib,
			    cudaMemcpyHostToDevice,stream[0]);
	}

	cudaStreamSynchronize(stream[1]);
	sgeqrf_(&rows, &ib, work_ref(i), &ldwork, tau+i, hwork, &lhwork, info);
	/* Form the triangular factor of the block reflector
	   H = H(i) H(i+1) . . . H(i+ib-1) */
	slarft_("F", "C", &rows, &ib, work_ref(i), &ldwork, tau+i, hwork, &ib);
	spanel_to_q('U', ib, work_ref(i), ldwork, hwork+ib*ib); 
	// send the current updated panel to the GPU
	cublasSetMatrix(rows, ib, sizeof(float), 
			work_ref(i), ldwork, a_ref(i,i), *lda);
        sq_to_panel('U', ib, work_ref(i), ldwork, hwork+ib*ib);

	nb = magma_get_sgeqrf_nb(k-i);
	//printf("%4d %4d\n", i, nb);
	if (i + ib < *n) {
	  // send T to the GPU
	  cublasSetMatrix(ib, ib, sizeof(float), hwork, ib, dwork, lddwork);

	  //if (i+nb < k-nx)
	  if (i+ib < k-nx)
	    /* Apply H' to A(i:m,i+ib:i+2*ib) from the left */
	    // if we can do one more step, first update T1
	    /*
	    magma_slarfb('F', 'C', rows, ib, &ib, a_ref(i,i), lda, dwork,
			 &lddwork, a_ref(i,i+ib), lda, dwork+ib, &lddwork);
	    */
	    magma_slarfb('F', 'C', rows, nb, &ib, a_ref(i,i), lda, dwork,
                         &lddwork, a_ref(i,i+ib), lda, dwork+ib, &lddwork);
	  else {
	    // otherwise, update until the end and fix the panel 
	    magma_slarfb('F', 'C', rows, *n-i-ib, &ib, a_ref(i,i), lda, dwork,
			 &lddwork, a_ref(i,i+ib), lda, dwork+ib, &lddwork);
	    cublasSetMatrix(ib, ib, sizeof(float),
			    work_ref(i), ldwork, a_ref(i,i), *lda);
	  }
	  old_i = i;
	  old_ib = ib;
	}
	// TTT
	//nb = magma_get_sgeqrf_nb(k-i);
      }  
   } else {
     i = 0;
   }
   
   /* Use unblocked code to factor the last or only block. */
   if (i < k) {
      ib   = *n-i;
      rows = *m-i;
      cublasGetMatrix(rows, ib, sizeof(float),
		      a_ref(i,i), *lda, work_ref(i), ldwork);
      sgeqrf_(&rows, &ib, work_ref(i), &ldwork, tau+i, hwork, &lhwork, info);
      cublasSetMatrix(rows, ib, sizeof(float),
		      work_ref(i), ldwork, a_ref(i,i), *lda);
   }
   return 0; 
  
/*     End of MAGMA_SGEQRF */

} /* magma_sgeqrf_ */

#undef a_ref
#undef work_ref
#undef t_ref
#undef min
#undef max
