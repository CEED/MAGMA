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

extern "C" magma_int_t
magma_sgeqrf(magma_int_t m_, magma_int_t n_, float *a, magma_int_t  lda_,  float  *tau,
             float *work, magma_int_t lwork_, float *da, magma_int_t *info )
{
/*  -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

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

    A       (input/output) REAL array, dimension (LDA,N)   
            On entry, the M-by-N matrix A.   
            On exit, the elements on and above the diagonal of the array   
            contain the min(M,N)-by-N upper trapezoidal matrix R (R is   
            upper triangular if m >= n); the elements below the diagonal,   
            with the array TAU, represent the orthogonal matrix Q as a   
            product of min(m,n) elementary reflectors (see Further   
            Details).   

            Higher performance is achieved if A is in pinned memory, e.g.
            allocated using cudaMallocHost.

    LDA     (input) INTEGER   
            The leading dimension of the array A.  LDA >= max(1,M).   

    TAU     (output) REAL array, dimension (min(M,N))   
            The scalar factors of the elementary reflectors (see Further   
            Details).   

    WORK    (workspace/output) REAL array, dimension (MAX(1,LWORK))   
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.   

	    Higher performance is achieved if WORK is in pinned memory, e.g.
            allocated using cudaMallocHost.

    LWORK   (input) INTEGER   
            The dimension of the array WORK.  LWORK >= N*NB, 
            where NB can be obtained through magma_get_sgeqrf_nb(M).   

            If LWORK = -1, then a workspace query is assumed; the routine   
            only calculates the optimal size of the WORK array, returns   
            this value as the first entry of the WORK array, and no error   
            message related to LWORK is issued.

    DA      (workspace)  REAL array on the GPU, dimension N*(M + NB), 
            where NB can be obtained through magma_get_sgeqrf_nb(M).
            (size to be reduced in upcoming versions).

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

   #define  a_ref(a_1,a_2) ( a+(a_2)*(*lda) + (a_1))
   #define da_ref(a_1,a_2) (da+(a_2)*ldda   + (a_1))
   #define min(a,b)  (((a)<(b))?(a):(b))
   #define max(a,b)  (((a)>(b))?(a):(b))

   int *m = &m_;
   int *n = &n_;
   int *lda = &lda_;
   int *lwork = &lwork_;

   int i, k, lddwork, old_i, old_ib;
   int nbmin, nx, ib, ldda;

   /* Function Body */
   *info = 0;
   int nb = magma_get_sgeqrf_nb(*m);
   
   int lwkopt = *n * nb;
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

   float *dwork = da + (*m)*(*n);

   static cudaStream_t stream[2];
   cudaStreamCreate(&stream[0]);
   cudaStreamCreate(&stream[1]);

   ldda = *m;
   nbmin = 2;
   nx = 192;
   lddwork = *n;

   if (nb >= nbmin && nb < k && nx < k) {
      /* Use blocked code initially */
      cudaMemcpy2DAsync(da_ref(0,nb),  ldda *sizeof(float),
                         a_ref(0,nb), (*lda)*sizeof(float),
                        sizeof(float)*(*m), (*n-nb), 
                        cudaMemcpyHostToDevice,stream[0]);

      for (i = 0; i < k-nx; i += nb) {
	ib = min(k-i, nb);
	if (i>0){
	  cudaMemcpy2DAsync(  a_ref(i,i), (*lda)*sizeof(float),
			      da_ref(i,i), ldda *sizeof(float),
			      sizeof(float)*(*m-i), ib,
			      cudaMemcpyDeviceToHost,stream[1]);
	  
	  cudaMemcpy2DAsync(  a_ref(0,i), (*lda)*sizeof(float),
			      da_ref(0,i), ldda *sizeof(float),
			      sizeof(float)*i, ib,
			      cudaMemcpyDeviceToHost,stream[0]);
	  
	  /* Apply H' to A(i:m,i+2*ib:n) from the left */
	  magma_slarfb('F','C', *m-old_i, *n-old_i-2*old_ib, old_ib,
		       da_ref(old_i, old_i), ldda, dwork, lddwork, 
		       da_ref(old_i, old_i+2*old_ib), ldda, 
		       dwork+old_ib, lddwork);
	}

	cudaStreamSynchronize(stream[1]);
	int rows = *m-i;

	sgeqrf_(&rows, &ib, a_ref(i,i), lda, tau+i, work, lwork, info);

	/* Form the triangular factor of the block reflector   
	   H = H(i) H(i+1) . . . H(i+ib-1) */
	slarft_("F", "C", &rows, &ib, a_ref(i,i), lda, tau+i,
		work, &ib);
	spanel_to_q('U', ib, a_ref(i,i), *lda, work+ib*ib); 
	cublasSetMatrix(rows, ib, sizeof(float), 
			a_ref(i,i), *lda, da_ref(i,i), ldda);
	sq_to_panel('U', ib, a_ref(i,i), *lda, work+ib*ib);

	if (i + ib < *n) {
	  cublasSetMatrix(ib, ib, sizeof(float), work, ib, dwork, lddwork);

	  if (i+ib < k-nx)
	    /* Apply H' to A(i:m,i+ib:i+2*ib) from the left */   
	    magma_slarfb('F','C', rows, ib, ib, da_ref(i,i), ldda, dwork,
			 lddwork, da_ref(i,i+ib), ldda, dwork+ib, lddwork);
	  else 
	    magma_slarfb('F','C',rows, *n-i-ib, ib, da_ref(i,i), ldda, dwork,
			 lddwork, da_ref(i,i+ib), ldda, dwork+ib, lddwork);
       
	  old_i = i;
	  old_ib = ib;
	}
      }  
   } else {
     i = 0;
   }
   
   /* Use unblocked code to factor the last or only block. */
   if (i < k) {
      ib = *n-i;
      if (i!=0)
	cublasGetMatrix(*m, ib, sizeof(float),
			da_ref(0,i), ldda, a_ref(0,i), *lda);
      int rows = *m-i;
      sgeqrf_(&rows, &ib, a_ref(i,i), lda, tau+i, work, lwork, info);
   }
   return 0; 
  
   /* End of MAGMA_SGEQRF */

} /* magma_sgeqrf */

#undef  a_ref
#undef da_ref
#undef min
#undef max

