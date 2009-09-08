/*
    -- MAGMA (version 0.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       June 2009
*/

#include "cuda_runtime_api.h"
#include "cublas.h"
#include "magma.h"
#include <stdio.h>

int 
magma_dgeqrf(int *m, int *n, double *a, int  *lda,  double  *tau,
             double *work, int *lwork, double *da, int *info )
{
/*  -- MAGMA (version 0.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       June 2009

    Purpose   
    =======   

    DGEQRF computes a QR factorization of a real M-by-N matrix A:   
    A = Q * R.   

    Arguments   
    =========   

    M       (input) INTEGER   
            The number of rows of the matrix A.  M >= 0.   

    N       (input) INTEGER   
            The number of columns of the matrix A.  N >= 0.   

    A       (input/output) DOUBLE array, dimension (LDA,N)   
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

    TAU     (output) DOUBLE array, dimension (min(M,N))   
            The scalar factors of the elementary reflectors (see Further   
            Details).   

    WORK    (workspace/output) DOUBLE array, dimension (MAX(1,LWORK))   
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.   

	    Higher performance is achieved if WORK is in pinned memory, e.g.
            allocated using cudaMallocHost.

    LWORK   (input) INTEGER   
            The dimension of the array WORK.  LWORK >= N*NB, 
            where NB can be obtained through magma_get_dgeqrf_nb(M).   

            If LWORK = -1, then a workspace query is assumed; the routine   
            only calculates the optimal size of the WORK array, returns   
            this value as the first entry of the WORK array, and no error   
            message related to LWORK is issued.

    DA      (workspace)  DOUBLE array on the GPU, dimension N*(M + NB), 
            where NB can be obtained through magma_get_dgeqrf_nb(M).
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

   int i, k, lddwork, old_i, old_ib;
   int nbmin, nx, ib, ldda;

   /* Function Body */
   *info = 0;
   int nb = magma_get_dgeqrf_nb(*m);
   
   int lwkopt = *n * nb;
   work[0] = (double) lwkopt;
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

   double *dwork = da + (*m)*(*n);

   static cudaStream_t stream[2];
   cudaStreamCreate(&stream[0]);
   cudaStreamCreate(&stream[1]);

   ldda = *m;
   nbmin = 2;
   nx = 192;
   lddwork = *n;

   if (nb >= nbmin && nb < k && nx < k) {
      /* Use blocked code initially */
      cudaMemcpy2DAsync(da_ref(0,nb),  ldda *sizeof(double),
                         a_ref(0,nb), (*lda)*sizeof(double),
                        sizeof(double)*(*m), (*n-nb), 
                        cudaMemcpyHostToDevice,stream[0]);

      for (i = 0; i < k-nx; i += nb) {
	ib = min(k-i, nb);
	if (i>0){
	  cudaMemcpy2DAsync(  a_ref(i,i), (*lda)*sizeof(double),
			      da_ref(i,i), ldda *sizeof(double),
			      sizeof(double)*(*m-i), ib,
			      cudaMemcpyDeviceToHost,stream[1]);
	  cudaMemcpy2DAsync(  a_ref(0,i), (*lda)*sizeof(double),
			      da_ref(0,i), ldda *sizeof(double),
			      sizeof(double)*i, ib,
			      cudaMemcpyDeviceToHost,stream[0]);

	  /* Apply H' to A(i:m,i+2*ib:n) from the left */
	  magma_dlarfb(*m-old_i, *n-old_i-2*old_ib, &old_ib, 
		       da_ref(old_i, old_i), &ldda, dwork, &lddwork, 
		       da_ref(old_i, old_i+2*old_ib), &ldda, 
		       dwork+old_ib, &lddwork);
	}

	cudaStreamSynchronize(stream[1]);
	int rows = *m-i;
	dgeqrf_(&rows, &ib, a_ref(i,i), lda, tau+i, work, lwork, info);
	/* Form the triangular factor of the block reflector   
	   H = H(i) H(i+1) . . . H(i+ib-1) */
	dlarft_("F", "C", &rows, &ib, a_ref(i,i), lda, tau+i,
		work, &ib);
	dpanel_to_q(ib, a_ref(i,i), *lda, work+ib*ib); 
	cublasSetMatrix(rows, ib, sizeof(double), 
			a_ref(i,i), *lda, da_ref(i,i), ldda);
	dq_to_panel(ib, a_ref(i,i), *lda, work+ib*ib);

	if (i + ib < *n) {
	  cublasSetMatrix(ib, ib, sizeof(double), work, ib, dwork, lddwork);

	  if (i+nb < k-nx)
	    /* Apply H' to A(i:m,i+ib:i+2*ib) from the left */
	    magma_dlarfb(rows, ib, &ib, da_ref(i,i), &ldda, dwork,
			 &lddwork, da_ref(i,i+ib), &ldda, dwork+ib, &lddwork);
	  else 
	    magma_dlarfb(rows, *n-i-ib, &ib, da_ref(i,i), &ldda, dwork,
			 &lddwork, da_ref(i,i+ib), &ldda, dwork+ib, &lddwork);
       
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
	cublasGetMatrix(*m, ib, sizeof(double),
			da_ref(0,i), ldda, a_ref(0,i), *lda);
      int rows = *m-i;
      dgeqrf_(&rows, &ib, a_ref(i,i), lda, tau+i, work, lwork, info);
   }
   return 0; 
  
/*     End of MAGMA_DGEQRF */

} /* magma_dgeqrf */

#undef  a_ref
#undef da_ref
#undef min
#undef max

