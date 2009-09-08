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
magma_sgelqf(int *m, int *n, float *a, int *lda, float *tau, 
	     float *work, int *lwork, float *da, int *info)
{
/*  -- MAGMA (version 0.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       June 2009

    Purpose   
    =======   

    SGELQF computes an LQ factorization of a real M-by-N matrix A:   
    A = L * Q.   

    Arguments   
    =========   

    M       (input) INTEGER   
            The number of rows of the matrix A.  M >= 0.   

    N       (input) INTEGER   
            The number of columns of the matrix A.  N >= 0.   

    A       (input/output) REAL array, dimension (LDA,N)   
            On entry, the M-by-N matrix A.   
            On exit, the elements on and below the diagonal of the array   
            contain the m-by-min(m,n) lower trapezoidal matrix L (L is   
            lower triangular if m <= n); the elements above the diagonal,   
            with the array TAU, represent the orthogonal matrix Q as a   
            product of elementary reflectors (see Further Details).   

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
            The dimension of the array WORK.  LWORK >= max(1,M).   
            For optimum performance LWORK >= M*NB, where NB is the   
            optimal blocksize.   

            If LWORK = -1, then a workspace query is assumed; the routine   
            only calculates the optimal size of the WORK array, returns   
            this value as the first entry of the WORK array, and no error   
            message related to LWORK is issued.

    DA      (workspace)  REAL array on the GPU, dimension M*(N + NB),
            where NB can be obtained through magma_get_sgeqrf_nb(M).
            (size to be reduced in upcoming versions).

    INFO    (output) INTEGER   
            = 0:  successful exit   
            < 0:  if INFO = -i, the i-th argument had an illegal value   

    Further Details   
    ===============   

    The matrix Q is represented as a product of elementary reflectors   

       Q = H(k) . . . H(2) H(1), where k = min(m,n).   

    Each H(i) has the form   

       H(i) = I - tau * v * v'   

    where tau is a real scalar, and v is a real vector with   
    v(1:i-1) = 0 and v(i) = 1; v(i+1:n) is stored on exit in A(i,i+1:n),   
    and tau in TAU(i).   

    =====================================================================    */

    #define  a_ref(a_1,a_2) ( a+(a_2)*(*lda) + (a_1))
    #define da_ref(a_1,a_2) (da+(a_2)*ldda   + (a_1))
    #define min(a,b)  (((a)<(b))?(a):(b))
    #define max(a,b)  (((a)>(b))?(a):(b))
    
    int rows, cols, i, k, ib, nx, nbmin, iinfo, ldwork;
    long int lquery;

    /* Function Body */
    *info = 0;
    int nb = magma_get_sgelqf_nb(*m); 

    work[0] = (float) *m * nb;
    lquery = *lwork == -1;
    if (*m < 0) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < max(1,*m)) {
	*info = -4;
    } else if (*lwork < max(1,*m) && ! lquery) {
	*info = -7;
    }
    if (*info != 0) {
	return 0;
    } else if (lquery) {
	return 0;
    }

    /*  Quick return if possible */
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
        /*  Use blocked code initially */
        cudaMemcpy2DAsync(da_ref(nb,0),  ldda *sizeof(float),
			   a_ref(nb,0), (*lda)*sizeof(float),
			  sizeof(float)*(*m-nb), *n,
			  cudaMemcpyHostToDevice,stream[0]);

	for (i = 0; i < k-nx; i += nb) {
	    ib = min(k-i, nb);
	    if ( i>0 ) {
	      cudaMemcpy2DAsync(  a_ref(i,i), (*lda)*sizeof(float),
				  da_ref(i,i), ldda *sizeof(float),
				  sizeof(float)*ib, *n-i,
				  cudaMemcpyDeviceToHost,stream[1]);

	      cudaMemcpy2DAsync(  a_ref(i,0), (*lda)*sizeof(float),
				  da_ref(i,0), ldda *sizeof(float),
				  sizeof(float)*ib, i,
				  cudaMemcpyDeviceToHost,stream[0]);

	      /* Apply H to A(i+ib:m,i:n) from the right */
              rows = *m - old_i - 2*old_ib;
              cols = *n - old_i;
	      magma_slarfb( "R", "N", "F", "R", &rows, &cols, &old_ib, 
			    da_ref(old_i, old_i), ldda, dwork, &lddwork, 
			    da_ref(old_i + 2*old_ib, old_i), ldda, 
			    dwork+old_ib, &lddwork);
	    }

	    cudaStreamSynchronize(stream[1]);
	    /* Compute the LQ factorization of the current block   
	       A(i:i+ib-1,i:n) */
	    rows = *m - i - ib;
	    cols = *n - i;
	    sgelqf_(&ib, &cols, a_ref(i, i), lda, tau+i, work, lwork, &iinfo);

	    if (rows > 0) {
	      /* Form the triangular factor of the block reflector   
		 H = H(i) H(i+1) . . . H(i+ib-1) */
	      slarft_("F", "R", &cols, &ib, a_ref(i,i), lda, tau+i, work, &ib);
	      spanel_to_q('L', ib, a_ref(i,i), *lda, work+ib*ib);
	      cublasSetMatrix(rows, ib, sizeof(float),
			      a_ref(i,i), *lda, da_ref(i,i), ldda);
	      sq_to_panel('L', ib, a_ref(i,i), *lda, work+ib*ib);
	      
	      // Send the triangular part on the GPU
	      cublasSetMatrix(ib,ib,sizeof(float), work, ib, dwork, lddwork);
	      
	      if (i+ib < k-nx)
		/* Apply H to A(i+ib:m,i:n) from the right */
		magma_slarfb("R", "N", "F", "R", &ib, &cols, &ib,
			     da_ref(i, i), ldda, dwork, &lddwork,
			     da_ref(i + ib, i), ldda, dwork+ib, &lddwork);
	      else
		magma_slarfb_("R", "N", "F", "R", &rows, &cols, &ib, 
			      da_ref(i, i), ldda, dwork, &lddwork,
			      da_ref(i + ib, i), ldda, dwork+ib, &lddwork);
	    }
	}
    } else {
      i = 0;
    }
    
    /* Use unblocked code to factor the last or only block. */
    if (i < k) {
       rows = *m - i;
       cols = *n - i;
       if (i!=0)
	 cublasGetMatrix(rows, *n, sizeof(float),
			 da_ref(i,0), ldda, a_ref(i,0), *lda);

       sgelqf_(&rows, &cols, a_ref(i, i), lda, tau+i, work, lwork, &iinfo);
    }

    work[0] = (float) *m * nb;
    return 0;

    /*     End of MAGMA_SGELQF */

} /* magma_sgelqf */

#undef  a_ref
#undef da_ref
#undef min
#undef max
