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

extern "C" int sorg2r_(int*, int*, int*, float*, int*, float*, float*, int*);
extern "C" void magma_slaset(int m, int n, float *A, int lda);



extern "C" int
magma_sorgqr_gpu(int *m, int *n, int *k, float *da, int *ldda, 
		 float *tau, float *dwork, int *info)
{
/*  -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

    Purpose   
    =======   

    SORGQR generates an M-by-N real matrix Q with orthonormal columns,   
    which is defined as the first N columns of a product of K elementary   
    reflectors of order M   

          Q  =  H(1) H(2) . . . H(k)   

    as returned by SGEQRF_GPU2.   

    Arguments   
    =========   

    M       (input) INTEGER   
            The number of rows of the matrix Q. M >= 0.   

    N       (input) INTEGER   
            The number of columns of the matrix Q. M >= N >= 0.   

    K       (input) INTEGER   
            The number of elementary reflectors whose product defines the   
            matrix Q. N >= K >= 0.   

    DA      (input/output) REAL array A on the GPU device, dimension (LDDA,N).   
            On entry, the i-th column must contain the vector which   
            defines the elementary reflector H(i), for i = 1,2,...,k, as   
            returned by SGEQRF in the first k columns of its array   
            argument A.   
            On exit, the M-by-N matrix Q.

    LDDA    (input) INTEGER   
            The first dimension of the array A. LDDA >= max(1,M).   

    TAU     (input) REAL array, dimension (K)   
            TAU(i) must contain the scalar factor of the elementary   
            reflector H(i), as returned by SGEQRF.   

    DWORK   (input) REAL work space array on the GPU device.
            This must be the 8th argument of magma_sgeqrf_gpu2.

    INFO    (output) INTEGER   
            = 0:  successful exit   
            < 0:  if INFO = -i, the i-th argument has an illegal value   
    =====================================================================    */
  
    #define da_ref(a_1,a_2) (da+(a_2)*(*ldda) + (a_1)) 
    #define t_ref(a_1)      (dwork+(a_1))
    #define min(a,b)        (((a)<(b))?(a):(b))
    #define max(a,b)        (((a)>(b))?(a):(b))
    
    int  i__1, i__2, i__3;
    static int i, ib, nb, ki, kk, iinfo;
    int lddwork = min(*m, *n);

    static cudaStream_t stream[2];
    cudaStreamCreate(&stream[0]);
    cudaStreamCreate(&stream[1]);

    nb = magma_get_sgeqrf_nb(*m);

    *info = 0;
    if (*m < 0) {
	*info = -1;
    } else if (*n < 0 || *n > *m) {
	*info = -2;
    } else if (*k < 0 || *k > *n) {
	*info = -3;
    } else if (*ldda < max(1,*m)) {
	*info = -5;
    } 
    if (*info != 0 || *n <= 0)
      return 0;

    if (nb >= 2 && nb < *k) 
      {
	/*  Use blocked code after the last block.   
	    The first kk columns are handled by the block method. */ 
	ki = (*k - nb - 1) / nb * nb;
	kk = min(*k, ki + nb);
	
	/* Set A(1:kk,kk+1:n) to zero. */ 
        magma_slaset(kk, *n-kk, da_ref(0,kk), *ldda);
      }
    else 
      kk = 0;
 
    /* Allocate work space on CPU in pinned memory */
    int lwork = (*n+*m) * nb;
    if (kk < *n)
      lwork = max(lwork, *n *nb + (*m-kk)*(*n-kk));
    float *work, *panel;
    cudaMallocHost( (void**)&work,  (lwork)*sizeof(float) );
    panel = work + (*n) * nb;

    /* Use unblocked code for the last or only block. */
    if (kk < *n) 
      {
	i__1 = *m - kk;
	i__2 = *n - kk;
	i__3 = *k - kk;
	cublasGetMatrix(i__1, i__2, sizeof(float),
			da_ref(kk, kk), *ldda, panel, i__1);
	sorg2r_(&i__1, &i__2, &i__3, panel, &i__1, &tau[kk], work, &iinfo);

	cublasSetMatrix(i__1, i__2, sizeof(float),
			panel, i__1, da_ref(kk, kk), *ldda);
      }
    
    if (kk > 0) 
      {
	/* Use blocked code */
	for (i = ki; i >= 0; i-=nb) 
	  {
	    ib = min(nb, *k - i);

	    /* Send current panel to the CPU for update */ 
	    i__2 = *m - i;
	    cudaMemcpy2DAsync(panel,         i__2  * sizeof(float), 
			      da_ref(i,i), (*ldda) * sizeof(float), 
			      sizeof(float)*i__2, ib,
			      cudaMemcpyDeviceToHost,stream[1]);

	    if (i + ib < *n) 
	      {
		/* Apply H to A(i:m,i+ib:n) from the left */
		i__3 = *n - i - ib;
		magma_slarfb('N', 'F', 'C', i__2, i__3, &ib, da_ref(i, i), ldda,
			     t_ref(i), &lddwork, da_ref(i, i+ib), ldda, 
			     dwork + 2*lddwork*nb, &lddwork);
	      }
   
	    /* Apply H to rows i:m of current block on the CPU */
	    cudaStreamSynchronize(stream[1]);
	    sorg2r_(&i__2, &ib, &ib, panel, &i__2, &tau[i], work, &iinfo);
	    cudaMemcpy2DAsync(da_ref(i,i), (*ldda) * sizeof(float),
			      panel,         i__2  * sizeof(float),
			      sizeof(float)*i__2, ib,
                              cudaMemcpyHostToDevice,stream[2]);

	    /* Set rows 1:i-1 of current block to zero */
            i__2 = i + ib;
	    magma_slaset(i, i__2 - i, da_ref(0,i), *ldda);
	  }
      }
    cublasFree(work);

    return 0;

    /* End of MAGMA_SORGQR_GPU */
} /* magma_sorgqr_gpu */

#undef da_ref
#undef t_ref
#undef min
#undef max

