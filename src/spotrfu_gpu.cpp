/*
    -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010
*/

#include <cuda_runtime_api.h>
#include <cublas.h>
#include "magma.h"
#include "magmablas.h"

extern "C" int 
magma_spotrfu_gpu(char *uplo, int *n, float *a, int *lda, float *work, 
		 int *info)
{
/*  -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

    Purpose   
    =======   

    SPOTRFU updates the Cholesky factorization of a real symmetric   
    positive definite matrix A to the factorization of an updated matrix
    obtained by adding NN rows (and columns) to A.   

    The factorization has the form   
       A = U**T * U,  if UPLO = 'U', or   
       A = L  * L**T,  if UPLO = 'L',   
    where U is an upper triangular matrix and L is lower triangular.   

    This is the block version of the algorithm, calling Level 3 BLAS.   

    Arguments   
    =========   

    UPLO    (input) CHARACTER*1   
            = 'U':  Upper triangle of A is stored;   
            = 'L':  Lower triangle of A is stored.   

    N       (input) INTEGER   
            The order of the original matrix A.  N >= 0.   

    NN      (input) INTEGER
            The number of added row (and columns). NN >=0.

    A       (input/output) REAL array on the GPU, dimension (LDA,N+NN)   
            On entry, the already factored N x N matrix, extended by NN
            new rows (and columns). If UPLO = 'U', the leading   
            (N+NN)-by-(N+NN) upper triangular part of A contains the upper   
            triangular part of the matrix A, and the strictly lower   
            triangular part of A is not referenced.  If UPLO = 'L', the   
            leading (N+NN)-by-(N+NN) lower triangular part of A contains 
            the lower triangular part of the matrix A, and the strictly 
            upper triangular part of A is not referenced.   

            On exit, if INFO = 0, the factor U or L from the Cholesky   
            factorization A = U**T*U or A = L*L**T.   

    LDA     (input) INTEGER   
            The leading dimension of the array A.  LDA >= max(1,N+NN).
            To benefit from coalescent memory accesses LDA must be
            dividable by 16.

    WORK    (workspace) REAL array, dimension at least (NN, NN)
            work array allocated with cudaMallocHost.

    INFO    (output) INTEGER   
            = 0:  successful exit   
            < 0:  if INFO = -i, the i-th argument had an illegal value   
            > 0:  if INFO = i, the leading minor of order i is not   
                  positive definite, and the factorization could not be   
                  completed.   

    =====================================================================    */

    #define a_ref(a_1,a_2) (a+(a_2)*a_dim1 + a_1)
    #define min(a,b)  (((a)<(b))?(a):(b))
    #define max(a,b)  (((a)>(b))?(a):(b))

    /* Table of constant values */
    static float c_b13 = -1.f;
    static float c_b14 = 1.f;
    
    /* System generated locals */
    int a_dim1, a_offset, i__1, i__2, i__3, i__4;
    /* Local variables */
    static int j;

    long int upper = lsame_(uplo, "U");
    *info = 0;
    if (! upper && ! lsame_(uplo, "L")) {
      *info = -1;
    } else if (*n < 0) {
      *info = -2;
    } else if (*lda < max(1,*n)) {
      *info = -4;
    }
    if (*info != 0)
      return 0;

    static int jb;

    static cudaStream_t stream[2];
    cudaStreamCreate(&stream[0]);
    cudaStreamCreate(&stream[1]);

    int n2 = (*n)*(*n);
    cublasStatus status;

    a_dim1 = *lda;
    a_offset = 1 + a_dim1 * 1;
    a -= a_offset;
    unsigned int timer = 0;

    int nb = magma_get_spotrf_nb(*n);

    if (nb <= 1 || nb >= *n) {
      /*  Use unblocked code. */
      cublasGetMatrix(*n, *n, sizeof(float), a + a_offset, *lda, work, *n);
      spotrf_(uplo, n, work, n, info);
      cublasSetMatrix(*n, *n, sizeof(float), work, *n, a + a_offset, *lda);
    } else {

        /* Use blocked code. */
	if (upper) {

            /* Compute the Cholesky factorization A = U'*U. */

	    i__1 = *n;
	    i__2 = nb;
	    for (j = 1; i__2 < 0 ? j >= i__1 : j <= i__1; j += i__2) {

               /* Update and factorize the current diagonal block and test   
                  for non-positive-definiteness. Computing MIN */
		i__3 = nb, i__4 = *n - j + 1;
		jb = min(i__3,i__4);
		i__3 = j - 1;
                cublasSsyrk('u', 't', jb, i__3, c_b13, a_ref(1,j),
                             *lda, c_b14, a_ref(j, j), *lda);
                cudaMemcpy2DAsync(work, jb*sizeof(float), a_ref(j,j), 
				  (*lda) * sizeof(float), 4*sizeof(float), 
				  jb, cudaMemcpyDeviceToHost,stream[1]);
		
		if (j + jb <= *n) {
                    /* Compute the current block row. */
		    i__3 = *n - j - jb + 1;
		    i__4 = j - 1;
                    cublasSgemm('T', 'N', jb, i__3, i__4,
                            c_b13, a_ref(1, j), *lda, a_ref(1, j + jb), *lda,
                            c_b14, a_ref(j, j + jb), *lda);
                 }
             
                 cudaStreamSynchronize(stream[1]);
                 spotrf_("Upper", &jb, work, &jb, info);
		 if (*info != 0) {
		   *info = *info + j - 1;
		   break;
		 }
                 cudaMemcpy2DAsync(a_ref(j,j), (*lda) * sizeof(float), work, 
				   jb*sizeof(float), sizeof(float)*jb, 
				   jb, cudaMemcpyHostToDevice,stream[0]);

                 if (j + jb <= *n)
                    cublasStrsm('L', 'U', 'T', 'N', jb, i__3,
                           c_b14, a_ref(j, j), *lda, a_ref(j, j + jb),*lda);
	    }
	} else {
	  /* Update the NxN Cholesky factorization A = L*L' to a
	     (N+NN) x (N+NN) matrix derived from A by adding NN rows. */
	  
	    cublasStrsm('R', 'L', 'T', 'N', *nn, *n, c_b14,
			a_ref(1, 1), *lda, a_ref(1 + *n, 1),*lda);
	    cublasSsyrk('l', 'n', *nn, *n, c_b13, a_ref(1 + *n,1),
			*lda, c_b14, a_ref(1 + *n, 1 + *n), *lda);
	    cudaMemcpy2DAsync(work, (*nn)*sizeof(float), a_ref(1+*n,1+*n),
			      (*lda) * sizeof(float), sizeof(float)*(*nn),
			      *nn, cudaMemcpyDeviceToHost,stream[1]);
	    cudaStreamSynchronize(stream[1]);
	    spotrf_("Lower", nn, work, nn, info);
	    cudaMemcpy2DAsync(a_ref(1+*n,1+*n), (*lda) * sizeof(float), work,
			      (*nn)*sizeof(float), sizeof(float)*(*nn),
			      *nn, cudaMemcpyHostToDevice,stream[0]);
	}
    }
    return 0;

/*     End of MAGMA_SPOTRFU_GPU */

} /* magma_spotrfu_gpu */

#undef a_ref
#undef min
#undef max
