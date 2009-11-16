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

int 
magma_cpotrf_gpu(char *uplo, int *n, float2 *a, int *lda, float2 *work, 
		 int *info)
{
/*  -- MAGMA (version 0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2009

    Purpose   
    =======   

    CPOTRF computes the Cholesky factorization of a complex symmetric   
    positive definite matrix A.   

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
            The order of the matrix A.  N >= 0.   

    A       (input/output) COMPLEX array on the GPU, dimension (LDA,N)   
            On entry, the symmetric matrix A.  If UPLO = 'U', the leading   
            N-by-N upper triangular part of A contains the upper   
            triangular part of the matrix A, and the strictly lower   
            triangular part of A is not referenced.  If UPLO = 'L', the   
            leading N-by-N lower triangular part of A contains the lower   
            triangular part of the matrix A, and the strictly upper   
            triangular part of A is not referenced.   

            On exit, if INFO = 0, the factor U or L from the Cholesky   
            factorization A = U**T*U or A = L*L**T.   

    LDA     (input) INTEGER   
            The leading dimension of the array A.  LDA >= max(1,N).   

    WORK    (workspace) COMPLEX array, dimension at least (nb, nb)
            where nb can be obtained through magma_get_cpotrf_nb(*n)
            Work array allocated with cudaMallocHost.

    INFO    (output) INTEGER   
            = 0:  successful exit   
            < 0:  if INFO = -i, the i-th argument had an illegal value   
            > 0:  if INFO = i, the leading minor of order i is not   
                  positive definite, and the factorization could not be   
                  completed.   

    =====================================================================   

       Test the input parameters.   

       Parameter adjustments */

    #define a_ref(a_1,a_2) (a+(a_2)*a_dim1 + a_1)
    #define min(a,b)  (((a)<(b))?(a):(b))
    #define max(a,b)  (((a)>(b))?(a):(b))

    /* Table of constant values */
    static int c__1 = 1;
    static int c_n1 = -1;
    static float2 cmone = {-1.0, 0.0};
    static float2 cone = {1.0, 0.0};
    
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

    int nb = magma_get_cpotrf_nb(*n);

    if (nb <= 1 || nb >= *n) {
      /*  Use unblocked code. */
      cublasGetMatrix(*n, *n, sizeof(float2), a + a_offset, *lda, work, *n);
      cpotrf_(uplo, n, work, n, info);
      cublasSetMatrix(*n, *n, sizeof(float2), work, *n, a + a_offset, *n);
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
                magmablas_cherk('u', 'c', jb, i__3, -1.f, a_ref(1,j),
                             *lda, 1.f, a_ref(j, j), *lda);
                cudaMemcpy2DAsync(work, jb*sizeof(float2), a_ref(j,j), 
				  (*lda) * sizeof(float2), 4*sizeof(float2), 
				  jb, cudaMemcpyDeviceToHost,stream[1]);
		
		if (j + jb <= *n) {
                    /* Compute the current block row. */
		    i__3 = *n - j - jb + 1;
		    i__4 = j - 1;
                    cublasCgemm('C', 'N', jb, i__3, i__4,
                            cmone, a_ref(1, j), *lda, a_ref(1, j + jb), *lda,
                            cone, a_ref(j, j + jb), *lda);
                 }
             
                 cudaStreamSynchronize(stream[1]);
                 cpotrf_("Upper", &jb, work, &jb, info);
		 if (*info != 0) {
		   *info = *info + j - 1;
		   break;
		 }
                 cudaMemcpy2DAsync(a_ref(j,j), (*lda) * sizeof(float2), work, 
				   jb*sizeof(float2), sizeof(float2)*jb, 
				   jb, cudaMemcpyHostToDevice,stream[0]);

                 if (j + jb <= *n)
                    magmablas_ctrsm('L', 'U', 'C', 'N', jb, i__3,
                           cone, a_ref(j, j), *lda, a_ref(j, j + jb),*lda);
	    }
	} else {
            //=========================================================
            // Compute the Cholesky factorization A = L*L'.
	    i__2 = *n;
	    i__1 = nb;
	    for (j = 1; i__1 < 0 ? j >= i__2 : j <= i__2; j += i__1) {
                //  Update and factorize the current diagonal block and test   
                //  for non-positive-definiteness. Computing MIN 
		i__3 = nb, i__4 = *n - j + 1;
		jb = min(i__3,i__4);
		i__3 = j - 1;
                magmablas_cherk('l', 'n', jb, i__3, -1.f, a_ref(j,1), 
                             *lda, 1.f, a_ref(j, j), *lda);
                cudaMemcpy2DAsync(work, jb*sizeof(float2), a_ref(j,j), 
				  (*lda) * sizeof(float2), sizeof(float2)*jb, 
				  jb, cudaMemcpyDeviceToHost,stream[1]);

                if (j + jb <= *n) {
                    i__3 = *n - j - jb + 1;
                    i__4 = j - 1;
                    cublasCgemm('N', 'C', i__3, jb, i__4,
                            cmone, a_ref(j + jb, 1), *lda, a_ref(j, 1), *lda,
                            cone, a_ref(j + jb, j), *lda);
                }

                cudaStreamSynchronize(stream[1]);
	        cpotrf_("Lower", &jb, work, &jb, info);
		if (*info != 0) {
                  *info = *info + j - 1;
                  break;
                }
	        cudaMemcpy2DAsync(a_ref(j,j), (*lda) * sizeof(float2), work, 
				  jb*sizeof(float2), sizeof(float2)*jb, 
				  jb, cudaMemcpyHostToDevice,stream[0]);
	        
		if (j + jb <= *n)
                    magmablas_ctrsm('R', 'L', 'C', 'N', i__3,
		          jb, cone, a_ref(j, j), *lda, a_ref(j + jb, j),*lda);
	    }

	}
    }
    return 0;

/*     End of MAGMA_CPOTRF_GPU */

} /* magma_cpotrf_gpu */

#undef a_ref
#undef min
#undef max
