/*
    -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

       @precisions normal z -> s d c

*/

#include <cuda_runtime_api.h>
#include <cublas.h>
#include "magma.h"
#include "magmablas.h"

extern "C" int 
magma_zpotrf_gpu(char uplo, magma_int_t n, double2 *a, magma_int_t lda, 
		 int *info)
{
/*  -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

    Purpose   
    =======   

    ZPOTRF computes the Cholesky factorization of a real hemmetric   
    positive definite matrix A.   

    The factorization has the form   
       A = U\*\*H * U,  if UPLO = 'U', or   
       A = L  * L\*\*H,  if UPLO = 'L',   
    where U is an upper triangular matrix and L is lower triangular.   

    This is the block version of the algorithm, calling Level 3 BLAS.   

    Arguments   
    =========   

    UPLO    (input) CHARACTER*1   
            = 'U':  Upper triangle of A is stored;   
            = 'L':  Lower triangle of A is stored.   

    N       (input) INTEGER   
            The order of the matrix A.  N >= 0.   

    A       (input/output) COMPLEX_16 array on the GPU, dimension (LDA,N)   
            On entry, the hemmetric matrix A.  If UPLO = 'U', the leading   
            N-by-N upper triangular part of A contains the upper   
            triangular part of the matrix A, and the strictly lower   
            triangular part of A is not referenced.  If UPLO = 'L', the   
            leading N-by-N lower triangular part of A contains the lower   
            triangular part of the matrix A, and the strictly upper   
            triangular part of A is not referenced.   

            On exit, if INFO = 0, the factor U or L from the Cholesky   
            factorization A = U\*\*H*U or A = L*L\*\*H.   

    LDA     (input) INTEGER   
            The leading dimension of the array A.  LDA >= max(1,N).
            To benefit from coalescent memory accesses LDA must be
            dividable by 16.

    INFO    (output) INTEGER   
            = 0:  successful exit   
            < 0:  if INFO = -i, the i-th argument had an illegal value   
            > 0:  if INFO = i, the leading minor of order i is not   
                  positive definite, and the factorization could not be   
                  completed.   

    =====================================================================   */

    /* Test the input parameters.   

       Parameter adjustments */

    #define a_ref(a_1,a_2) (a+(a_2)*a_dim1 + a_1)
    #define min(a,b)  (((a)<(b))?(a):(b))
    #define max(a,b)  (((a)>(b))?(a):(b))

    char uplo_[2] = {uplo, 0};

    /* Table of constant values */
    double2 c_one = MAGMA_Z_ONE;
    double2 c_neg_one = MAGMA_Z_NEG_ONE;
    
    /* System generated locals */
    int a_dim1, a_offset, i__1, i__2, i__3, i__4;
    /* Local variables */
    static int j;

    long int upper = lapackf77_lsame(uplo_, "U");
    *info = 0;
    if (! upper && ! lapackf77_lsame(uplo_, "L")) {
      *info = -1;
    } else if (n < 0) {
      *info = -2;
    } else if (lda < max(1,n)) {
      *info = -4;
    }
    if (*info != 0)
      return 0;

    static int jb;

    static cudaStream_t stream[2];
    cudaStreamCreate(&stream[0]);
    cudaStreamCreate(&stream[1]);

    a_dim1 = lda;
    a_offset = 1 + a_dim1 * 1;
    a -= a_offset;

    int nb = magma_get_zpotrf_nb(n);

    double2 *work;
    cudaMallocHost( (void**)&work,  nb*nb*sizeof(double2) );

    if (nb <= 1 || nb >= n) {
      /*  Use unblocked code. */
      cublasGetMatrix(n, n, sizeof(double2), a + a_offset, lda, work, n);
      lapackf77_zpotrf(uplo_, &n, work, &n, info);
      cublasSetMatrix(n, n, sizeof(double2), work, n, a + a_offset, lda);
    } else {

        /* Use blocked code. */
	if (upper) {

            /* Compute the Cholesky factorization A = U'*U. */

	    i__1 = n;
	    i__2 = nb;
	    for (j = 1; i__2 < 0 ? j >= i__1 : j <= i__1; j += i__2) {

               /* Update and factorize the current diagonal block and test   
                  for non-positive-definiteness. Computing MIN */
		i__3 = nb, i__4 = n - j + 1;
		jb = min(i__3,i__4);
		i__3 = j - 1;
                cublasZherk('u', 't', jb, i__3, -1.0, a_ref(1,j),
                             lda, 1.0, a_ref(j, j), lda);
                cudaMemcpy2DAsync(work, jb*sizeof(double2), a_ref(j,j), 
				  (lda) * sizeof(double2), 4*sizeof(double2), 
				  jb, cudaMemcpyDeviceToHost,stream[1]);
		
		if (j + jb <= n) {
                    /* Compute the current block row. */
		    i__3 = n - j - jb + 1;
		    i__4 = j - 1;
                    cublasZgemm('T', 'N', jb, i__3, i__4,
                            c_neg_one, a_ref(1, j), lda, a_ref(1, j + jb), lda,
                            c_one, a_ref(j, j + jb), lda);
                 }
             
                 cudaStreamSynchronize(stream[1]);
                 lapackf77_zpotrf("Upper", &jb, work, &jb, info);
		 if (*info != 0) {
		   *info = *info + j - 1;
		   break;
		 }
                 cudaMemcpy2DAsync(a_ref(j,j), (lda) * sizeof(double2), work, 
				   jb*sizeof(double2), sizeof(double2)*jb, 
				   jb, cudaMemcpyHostToDevice,stream[0]);

                 if (j + jb <= n)
                    cublasZtrsm('L', 'U', 'T', 'N', jb, i__3,
                           c_one, a_ref(j, j), lda, a_ref(j, j + jb),lda);
	    }
	} else {
            //=========================================================
            // Compute the Cholesky factorization A = L*L'.
	    i__2 = n;
	    i__1 = nb;
	    for (j = 1; i__1 < 0 ? j >= i__2 : j <= i__2; j += i__1) {
                //  Update and factorize the current diagonal block and test   
                //  for non-positive-definiteness. Computing MIN 
		i__3 = nb, i__4 = n - j + 1;
		jb = min(i__3,i__4);
		i__3 = j - 1;
                cublasZherk('l', 'n', jb, i__3, -1.0, a_ref(j,1), 
                             lda, 1.0, a_ref(j, j), lda);
                cudaMemcpy2DAsync(work, jb*sizeof(double2), a_ref(j,j), 
				  (lda) * sizeof(double2), sizeof(double2)*jb, 
				  jb, cudaMemcpyDeviceToHost,stream[1]);

                if (j + jb <= n) {
                    i__3 = n - j - jb + 1;
                    i__4 = j - 1;
                    cublasZgemm('N', 'T', i__3, jb, i__4,
                            c_neg_one, a_ref(j + jb, 1), lda, a_ref(j, 1), lda,
                            c_one, a_ref(j + jb, j), lda);
                }

                cudaStreamSynchronize(stream[1]);
	        lapackf77_zpotrf("Lower", &jb, work, &jb, info);
		if (*info != 0) {
                  *info = *info + j - 1;
                  break;
                }
	        cudaMemcpy2DAsync(a_ref(j,j), (lda) * sizeof(double2), work, 
				  jb*sizeof(double2), sizeof(double2)*jb, 
				  jb, cudaMemcpyHostToDevice,stream[0]);
	        
		if (j + jb <= n)
		  cublasZtrsm('R', 'L', 'T', 'N', i__3, jb, c_one, 
		  //magmablas_ztrsm('R', 'L', 'T', 'N', i__3, jb, c_one,
				  a_ref(j, j), lda, a_ref(j + jb, j),lda);
	    }

	}
    }

    cublasFree(work);
    return 0;

    /* End of MAGMA_ZPOTRF_GPU */

} /* magma_zpotrf_gpu */

#undef a_ref
#undef min
#undef max
