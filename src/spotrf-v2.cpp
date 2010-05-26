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

#define cublasStrsm magmablas_strsm
#define cublasSgemm magmablas_sgemm

extern "C" int 
magma_spotrf2(char *uplo, int *n, float *a, int *lda, int *info)
{
/*  -- MAGMA (version 0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2009

    Purpose   
    =======   

    SPOTRF computes the Cholesky factorization of a real symmetric   
    positive definite matrix A. This version does not require work
    space on the GPU passed as input. GPU memory is allocated in the
    routine.

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

    A       (input/output) REAL array, dimension (LDA,N)   
            On entry, the symmetric matrix A.  If UPLO = 'U', the leading   
            N-by-N upper triangular part of A contains the upper   
            triangular part of the matrix A, and the strictly lower   
            triangular part of A is not referenced.  If UPLO = 'L', the   
            leading N-by-N lower triangular part of A contains the lower   
            triangular part of the matrix A, and the strictly upper   
            triangular part of A is not referenced.   

            On exit, if INFO = 0, the factor U or L from the Cholesky   
            factorization A = U**T*U or A = L*L**T.   

            Higher performance is achieved if A is in pinned memory, e.g.
            allocated using cudaMallocHost.

    LDA     (input) INTEGER   
            The leading dimension of the array A.  LDA >= max(1,N).   

    WORK    (workspace) REAL array on the GPU, dimension (N, N)
            (size to be reduced in upcoming versions).

    INFO    (output) INTEGER   
            = 0:  successful exit   
            < 0:  if INFO = -i, the i-th argument had an illegal value 
                  if INFO = -5, the GPU memory allocation failed 
            > 0:  if INFO = i, the leading minor of order i is not   
                  positive definite, and the factorization could not be   
                  completed.   

    =====================================================================    */

    #define  a_ref(a_1,a_2) (a+(a_2)*a_dim1 + a_1)
    #define da_ref(a_1,a_2) (work+(a_2)*ldda + a_1)
    #define min(a,b)  (((a)<(b))?(a):(b))
    #define max(a,b)  (((a)>(b))?(a):(b))
    
    /* System generated locals */
    int a_dim1, a_offset, i__3, i__4, ldda;
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

    static cudaStream_t stream[3];
    cudaStreamCreate(&stream[0]);
    cudaStreamCreate(&stream[1]);
    cudaStreamCreate(&stream[2]);

    cublasStatus status;

    a_dim1 = *lda;
    ldda   = *n;
    
    if (ldda %32 != 0)
      ldda = (ldda/32)*32 + 32;

    float *work;
    status = cublasAlloc((*n)*ldda, sizeof(float), (void**)&work);
    if (status != CUBLAS_STATUS_SUCCESS) {
      *info = -5;
      return 0;
    }

    a_offset = 1 + a_dim1 * 1;
    a    -= a_offset;
    work -= (1 + (*n));

    int nb = magma_get_spotrf_nb(*n);

    if (nb <= 1 || nb >= *n) {
	spotrf_(uplo, n, a_ref(1, 1), lda, info);
    } else {

        /* Use hybrid blocked code. */
	if (upper) {
            /* Compute the Cholesky factorization A = U'*U. */
	    for (j = 1; j <= *n; j += nb) {
               /* Update and factorize the current diagonal block and test   
                  for non-positive-definiteness. Computing MIN */
		i__4 = *n - j + 1;
		jb = min(nb,i__4);
		i__3 = j - 1;
		cublasSetMatrix(jb, i__4, sizeof(float),
                                a_ref(j,j), *lda, da_ref(j,j), ldda);
                cublasSsyrk('u', 't', jb, i__3, -1.f, da_ref(1,j),
                             ldda, 1.f, da_ref(j, j), ldda);
                cudaMemcpy2DAsync(  a_ref(1,j), (*lda)*sizeof(float), 
				   da_ref(1,j), ldda *sizeof(float), 
				    sizeof(float)*(j+jb-1), jb,
				    cudaMemcpyDeviceToHost,stream[1]);
		
		if (j + jb <= *n) {
		  i__3 = *n - j - jb + 1;
		  i__4 = j - 1;
		  cublasSgemm('T', 'N', jb, i__3, i__4,
			  -1.f, da_ref(1, j), ldda, da_ref(1, j + jb), ldda,
			  1.f, da_ref(j, j + jb), ldda);
		}
             
		cudaStreamSynchronize(stream[1]);
		spotrf_("Upper", &jb, a_ref(j,j), lda, info);
		if (*info != 0) {
		  *info = *info + j - 1;
		  break;
		}
		cudaMemcpy2DAsync(da_ref(j,j),  ldda * sizeof(float), 
				  a_ref( j,j), (*lda)* sizeof(float), 
				  sizeof(float)*jb, jb, 
				  cudaMemcpyHostToDevice,stream[0]);
		
		if (j + jb <= *n)
		  cublasStrsm('L', 'U', 'T', 'N', jb, i__3,
			      1.f, da_ref(j,j), ldda, da_ref(j, j+jb), ldda);
	    }
	} else {
            //=========================================================
            // Compute the Cholesky factorization A = L*L'.
	    for (j = 1; j <= *n; j += nb) {
                //  Update and factorize the current diagonal block and test   
                //  for non-positive-definiteness. Computing MIN 
		i__4 = *n - j + 1;
		jb = min(nb,i__4);
		i__3 = j - 1;
                cublasSetMatrix(i__4, jb, sizeof(float), 
				a_ref(j,j), *lda, da_ref(j,j), ldda);
                cublasSsyrk('l', 'n', jb, i__3, -1.f, da_ref(j,1), ldda, 
                            1.f, da_ref(j, j), ldda);
		/*
		cudaMemcpy2DAsync( a_ref(j,1), (*lda)*sizeof(float), 
				   da_ref(j,1),  ldda *sizeof(float), 
				   sizeof(float)*jb, j+jb-1, 
				   cudaMemcpyDeviceToHost,stream[1]);
		*/
		cudaMemcpy2DAsync( a_ref(j,j), (*lda)*sizeof(float),
                                   da_ref(j,j),  ldda *sizeof(float),
                                   sizeof(float)*jb, jb,
                                   cudaMemcpyDeviceToHost,stream[1]);
		cudaMemcpy2DAsync( a_ref(j,1), (*lda)*sizeof(float),
                                   da_ref(j,1),  ldda *sizeof(float),
                                   sizeof(float)*jb, j-1,
                                   cudaMemcpyDeviceToHost,stream[2]);

                if (j + jb <= *n) {
                    i__3 = *n - j - jb + 1;
                    i__4 = j - 1;
                    cublasSgemm('N', 'T', i__3, jb, i__4,
                            -1.f, da_ref(j + jb, 1), ldda, da_ref(j, 1), ldda,
                            1.f, da_ref(j + jb, j), ldda);
                }
		
                cudaStreamSynchronize(stream[1]);
	        spotrf_("Lower", &jb, a_ref(j, j), lda, info);
		if (*info != 0){
                  *info = *info + j - 1;
		  break;
		}
	        cudaMemcpy2DAsync(da_ref(j,j),  ldda * sizeof(float), 
				  a_ref( j,j), (*lda)* sizeof(float), 
				  sizeof(float)*jb, jb, 
				  cudaMemcpyHostToDevice,stream[0]);
	        
		if (j + jb <= *n)
		  cublasStrsm('R', 'L', 'T', 'N', i__3, jb, 1.f, 
			      da_ref(j, j), ldda, da_ref(j + jb, j), ldda);
	    }
	}
    }
    
    work += 1 + (*n);
    cublasFree(work);
    
    return 0;

/*     End of MAGMA_SPOTRF */

} /* magma_spotrf */

#undef a_ref
#undef da_ref
#undef min
#undef max

