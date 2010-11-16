/*
    -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

       @precisions normal z -> s d c

*/

#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include "magma.h"
#include "magmablas.h"

#define A(i, j)  (a   +(j)*lda  + (i))
#define dA(i, j) (work+(j)*ldda + (i))

extern "C" int 
magma_zpotrf(char uplo, magma_int_t n, cuDoubleComplex *a, magma_int_t lda, int *info)
{
/*  -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

    Purpose   
    =======   

    ZPOTRF computes the Cholesky factorization of a real hemmetric   
    positive definite matrix A. This version does not require work
    space on the GPU passed as input. GPU memory is allocated in the
    routine.

    The factorization has the form   
       A = U\*\*H * U,  if UPLO = 'U', or   
       A = L  * L\*\*H, if UPLO = 'L',   
    where U is an upper triangular matrix and L is lower triangular.   

    This is the block version of the algorithm, calling Level 3 BLAS.   

    Arguments   
    =========   

    UPLO    (input) CHARACTER*1   
            = 'U':  Upper triangle of A is stored;   
            = 'L':  Lower triangle of A is stored.   

    N       (input) INTEGER   
            The order of the matrix A.  N >= 0.   

    A       (input/output) COMPLEX_16 array, dimension (LDA,N)   
            On entry, the hemmetric matrix A.  If UPLO = 'U', the leading   
            N-by-N upper triangular part of A contains the upper   
            triangular part of the matrix A, and the strictly lower   
            triangular part of A is not referenced.  If UPLO = 'L', the   
            leading N-by-N lower triangular part of A contains the lower   
            triangular part of the matrix A, and the strictly upper   
            triangular part of A is not referenced.   

            On exit, if INFO = 0, the factor U or L from the Cholesky   
            factorization A = U\*\*H*U or A = L*L\*\*H.   

            Higher performance is achieved if A is in pinned memory, e.g.
            allocated using cudaMallocHost.

    LDA     (input) INTEGER   
            The leading dimension of the array A.  LDA >= max(1,N).   

    INFO    (output) INTEGER   
            = 0:  successful exit   
            < 0:  if INFO = -i, the i-th argument had an illegal value 
                  if INFO = -6, the GPU memory allocation failed 
            > 0:  if INFO = i, the leading minor of order i is not   
                  positive definite, and the factorization could not be   
                  completed.   

    =====================================================================    */

    #define min(a,b)  (((a)<(b))?(a):(b))
    #define max(a,b)  (((a)>(b))?(a):(b))

    /* Local variables */
    char uplo_[2] = {uplo, 0};
    int  ldda;
    static int j, jb;
    cuDoubleComplex c_one     = MAGMA_Z_ONE;
    cuDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;

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

    static cudaStream_t stream[3];
    cudaStreamCreate(&stream[0]);
    cudaStreamCreate(&stream[1]);
    cudaStreamCreate(&stream[2]);

    cublasStatus status;

    ldda   = ((n+31)/32)*32;
    
    cuDoubleComplex *work;
    status = cublasAlloc((n)*ldda, sizeof(cuDoubleComplex), (void**)&work);
    if (status != CUBLAS_STATUS_SUCCESS) {
	*info = -6;
	return 0;
    }

    int nb = magma_get_zpotrf_nb(n);

    if (nb <= 1 || nb >= n) {
	lapackf77_zpotrf(uplo_, &n, a, &lda, info);
    } else {

        /* Use hybrid blocked code. */
	if (upper) {
            /* Compute the Cholesky factorization A = U'*U. */
	    for (j=0; j<n; j += nb) {
                /* Update and factorize the current diagonal block and test   
                   for non-positive-definiteness. Computing MIN */
                jb = min(nb, (n-j));
		cublasSetMatrix(jb, (n-j), sizeof(cuDoubleComplex), 
                                A(j, j), lda, dA(j, j), ldda);

                cublasZherk(MagmaUpper, MagmaConjTrans, jb, j, 
                            -1.f, dA(0, j), ldda, 
                            1.f,  dA(j, j), ldda);

                cudaMemcpy2DAsync(  A(0, j), lda *sizeof(cuDoubleComplex), 
				   dA(0, j), ldda*sizeof(cuDoubleComplex), 
                                    sizeof(cuDoubleComplex)*(j+jb), jb,
				    cudaMemcpyDeviceToHost, stream[1]);
		
		if ( (j+jb) < n) {
                    cublasZgemm(MagmaConjTrans, MagmaNoTrans, 
                                jb, (n-j-jb), j,
                                c_neg_one, dA(0, j   ), ldda, 
                                           dA(0, j+jb), ldda,
                                c_one,     dA(j, j+jb), ldda);
		}
             
		cudaStreamSynchronize(stream[1]);
		lapackf77_zpotrf(MagmaUpperStr, &jb, A(j, j), &lda, info);
		if (*info != 0) {
		  *info = *info + j;
		  break;
		}
		cudaMemcpy2DAsync(dA(j, j), ldda * sizeof(cuDoubleComplex), 
				   A(j, j), lda  * sizeof(cuDoubleComplex), 
				  sizeof(cuDoubleComplex)*jb, jb, 
				  cudaMemcpyHostToDevice,stream[0]);
		
		if ( (j+jb) < n )
                  cublasZtrsm(MagmaLeft, MagmaUpper, MagmaConjTrans, MagmaNonUnit, 
                              jb, (n-j-jb),
                              c_one, dA(j, j   ), ldda, 
                                     dA(j, j+jb), ldda);
	    }
	} else {
            //=========================================================
            // Compute the Cholesky factorization A = L*L'.
	    for (j=0; j<n; j+=nb) {
                //  Update and factorize the current diagonal block and test   
                //  for non-positive-definiteness. Computing MIN 
		jb = min(nb, (n-j));
                cublasSetMatrix((n-j), jb, sizeof(cuDoubleComplex), 
				A(j, j), lda, dA(j, j), ldda);

                cublasZherk(MagmaLower, MagmaNoTrans, jb, j,
                            -1.f, dA(j, 0), ldda, 
                            1.f,  dA(j, j), ldda);
		/*
		cudaMemcpy2DAsync( A(j, 0), lda *sizeof(cuDoubleComplex), 
				   dA(j,0), ldda*sizeof(cuDoubleComplex), 
				   sizeof(cuDoubleComplex)*jb, j+jb, 
				   cudaMemcpyDeviceToHost,stream[1]);
		*/
		cudaMemcpy2DAsync( A(j,j),  lda *sizeof(cuDoubleComplex),
                                   dA(j,j), ldda*sizeof(cuDoubleComplex),
                                   sizeof(cuDoubleComplex)*jb, jb,
                                   cudaMemcpyDeviceToHost,stream[1]);
		cudaMemcpy2DAsync( A(j, 0),  lda *sizeof(cuDoubleComplex),
                                   dA(j, 0), ldda*sizeof(cuDoubleComplex),
                                   sizeof(cuDoubleComplex)*jb, j,
                                   cudaMemcpyDeviceToHost,stream[2]);

                if ( (j+jb) < n) {
                    cublasZgemm( MagmaNoTrans, MagmaConjTrans, 
                                 (n-j-jb), jb, j,
                                 c_neg_one, dA(j+jb, 0), ldda, 
                                            dA(j,    0), ldda,
                                 c_one,     dA(j+jb, j), ldda);
                }
		
                cudaStreamSynchronize(stream[1]);
	        lapackf77_zpotrf(MagmaLowerStr, &jb, A(j, j), &lda, info);
		if (*info != 0){
                    *info = *info + j;
                    break;
		}
	        cudaMemcpy2DAsync( dA(j, j), ldda*sizeof(cuDoubleComplex), 
				   A(j, j),  lda *sizeof(cuDoubleComplex), 
                                   sizeof(cuDoubleComplex)*jb, jb, 
                                   cudaMemcpyHostToDevice,stream[0]);
	        
		if ( (j+jb) < n)
                    cublasZtrsm(MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit, 
                                (n-j-jb), jb, 
                                c_one, dA(j,    j), ldda, 
                                       dA(j+jb, j), ldda);
	    }
	}
    }
    
    cublasFree(work);
    
    return 0;

    /* End of MAGMA_ZPOTRF */
} /* magma_zpotrf */

