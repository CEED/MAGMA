/*
    -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010
*/

#include "cuda.h"
#include "cublas.h"
#include "cuda_runtime_api.h"
#include "magma.h"
#include "magmablas.h"
#include <stdio.h>

extern "C" void
magmablas_stranspose2(float *, int, float *, int, int, int);

extern "C" void 
magmablas_spermute_long2(float *, int, int *, int, int);

extern "C" magma_int_t 
magma_sgetrf_gpu(magma_int_t m_, magma_int_t n_, float *a, magma_int_t lda_, magma_int_t *ipiv, magma_int_t *info)
{
/*  -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

    Purpose   
    =======   

    SGETRF computes an LU factorization of a general M-by-N matrix A   
    using partial pivoting with row interchanges.   

    The factorization has the form   
       A = P * L * U   
    where P is a permutation matrix, L is lower triangular with unit   
    diagonal elements (lower trapezoidal if m > n), and U is upper   
    triangular (upper trapezoidal if m < n).   

    This is the right-looking Level 3 BLAS version of the algorithm.   

    Arguments   
    =========   

    M       (input) INTEGER   
            The number of rows of the matrix A.  M >= 0.   

    N       (input) INTEGER   
            The number of columns of the matrix A.  N >= 0.   

    A       (input/output) REAL array on the GPU, dimension (LDA,N). 
            On entry, the M-by-N matrix to be factored.   
            On exit, the factors L and U from the factorization   
            A = P*L*U; the unit diagonal elements of L are not stored.   

    LDA     (input) INTEGER   
            The leading dimension of the array A.  LDA >= max(1,M).   

    IPIV    (output) INTEGER array, dimension (min(M,N))   
            The pivot indices; for 1 <= i <= min(M,N), row i of the   
            matrix was interchanged with row IPIV(i).

    INFO    (output) INTEGER   
            = 0:  successful exit   
            < 0:  if INFO = -i, the i-th argument had an illegal value
                  if INFO = -7, internal GPU memory allocation failed.   
            > 0:  if INFO = i, U(i,i) is exactly zero. The factorization   
                  has been completed, but the factor U is exactly   
                  singular, and division by zero will occur if it is used   
                  to solve a system of equations.   

    =====================================================================    */

#define inAT(i,j) (dAT + (i)*nb*ldda + (j)*nb)
#define max(a,b)  (((a)>(b))?(a):(b))
#define min(a,b)  (((a)<(b))?(a):(b))

    int *m = &m_;
    int *n = &n_;
    int *lda = &lda_;

    /* Function Body */

    *info = 0;
    int iinfo, nb = magma_get_sgetrf_nb(*m);

    if (*m < 0)
      *info = -1;
    else if (*n < 0)
      *info = -2;
    else if (*lda < max(1,*m))
      *info = -4;
    
    if (*info != 0)
      return 0;

    /* Quick return if possible */
    if (*m == 0 || *n == 0)
      return 0;

    int maxm, maxn, mindim = min(*m, *n);
    int i, rows, cols, s = mindim/nb, ldda, lddwork;
    float *dAT, *dA, *work;

    if (nb <= 1 || nb >= min(*m,*n)) {
      /* Use CPU code. */
      work = (float*)malloc(maxm * (*n) * sizeof(float));
      cublasGetMatrix(*m, *n, sizeof(float), a, *lda, work, maxm);
      sgetrf_(m, n, work, &maxm, ipiv, info);
      cublasSetMatrix(*m, *n, sizeof(float), work, maxm, a, *lda);
      free(work);
    } else {
      /* Use hybrid blocked code. */
      maxm = ((*m + 31)/32)*32;
      maxn = ((*n + 31)/32)*32;

      ldda    = maxn;
      lddwork = maxm;

      dAT = a;

      cublasStatus status;
      status = cublasAlloc(nb*maxm, sizeof(float), (void**)&dA);
      if (status != CUBLAS_STATUS_SUCCESS) {
        *info = -7;
        return 0;
      }
      
      if ((*m == *n) && (*m % 32 == 0) && (*lda%32 == 0))
	magmablas_sinplace_transpose( dAT, *lda, ldda );
      else {
	status = cublasAlloc(maxm*maxn, sizeof(float), (void**)&dAT);
	if (status != CUBLAS_STATUS_SUCCESS) {
	  *info = -7;
	  return 0;
	}
	magmablas_stranspose2( dAT, ldda, a, *lda, *m, *n );
      }
      
      cudaMallocHost( (void**)&work, maxm*nb*sizeof(float) );
      if (work == 0){
        *info = -7;
        return 0;
      }

      for( i = 0; i < s; i++ )
        {
	  // download i-th panel
	  cols = maxm - i*nb;
	  magmablas_stranspose( dA, cols, inAT(i,i), ldda, nb, cols );
	  cublasGetMatrix( *m-i*nb, nb, sizeof(float), dA, cols, work, lddwork); 

	  // make sure that gpu queue is empty
	  cuCtxSynchronize();

	  if (i>0){
	    cublasStrsm( 'R', 'U', 'N', 'U', *n - (i+1)*nb, nb, 1, 
			 inAT(i-1,i-1), ldda, inAT(i-1,i+1), ldda ); 
	    cublasSgemm( 'N', 'N', *n-(i+1)*nb, *m-i*nb, nb, -1, 
			 inAT(i-1,i+1), ldda, inAT(i,i-1), ldda, 1, 
			 inAT(i,i+1), ldda );
	  }
	  
	  // do the cpu part
	  rows = *m - i*nb;
	  sgetrf_( &rows, &nb, work, &lddwork, ipiv+i*nb, &iinfo);
	  if (*info == 0 && iinfo > 0)
	    *info = iinfo + i*nb;

	  magmablas_spermute_long2( dAT, ldda, ipiv, nb, i*nb );

	  // upload i-th panel
	  cublasSetMatrix( *m-i*nb, nb, sizeof(float), work, lddwork, dA, cols);
	  magmablas_stranspose( inAT(i,i), ldda, dA, cols, cols, nb);

	  // do the small non-parallel computations
	  if (s > (i+1)){
	    cublasStrsm( 'R', 'U', 'N', 'U', nb, nb, 1, inAT(i,i), ldda, 
			 inAT(i, i+1), ldda);
	    cublasSgemm( 'N', 'N', nb, *m-(i+1)*nb, nb, -1, inAT(i,i+1), ldda,
			 inAT(i+1,i), ldda, 1, inAT(i+1,i+1), ldda );
	  }
	  else{
	    cublasStrsm( 'R', 'U', 'N', 'U', *n-s*nb, nb, 1, inAT(i,i), ldda,
                         inAT(i, i+1), ldda);
	    cublasSgemm( 'N', 'N', *n-(i+1)*nb, *m-(i+1)*nb, nb, 
			 -1, inAT(i,i+1), ldda,
			 inAT(i+1,i), ldda, 1, inAT(i+1,i+1), ldda );
	  }
	}

      int nb0 = min(*m - s*nb, *n - s*nb);
      rows = *m - s*nb;
      cols = maxm - s*nb;

      magmablas_stranspose2( dA, cols, inAT(s,s), ldda, nb0, rows);
      cublasGetMatrix(rows, nb0, sizeof(float), dA, cols, work, lddwork); 

      // make sure that gpu queue is empty
      cuCtxSynchronize();
	  
      // do the cpu part
      sgetrf_( &rows, &nb0, work, &lddwork, ipiv+s*nb, &iinfo);
      if (*info == 0 && iinfo > 0)
	*info = iinfo + s*nb;
      magmablas_spermute_long2( dAT, ldda, ipiv, nb0, s*nb );

      // upload i-th panel
      cublasSetMatrix(rows, nb0, sizeof(float), work, lddwork, dA, cols);
      magmablas_stranspose2( inAT(s,s), ldda, dA, cols, rows, nb0);

      cublasStrsm( 'R', 'U', 'N', 'U', *n-s*nb-nb0, nb0,
		   1, inAT(s,s), ldda, inAT(s, s)+nb0, ldda);

      if ((*m == *n) && (*m % 32 == 0) && (*lda%32 == 0))
        magmablas_sinplace_transpose( dAT, *lda, ldda );
      else {
	magmablas_stranspose2( a, *lda, dAT, ldda, *n, *m );
	cublasFree(dAT);
      }

      cublasFree(work);
      cublasFree(dA);
    }
    return 0;
    
    /* End of MAGMA_SGETRF_GPU */
}

#undef inAT
#undef max
#undef min
