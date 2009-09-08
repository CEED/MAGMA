/*
    -- MAGMA (version 0.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       June 2009
*/

#include "cuda.h"
#include "cublas.h"
#include "magma.h"
#include "magmablas.h"
#include <stdio.h>

int 
magma_sgetrf(int *m, int *n, float *a, int *lda, 
	     int *ipiv, float *work, float *da, int *info)
{
/*  -- MAGMA (version 0.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       June 2009

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

    A       (input/output) REAL array, dimension (LDA,N)   
            On entry, the M-by-N matrix to be factored.   
            On exit, the factors L and U from the factorization   
            A = P*L*U; the unit diagonal elements of L are not stored.   

            Higher performance is achieved if A is in pinned memory, e.g.
            allocated using cudaMallocHost.

    LDA     (input) INTEGER   
            The leading dimension of the array A.  LDA >= max(1,M).   

    IPIV    (output) INTEGER array, dimension (min(M,N))   
            The pivot indices; for 1 <= i <= min(M,N), row i of the   
            matrix was interchanged with row IPIV(i).   

    WORK    (workspace/output) REAL array, dimension >= N*NB,
            where NB can be obtained through magma_get_sgetrf_nb(M).

            Higher performance is achieved if WORK is in pinned memory, e.g.
            allocated using cudaMallocHost.

    DA      (workspace)  REAL array on the GPU, dimension 
            (max(M, N)+ k1)^2 + (M + k2)*NB + 2*NB^2,
            where NB can be obtained through magma_get_sgetrf_nb(M).
            k1 < 32 and k2 < 32 are such that 
            (max(M, N) + k1)%32==0 and (M+k2)%32==0.

    INFO    (output) INTEGER   
            = 0:  successful exit   
            < 0:  if INFO = -i, the i-th argument had an illegal value   
            > 0:  if INFO = i, U(i,i) is exactly zero. The factorization   
                  has been completed, but the factor U is exactly   
                  singular, and division by zero will occur if it is used   
                  to solve a system of equations.   

    =====================================================================    */

#define inAT(i,j) (dAT + (i)*nb*ldda + (j)*nb)
#define max(a,b)  (((a)>(b))?(a):(b))
#define min(a,b)  (((a)<(b))?(a):(b))

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

    if (nb <= 1 || nb >= min(*m,*n)) {
      /* Use CPU code. */
      sgetrf_(m, n, a, lda, ipiv, info);
    } else {
      /* Use hybrid blocked code. */
      int maxm, mindim = min(*m, *n), maxdim = max(*m, *n);
      int i, rows, cols, s = mindim/nb;

      if ((maxdim % 32) != 0)
	maxdim = (maxdim/32)*32+32;
      if ((*m % 32) != 0)
	maxm = (*m/32)*32 + 32;
      else
	maxm = *m;

      int ldda = maxdim;

      float *W   = a;
      float *dAT = da;
      float *dA  = da + maxdim*maxdim;

      cublasSetMatrix( *m, *n, sizeof(float), a, *lda, dAT, ldda);
      magmablas_sinplace_transpose( dAT, ldda, ldda );
      for( i = 0; i < s; i++ )
        {
	  // download i-th panel
	  cols = maxm - i*nb;
	  magmablas_stranspose( dA, cols, inAT(i,i), ldda, nb, cols );
	  cublasGetMatrix( *m-i*nb, nb, sizeof(float), dA, cols, W, *lda); 

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
	  sgetrf_( &rows, &nb, W, lda, ipiv+i*nb, &iinfo);
	  if (*info == 0 && iinfo > 0)
	    *info = iinfo + i*nb;
	  magmablas_spermute_long( dAT, ldda, ipiv, nb, i*nb );

	  // upload i-th panel
	  cublasSetMatrix( *m-i*nb, nb, sizeof(float), W, *lda, dA, cols);
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
      magmablas_sinplace_transpose( dAT, ldda, ldda );
      cublasGetMatrix( *m, *n, sizeof(float), dAT, ldda, a, *lda); 
      
      rows = *m - s * nb;
      cols = *n - s * nb;
      sgetrf_( &rows, &cols, a+s*nb*(*lda)+s*nb, lda, ipiv+s*nb, &iinfo);
      if (*info == 0 && iinfo > 0)
	*info = iinfo + s*nb;
      for(i=s*nb; i<min(*m, *n); i++)
	ipiv[i] += s*nb;
      
      cols = s*nb+1;
      rows = min(*m, *n);
      int one = 1, pp = s*nb;
      slaswp_(&pp, a, lda, &cols, &rows, ipiv, &one);
    }
      
    return 0;

/*     End of MAGMA_SGETRF */

} /* magma_sgetrf */

#undef inAT
#undef max
#undef min
