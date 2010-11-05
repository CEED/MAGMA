/*
    -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

       @precisions normal z -> s d c

*/

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include "magma.h"
#include "magmablas.h"

extern "C" void
magmablas_ztranspose2(double2 *, int, double2 *, int, int, int);

extern "C" void 
magmablas_zpermute_long2(double2 *, int, int *, int, int);

extern "C" magma_int_t 
magma_zgetrf_gpu(magma_int_t m, magma_int_t n, double2 *a, magma_int_t lda,
		 magma_int_t *ipiv, magma_int_t *info)
{
/*  -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

    Purpose   
    =======   

    ZGETRF computes an LU factorization of a general M-by-N matrix A   
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

    A       (input/output) COMPLEX_16 array on the GPU, dimension (LDA,N). 
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

    double2 c_one = MAGMA_Z_ONE;
    double2 c_neg_one = MAGMA_Z_NEG_ONE;

    int iinfo, nb;
    int maxm, maxn, mindim;
    int i, rows, cols, s, ldda, lddwork;
    double2 *dAT, *dA, *work;

    /* Check arguments */
    *info = 0;
    if (m < 0)
	*info = -1;
    else if (n < 0)
	*info = -2;
    else if (lda < max(1,m))
	*info = -4;
    
    if (*info != 0)
      return 0;

    /* Quick return if possible */
    if (m == 0 || n == 0)
      return 0;

    /* Function Body */
    mindim = min(m, n);
    nb     = magma_get_zgetrf_nb(m);
    s      = mindim / nb;

    if (nb <= 1 || nb >= min(m,n)) {
	/* Use CPU code. */
	work = (double2*)malloc(maxm * n * sizeof(double2));
	cublasGetMatrix(m, n, sizeof(double2), a, lda, work, maxm);
	zgetrf_(&m, &n, work, &maxm, ipiv, info);
	cublasSetMatrix(m, n, sizeof(double2), work, maxm, a, lda);
	free(work);
    } 
    else {
	/* Use hybrid blocked code. */
	maxm = ((m + 31)/32)*32;
	maxn = ((n + 31)/32)*32;

	ldda    = maxn;
	lddwork = maxm;

	dAT = a;

	cublasStatus status;
	status = cublasAlloc(nb*maxm, sizeof(double2), (void**)&dA);
	if (status != CUBLAS_STATUS_SUCCESS)
	    return -7;
	
	if ((m == n) && (m % 32 == 0) && (lda%32 == 0))
	    magmablas_zinplace_transpose( dAT, lda, ldda );
	else {
	    status = cublasAlloc(maxm*maxn, sizeof(double2), (void**)&dAT);
	    if (status != CUBLAS_STATUS_SUCCESS)
		return -7;
	    magmablas_ztranspose2( dAT, ldda, a, lda, m, n );
	}
      
	cudaMallocHost( (void**)&work, maxm*nb*sizeof(double2) );
	if (work == 0)
	    return -7;

	for( i=0; i<s; i++ )
	{
	    // download i-th panel
	    cols = maxm - i*nb;
	    magmablas_ztranspose( dA, cols, inAT(i,i), ldda, nb, cols );
	    cublasGetMatrix( m-i*nb, nb, sizeof(double2), dA, cols, work, lddwork); 
	    
	    // make sure that gpu queue is empty
	    cuCtxSynchronize();
	    
	    if ( i>0 ){
		cublasZtrsm( 'R', 'U', 'N', 'U', n - (i+1)*nb, nb, c_one, 
			     inAT(i-1,i-1), ldda, inAT(i-1,i+1), ldda ); 
		cublasZgemm( 'N', 'N', n-(i+1)*nb, m-i*nb, nb, c_neg_one, 
			     inAT(i-1,i+1), ldda, inAT(i,i-1), ldda, c_one, 
			     inAT(i,i+1), ldda );
	    }
	  
	    // do the cpu part
	    rows = m - i*nb;
	    zgetrf_( &rows, &nb, work, &lddwork, ipiv+i*nb, &iinfo);
	    if ( (*info == 0) && (iinfo > 0) )
		*info = iinfo + i*nb;
	    
	    magmablas_zpermute_long2( dAT, ldda, ipiv, nb, i*nb );

	    // upload i-th panel
	    cublasSetMatrix( m-i*nb, nb, sizeof(double2), work, lddwork, dA, cols);
	    magmablas_ztranspose( inAT(i,i), ldda, dA, cols, cols, nb);

	    // do the small non-parallel computations
	    if ( s > (i+1) ) {
		cublasZtrsm( 'R', 'U', 'N', 'U', nb, nb, c_one, inAT(i,i), ldda, 
			     inAT(i, i+1), ldda);
		cublasZgemm( 'N', 'N', nb, m-(i+1)*nb, nb, c_neg_one, inAT(i,i+1), ldda,
			     inAT(i+1,i), ldda, c_one, inAT(i+1,i+1), ldda );
	    }
	    else {
		cublasZtrsm( 'R', 'U', 'N', 'U', n-s*nb, nb, c_one, inAT(i,i), ldda,
			     inAT(i, i+1), ldda);
		cublasZgemm( 'N', 'N', n-(i+1)*nb, m-(i+1)*nb, nb, 
			     c_neg_one, inAT(i,i+1), ldda,
			     inAT(i+1,i), ldda, c_one, inAT(i+1,i+1), ldda );
	    }
	}

	int nb0 = min(m - s*nb, n - s*nb);
	rows = m - s*nb;
	cols = maxm - s*nb;
	
	magmablas_ztranspose2( dA, cols, inAT(s,s), ldda, nb0, rows);
	cublasGetMatrix(rows, nb0, sizeof(double2), dA, cols, work, lddwork); 

	// make sure that gpu queue is empty
	cuCtxSynchronize();
	
	// do the cpu part
	zgetrf_( &rows, &nb0, work, &lddwork, ipiv+s*nb, &iinfo);
	if ( (*info == 0) && (iinfo > 0) )
	    *info = iinfo + s*nb;
	magmablas_zpermute_long2( dAT, ldda, ipiv, nb0, s*nb );

	// upload i-th panel
	cublasSetMatrix(rows, nb0, sizeof(double2), work, lddwork, dA, cols);
	magmablas_ztranspose2( inAT(s,s), ldda, dA, cols, rows, nb0);

	cublasZtrsm( 'R', 'U', 'N', 'U', n-s*nb-nb0, nb0,
		     c_one, inAT(s,s), ldda, inAT(s, s)+nb0, ldda);

	if ((m == n) && (m % 32 == 0) && (lda%32 == 0))
	    magmablas_zinplace_transpose( dAT, lda, ldda );
	else {
	    magmablas_ztranspose2( a, lda, dAT, ldda, n, m );
	    cublasFree(dAT);
	}
	
	cublasFree(work);
	cublasFree(dA);
    }
    return 0;
    
    /* End of MAGMA_ZGETRF_GPU */
}

#undef inAT
#undef max
#undef min
