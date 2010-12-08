/*
    -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

       @precisions mixed zc -> ds

*/

#include <stdio.h>
#include <cuda.h>
#include <cublas.h>
#include "magma.h"
#include "magmablas.h"

extern "C" magma_int_t 
magma_zcgetrs_gpu(magma_int_t n, magma_int_t nrhs, 
                  cuFloatComplex  *dA, magma_int_t ldda, 
		  magma_int_t     *ipiv, 
                  cuDoubleComplex *dB, magma_int_t lddb, 
                  cuFloatComplex  *dX, magma_int_t lddx,
                  magma_int_t *info)
{
/*  -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

    Purpose   
    =======   

    ZCGETRS solves a system of linear equations   
       A * X = B  or  A' * X = B   
    with a general N-by-N matrix A using the LU factorization computed   
    by MAGMA_ZGETRF_GPU. B is in double, A and X in single precision. This 
    routine is used in the mixed precision iterative solver magma_zcgesv.

    Arguments   
    =========   

    N       (input) INTEGER   
            The order of the matrix A.  N >= 0.   

    NRHS    (input) INTEGER   
            The number of right hand sides, i.e., the number of columns   
            of the matrix B.  NRHS >= 0.   

    A       (input) REAL array on the GPU, dimension (LDDA,N)   
            The factors L and U from the factorization A = P*L*U   
            as computed by ZGETRF.   

    LDDA     (input) INTEGER   
            The leading dimension of the array A.  LDDA >= max(1,N).   

    IPIV    (input) INTEGER array on the GPU, dimension (N)   
            The pivot indices from ZGETRF_GPU; Row i of the   
            matrix was moved to row IPIV(i).

    X       (output) REAL array on the GPU, dimension (LDDB,NRHS)
            On exit, the solution matrix X.

    B       (input) DOUBLE PRECISION array on the GPU, dimension (LDDB,NRHS)   
            On entry, the right hand side matrix B.  

    LDDB     (input) INTEGER   
            The leading dimension of the arrays X and B.  LDDB >= max(1,N).   

    INFO    (output) INTEGER   
            = 0:  successful exit   
            < 0:  if INFO = -i, the i-th argument had an illegal value   

    =====================================================================    */
    cuFloatComplex cone = MAGMA_C_ONE;

    *info = 0;
    if (n < 0) {
	*info = -1;
    } else if (nrhs < 0) {
	*info = -2;
    } else if (ldda < n) {
	*info = -4;
    } else if (lddb < n) {
	*info = -7;
    } else if (lddx < n) {
	*info = -9;
    } else if (lddx != lddb) { /* TODO: remove it when zclaswp will have the correct interface */
	*info = -9;
    }
    if (*info != 0) {
	return 0;
    }

    /* Quick return if possible */
    if (n == 0 || nrhs == 0) {
	return 0;
    }
    
    /* Get X by row applying interchanges to B and cast to single */
    /* 
     * TODO: clean zclaswp interface to have interface closer to zlaswp 
     */
    //magmablas_zclaswp(nrhs, dB, lddb, dX, lddbx, 1, n, ipiv);
    magmablas_zclaswp(nrhs, dB, lddb, dX, n, ipiv);

    /* Solve L*X = B, overwriting B with X. */
    cublasCtrsm( MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit, 
		 n, nrhs, cone, dA, ldda, dX, lddx);
    
    /* Solve U*X = B, overwriting B with X. */
    cublasCtrsm( MagmaLeft, MagmaUpper, MagmaNoTrans, MagmaNonUnit, 
		 n, nrhs, cone, dA, ldda, dX, lddx);
    
    return 0;
    /* End of MAGMA_ZCGETRS */
    
} /* magma_zcgetrs */
