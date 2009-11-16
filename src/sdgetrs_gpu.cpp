/*
    -- MAGMA (version 0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2009
*/

#include "cuda.h"
#include "cublas.h"
#include "magma.h"
#include "magmablas.h"
#include <stdio.h>

int 
magma_sdgetrs_gpu(int *n, int *nrhs, float *a, int *lda, 
		  int *ipiv, float *x, double *b, int *ldb, int *info)
{
/*  -- MAGMA (version 0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2009

    Purpose   
    =======   

    SDGETRS solves a system of linear equations   
       A * X = B  or  A' * X = B   
    with a general N-by-N matrix A using the LU factorization computed   
    by MAGMA_SGETRF_GPU. B is in double, A and X in single precision. This 
    routine is used in the mixed precision iterative solver magma_dsgesv.

    Arguments   
    =========   

    N       (input) INTEGER   
            The order of the matrix A.  N >= 0.   

    NRHS    (input) INTEGER   
            The number of right hand sides, i.e., the number of columns   
            of the matrix B.  NRHS >= 0.   

    A       (input) REAL array on the GPU, dimension (LDA,N)   
            The factors L and U from the factorization A = P*L*U   
            as computed by SGETRF.   

    LDA     (input) INTEGER   
            The leading dimension of the array A.  LDA >= max(1,N).   

    IPIV    (input) INTEGER array on the GPU, dimension (N)   
            The pivot indices from SGETRF_GPU; Row i of the   
            matrix was moved to row IPIV(i).

    X       (output) REAL array on the GPU, dimension (LDB,NRHS)
            On exit, the solution matrix X.

    B       (input) DOUBLE PRECISION array on the GPU, dimension (LDB,NRHS)   
            On entry, the right hand side matrix B.  

    LDB     (input) INTEGER   
            The leading dimension of the arrays X and B.  LDB >= max(1,N).   

    INFO    (output) INTEGER   
            = 0:  successful exit   
            < 0:  if INFO = -i, the i-th argument had an illegal value   

    =====================================================================    */

  *info = 0;
  if (*n < 0) {
    *info = -1;
  } else if (*nrhs < 0) {
    *info = -2;
  } else if (*lda < *n) {
    *info = -4;
  } else if (*ldb < *n) {
    *info = -8;
  }
  if (*info != 0) {
    return 0;
  }

  /* Quick return if possible */
  if (*n == 0 || *nrhs == 0) {
    return 0;
  }
  /* Get X by row applying interchanges to B and cast to single */
  magmablas_sdlaswp(*nrhs, b, *ldb, x, *n, ipiv);

  /* Solve L*X = B, overwriting B with X. */
  float fone = 1.;
  magmablas_strsm('L','L','N','U', *n, *nrhs, fone, a, *lda, x, *ldb);

  /* Solve U*X = B, overwriting B with X. */
  magmablas_strsm('L','U','N','N', *n, *nrhs, fone, a, *lda, x, *ldb);

  return 0;
  /*     End of MAGMA_SDGETRS */

} /* magma_sdgetrs */
