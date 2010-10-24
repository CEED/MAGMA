/*
    -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010
*/

#include "cuda_runtime_api.h"
#include "cublas.h"
#include "magma.h"
#include <stdio.h>
#include <stdlib.h>

extern "C" magma_int_t
magma_sgetrs_gpu(char trans_, magma_int_t n, magma_int_t nrhs, float *a , magma_int_t lda,
		 magma_int_t *ipiv, float *b, magma_int_t ldb, magma_int_t *info, float *hwork)
{
/*  -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

    Purpose
    =======

    Solves a system of linear equations
      A * X = B  or  A' * X = B
    with a general N-by-N matrix A using the LU factorization computed by SGETRF_GPU.

    Arguments
    =========

    TRANS   (input) CHARACTER*1
            Specifies the form of the system of equations:
            = 'N':  A * X = B  (No transpose)
            = 'T':  A'* X = B  (Transpose)
            = 'C':  A'* X = B  (Conjugate transpose = Transpose)
 
    N       (input) INTEGER
            The order of the matrix A.  N >= 0.
 
    NRHS    (input) INTEGER
            The number of right hand sides, i.e., the number of columns
            of the matrix B.  NRHS >= 0.

    A       (input) REAL array on the GPU, dimension (LDA,N)
            The factors L and U from the factorization A = P*L*U as computed 
            by SGETRF_GPU.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    IPIV    (input) INTEGER array, dimension (N)
            The pivot indices from SGETRF; for 1<=i<=N, row i of the
            matrix was interchanged with row IPIV(i).

    B       (input/output) REAL array on the GPU, dimension (LDB,NRHS)
            On entry, the right hand side matrix B.
            On exit, the solution matrix X.

    LDB     (input) INTEGER
            The leading dimension of the array B.  LDB >= max(1,N).

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value

    HWORK   (workspace) REAL array, dimension N*NRHS
    =====================================================================    */

    #define max(a,b)  (((a)>(b))?(a):(b))

    char trans[2] = {trans_, 0};

    long int notran = lsame_(trans, "N");
    *info = 0;
    if (! notran && ! lsame_(trans, "T") && ! lsame_(trans, "C")) {
      *info = -1;
    } else if (n < 0) {
      *info = -2;
    } else if (nrhs < 0) {
      *info = -3;
    } else if (lda < max(1,n)) {
      *info = -5;
    } else if (ldb < max(1,n)) {
      *info = -8;
    }
    if (*info != 0) {
      return 0;
    }

    /* Quick return if possible */
    if (n == 0 || nrhs == 0) {
      return 0;
    }

    if (notran) {
      /* Solve A * X = B. */
      cublasGetMatrix( n, nrhs, sizeof(float), b,ldb ,hwork, n);
      int k1 = 1 ;
      int k2 = n;
      int k3 = 1 ;
      slaswp_(&nrhs, hwork, &n, &k1, &k2, ipiv, &k3);
      cublasSetMatrix( n, nrhs, sizeof(float), hwork, n, b, ldb);
      
      cublasStrsm('L','L','N','U', n , nrhs, 1.0, a , lda , b , ldb );
      cublasStrsm('L','U','N','N', n , nrhs, 1.0, a , lda , b , ldb );
    } else {
      /* Solve A' * X = B. */
      cublasStrsm('L','U','T','N', n , nrhs, 1.0, a , lda , b , ldb );
      cublasStrsm('L','L','T','U', n , nrhs, 1.0, a , lda , b , ldb );

      cublasGetMatrix( n, nrhs, sizeof(float), b,ldb , hwork , n );
      int k1 = 1 ;
      int k2 = n;
      int k3 = -1;
      slaswp_(&nrhs, hwork, &n, &k1, &k2, ipiv , &k3);
      cublasSetMatrix( n, nrhs, sizeof(float), hwork,n, b,ldb);
    }
}

#undef max
