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
#include <stdio.h>
#include <stdlib.h>

#define cublasDtrsm magmablas_dtrsm
#define cublasDgemm magmablasDgemm

extern "C" int
magma_dgetrs_gpu(char *trans , int n, int nrhs, double *a , int lda,
		 int *ipiv, double *b, int ldb, int *info, double *hwork)
{
/*  -- MAGMA (version 0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2009

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

    A       (input) DOUBLE REAL array on the GPU, dimension (LDA,N)
            The factors L and U from the factorization A = P*L*U as computed 
            by SGETRF_GPU.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    IPIV    (input) INTEGER array, dimension (N)
            The pivot indices from SGETRF; for 1<=i<=N, row i of the
            matrix was interchanged with row IPIV(i).

    B       (input/output) DOUBLE REAL array on the GPU, dimension (LDB,NRHS)
            On entry, the right hand side matrix B.
            On exit, the solution matrix X.

    LDB     (input) INTEGER
            The leading dimension of the array B.  LDB >= max(1,N).

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value

    HWORK   (workspace) DOUBLE REAL array, dimension N*NRHS
    =====================================================================    */

    #define max(a,b)  (((a)>(b))?(a):(b))

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
      cublasGetMatrix( n, nrhs, sizeof(double), b,n ,hwork, n);
      int k1 = 1 ;
      int k2 = n;
      int k3 = 1 ;
      dlaswp_(&nrhs, hwork, &ldb, &k1, &k2, ipiv, &k3);
      cublasSetMatrix( n, nrhs, sizeof(double), hwork, n, b, n);
      
      cublasDtrsm('L','L','N','U', n , nrhs, 1.0, a , lda , b , ldb );
      cublasDtrsm('L','U','N','N', n , nrhs, 1.0, a , lda , b , ldb );
    } else {
      /* Solve A' * X = B. */
      cublasDtrsm('L','U','T','N', n , nrhs, 1.0, a , lda , b , ldb );
      cublasDtrsm('L','L','T','U', n , nrhs, 1.0, a , lda , b , ldb );

      cublasGetMatrix( n, nrhs, sizeof(double), b,n , hwork , n );
      int k1 = 1 ;
      int k2 = n;
      int k3 = -1;
      dlaswp_(&nrhs, hwork, &ldb, &k1, &k2, ipiv , &k3);
      cublasSetMatrix( n, nrhs, sizeof(double), hwork, n , b , n);
    }
}

#undef max
