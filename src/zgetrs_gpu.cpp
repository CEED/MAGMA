/*
    -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

       @precisions normal z -> s d c

*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include "magma.h"

extern "C" magma_int_t
magma_zgetrs_gpu(char trans, magma_int_t n, magma_int_t nrhs, 
                 cuDoubleComplex *dA, magma_int_t ldda,
		 magma_int_t *ipiv, 
                 cuDoubleComplex *dB, magma_int_t lddb, 
                 magma_int_t *info)
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
    with a general N-by-N matrix A using the LU factorization computed by ZGETRF_GPU.

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

    A       (input) COMPLEX_16 array on the GPU, dimension (LDA,N)
            The factors L and U from the factorization A = P*L*U as computed
            by ZGETRF_GPU.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    IPIV    (input) INTEGER array, dimension (N)
            The pivot indices from ZGETRF; for 1<=i<=N, row i of the
            matrix was interchanged with row IPIV(i).

    B       (input/output) COMPLEX_16 array on the GPU, dimension (LDB,NRHS)
            On entry, the right hand side matrix B.
            On exit, the solution matrix X.

    LDB     (input) INTEGER
            The leading dimension of the array B.  LDB >= max(1,N).

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value

    HWORK   (workspace) COMPLEX_16 array, dimension N*NRHS
    =====================================================================    */

    #define max(a,b)  (((a)>(b))?(a):(b))

    cuDoubleComplex c_one = MAGMA_Z_ONE;
    cuDoubleComplex *work = NULL;
    char trans_[2] = {trans, 0};

    long int notran = lapackf77_lsame(trans_, "N");
    *info = 0;
    if ( (! notran) && 
         (! lapackf77_lsame(trans_, "T")) && 
         (! lapackf77_lsame(trans_, "C")) ) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (nrhs < 0) {
        *info = -3;
    } else if (ldda < max(1,n)) {
        *info = -5;
    } else if (lddb < max(1,n)) {
        *info = -8;
    }
    if (*info != 0) {
        return 0;
    }

    /* Quick return if possible */
    if (n == 0 || nrhs == 0) {
        return 0;
    }

    work = (cuDoubleComplex*)malloc(n * nrhs * sizeof(cuDoubleComplex));
    if ( !work ) {
        return -7;
    }
      
    if (notran) {
        /* Solve A * X = B. */
        cublasGetMatrix( n, nrhs, sizeof(cuDoubleComplex), dB, lddb, work, n);
        int k1 = 1 ;
        int k2 = n;
        int k3 = 1 ;
        lapackf77_zlaswp(&nrhs, work, &n, &k1, &k2, ipiv, &k3);
        cublasSetMatrix( n, nrhs, sizeof(cuDoubleComplex), work, n, dB, lddb);

        cublasZtrsm(MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit,    n, nrhs, c_one, dA, ldda, dB, lddb );
        cublasZtrsm(MagmaLeft, MagmaUpper, MagmaNoTrans, MagmaNonUnit, n, nrhs, c_one, dA, ldda, dB, lddb );
    } else {
        /* Solve A' * X = B. */
        cublasZtrsm(MagmaLeft, MagmaUpper, MagmaConjTrans, MagmaNonUnit, n, nrhs, c_one, dA, ldda, dB, lddb );
        cublasZtrsm(MagmaLeft, MagmaLower, MagmaConjTrans, MagmaUnit,    n, nrhs, c_one, dA, ldda, dB, lddb );

        cublasGetMatrix( n, nrhs, sizeof(cuDoubleComplex), dB, lddb, work, n );
        int k1 = 1 ;
        int k2 = n;
        int k3 = -1;
        lapackf77_zlaswp(&nrhs, work, &n, &k1, &k2, ipiv, &k3);
        cublasSetMatrix( n, nrhs, sizeof(cuDoubleComplex), work, n, dB, lddb);
    }
    free(work);

    return 0;
}

#undef max
