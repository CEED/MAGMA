/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       @author Adrien REMY

       @precisions normal z -> s d c

*/
#include "common_magma.h"

/**
    Purpose
    -------
    Solves a system of linear equations
       A * X = B
    where A is an n-by-n hermitian matrix and X and B are n-by-nrhs matrices.
    The LU decomposition with no pivoting is
    used to factor A as
    The factorization has the form
       A = U^H * D * U,  if UPLO = 'U', or
       A = L  * D * L^H, if UPLO = 'L',
    where U is an upper triangular matrix, L is lower triangular, and
    D is a diagonal matrix.
    The factored form of A is then
    used to solve the system of equations A * X = B.
    
    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  Upper triangle of A is stored;
      -     = MagmaLower:  Lower triangle of A is stored.

    @param[in]
    n       INTEGER
            The order of the matrix A.  n >= 0.

    @param[in]
    nrhs    INTEGER
            The number of right hand sides, i.e., the number of columns
            of the matrix B.  nrhs >= 0.

    @param[in,out]
    dA       COMPLEX_16 array, dimension (ldda,n).
            On entry, the n-by-n matrix to be factored.
            On exit, the factors L and U from the factorization
            A = L*U; the unit diagonal elements of L are not stored.

    @param[in]
    ldda     INTEGER
            The leading dimension of the array A.  ldda >= max(1,n).

    @param[in,out]
    dB       COMPLEX_16 array, dimension (lddb,nrhs)
            On entry, the right hand side matrix B.
            On exit, the solution matrix X.

    @param[in]
    lddb     INTEGER
            The leading dimension of the array B.  ldb >= max(1,n).

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value

    @ingroup magma_zhesv_driver
    ********************************************************************/




extern "C" magma_int_t
magma_zhesv_nopiv_gpu(magma_uplo_t uplo,  magma_int_t n, magma_int_t nrhs,
                 magmaDoubleComplex_ptr dA, magma_int_t ldda,
                 magmaDoubleComplex_ptr dB, magma_int_t lddb,
                 magma_int_t *info)
{
    magma_int_t ret;

    *info = 0;
    int   upper = (uplo == MagmaUpper);
    if (! upper && uplo != MagmaLower) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (nrhs < 0) {
        *info = -3;
    } else if (ldda < max(1,n)) {
        *info = -5;
    } else if (lddb < max(1,n)) {
        *info = -7;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return MAGMA_ERR_ILLEGAL_VALUE;
    }

    /* Quick return if possible */
    if (n == 0 || nrhs == 0) {
        return MAGMA_SUCCESS;
    }

    ret = magma_zhetrf_nopiv_gpu(uplo, n, dA, ldda, info);
    if ( (ret != MAGMA_SUCCESS) || (*info != 0) ) {
        return ret;
    }
        
    ret = magma_zhetrs_nopiv_gpu( uplo, n, nrhs, dA, ldda, dB, lddb, info );
    if ( (ret != MAGMA_SUCCESS) || (*info != 0) ) {
        return ret;
    }

    
    return ret;
}
