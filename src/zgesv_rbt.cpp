/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

*/
#include "common_magma.h"

/**
    Purpose
    -------
    Solves a system of linear equations
       A * X = B
    where A is a general N-by-N matrix and X and B are N-by-NRHS matrices.
    Random Butterfly Tranformation is applied on A and B, then 
    the LU decomposition with no pivoting is
    used to factor A as
       A = L * U,
    where L is unit lower triangular, and U is
    upper triangular.  The factored form of A is then used to solve the
    system of equations A * X = B.

    Arguments
    ---------
    @param[in]
    n       INTEGER
            The order of the matrix A.  N >= 0.

    @param[in]
    nrhs    INTEGER
            The number of right hand sides, i.e., the number of columns
            of the matrix B.  NRHS >= 0.

    @param[in,out]
    A       COMPLEX_16 array, dimension (LDA,N).
            On entry, the M-by-N matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    
    @param[in,out]
    B       COMPLEX_16 array, dimension (LDB,NRHS)
            On entry, the right hand side matrix B.
            On exit, the solution matrix X.

    @param[in]
    ldb     INTEGER
            The leading dimension of the array B.  LDB >= max(1,N).

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value

    @ingroup magma_zgesv_driver
    ********************************************************************/


extern "C" magma_int_t 
magma_zgesv_rbt(
    magma_int_t N, magma_int_t NRHS, 
    magmaDoubleComplex *A, magma_int_t lda, 
    magmaDoubleComplex *B, magma_int_t ldb, 
    magma_int_t *info)
{
    magma_int_t ret;

    /* Function Body */
    *info = 0;
    if (N < 0) {
        *info = -1;
    } else if (NRHS < 0) {
        *info = -2;
    } else if (lda < max(1,N)) {
        *info = -4;
    } else if (ldb < max(1,N)) {
        *info = -7;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );

        return MAGMA_ERR_ILLEGAL_VALUE;
    }

    /* Quick return if possible */
    if (NRHS == 0 || N == 0)
        return MAGMA_SUCCESS;


    magma_int_t NN = N + ((4-(N % 4))%4);
    magmaDoubleComplex *d_A, *h_u, *h_v, *d_b;
    magma_int_t n2;

    n2 = NN*NN;

    if (MAGMA_SUCCESS != magma_zmalloc( &d_A, n2 )) {
        return MAGMA_ERR_DEVICE_ALLOC;
    }
    if (MAGMA_SUCCESS != magma_zmalloc( &d_b, NN*NRHS )) {
        return MAGMA_ERR_DEVICE_ALLOC;
    }

    
    if (MAGMA_SUCCESS != magma_zmalloc_cpu( &h_u, 2*NN )) {
        return MAGMA_ERR_HOST_ALLOC;
    }

    if (MAGMA_SUCCESS != magma_zmalloc_cpu( &h_v, 2*NN )) {
        return MAGMA_ERR_HOST_ALLOC;
    }

    magmablas_zlaset(MagmaFull, NN, NN, MAGMA_Z_ZERO, MAGMA_Z_ONE, d_A, NN);

    /* Send matrix on the GPU*/
    magma_zsetmatrix(N, N, A, lda, d_A, NN);

    /* Send b on the GPU*/
    magma_zsetmatrix(N, NRHS, B, ldb, d_b, NN);
    ret = magma_zgerbt_gpu('N', NN, NRHS, d_A, NN, d_b, NN, h_u, h_v);
    if(ret != MAGMA_SUCCESS)  {
        return ret;
    }
    
    /* Solve the system U^TAV.y = U^T.b on the GPU*/ 
    magma_zgesv_nopiv_gpu( NN, NRHS, d_A, NN, d_b, NN, info);

    /* The solution of A.x = b is Vy computed on the GPU */
    magmaDoubleComplex *d_v;

    if (MAGMA_SUCCESS != magma_zmalloc( &d_v, 2*NN )) {
        return MAGMA_ERR_DEVICE_ALLOC;
    }

    magma_zsetvector(2*NN, h_v, 1, d_v, 1);
    
    for(int i = 0; i < NRHS; i++)
        magmablas_zprbt_mv(NN, d_v, d_b+(i*NN));

    magma_zgetmatrix(N, NRHS, d_b, NN, B, ldb);

    magma_free_cpu( h_u);
    magma_free_cpu( h_v);

    magma_free( d_A );
    magma_free( d_v );
    magma_free( d_b );

    return(MAGMA_SUCCESS);
}


