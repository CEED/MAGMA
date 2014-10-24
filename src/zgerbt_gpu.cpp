/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Adrien REMY
*/
#include "common_magma.h"

#define PRECISION_z
#define COMPLEX



void 
init_butterfly(magmaDoubleComplex* u, magmaDoubleComplex* v, magma_int_t N)
{

    magma_int_t idx;
    double u1, v1;
    for (idx=0; idx<N; idx++){
        u1 = exp((((rand() * 1.0)/RAND_MAX)-0.5)/10);
        v1 = exp((((rand() * 1.0)/RAND_MAX)-0.5)/10);
        u[idx] = MAGMA_Z_MAKE(u1,u1);

        v[idx] = MAGMA_Z_MAKE(v1,v1);
    }
}


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
    gen     CHAR
     -         = "N" or "n"     new matrices are generated for U and V
     -       = "O" or "o"     matrices U and V given as parameter are used

    
    @param[in]
    N       INTEGER
            The order of the matrix A.  N >= 0.

    @param[in]
    NRHS    INTEGER
            The number of right hand sides, i.e., the number of columns
            of the matrix B.  NRHS >= 0.

    @param[in,out]
    dA       COMPLEX_16 array, dimension (LDA,N).
            On entry, the M-by-N matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    
    @param[in,out]
    dB       COMPLEX_16 array, dimension (LDB,NRHS)
            On entry, the right hand side matrix B.
            On exit, the solution matrix X.

    @param[in]
    ldb     INTEGER
            The leading dimension of the array B.  LDB >= max(1,N).

    @param[in,out]
    U        COMPLEX_16 array, dimension (2,N)
            Random butterfly matrix, if gen = 'N' U is generated and returned as output
        else we use U given as input
        CPU memory
    @param[in,out]
    V        COMPLEX_16 array, dimension (2,N)
            Random butterfly matrix, if gen = 'N' V is generated and returned as output
        else we use U given as input
        CPU memory

 ********************************************************************/



extern "C" 
magma_int_t 
magma_zgerbt_gpu(char gen, magma_int_t N, magma_int_t NRHS, 
        magmaDoubleComplex *dA, magma_int_t lda, 
        magmaDoubleComplex *dB, magma_int_t ldb, 
        magmaDoubleComplex *U, magmaDoubleComplex *V)
{
    magma_int_t ret;

    /* Function Body */
    ret = 0;
    if (N < 0) {
        ret = -1;
    } else if (NRHS < 0) {
        ret = -2;
    } else if (lda < max(1,N)) {
        ret = -4;
    } else if (ldb < max(1,N)) {
        ret = -7;
    }
    if (ret != 0) {
        magma_xerbla( __func__, -(ret) );

        return MAGMA_ERR_ILLEGAL_VALUE;
    }

    /* Quick return if possible */
    if (NRHS == 0 || N == 0)
        return MAGMA_SUCCESS;


    char            trans_[2] = {gen, 0};
    magma_int_t    notran = lapackf77_lsame(trans_, "N");



    magma_int_t n2;

    n2 = N*N;


    magmaDoubleComplex *d_u, *d_v;

    /* Allocate memory for the buterfly matrices */
    if (MAGMA_SUCCESS != magma_zmalloc( &d_u, 2*N )) {
        return MAGMA_ERR_DEVICE_ALLOC;
    }
    if (MAGMA_SUCCESS != magma_zmalloc( &d_v, 2*N )) {
        return MAGMA_ERR_DEVICE_ALLOC;
    }

    /* Initialize Butterfly matrix on the CPU*/
    if(notran)
        init_butterfly(U,V,2*N);

    /* Copy the butterfly to the GPU */
    magma_zsetvector( 2*N, U, 1, d_u, 1);
    magma_zsetvector( 2*N, V, 1, d_v, 1);

    /* Perform Partial Random Butterfly Transformation on the GPU*/
    magmablas_zprbt(N, dA, lda, d_u, d_v);

    /* Compute U^T.b on the GPU*/
    for(int i= 0; i < NRHS; i++)
        magmablas_zprbt_mtv(N, d_u, dB+(i*ldb));

    magma_free( d_u );
    magma_free( d_v );

    return MAGMA_SUCCESS;

}


