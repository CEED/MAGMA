/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> s d c

*/
#include "common_magma.h"

extern "C" magma_int_t
magma_zgeqrs3_gpu(magma_int_t m, magma_int_t n, magma_int_t nrhs,
                  magmaDoubleComplex *dA,    magma_int_t ldda, 
                  magmaDoubleComplex *tau,   magmaDoubleComplex *dT, 
                  magmaDoubleComplex *dB,    magma_int_t lddb, 
                  magmaDoubleComplex *hwork, magma_int_t lwork, 
                  magma_int_t *info)
{
/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======
    Solves the least squares problem
           min || A*X - C ||
    using the QR factorization A = Q*R computed by ZGEQRF3_GPU.

    Arguments
    =========
    M       (input) INTEGER
            The number of rows of the matrix A. M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix A. M >= N >= 0.

    NRHS    (input) INTEGER
            The number of columns of the matrix C. NRHS >= 0.

    A       (input) COMPLEX_16 array on the GPU, dimension (LDDA,N)
            The i-th column must contain the vector which defines the
            elementary reflector H(i), for i = 1,2,...,n, as returned by
            ZGEQRF3_GPU in the first n columns of its array argument A.

    LDDA    (input) INTEGER
            The leading dimension of the array A, LDDA >= M.

    TAU     (input) COMPLEX_16 array, dimension (N)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by MAGMA_ZGEQRF_GPU.

    DB      (input/output) COMPLEX_16 array on the GPU, dimension (LDDB,NRHS)
            On entry, the M-by-NRHS matrix C.
            On exit, the N-by-NRHS solution matrix X.

    DT      (input) COMPLEX_16 array that is the output (the 6th argument)
            of magma_zgeqrf_gpu of size
            2*MIN(M, N)*NB + ((N+31)/32*32 )* MAX(NB, NRHS). 
            The array starts with a block of size MIN(M,N)*NB that stores 
            the triangular T matrices used in the QR factorization, 
            followed by MIN(M,N)*NB block storing the diagonal block 
            matrices for the R matrix, followed by work space of size 
            ((N+31)/32*32 )* MAX(NB, NRHS).

    LDDB    (input) INTEGER
            The leading dimension of the array DB. LDDB >= M.

    HWORK   (workspace/output) COMPLEX_16 array, dimension (LWORK)
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.

    LWORK   (input) INTEGER
            The dimension of the array WORK,
            LWORK >= (M - N + NB)*(NRHS + NB) + NRHS*NB,
            where NB is the blocksize given by magma_get_zgeqrf_nb( M ).

            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the HWORK array, returns
            this value as the first entry of the WORK array.

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
    =====================================================================    */

   #define a_ref(a_1,a_2) (dA+(a_2)*(ldda) + (a_1))
   #define d_ref(a_1)     (dT+(lddwork+(a_1))*nb)

    magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    magma_int_t k, lddwork;

    magma_int_t nb     = magma_get_zgeqrf_nb(m);
    magma_int_t lwkopt = (m - n + nb)*(nrhs + nb) + nrhs*nb;
    int lquery = (lwork == -1);

    hwork[0] = MAGMA_Z_MAKE( (double)lwkopt, 0. );

    *info = 0;
    if (m < 0)
        *info = -1;
    else if (n < 0 || m < n)
        *info = -2;
    else if (nrhs < 0)
        *info = -3;
    else if (ldda < max(1,m))
        *info = -5;
    else if (lddb < max(1,m))
        *info = -8;
    else if (lwork < lwkopt && ! lquery)
        *info = -10;

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }
    else if (lquery)
        return *info;

    k = min(m,n);
    if (k == 0) {
        hwork[0] = c_one;
        return *info;
    }
    lddwork= k;

    /* B := Q' * B */
    magma_zunmqr_gpu( MagmaLeft, MagmaConjTrans, 
                      m, nrhs, n,
                      a_ref(0,0), ldda, tau, 
                      dB, lddb, hwork, lwork, dT, nb, info );
    if ( *info != 0 ) {
        return *info;
    }

    /* Solve R*X = B(1:n,:) 
       1. Move the block diagonal submatrices from d_ref to R
       2. Solve 
       3. Restore the data format moving data from R back to d_ref 
    */
    magmablas_zswapdblk(k, nb, a_ref(0,0), ldda, 1, d_ref(0), nb, 0);
    if ( nrhs == 1 ) {
        magma_ztrsv(MagmaUpper, MagmaNoTrans, MagmaNonUnit,
                    n, a_ref(0,0), ldda, dB, 1);
    } else {
        magma_ztrsm(MagmaLeft, MagmaUpper, MagmaNoTrans, MagmaNonUnit,
                    n, nrhs, c_one, a_ref(0,0), ldda, dB, lddb);
    }
    magmablas_zswapdblk(k, nb, d_ref(0), nb, 0, a_ref(0,0), ldda, 1);

    return *info;
}

#undef a_ref
#undef d_ref
