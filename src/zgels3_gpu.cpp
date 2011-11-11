/*
    -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

       @precisions normal z -> s d c

*/
#include "common_magma.h"

extern "C" magma_int_t
magma_zgels3_gpu( char trans, magma_int_t m, magma_int_t n, magma_int_t nrhs,
                  cuDoubleComplex *dA,    magma_int_t ldda, 
                  cuDoubleComplex *dB,    magma_int_t lddb, 
                  cuDoubleComplex *hwork, magma_int_t lwork, 
                  magma_int_t *info)
{
/*  -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

    Purpose
    =======
    Solves the overdetermined, least squares problem
           min || A*X - C ||
    using the QR factorization A.
    The underdetermined problem (m < n) is not currently handled.


    Arguments
    =========
    TRANS   (input) CHARACTER*1
            = 'N': the linear system involves A.
            Only trans='N' is currently handled.

    M       (input) INTEGER
            The number of rows of the matrix A. M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix A. M >= N >= 0.

    NRHS    (input) INTEGER
            The number of columns of the matrix C. NRHS >= 0.

    A       (input/output) COMPLEX_16 array, dimension (LDA,N)
            On entry, the M-by-N matrix A.
            On exit, A is overwritten by details of its QR
            factorization as returned by ZGEQRF3.

    LDDA    (input) INTEGER
            The leading dimension of the array A, LDDA >= M.

    DB      (input/output) COMPLEX_16 array on the GPU, dimension (LDDB,NRHS)
            On entry, the M-by-NRHS matrix C.
            On exit, the N-by-NRHS solution matrix X.

    LDDB    (input) INTEGER
            The leading dimension of the array DB. LDDB >= M.

    HWORK   (workspace/output) COMPLEX_16 array, dimension MAX(1,LWORK).
            On exit, if INFO = 0, HWORK(1) returns the optimal LWORK.

    LWORK   (input) INTEGER
            The dimension of the array HWORK, LWORK >= max(1,NRHS).
            For optimum performance LWORK >= (M-N+NB)*(NRHS + 2*NB), where 
            NB is the blocksize given by magma_get_zgeqrf_nb( M ).

            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the HWORK array, returns
            this value as the first entry of the HWORK array.

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
    =====================================================================    */

   #define a_ref(a_1,a_2) (dA+(a_2)*(ldda) + (a_1))

    cuDoubleComplex *dT, *tau;
    magma_int_t k, ret;

    magma_int_t nb     = magma_get_zgeqrf_nb(m);
    magma_int_t lwkopt = (m-n+nb)*(nrhs+2*nb);
    long int lquery = (lwork == -1);

    hwork[0] = MAGMA_Z_MAKE( (double)lwkopt, 0. );

    *info = 0;
    /* For now, N is the only case working */
    if ( (trans != 'N') && (trans != 'n' ) )
        *info = -1;
    else if (m < 0)
        *info = -2;
    else if (n < 0 || m < n) /* LQ is not handle for now*/
        *info = -3;
    else if (nrhs < 0)
        *info = -4;
    else if (ldda < max(1,m))
        *info = -6;
    else if (lddb < max(1,m))
        *info = -8;
    else if (lwork < lwkopt && ! lquery)
        *info = -10;

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return MAGMA_ERR_ILLEGAL_VALUE;
    }
    else if (lquery)
        return MAGMA_SUCCESS;

    k = min(m,n);
    if (k == 0) {
        hwork[0] = MAGMA_Z_ONE;
        return MAGMA_SUCCESS;
    }

    /*
     * Allocate temporary buffers
     */
    int ldtwork = ( 2*k + ((n+31)/32)*32 )*nb;
    if (nb < nrhs)
      ldtwork = ( 2*k + ((n+31)/32)*32 )*nrhs;
    if( CUBLAS_STATUS_SUCCESS != cublasAlloc(ldtwork, 
                                             sizeof(cuDoubleComplex), (void**)&dT) ) {
        return MAGMA_ERR_CUBLASALLOC;
    }
    
    tau = (cuDoubleComplex*) malloc( k * sizeof(cuDoubleComplex) );
    if( tau == NULL ) {
        cublasFree(dT);
        return MAGMA_ERR_ALLOCATION;
    }

    ret = magma_zgeqrf3_gpu( m, n, dA, ldda, tau, dT, info );
    if ( (ret != MAGMA_SUCCESS) || (*info != 0) ) {
        cublasFree(dT);
        free(tau);
        return ret;
    }

    ret = magma_zgeqrs3_gpu(m, n, nrhs,
                           dA, ldda, tau, dT, 
                           dB, lddb, hwork, lwork, info);
    if ( (ret != MAGMA_SUCCESS) || (*info != 0) ) {
        cublasFree(dT);
        free(tau);
        return ret;
    }

    cublasFree(dT);
    free(tau);
    return MAGMA_SUCCESS;
}

#undef a_ref

