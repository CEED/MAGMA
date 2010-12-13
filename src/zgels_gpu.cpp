/*
    -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

       @precisions normal z -> s d c

*/

#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include "magma.h"

extern "C" magma_int_t
magma_zgels_gpu( char trans, magma_int_t m, magma_int_t n, magma_int_t nrhs,
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

    Solves the least squares problem
           min || A*X - C ||
    using the QR factorization A = Q*R computed by ZGEQRF_GPU2.


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
            ZGEQRF_GPU2 in the first n columns of its array argument A.

    LDDA     (input) INTEGER
            The leading dimension of the array A, LDDA >= M.

    TAU     (input) COMPLEX_16 array, dimension (N)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by MAGMA_ZGEQRF_GPU.

    DB       (input/output) COMPLEX_16 array on the GPU, dimension (LDDB,NRHS)
            On entry, the M-by-NRHS matrix C.
            On exit, the N-by-NRHS solution matrix X.

    LDDB     (input) INTEGER
            The leading dimension of the array DB. LDDB >= M.

    HWORK    (workspace/output) COMPLEX_16 array, dimension (LWORK)
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.

    LWORK   (input) INTEGER
            The dimension of the array WORK, LWORK >= max(1,NRHS).
            For optimum performance LWORK >= (M-N+NB+2*NRHS)*NB, where NB is
            the blocksize given by magma_get_zgeqrf_nb( M ).

            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the HWORK array, returns
            this value as the first entry of the WORK array.

    DT      (input) COMPLEX_16 array that is the output (the 9th argument)
            of magma_zgeqrf_gpu.

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value

    =====================================================================    */

   #define a_ref(a_1,a_2) (dA+(a_2)*(ldda) + (a_1))
   #define t_ref(a_1)     (dT+(a_1))
   #define d_ref(a_1)     (dT+(lddwork+(a_1))*nb)
   #define min(a,b)       (((a)<(b))?(a):(b))
   #define max(a,b)       (((a)>(b))?(a):(b))

    cuDoubleComplex c_zero    = MAGMA_Z_ZERO;
    cuDoubleComplex c_one     = MAGMA_Z_ONE;
    cuDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    cuDoubleComplex *dT, *dwork, *tau;
    int i, k, lddwork, rows, ib;

    /* Function Body */
    int nb     = magma_get_zgeqrf_nb(m);
    int lwkopt = (m-n+nb+2*(nrhs)) * nb;
    long int lquery = (lwork == -1);

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

    if (*info != 0)
        return 0;
    else if (lquery)
        return 0;

    k = min(m,n);
    if (k == 0) {
        hwork[0] = c_one;
        return 0;
    }

    /*
     * Allocate temporary buffers
     */
    if( CUBLAS_STATUS_SUCCESS != cublasAlloc(( 2*k + ((n+31)/32)*32 )*nb, sizeof(cuDoubleComplex), (void**)&dT) ) {
        magma_xerbla("magma_zgels_gpu", info);
	return -7;
    }
    
    tau = (cuDoubleComplex*) malloc( k * sizeof(cuDoubleComplex) );
    if( tau == NULL ) {
	cublasFree(dT);
        magma_xerbla("magma_zgels_gpu", info);
	return -7;
    }

    magma_zgeqrf_gpu( m, n, dA, ldda, tau, dT, info );
    if ( *info != 0 ) {
	cublasFree(dT);
	free(tau);
	magma_xerbla("magma_zgels_gpu", info);
	return 0;
    }

    magma_zgeqrs_gpu(m, n, nrhs, 
		     dA, ldda, tau, dT, 
		     dB, lddb, hwork, lwork, info);

    cublasFree(dT);
    free(tau);

    return 0;
}

#undef a_ref
#undef t_ref
#undef d_ref
