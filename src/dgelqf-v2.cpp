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
#include "magmablas.h"
#include <stdio.h>

extern "C" int
magma_dgelqf2(int *m, int *n, double *a, int *lda, double *tau, 
	      double *work, int *lwork, double *da, int *info)
{
/*  -- MAGMA (version 0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2009

    Purpose   
    =======   

    DGELQF computes an LQ factorization of a real M-by-N matrix A:   
    A = L * Q.   

    Arguments   
    =========   

    M       (input) INTEGER   
            The number of rows of the matrix A.  M >= 0.   

    N       (input) INTEGER   
            The number of columns of the matrix A.  N >= 0.   

    A       (input/output) DOUBLE REAL array, dimension (LDA,N)   
            On entry, the M-by-N matrix A.   
            On exit, the elements on and below the diagonal of the array   
            contain the m-by-min(m,n) lower trapezoidal matrix L (L is   
            lower triangular if m <= n); the elements above the diagonal,   
            with the array TAU, represent the orthogonal matrix Q as a   
            product of elementary reflectors (see Further Details).   

            Higher performance is achieved if A is in pinned memory, e.g.
            allocated using cudaMallocHost.

    LDA     (input) INTEGER   
            The leading dimension of the array A.  LDA >= max(1,M).   

    TAU     (output) DOUBLE REAL array, dimension (min(M,N))   
            The scalar factors of the elementary reflectors (see Further   
            Details).   

    WORK    (workspace/output) DOUBLE REAL array, dimension (MAX(1,LWORK))   
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.   

            Higher performance is achieved if WORK is in pinned memory, e.g.
            allocated using cudaMallocHost.

    LWORK   (input) INTEGER   
            The dimension of the array WORK.  LWORK >= max(1,M).   
            For optimum performance LWORK >= M*NB, where NB is the   
            optimal blocksize.   

            If LWORK = -1, then a workspace query is assumed; the routine   
            only calculates the optimal size of the WORK array, returns   
            this value as the first entry of the WORK array, and no error   
            message related to LWORK is issued.

    DA      (workspace)  DOUBLE REAL array on the GPU, dimension M*(N + NB),
            where NB can be obtained through magma_get_dgeqrf_nb(M).
            (size to be reduced in upcoming versions).

    INFO    (output) INTEGER   
            = 0:  successful exit   
            < 0:  if INFO = -i, the i-th argument had an illegal value   

    Further Details   
    ===============   

    The matrix Q is represented as a product of elementary reflectors   

       Q = H(k) . . . H(2) H(1), where k = min(m,n).   

    Each H(i) has the form   

       H(i) = I - tau * v * v'   

    where tau is a real scalar, and v is a real vector with   
    v(1:i-1) = 0 and v(i) = 1; v(i+1:n) is stored on exit in A(i,i+1:n),   
    and tau in TAU(i).   

    =====================================================================    */

    #define  a_ref(a_1,a_2) ( a+(a_2)*(*lda) + (a_1))
    #define da_ref(a_1,a_2) (da+(a_2)*ldda   + (a_1))
    #define min(a,b)  (((a)<(b))?(a):(b))
    #define max(a,b)  (((a)>(b))?(a):(b))
    
    int rows, cols, i, k, ib, nx, nbmin, iinfo;
    int ldda, lddwork, old_i, old_ib;
    long int lquery;

    /* Function Body */
    *info = 0;
    int nb = magma_get_dgelqf_nb(*m); 

    work[0] = (double) *m * nb;
    lquery = *lwork == -1;
    if (*m < 0) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < max(1,*m)) {
	*info = -4;
    } else if (*lwork < max(1,*m) && ! lquery) {
	*info = -7;
    }
    if (*info != 0) {
	return 0;
    } else if (lquery) {
	return 0;
    }

    /*  Quick return if possible */
    k = min(*m,*n);
    if (k == 0) {
	work[0] = 1.f;
	return 0;
    }

    double *dwork = da + (*m)*(*n);

    static cudaStream_t stream[2];
    cudaStreamCreate(&stream[0]);
    cudaStreamCreate(&stream[1]);

    ldda = *m;
    nbmin = 2;
    nx = 192;
    lddwork = *n;

    if (*m == *n){
      cublasSetMatrix( *m, *n, sizeof(double), a, *lda, da, ldda);
      magmablas_dinplace_transpose( da, ldda, ldda );
      
      magma_dgeqrf_gpu(m, n, da, m, tau, work, lwork, dwork, &iinfo);

      magmablas_dinplace_transpose( da, ldda, ldda );
      cublasGetMatrix( *m, *n, sizeof(double), da, ldda, a, *lda); 
    }

    work[0] = (double) *m * nb;
    return 0;

    /*     End of MAGMA_DGELQF */

} /* magma_dgelqf */

#undef  a_ref
#undef da_ref
#undef min
#undef max
