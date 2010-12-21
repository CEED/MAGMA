/*
    -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

       @precisions normal z -> s d c

*/

#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include "magma.h"

#define min(a,b)  (((a)<(b))?(a):(b))
#define max(a,b)  (((a)>(b))?(a):(b))

#include <pthread.h>

typedef struct {
  int flag;
  int nthreads;
  int nb;
  int ob;
  int fb;
  int np_gpu;
  int m;
  int n;
  int lda;
  cuDoubleComplex *a;
  cuDoubleComplex *t;
  pthread_t *thread;
  cuDoubleComplex **p;
  cuDoubleComplex *w;
} MAGMA_GLOBALS;



extern MAGMA_GLOBALS MG;

extern "C" magma_int_t
magma_zgeqrf3(magma_int_t m, magma_int_t n, 
             cuDoubleComplex *a,    magma_int_t lda, cuDoubleComplex *tau, 
             cuDoubleComplex *work, magma_int_t lwork,
             magma_int_t *info )
{
/*  -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

    Purpose   
    =======   

    SGEQRF computes a QR factorization of a real M-by-N matrix A:   
    A = Q * R.   

    Arguments   
    =========   

    M       (input) INTEGER   
            The number of rows of the matrix A.  M >= 0.   

    N       (input) INTEGER   
            The number of columns of the matrix A.  N >= 0.   

    A       (input/output) REAL array, dimension (LDA,N)   
            On entry, the M-by-N matrix A.   
            On exit, the elements on and above the diagonal of the array   
            contain the min(M,N)-by-N upper trapezoidal matrix R (R is   
            upper triangular if m >= n); the elements below the diagonal,   
            with the array TAU, represent the orthogonal matrix Q as a   
            product of min(m,n) elementary reflectors (see Further   
            Details).   

    LDA     (input) INTEGER   
            The leading dimension of the array A.  LDA >= max(1,M).   

    TAU     (output) REAL array, dimension (min(M,N))   
            The scalar factors of the elementary reflectors (see Further   
            Details).   

    WORK    (workspace/output) REAL array, dimension (MAX(1,LWORK))   
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.   

    LWORK   (input) INTEGER   
            The dimension of the array WORK.  LWORK >= N*NB. 

            If LWORK = -1, then a workspace query is assumed; the routine   
            only calculates the optimal size of the WORK array, returns   
            this value as the first entry of the WORK array, and no error   
            message related to LWORK is issued.

    INFO    (output) INTEGER   
            = 0:  successful exit   
            < 0:  if INFO = -i, the i-th argument had an illegal value   

    Further Details   
    ===============   

    The matrix Q is represented as a product of elementary reflectors   

       Q = H(1) H(2) . . . H(k), where k = min(m,n).   

    Each H(i) has the form   

       H(i) = I - tau * v * v'   

    where tau is a real scalar, and v is a real vector with   
    v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in A(i+1:m,i),   
    and tau in TAU(i).   

    ====================================================================    */

  cuDoubleComplex c_one = MAGMA_Z_ONE;

  int k, ib;

  *info = 0;

  /* Check arguments */
  int lwkopt = n * MG.nb;
  work[0] = MAGMA_Z_MAKE( (double)lwkopt, 0 );
  long int lquery = (lwork == -1);
  if (m < 0) {
    *info = -1;
  } else if (n < 0) {
    *info = -2;
  } else if (lda < max(1,m)) {
    *info = -4;
  } else if (lwork < max(1,n) && ! lquery) {
    *info = -7;
  }
  if (*info != 0)
    return 0;
  else if (lquery)
    return 0;
  k = min(m,n);
  if (k == 0) {
    work[0] = c_one;
    return 0;
  }

  int M=MG.nthreads*MG.ob;
  int N=MG.nthreads*MG.ob;

  if (MG.m > MG.n) {
    M = MG.m - (MG.n-MG.nthreads*MG.ob);
  }

  /* Use MAGMA code to factor left portion of matrix, waking up threads 
	 along the way to perform updates on the right portion of matrix */
  magma_zgeqrf2(m,n-MG.nthreads*MG.ob, a, m, tau, work, lwork, info);

  /* Wait for all update threads to finish */
  for (k = 0; k < MG.nthreads; k++){
    pthread_join(MG.thread[k], NULL);
  }

  for (k = 0; k < MG.np_gpu-1; k++){
    ib = min(MG.nb,(n-MG.nthreads*MG.ob)-MG.nb*k);
    zq_to_panel(MagmaUpper, ib, a+k*MG.nb*lda+k*MG.nb, lda, MG.w+MG.nb*MG.nb*k);
  }


  MG.nb = MG.fb;

  MG.flag = 1;

  /* Use MAGMA code to perform final factorization if necessary */
  if (MG.m >= (MG.n - (MG.nthreads*MG.ob))) {
    magma_zgeqrf2(M, N, a+(n-MG.nthreads*MG.ob)*m+(n-MG.nthreads*MG.ob), m, 
                  &tau[n-MG.nthreads*MG.ob], work, lwork, info);
  }
 
}

#undef min
#undef max
