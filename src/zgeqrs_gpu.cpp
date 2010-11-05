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
magma_zgeqrs_gpu(magma_int_t m_, magma_int_t n_, magma_int_t nrhs_, 
		 double2 *a, magma_int_t lda_, double2 *tau, double2 *c, magma_int_t ldc_, 
		 double2 *work, magma_int_t *lwork, double2 *td, magma_int_t *info)
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

    A       (input) COMPLEX_16 array on the GPU, dimension (LDA,N)   
            The i-th column must contain the vector which defines the   
            elementary reflector H(i), for i = 1,2,...,n, as returned by   
            ZGEQRF_GPU2 in the first n columns of its array argument A.

    LDA     (input) INTEGER   
            The leading dimension of the array A, LDA >= M.

    TAU     (input) COMPLEX_16 array, dimension (N)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by MAGMA_ZGEQRF_GPU2.

    C       (input/output) COMPLEX_16 array on the GPU, dimension (LDC,NRHS)   
            On entry, the M-by-NRHS matrix C.
            On exit, the N-by-NRHS solution matrix X.

    LDC     (input) INTEGER   
            The leading dimension of the array C. LDC >= M.   

    WORK    (workspace/output) COMPLEX_16 array, dimension (LWORK)   
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.   

    LWORK   (input) INTEGER   
            The dimension of the array WORK, LWORK >= max(1,NRHS).   
            For optimum performance LWORK >= (M-N+NB+2*NRHS)*NB, where NB is 
            the blocksize given by magma_get_zgeqrf_nb( M ).

            If LWORK = -1, then a workspace query is assumed; the routine   
            only calculates the optimal size of the WORK array, returns   
            this value as the first entry of the WORK array.   

    TD      (input) COMPLEX_16 array that is the output (the 9th argument)
            of magma_zgeqrf_gpu2.

    INFO    (output) INTEGER   
            = 0:  successful exit   
            < 0:  if INFO = -i, the i-th argument had an illegal value   

    =====================================================================    */

   #define a_ref(a_1,a_2) ( a+(a_2)*(*lda) + (a_1))
   #define t_ref(a_1)     (td+(a_1))
   #define d_ref(a_1)     (td+(lddwork+(a_1))*nb)
   #define min(a,b)       (((a)<(b))?(a):(b))
   #define max(a,b)       (((a)>(b))?(a):(b))

   double2 c_zero = MAGMA_Z_ZERO;
   double2 c_one = MAGMA_Z_ONE;
   double2 c_neg_one = MAGMA_Z_NEG_ONE;

   int *m = &m_;
   int *n = &n_;
   int *nrhs = &nrhs_;
   int *lda = &lda_;
   int *ldc = &ldc_;

   double2 *dwork;
   int i, k, lddwork, rows, ib;

   /* Function Body */
   *info = 0;
   int nb = magma_get_zgeqrf_nb(*m);
   
   int lwkopt = (*m-*n+nb+2*(*nrhs)) * nb;
   MAGMA_Z_SET2REAL( work[0], (double) lwkopt );
   long int lquery = *lwork == -1;
   if (*m < 0)
     *info = -1;
   else if (*n < 0 || *m < *n)
     *info = -2;
   else if (*nrhs < 0)
     *info = -3;
   else if (*lda < max(1,*m))
     *info = -5;
   else if (*ldc < max(1,*m))
     *info = -8;
   else if (*lwork < lwkopt && ! lquery)
     *info = -10;
   
   if (*info != 0)
     return 0;
   else if (lquery)
     return 0;

   k = min(*m,*n);
   if (k == 0) {
     work[0] = c_one;
     return 0;
   }

   magma_zunmqr_gpu('L', 'T', m_, nrhs_, n_,
                    a_ref(0,0), lda_, tau, c, ldc_,
                    work, lwork, td, nb, info);

   lddwork= k;
   dwork = td+2*lddwork*nb;

   i    = (k-1)/nb * nb;
   ib   = *n-i;
   rows = *m-i;
   double2 one = MAGMA_Z_ONE;
   ztrsm_("l", "u", "n", "n", &ib, nrhs, &one, work, &rows,
	  work+rows*ib, &rows);
   
   // update the solution vector
   cublasSetMatrix(rows, *nrhs, sizeof(double2),
		   work+rows*ib, rows, dwork+i, *ldc);
   
   // update c
   if (*nrhs == 1)
     cublasZgemv('n', i, ib, c_neg_one, a_ref(0, i), *lda,
		 dwork + i, 1, c_one, c, 1);
   else
     cublasZgemm('n', 'n', i, *nrhs, ib, c_neg_one, a_ref(0, i), *lda,
		 dwork + i, *ldc, c_one, c, *ldc);

   int start = i-nb;
   if (nb < k) {
     for (i = start; i >=0; i -= nb) {
       ib = min(k-i, nb);
       rows = *m -i;

       if (i + ib < *n) {
	 if (*nrhs == 1)
	   {
	     cublasZgemv('n', ib, ib, c_one, d_ref(i), ib,
			 c+i, 1, c_zero, dwork + i, 1);
	     cublasZgemv('n', i, ib, c_neg_one, a_ref(0, i), *lda,
			 dwork + i, 1, c_one, c, 1);
	   }
	 else
	   {
	     cublasZgemm('n', 'n', ib, *nrhs, ib, c_one, d_ref(i), ib,
                         c+i, *ldc, c_zero, dwork + i, *ldc);
             cublasZgemm('n', 'n', i, *nrhs, ib, c_neg_one, a_ref(0, i), *lda,
                         dwork + i, *ldc, c_one, c, *ldc);
	   }
       }
     }
   }

   if (*nrhs==1)
     cublasZcopy(*n, dwork, 1, c, 1);
   else
     cudaMemcpy2D(c, (*ldc)*sizeof(double2),
		  dwork, (*ldc)*sizeof(double2),
		  (*n)*sizeof(double2), *nrhs, cudaMemcpyDeviceToDevice);

   return 0; 
}

#undef a_ref
#undef t_ref
#undef d_ref
