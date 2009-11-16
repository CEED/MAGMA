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
#include <stdio.h>
#include <stdlib.h>


int 
magma_dgeqrs_gpu(int *m, int *n, int *nrhs, 
		 double *a, int *lda, double *tau, double *c, int *ldc, 
		 double *work, int *lwork, double *td, int *info)
{
/*  -- MAGMA (version 0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2009

    Purpose   
    =======   

    Solve the least squares problem
           min || A*X - C ||
    using the QR factorization A = Q*R computed by SGEQRF_GPU2.


    Arguments   
    =========

    M       (input) INTEGER   
            The number of rows of the matrix A. M >= 0.   

    N       (input) INTEGER
            The number of columns of the matrix A. M >= N >= 0.

    NRHS    (input) INTEGER   
            The number of columns of the matrix C. NRHS >= 0.   

    A       (input) DOUBLE REAL array on the GPU, dimension (LDA,N)   
            The i-th column must contain the vector which defines the   
            elementary reflector H(i), for i = 1,2,...,n, as returned by   
            SGEQRF_GPU2 in the first n columns of its array argument A.

    LDA     (input) INTEGER   
            The leading dimension of the array A, LDA >= M.

    TAU     (input) DOUBLE REAL array, dimension (N)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by MAGMA_DGEQRF_GPU2.

    C       (input/output) DOUBLE REAL array on the GPU, dimension (LDC,NRHS)   
            On entry, the M-by-NRHS matrix C.
            On exit, the N-by-NRHS solution matrix X.

    LDC     (input) INTEGER   
            The leading dimension of the array C. LDC >= M.   

    WORK    (workspace/output) DOUBLE REAL array, dimension (LWORK)   
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.   

    LWORK   (input) INTEGER   
            The dimension of the array WORK, LWORK >= max(1,NRHS).   
            For optimum performance LWORK >= (M-N+NB+2*NRHS)*NB, where NB is 
            the blocksize given by magma_get_dgeqrf_nb( M ).

            If LWORK = -1, then a workspace query is assumed; the routine   
            only calculates the optimal size of the WORK array, returns   
            this value as the first entry of the WORK array.   

    TD      (input) DOUBLE REAL array that is the output (the 9th argument)
            of magma_dgeqrf_gpu2.

    INFO    (output) INTEGER   
            = 0:  successful exit   
            < 0:  if INFO = -i, the i-th argument had an illegal value   

    =====================================================================    */

   #define a_ref(a_1,a_2) ( a+(a_2)*(*lda) + (a_1))
   #define t_ref(a_1)     (td+(a_1))
   #define d_ref(a_1)     (td+(lddwork+(a_1))*nb)
   #define min(a,b)       (((a)<(b))?(a):(b))
   #define max(a,b)       (((a)>(b))?(a):(b))

   double *dwork;
   int i, k, lddwork, rows, ib;

   /* Function Body */
   *info = 0;
   int nb = magma_get_dgeqrf_nb(*m);
   
   int lwkopt = (*m-*n+nb+2*(*nrhs)) * nb;
   work[0] = (double) lwkopt;
   long int lquery = *lwork == -1;
   if (*m < 0)
     *info = -1;
   else if (*n < 0)
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
     work[0] = 1.f;
     return 0;
   }

   lddwork= k;
   dwork = td+2*lddwork*nb;

   if (nb < k) {
      /* Use blocked code initially */
      for (i = 0; i < k-nb; i += nb) {
	ib = min(k-i, nb);
	rows = *m -i;

	if (i + ib < *n)
	  magma_dlarfb('F','C',rows, *nrhs, &ib, a_ref(i,i), lda, t_ref(i),
		       &lddwork, c + i, ldc, dwork, nrhs);
      }
   } else {
     i = 0;
   }
   
   /* Use unblocked code to multiply the last or only block. */
   if (i < k) {
      ib   = *n-i;
      rows = *m-i;

      cublasGetMatrix(rows, ib, sizeof(double), a_ref(i,i), *lda, work, rows);
      cublasGetMatrix(rows, *nrhs, sizeof(double), c+i, *ldc, 
		      work+rows*ib, rows);
      int lhwork = *lwork - rows*(ib+*nrhs);
      dormqr_("l", "t", &rows, nrhs, &ib, work, &rows,
	      tau+i, work+rows*ib, &rows, work+rows*(ib+*nrhs), &lhwork, info);
    
      cublasSetMatrix(rows, *nrhs, sizeof(double),
                      work+rows*ib, rows, c+i, *ldc);

      double one = 1.;
      dtrsm_("l", "u", "n", "n", &rows, nrhs, &one, work, &rows,
	     work+rows*ib, &rows);

      // update the solution vector
      cublasSetMatrix(rows, *nrhs, sizeof(double),
		      work+rows*ib, rows, dwork+i, *ldc);

      // update c
      cublasDgemv('n', i, ib, -1.f, a_ref(0, i), *lda,
		  dwork + i, 1, 1.f, c, 1);
   }
   
   int start = i-nb;
   if (nb < k) {
     for (i = start; i >=0; i -= nb) {
       ib = min(k-i, nb);
       rows = *m -i;

       if (i + ib < *n) {
	 cublasDgemv('n', ib, ib, 1.f, d_ref(i), ib,
		     c+i, 1, 0.f, dwork + i, 1);
	 cublasDgemv('n', i, ib, -1.f, a_ref(0, i), *lda,
		     dwork + i, 1, 1.f, c, 1);
       }
     }
   }

   cublasDcopy(*n, dwork, 1, c, 1);

   return 0; 
}

#undef a_ref
#undef t_ref
#undef d_ref
#undef min
#undef max
