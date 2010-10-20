/*
    -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010
*/

#include "cuda_runtime_api.h"
#include "cublas.h"
#include "magma.h"
#include "magmablas.h"
#include <stdio.h>

extern "C" int 
magma_sgessm(int M, int N, int K, int IB, int *IPIV,
	     float *L, int LDL, float *A, int LDA)
{
/*  -- MAGMA (version 1.0) --
       Univ. of Tennessee, Univ. of California Berkeley
       November 2010

    Purpose
    =======

    SGESSM applies the factor L computed by magma_sgetrf_gpu to
    M-by-N tile A.

    Arguments
    =========

    M       (input) INTEGER
            The number of rows of the tile A.  M >= 0.
 
    N       (input) INTEGER
            The number of columns of the tile A.  N >= 0.
 
    K       (input) INTEGER
   
    IB      (input) INTEGER
            The inner-blocking size.  IB >= 0.
 
    IPIV    (output) INTEGER array on the CPU
            Pivots as returned by magma_sgetrf_gpu.
 
    DL      (input) REAL array on the GPU
            The NB-by-NB lower triangular tile.
 
    LDDL    (input) INTEGER
            The leading dimension of the array L.  LDL >= max(1,NB).
  
    DA      (input/output) REAL array on the GPU.
            On entry, the M-by-N tile A.
            On exit, updated by the application of L.
  
    LDDA    (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).
 
    ===================================================================      */

  #define min(a,b)  (((a)<(b))?(a):(b))

  static float zone  = 1.0;
  static float mzone =-1.0;
  static int   ione  = 1;

  int i, sb;
  int tmp, tmp2;

  /* Quick return */
  if ( (M == 0) || (N == 0) || (K == 0) || (IB == 0))
    return 0;

  for(i=0; i<K; i+=IB) {
    sb = min(IB, K-i);

    /*
     * Apply interchanges to columns I*IB+1:IB*( I+1 )+1.
     */
    tmp  = i+1;
    tmp2 = i+sb;
    for(int j=tmp; j<=tmp2; j++)
      cublasSswap(N, &A[j], LDA, &A[ IPIV[j] ], LDA);
    // lapack_slaswp( N, A, LDA, tmp, tmp2, IPIV, ione );
    
    /*
     * Compute block row of U.
     */
    cublasStrsm('L', 'L', 'N', 'U',
		sb, N, (zone), &L[LDL*i+i], LDL, &A[i], LDA);
    
    if ( (i+sb) < M ) {
      /*
       * Update trailing submatrix.
       */
      cublasSgemm('N', 'N',  M-( i+sb ), N, sb, (mzone), &L[LDL*i+(i+sb)], LDL, 
		  &A[i], LDA, (zone), &A[i+sb], LDA );
    }
  }
  return 0;
}

#undef min
