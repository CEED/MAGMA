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

extern "C" 
int magma_sssssm(int, int, int, int, int, float *, int,
		 float *, int, float *, int, float *, int, int *);

extern "C" int 
magma_ststrf(int M, int N, int IB, int NB,
	     float *U, int LDU,
	     float *A, int LDA,
	     float *L, int LDL,
	     int *IPIV,
	     float *WORK, int LDWORK,
	     int *INFO)
{
/*  -- MAGMA (version 1.0) --
       Univ. of Tennessee, Univ. of California Berkeley
       November 2010

    Purpose
    =======

    STSTRF computes an LU factorization of a matrix formed by an upper 
    triangular NB-by-N tile U on top of a M-by-N tile A using partial 
    pivoting with row interchanges.
 
    This is the right-looking Level 2.5 BLAS version of the algorithm.

    Arguments
    =========
    
    M       (input) INTEGER
            The number of rows of the tile A.  M >= 0.
 
    N       (input) INTEGER
            The number of columns of the tile A.  N >= 0.
 
    IB      (input) INTEGER
            The inner-blocking size.  IB >= 0.
 
    DU      (input/output) REAL array on the GPU
            On entry, the NB-by-N upper triangular tile.
            On exit, the new factor U from the factorization
 
    LDDU    (input) INTEGER
            The leading dimension of the array U.  LDU >= max(1,NB).
 
    DA      (input/output) REAL array on the GPU
            On entry, the M-by-N tile to be factored.
            On exit, the factor L from the factorization
 
    LDDA    (input) INTEGER  
            The leading dimension of the array A.  LDA >= max(1,M).
 
    DL      (input/output) REAL array on the GPU
            On entry, the NB-by-N lower triangular tile.
            On exit, the interchanged rows formthe tile A in case of pivoting.

    LDDL    (input) INTEGER
            The leading dimension of the array L.  LDL >= max(1,NB).
 
    IPIV    (output) INTEGER array on the CPU
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            tile U was interchanged with row IPIV(i) of the tile A.
 
    DWORK   (input/output) Real array on the GPU
            Work space.
 
    LDDWORK (input) INTEGER
            The dimension of the array DWORK.
 
    ===================================================================      */

  
  #define min(a,b)  (((a)<(b))?(a):(b))

  static float zzero = 0.0;
  static float mzone =-1.0;

  float alpha;
  int i, j, ii, sb;
  int im, ip;

  *INFO = 0;
 
  /*
   * Quick return
   */
  if ( (M == 0) || (N == 0) || (IB == 0))
    return 0;

  ip = 0;
  
  for(ii=0; ii<N; ii+=IB) {
    sb = min( N-ii, IB );

    for(i=0; i<sb; i++) {
      im = cublasIsamax( M, &A[LDA*(ii+i)], 1 );
      IPIV[ip] = ii+i+1;
      
      if( fabsf( A[LDA*(ii+i)+im] ) > fabsf( U[LDU*(ii+i)+ii+i] ) ) {
	/*
	 * Swap behind.
	 */
	cublasSswap(i, &L[LDL*ii+i], LDL, &WORK[im], LDWORK );
	
	/*
	 * Swap ahead.
	 */
	cublasSswap(sb-i, &U[LDU*(ii+i)+ii+i], LDU, &A[LDA*(ii+i)+im], LDA );
	
	/*
	 * Set IPIV.
	 */
	IPIV[ip] = NB + im + 1;
	
	for(j=0; j<i; j++) {
	  A[LDA*(ii+j)+im] = zzero;
	}
      }
      
      if ((*INFO == 0) && (fabsf(A[LDA*(ii+i)+im]) == zzero)
	  &&  (fabsf(U[LDU*(ii+i)+ii+i]) == zzero)) {
	*INFO = ii+i+1;
      }
      
      alpha = ((float)1. / U[LDU*(ii+i)+ii+i]);
      cublasSscal(M, (alpha), &A[LDA*(ii+i)], 1);
      cublasScopy(M, &A[LDA*(ii+i)], 1, &WORK[LDWORK*i], 1);
      cublasSger(M, sb-i-1, (mzone), &A[LDA*(ii+i)], 1,
		&U[LDU*(ii+i+1)+ii+i], LDU, &A[LDA*(ii+i+1)], LDA );
      
      ip = ip+1;
    }
    
    /*
     * Apply the subpanel to the rest of the panel.
     */
    if( (ii+i) < N ) {
      for(j=ii; j<ii+sb; j++) {
	if ( IPIV[j] <= NB ) {
	  IPIV[j] = IPIV[j] - ii;
	}
      }
      
      magma_sssssm(NB, M, N-( ii+sb ), sb, sb,
		   &U[LDU*(ii+sb)+ii], LDU,
		   &A[LDA*(ii+sb)],    LDA,
		   &L[LDL*ii],         LDL,
		   WORK, LDWORK, &IPIV[ii]);
      
      for(j=ii; j<(ii+sb); j++) {
	if (IPIV[j] <= NB) {
	  IPIV[j] = IPIV[j] + ii;
	}
      }
    }
  }
  return 0;
}

#undef min

