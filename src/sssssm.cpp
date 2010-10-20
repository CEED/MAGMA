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
magma_sssssm(int M1, int M2, int N, int IB, int K,
	     float *A1, int LDA1,
	     float *A2, int LDA2,
	     float *L1, int LDL1,
	     float *L2, int LDL2,
	     int *IPIV)
{
/*  -- MAGMA (version 1.0) --
       Univ. of Tennessee, Univ. of California Berkeley
       November 2010

    Purpose
    =======

    SSSSSM applies the blocked update transformations from a tile LU algorithm 
    to a m+k by n matrix /A1\ , from the left.
                         \A2/
    A1 is k by n, A2 m by n.

    Arguments
    =========

    M1      (input) INTEGER
            The number of rows of the matrix A1.

    M2      (input) INTEGER
            The number of rows of the matrix A2.

    N       (input) INTEGER
            The number of columns of the matrix A1/A2.

    IB      (input) INTEGER
            The inner-blocking size. IB >= 0.

    K       (input) INTEGER

    DA1     (input/output) REAL array on the GPU, dimension (LDA1,N)
            On entry, the M1 by N matrix A1.
            On exit, /A1\ is overwritten by the transformed /A1\
                     \A2/                                   \A2/.

    LDA1    (input) INTEGER
            The leading dimension of the array DA1. LDA1 >= max(1,M1).

    DA2     (input/output) REAL array on the GPU, dimension (LDA2,N)
            On entry, the M2 by N matrix A2.
            On exit, /A1\ is overwritten by the transformed /A1\
                     \A2/                                   \A2/.

    LDA2    (input) INTEGER
            The leading dimension of the array DA2. LDA2 >= max(1,M2).

    DL1     (input) REAL array on the GPU, dimension (LDL1,N)
            On entry, the M1 by N matrix L1.

    LDL1    (input) INTEGER
            The leading dimension of the array DL1. LDL1 >= max(1,M1).

    DL2     (input) REAL array on the GPU, dimension (LDL2,N)
            On entry, the M2 by N matrix L2.

    LDL2    (input) INTEGER
            The leading dimension of the array DL2. LDL2 >= max(1,M2).

    IPIV    (input) INTEGER array on the CPU, dimension M1. 
            Pivoting indexes as returned by the ststrf routine.

    WORK    (workspace) REAL array, dimension (LDWORK,N)

    LDWORK  (input) INTEGER
            The leading dimension of the array WORK. LDWORK >= max(1,2*K);

    ===================================================================      */

  #define min(a,b)  (((a)<(b))?(a):(b))
  #define max(a,b)  (((a)>(b))?(a):(b))

  static float zone  = 1.0;
  static float mzone =-1.0;

  int i, ii, sb;
  int im, ip;

  ip = 0;
  
  for(ii=0; ii<K; ii+=IB) {
    sb = min( K-ii, IB );

    for(i=0; i<IB; i++) {
      im = IPIV[ip]-1;
      
      if (im != (ii+i)) {
	im = im - M1;
	cublasSswap(N, &A1[ii+i], LDA1, &A2[im], LDA2 );
      }
      ip = ip + 1;
    }
    
    cublasStrsm('L', 'L', 'N', 'U',
		sb, N, (zone),
		&L1[LDL1*ii], LDL1,
		&A1[ii],      LDA1);
    
    cublasSgemm('N', 'N',
		M2, N, sb,
		(mzone), &L2[LDL2*ii], LDL2,
		&A1[ii], LDA1,
		(zone), A2, LDA2);
  }

  return 0;
}

#undef min
#undef max

