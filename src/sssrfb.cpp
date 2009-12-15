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
magma_sssrfb(int m, int n, int *k, float *dv, int *ldv, float *dt, int *ldt, 
	     float *da1, int *lda1, float *da2, int *lda2, 
	     float *dwork, int *ldwork)
{
/*  -- MAGMA (version 0.2) --
       Univ. of Tennessee, Univ. of California Berkeley
       November 2009

    Purpose
    =======

    SSSRFB applies a real block reflector H or its transpose H' to a
    to a m+k by n matrix /A1\ , from the left.
                         \A2/
    A1 is k by n, A2 m by n, and H =   I - / I \  T  / I \' .
                                           \ V /     \ V /

    Arguments
    =========

    M       (input) INTEGER
            The number of rows of the matrix C.

    N       (input) INTEGER
            The number of columns of the matrix C.

    K       (input) INTEGER
            The order of the matrix T (= the number of elementary
            reflectors whose product defines the block reflector).

    V       (input) REAL array, dimension (LDV,K)
            The matrix V. See further details.

    LDV     (input) INTEGER
            The leading dimension of the array V. LDV >= max(1,M);

    T       (input) REAL array, dimension (LDT,K)
            The triangular k by k matrix T in the representation of the
            block reflector.

    LDT     (input) INTEGER
            The leading dimension of the array T. LDT >= K.

    DA1     (input/output) REAL array, dimension (LDA1,N)
            On entry, the k by n matrix A1.
            On exit, /A1\ is overwritten by H /A1\
                     \A2/                     \A2/.
    
    LDA2     (input) INTEGER
            The leading dimension of the array DA1. LDA1 >= max(1,K).

    DA2     (input/output) REAL array, dimension (LDA2,N)
            On entry, the m by n matrix A2.
            On exit, /A1\ is overwritten by H /A1\
                     \A2/                     \A2/.

    LDA2     (input) INTEGER
            The leading dimension of the array A2. LDA2 >= max(1,M).

    WORK    (workspace) REAL array, dimension (LDWORK,N)

    LDWORK  (input) INTEGER
            The leading dimension of the array WORK. LDWORK >= max(1,2*K);

    ===================================================================      */

  #define dwork_ref(a_1,a_2) (dwork+(a_2)*(*ldwork) + a_1)
  #define da2_ref(a_1,a_2)   (da2+(a_2)*(*lda2) + a_1)
  #define dv_ref(a_1,a_2)    (dv+(a_2)*(*ldv) + a_1)

  /* Function Body */
  if (m <= 0 || n <= 0) {
    return 0;
  }

  /* 1. dwork = A1 where A1 is of dimension k by n */
  cudaMemcpy2D(dwork, (*ldwork) * sizeof(float),
	       da1  , (*lda1)   * sizeof(float),
	       sizeof(float)*(*k), n,
	       cudaMemcpyDeviceToDevice);
  
  /* 2. dwork = dwork + V' A2.
        TTT : steps 1 & 2 are going to be fused in one kernel by Rajib */
  cublasSgemm('t', 'n', *k, n, m, 1.f, dv_ref(0, 0), *ldv,
	      da2_ref(0,0), *lda2, 1.f, dwork, *ldwork);

  /* 3. (dwork+k) = T dwork 
        T is triangular, assumed to have 0s in the unused part */
  cublasSgemm('t', 'n', *k, n, *k, 1.f, dt, *ldt, dwork, *ldwork,
	      0.f, dwork+(*k), *ldwork);

  /* 4. A1 = A1 - (dwork+k)
        TTT : steps 3 & 4 are going to be fused by Rajib 
	      for now I have this for completeness and testing
              (as it is going to be extremely slow in the loop) */
  for(int i=0; i<n; i++)
    cublasSaxpy(*k, -1.f, dwork+(*k) + i*(*ldwork), 1, da1+i*(*lda1), 1);

  /* 5. A2 = A2 - V (dwork+k) */
  cublasSgemm('n', 'n', m, n, *k, -1.f, dv_ref(0, 0), *ldv,
	      dwork+(*k), *ldwork, 1.f, da2_ref(0,0), *lda2);
  
  return 0;

} /* magma_sssrfb */

#undef dv_ref
#undef da2_ref
#undef dwork_ref
