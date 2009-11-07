/*
    -- MAGMA (version 0.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       June 2009
*/

#include "cuda_runtime_api.h"
#include "cublas.h"
#include "magma.h"
#include <stdio.h>
#include "magmablas.h"
int 
magma_clarfb(char direct, char storev,
	     int m, int n, int *k, float2 *dv, int *ldv, float2 *dt,
	     int *ldt, float2 *dc, int *ldc, float2 *dwork, int *ldwork)
{
/*  -- MAGMA (version 0.1) --
       Univ. of Tennessee, Univ. of California Berkeley
       June 2009

    Purpose
    =======

    CLARFB applies a real block reflector H or its transpose H' to a
    real m by n matrix C, from the left.

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

    C       (input/output) REAL array, dimension (LDC,N)
            On entry, the m by n matrix C.
            On exit, C is overwritten by H*C.

    LDC     (input) INTEGER
            The leading dimension of the array C. LDA >= max(1,M).

    WORK    (workspace) REAL array, dimension (LDWORK,K)

    LDWORK  (input) INTEGER
            The leading dimension of the array WORK. LDWORK >= max(1,N);

    ===================================================================      */

#define dwork_ref(a_1,a_2) (dwork+(a_2)*(*ldwork) + a_1)
#define dc_ref(a_1,a_2)    (dc+(a_2)*(*ldc) + a_1)
#define dv_ref(a_1,a_2)    (dv+(a_2)*(*ldv) + a_1)

  /* Function Body */
  if (m <= 0 || n <= 0) {
    return 0;
  }
  float2 cone = {1.f,0.f}, czero = {0.f, 0.f}, cmone = {-1.f, 0.f};
  
  if (storev == 'c' || storev == 'C'){
    cublasCgemm('c', 'n', n, *k, m, cone, dc_ref(0, 0), *ldc,
		dv_ref(0,0), *ldv, czero, dwork, *ldwork);
    
    if (direct == 'F' || direct =='f')
      magmablas_ctrmm('r', 'u', 'n', 'n',
		      n, *k, cone, dt, *ldt, dwork, *ldwork);
    else
      magmablas_ctrmm('r', 'l', 'n', 'n',
		      n, *k, cone, dt, *ldt, dwork, *ldwork);
    
    cublasCgemm('n', 'c', m, n, *k, cmone, dv_ref(0, 0), *ldv,
		dwork, *ldwork, cone, dc_ref(0,0), *ldc);
  }
  else {
    cublasCgemm('n', 'c', m, *k, n, cone, dc_ref(0, 0), *ldc,
                dv_ref(0,0), *ldv, czero, dwork, *ldwork);
    
    magmablas_ctrmm('r', 'u', 'n', 'n',
		    m, *k, cone, dt, *ldt, dwork, *ldwork);
    
    cublasCgemm('n', 'n', m, n, *k, cmone, 
		dwork, *ldwork,
		dv_ref(0, 0), *ldv, 
		cone, dc_ref(0,0), *ldc);

  }

  return 0;

} /* magma_clarfb */

#undef dv_ref
#undef dc_ref
#undef dwork_ref
