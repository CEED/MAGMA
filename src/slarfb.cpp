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
magma_slarfb(char direct, char storev,
	     int m, int n, int *k, float *dv, int *ldv, float *dt,
	     int *ldt, float *dc, int *ldc, float *dwork, int *ldwork)
{
/*  -- MAGMA (version 1.0) --
       Univ. of Tennessee, Univ. of California Berkeley
       November 2010

    Purpose
    =======

    SLARFB applies a real block reflector H or its transpose H' to a
    real m by n matrix C, from the left.

    Arguments
    =========

    DIRECT  (input) CHARACTER
            Indicates how H is formed from a product of elementary
            reflectors
            = 'F': H = H(1) H(2) . . . H(k) (Forward)
            = 'B': H = H(k) . . . H(2) H(1) (Backward)

    STOREV  (input) CHARACTER
            Indicates how the vectors which define the elementary
            reflectors are stored:
            = 'C': Columnwise
            = 'R': Rowwise

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
 
  if (storev == 'c' || storev == 'C'){
    /*
    if (n==1 && m%32==0){
      // This is used when we have to apply H on only one vector 
      magmablas_sgemvt(m, *k, 1., dv_ref(0,0), *ldv, dc_ref(0, 0), dwork);
      printf("m= %d, n = %d, ldwork = %d\n", m, *k, *ldwork);
    }
    else
    */
    //TimeStruct start, end;
    //start = get_current_time();
    cublasSgemm('t', 'n', n, *k, m, 1.f, dc_ref(0, 0), *ldc,
		dv_ref(0,0), *ldv, 0.f, dwork, *ldwork);
    
    if (direct == 'F' || direct =='f')
      cublasStrmm('r', 'u', 'n', 'n',
		  n, *k, 1.f, dt, *ldt, dwork, *ldwork);
    else
      cublasStrmm('r', 'l', 'n', 'n',
		  n, *k, 1.f, dt, *ldt, dwork, *ldwork);

    cublasSgemm('n', 't', m, n, *k, -1.f, dv_ref(0, 0), *ldv,
		dwork, *ldwork, 1.f, dc_ref(0,0), *ldc);
    //end = get_current_time();
    //if (n!=*k)
    //printf("%5d %5d  %7.2f\n",
    //	   m, n, (4.*n*(*k)*m+n*(*k)*(*k))/(1.e6*GetTimerValue(start,end)));
  }
  else {
    cublasSgemm('n', 't', m, *k, n, 1.f, dc_ref(0, 0), *ldc,
                dv_ref(0,0), *ldv, 0.f, dwork, *ldwork);
    
    cublasStrmm('r', 'u', 'n', 'n',
		m, *k, 1.f, dt, *ldt, dwork, *ldwork);
    
    cublasSgemm('n', 'n', m, n, *k, -1.f, 
		dwork, *ldwork,
		dv_ref(0, 0), *ldv, 
		1.f, dc_ref(0,0), *ldc);
    /*
    float one = 1.f, zero = 0.f, mone = -1.f;
    sgemm_("n", "t", &m, k, &n, &one, dc_ref(0, 0), ldc,
	  dv_ref(0,0), ldv, &zero, dwork, ldwork);

    strmm_("r", "u", "n", "n",
	   &m, k, &one, dt, ldt, dwork, ldwork);

    sgemm_("n", "n", &m, &n, k, &mone,
                dwork, ldwork,
                dv_ref(0, 0), ldv,
      		&one, dc_ref(0,0), ldc);
    */
  }
  return 0;

} /* magma_slarfb */

#undef dv_ref
#undef dc_ref
#undef dwork_ref
