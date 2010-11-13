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
#include "magmablas.h"
 
extern "C" magma_int_t 
magma_zlarfb(char direct, char storev,
	     magma_int_t m, magma_int_t n, magma_int_t k, double2 *dv, magma_int_t ldv, double2 *dt,
	     magma_int_t ldt, double2 *dc, magma_int_t ldc, double2 *dwork, magma_int_t ldwork)
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

    DV      (input) COMPLEX_16 array, dimension (LDV,K)
            The matrix V. See further details.

    LDV     (input) INTEGER
            The leading dimension of the array V. LDV >= max(1,M);

    DT      (input) COMPLEX_16 array, dimension (LDT,K)
            The triangular k by k matrix T in the representation of the
            block reflector.

    LDT     (input) INTEGER
            The leading dimension of the array T. LDT >= K.

    DC      (input/output) COMPLEX_16 array, dimension (LDC,N)
            On entry, the m by n matrix C.
            On exit, C is overwritten by H*C.

    LDC     (input) INTEGER
            The leading dimension of the array C. LDA >= max(1,M).

    WORK    (workspace) COMPLEX_16 array, dimension (LDWORK,K)

    LDWORK  (input) INTEGER
            The leading dimension of the array WORK. LDWORK >= max(1,N);

    ===================================================================      */

  #define dwork_ref(a_1,a_2) (dwork+(a_2)*(ldwork) + a_1)
  #define dc_ref(a_1,a_2)    (dc+(a_2)*(ldc) + a_1)
  #define dv_ref(a_1,a_2)    (dv+(a_2)*(ldv) + a_1)

  double2 c_zero = MAGMA_Z_ZERO;
  double2 c_one = MAGMA_Z_ONE;
  double2 c_neg_one = MAGMA_Z_NEG_ONE;

  /* Function Body */
  if (m <= 0 || n <= 0) {
    return 0;
  }
 
  if (storev == 'c' || storev == 'C'){
    /*
    if (n==1 && m%32==0){
      // This is used when we have to apply H on only one vector 
      magmablas_zgemvt(m, k, 1., dv_ref(0,0), ldv, dc_ref(0, 0), dwork);
      printf("m= %d, n = %d, ldwork = %d\n", m, k, ldwork);
    }
    else
    */
    //TimeStruct start, end;
    //start = get_current_time();
    cublasZgemm('t', 'n', n, k, m, c_one, dc_ref(0, 0), ldc,
		dv_ref(0,0), ldv, c_zero, dwork, ldwork);
    
    if (direct == 'F' || direct =='f')
      cublasZtrmm('r', 'u', 'n', 'n',
		  n, k, c_one, dt, ldt, dwork, ldwork);
    else
      cublasZtrmm('r', 'l', 'n', 'n',
		  n, k, c_one, dt, ldt, dwork, ldwork);

    cublasZgemm('n', 't', m, n, k, c_neg_one, dv_ref(0, 0), ldv,
		dwork, ldwork, c_one, dc_ref(0,0), ldc);
    //end = get_current_time();
    //if (n!=k)
    //printf("%5d %5d  %7.2f\n",
    //	   m, n, (4.*n*(k)*m+n*(k)*(k))/(1.e6*GetTimerValue(start,end)));
  }
  else {
    cublasZgemm('n', 't', m, k, n, c_one, dc_ref(0, 0), ldc,
                dv_ref(0,0), ldv, c_zero, dwork, ldwork);
    
    cublasZtrmm('r', 'u', 'n', 'n',
		m, k, c_one, dt, ldt, dwork, ldwork);
    
    cublasZgemm('n', 'n', m, n, k, c_neg_one, 
		dwork, ldwork,
		dv_ref(0, 0), ldv, 
		c_one, dc_ref(0,0), ldc);
    /*
    double2 one = 1.f, zero = 0.f, mone = -1.f;
    blasf77_zgemm("n", "t", &m, k, &n, &one, dc_ref(0, 0), ldc,
	  dv_ref(0,0), ldv, &zero, dwork, ldwork);

    blasf77_ztrmm("r", "u", "n", "n",
	   &m, k, &one, dt, ldt, dwork, ldwork);

    blasf77_zgemm("n", "n", &m, &n, k, &mone,
                dwork, ldwork,
                dv_ref(0, 0), ldv,
      		&one, dc_ref(0,0), ldc);
    */
  }
  return 0;

} /* magma_zlarfb */

#undef dv_ref
#undef dc_ref
#undef dwork_ref
