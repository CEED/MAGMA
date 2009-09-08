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

int 
magma_dlarfb(int m, int n, int *k, double *dv, int *ldv, double *dt,
	    int *ldt, double *dc, int *ldc, double *dwork, int *ldwork)
{
/*  -- MAGMA (version 0.1) --
       Univ. of Tennessee, Univ. of California Berkeley
       June 2009

    Purpose
    =======

    DLARFB applies a real block reflector H or its transpose H' to a
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

    V       (input) DOUBLE array, dimension (LDV,K)
            The matrix V. See further details.

    LDV     (input) INTEGER
            The leading dimension of the array V. LDV >= max(1,M);

    T       (input) DOUBLE array, dimension (LDT,K)
            The triangular k by k matrix T in the representation of the
            block reflector.

    LDT     (input) INTEGER
            The leading dimension of the array T. LDT >= K.

    C       (input/output) DOUBLE array, dimension (LDC,N)
            On entry, the m by n matrix C.
            On exit, C is overwritten by H*C.

    LDC     (input) INTEGER
            The leading dimension of the array C. LDA >= max(1,M).

    WORK    (workspace) DOUBLE array, dimension (LDWORK,K)

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

  cublasDgemm('t', 'n', n, *k, m, 1.f, dc_ref(0, 0), *ldc,
	      dv_ref(0,0), *ldv, 0.f, dwork, *ldwork);

  cublasDtrmm('r', 'u', 'n', 'n',
	      n, *k, 1.f, dt, *ldt, dwork, *ldwork);

  cublasDgemm('n', 't', m, n, *k, -1.f, dv_ref(0, 0), *ldv,
	      dwork, *ldwork, 1.f, dc_ref(0,0), *ldc);

  return 0;

} /* magma_dlarfb */

#undef dv_ref
#undef dc_ref
#undef dwork_ref
