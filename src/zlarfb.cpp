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

// === Define what BLAS to use ============================================
#define PRECISION_z
#if (defined(PRECISION_s) || defined(PRECISION_d))
  #define cublasZgemm magmablas_zgemm
#endif
// === End defining what BLAS to use =======================================

extern "C" magma_int_t
magma_zlarfb(char side, char trans, char direct, char storev,
	     magma_int_t m, magma_int_t n, magma_int_t k,
             cuDoubleComplex *dv, magma_int_t ldv, cuDoubleComplex *dt,
	     magma_int_t ldt,
             cuDoubleComplex *dc, magma_int_t ldc, 
	     cuDoubleComplex *dwork, magma_int_t ldwork)
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

    SIDE    (input) CHARACTER
            = 'L': apply H or H' from the Left
            = 'R': apply H or H' from the Right (Not implemented)

    TRANS   (input) CHARACTER
            = 'N': apply H  (No transpose)      (Not implemented)
            = 'C': apply H' (Conjugate transpose)

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

#define dwork_ref(a_1,a_2) (dwork+(a_2)*(ldwork)+ a_1)
#define dc_ref(a_1,a_2)    (dc   +(a_2)*(ldc)   + a_1)
#define dv_ref(a_1,a_2)    (dv   +(a_2)*(ldv)   + a_1)

    cuDoubleComplex c_zero    = MAGMA_Z_ZERO;
    cuDoubleComplex c_one     = MAGMA_Z_ONE;
    cuDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;

    /* Function Body */
    if (m <= 0 || n <= 0) {
        return 0;
    }

    char transt;
    if (trans == 'N' || trans == 'n')
      transt = MagmaConjTrans;
    else
      transt = MagmaNoTrans;

    if ( ( side  == 'r' || side  == 'R') ) 
      {
        magma_int_t info = -1;
        fprintf(stderr, "The case (side == right) is not implemented\n");
        magma_xerbla(__func__, &info);
        return 0;
      }

    if ( storev == 'c' || storev == 'C') {
        /*
          if (n==1 && m%32==0){
          // This is used when we have to apply H on only one vector
          magmablas_zgemvt(m, k, 1., dv_ref(0,0), ldv, dc_ref(0, 0), dwork);
          printf("m= %d, n = %d, ldwork = %d\n", m, k, ldwork);
          }
          else
        */
        cublasZgemm( MagmaConjTrans, MagmaNoTrans,
                     n, k, m,
                     c_one,  dc_ref(0, 0), ldc,
                             dv_ref(0, 0), ldv,
                     c_zero, dwork,        ldwork);

        if (direct == 'F' || direct =='f')
            cublasZtrmm( MagmaRight, MagmaUpper, transt, MagmaNonUnit,
                         n, k, 
                         c_one, dt,    ldt, 
                                dwork, ldwork);
        else
            cublasZtrmm( MagmaRight, MagmaLower, transt, MagmaNonUnit,
                         n, k, 
                         c_one, dt,    ldt, 
                                dwork, ldwork);

        cublasZgemm( MagmaNoTrans, MagmaConjTrans, 
                     m, n, k, 
                     c_neg_one, dv_ref(0, 0), ldv,
                                dwork,        ldwork, 
                     c_one,     dc_ref(0,0),  ldc);
    }
    else {
        cublasZgemm( MagmaNoTrans, MagmaConjTrans, 
                     m, k, n, 
                     c_one,  dc_ref(0, 0), ldc,
                             dv_ref(0, 0), ldv, 
                     c_zero, dwork,        ldwork);

        cublasZtrmm( MagmaRight, MagmaUpper, transt, MagmaNonUnit,
                     m, k, 
                     c_one, dt,    ldt, 
                            dwork, ldwork);

        cublasZgemm( MagmaNoTrans, MagmaNoTrans, 
                     m, n, k, 
                     c_neg_one, dwork,        ldwork,
                                dv_ref(0, 0), ldv,
                      c_one,    dc_ref(0, 0), ldc);
    }
    return 0;

} /* magma_zlarfb */
