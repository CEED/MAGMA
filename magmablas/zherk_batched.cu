/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

       @author Jakub Kurzak
       @author Stan Tomov
       @author Mark Gates
       @author Azzam Haidar

       [zcds]gemm_fermi.cu          defines the CPU driver.
       [zcds]gemm_fermi_kernels.h   defines the block sizes for each precision.
       gemm_stencil_defs.h          defines types and functions for precision-independent code.
       
       These files are included multiple times, once for each transpose version.
       herk_stencil.cuh             defines the GPU kernel (device function).
       herk_kernel_batched.cuh              defines the GPU kernel (global function).
       
       The batched version uses herk_kernel_batched.cuh instead of herk_kernel.cuh.
*/
#include "common_magma.h"
#include "commonblas_z.h"

#define PRECISION_z

///////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////

/**
    Purpose
    -------
    ZGEMM performs one of the matrix-matrix operations
    
        C = alpha*op( A )*op( B ) + beta*C,
    
    where op( X ) is one of
    
        op( X ) = X   or   op( X ) = X**T   or   op( X ) = X**H,
    
    alpha and beta are scalars, and A, B and C are matrices, with
    op( A ) an m by k matrix, op( B ) a k by n matrix and C an m by n matrix.
    
    Parameters
    ----------
    @param[in]
    TRANSA  CHARACTER*1.
            On entry, TRANSA specifies the form of op( A ) to be used in
            the matrix multiplication as follows:
      -     = 'N':  op( A ) = A.
      -     = 'T':  op( A ) = A**T.
      -     = 'C':  op( A ) = A**H.
    
    @param[in]
    TRANSB  CHARACTER*1.
            On entry, TRANSB specifies the form of op( B ) to be used in
            the matrix multiplication as follows:
      -     = 'N':  op( B ) = B.
      -     = 'T':  op( B ) = B**T.
      -     = 'C':  op( B ) = B**H.
    
    @param[in]
    m       INTEGER.
            On entry,  M  specifies  the number  of rows  of the  matrix
            op( d_A )  and of the  matrix d_C.  M  must  be at least  zero.
    
    @param[in]
    n       INTEGER.
            On entry,  N  specifies the number  of columns of the matrix
            op( d_B ) and the number of columns of the matrix d_C. N must be
            at least zero.
    
    @param[in]
    k       INTEGER.
            On entry,  K  specifies  the number of columns of the matrix
            op( d_A ) and the number of rows of the matrix op( d_B ). K must
            be at least  zero.
    
    @param[in]
    alpha   COMPLEX_16
            On entry, ALPHA specifies the scalar alpha.
    
    @param[in]
    d_A     COMPLEX_16 array of DIMENSION ( LDA, ka ), where ka is
            k  when  TRANSA = MagmaNoTrans,  and is  m  otherwise.
            Before entry with  TRANSA = MagmaNoTrans,  the leading  m by k
            part of the array d_A must contain the matrix d_A, otherwise
            the leading  k by m  part of the array d_A must contain  the
            matrix d_A.
    
    @param[in]
    lda     INTEGER.
            On entry, LDA specifies the first dimension of A as declared
            in the calling (sub) program. When  TRANSA = MagmaNoTrans then
            LDA must be at least  max( 1, m ), otherwise  LDA must be at
            least  max( 1, k ).
    
    @param[in]
    d_B     COMPLEX_16 array of DIMENSION ( LDB, kb ), where kb is
            n  when  TRANSB = MagmaNoTrans,  and is  k  otherwise.
            Before entry with  TRANSB = MagmaNoTrans,  the leading  k by n
            part of the array d_B must contain the matrix d_B, otherwise
            the leading  n by k  part of the array d_B must contain  the
            matrix d_B.
    
    @param[in]
    ldb     INTEGER.
            On entry, LDB specifies the first dimension of d_B as declared
            in the calling (sub) program. When  TRANSB = MagmaNoTrans then
            LDB must be at least  max( 1, k ), otherwise  LDB must be at
            least  max( 1, n ).
    
    @param[in]
    beta    COMPLEX_16.
            On entry,  BETA  specifies the scalar  beta.  When  BETA  is
            supplied as zero then d_C need not be set on input.
    
    @param[in,out]
    d_C     COMPLEX_16 array of DIMENSION ( LDC, n ).
            Before entry, the leading  m by n  part of the array  d_C must
            contain the matrix  d_C,  except when  beta  is zero, in which
            case d_C need not be set on entry.
            On exit, the array  d_C  is overwritten by the  m by n  matrix
            ( alpha*op( d_A )*op( d_B ) + beta*d_C ).
    
    @param[in]
    ldc     INTEGER.
            On entry, LDC specifies the first dimension of d_C as declared
            in  the  calling  (sub)  program.   LDC  must  be  at  least
            max( 1, m ).

    @ingroup magma_zblas3
    ********************************************************************/
extern "C" void
magmablas_zherk_batched(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k,
    double alpha,
    magmaDoubleComplex **d_Aarray, magma_int_t lda,
    double beta,
    magmaDoubleComplex **d_Carray, magma_int_t ldc, magma_int_t batchCount )
{

    if( k <= 32 ) {
        magmablas_zherk_batched_k32(
                  uplo, trans, n, k,
                  alpha, d_Aarray, lda,
                  beta,  d_Carray, ldc,
                  batchCount );
    }
    else{
        magmablas_zherk_batched_lg(
                  uplo, trans, n, k,
                  alpha, d_Aarray, lda,
                  beta,  d_Carray, ldc,
                  batchCount );
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
