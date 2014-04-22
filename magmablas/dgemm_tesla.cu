/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal d -> s
*/
#include "common_magma.h"
#include "commonblas_d.h"

/**
    Purpose
    -------
    
    DGEMM performs one of the matrix-matrix operations
    
        C = alpha*op( A )*op( B ) + beta*C,
    
    where op( X ) is one of
    
        op( X ) = X   or   op( X ) = X**T,
    
    alpha and beta are scalars, and A, B and C are matrices, with op( A )
    an m by k matrix,  op( B ) a k by n matrix and C an m by n matrix.
    
    Parameters
    ----------
    
    @param[in]
    transA  CHARACTER*1.
            On entry, transA specifies the form of op( A ) to be used in
            the matrix multiplication as follows:
      -     = 'N':  op( A ) = A.
      -     = 'T':  op( A ) = A**T.
      -     = 'C':  op( A ) = A**T.
    
    @param[in]
    transB  CHARACTER*1.
            On entry, transB specifies the form of op( B ) to be used in
            the matrix multiplication as follows:
      -     = 'N':  op( B ) = B.
      -     = 'T':  op( B ) = B**T.
      -     = 'C':  op( B ) = B**T.
    
    @param[in]
    m       INTEGER.
            On entry,  M  specifies  the number  of rows  of the  matrix
            op( A )  and of the  matrix  C.  M  must  be at least  zero.
    
    @param[in]
    n       INTEGER.
            On entry,  N  specifies the number  of columns of the matrix
            op( B ) and the number of columns of the matrix C. N must be
            at least zero.
    
    @param[in]
    k       INTEGER.
            On entry,  K  specifies  the number of columns of the matrix
            op( A ) and the number of rows of the matrix op( B ). K must
            be at least  zero.
    
    @param[in]
    alpha   DOUBLE PRECISION.
            On entry, ALPHA specifies the scalar alpha.
    
    @param[in]
    A       DOUBLE PRECISION array of DIMENSION ( LDA, ka ), where ka is
            k  when  transA = 'N' or 'n',  and is  m  otherwise.
            Before entry with  transA = 'N' or 'n',  the leading  m by k
            part of the array  A  must contain the matrix  A,  otherwise
            the leading  k by m  part of the array  A  must contain  the
            matrix A.
    
    @param[in]
    lda     INTEGER.
            On entry, LDA specifies the first dimension of A as declared
            in the calling (sub) program. When  transA = 'N' or 'n' then
            LDA must be at least  max( 1, m ), otherwise  LDA must be at
            least  max( 1, k ).
    
    @param[in]
    B       DOUBLE PRECISION array of DIMENSION ( LDB, kb ), where kb is
            n  when  transB = 'N' or 'n',  and is  k  otherwise.
            Before entry with  transB = 'N' or 'n',  the leading  k by n
            part of the array  B  must contain the matrix  B,  otherwise
            the leading  n by k  part of the array  B  must contain  the
            matrix B.
    
    @param[in]
    ldb     INTEGER.
            On entry, LDB specifies the first dimension of B as declared
            in the calling (sub) program. When  transB = 'N' or 'n' then
            LDB must be at least  max( 1, k ), otherwise  LDB must be at
            least  max( 1, n ).
    
    @param[in]
    beta    DOUBLE PRECISION.
            On entry,  BETA  specifies the scalar  beta.  When  BETA  is
            supplied as zero then C need not be set on input.
    
    @param[in,out]
    C       DOUBLE PRECISION array of DIMENSION ( LDC, n ).
            Before entry, the leading  m by n  part of the array  C must
            contain the matrix  C,  except when  beta  is zero, in which
            case C need not be set on entry.
            On exit, the array  C  is overwritten by the  m by n  matrix
            ( alpha*op( A )*op( B ) + beta*C ).
    
    @param[in]
    ldc     INTEGER.
            On entry, LDC specifies the first dimension of C as declared
            in  the  calling  (sub)  program.   LDC  must  be  at  least
            max( 1, m ).

    @ingroup magma_dblas3
    ********************************************************************/
extern "C" void
magmablas_dgemm_tesla(
    magma_trans_t transA, magma_trans_t transB, magma_int_t m, magma_int_t n, magma_int_t k,
    double alpha,
    const double *A, magma_int_t lda,
    const double *B, magma_int_t ldb,
    double beta,
    double *C, magma_int_t ldc )
{
    if ( m == 0 || n == 0 || ((alpha == 0.0 || k == 0) && beta == 1.0) ) {
        return;
    }
    if ( alpha == 0.0 ) {
        if ( beta == 0.0 ) {
            magmablas_dgemm_ab_0(
                C, A, B, m, n, k, lda, ldb, ldc, alpha, beta );
            return;
        }
        else {
            magmablas_dgemm_a_0(
                C, A, B, m, n, k, lda, ldb, ldc, alpha, beta );
            return;
        }
    }
    
    if ( ldc < m ) return;  /* TODO: error */
    if ( transA == MagmaNoTrans ) {
        if ( transB == MagmaNoTrans ) {
            if ( lda < m ) return;  /* TODO: error */
            if ( ldb < k ) return;  /* TODO: error */
            /*=======================================================================
              ===================C = alpha * A * B + beta * C =======================
              =======================================================================*/
            if ( m > 512 && n > 512 ) {
                if ( m % 64 == 0 && n % 16 == 0 && k % 16 == 0 )
                    magmablas_dgemm_N_N_64_16_16_16_4_special(
                        C, A, B, m, n, k, lda, ldb, ldc, alpha, beta );
                else
                    magmablas_dgemm_N_N_64_16_16_16_4(
                        C, A, B, m, n, k, lda, ldb, ldc, alpha, beta );
            }
            else {
                cublasDgemm(
                    lapacke_trans_const(transA), lapacke_trans_const(transB),
                    m, n, k, alpha, A, lda, B, ldb, beta, C, ldc );
            }
        }
        else {
            if ( lda < m ) return;  /* TODO: error */
            if ( ldb < n ) return;  /* TODO: error */
            /*=======================================================================
              ===================C = alpha * A * B^T + beta * C======================
              =======================================================================*/
            if ( m > 512 && n > 512 ) {
                //if ( m % 64 == 0 && n % 16 == 0 && k % 4 == 0 )
                //    magmablas_dgemm_N_T_64_16_4_16_4(
                //        C, A, B, m, n, k, lda, ldb, ldc, alpha, beta );
                //else
                    magmablas_dgemm_N_T_64_16_4_16_4(
                        C, A, B, m, n, k, lda, ldb, ldc, alpha, beta );
            }
            else {
                cublasDgemm(
                    lapacke_trans_const(transA), lapacke_trans_const(transB),
                    m, n, k, alpha, A, lda, B, ldb, beta, C, ldc );
            }
        }
    }
    else {
        if ( transB == MagmaNoTrans ) {
            if ( lda < k ) return;  /* TODO: error */
            if ( ldb < k ) return;  /* TODO: error */
            /*=======================================================================
              ===================C = alpha * A^T * B + beta * C======================
              =======================================================================*/
            if ( m > 512 && n > 512 ) {
                //if ( m % 32 == 0 && n % 32 == 0 && k % 8 == 0 )
                //    magmablas_dgemm_T_N_32_32_8_8_8(
                //        C, A, B, m, n, k, lda, ldb, ldc, alpha, beta );
                //else
                    magmablas_dgemm_T_N_32_32_8_8_8(
                        C, A, B, m, n, k, lda, ldb, ldc, alpha, beta );
            }
            else {
                cublasDgemm(
                    lapacke_trans_const(transA), lapacke_trans_const(transB),
                    m, n, k, alpha, A, lda, B, ldb, beta, C, ldc );
            }
        }
        else {
            if ( lda < k ) return;  /* TODO: error */
            if ( ldb < n ) return;  /* TODO: error */
            /*=======================================================================
              ===================C = alpha * A^T * B^T + beta * C====================
              =======================================================================*/
            if ( m > 512 && n > 512 ) {
                if ( m % 64 == 0 && n % 16 == 0 && k % 16 == 0 )
                    magmablas_dgemm_T_T_64_16_16_16_4_special(
                        C, B, A, n, m, k, ldb, lda, ldc, alpha, beta );
                else
                    magmablas_dgemm_T_T_64_16_16_16_4(
                        C, B, A, n, m, k, ldb, lda, ldc, alpha, beta );
            }
            else {
                cublasDgemm(
                    lapacke_trans_const(transA), lapacke_trans_const(transB),
                    m, n, k, alpha, A, lda, B, ldb, beta, C, ldc );
            }
        }
    }
}
