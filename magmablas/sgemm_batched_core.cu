/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Jakub Kurzak
       @author Stan Tomov
       @author Mark Gates
       @author Azzam Haidar
       @author Ahmad Abdelfattah
       
*/
#include "magma_internal.h"

#define PRECISION_s

#include "gemm_template_kernel_batched.cuh"
#include "gemm_config/sgemm_param_nn.h"
#include "gemm_config/sgemm_param_nt.h"
#include "gemm_config/sgemm_param_tn.h"
#include "gemm_config/sgemm_param_tt.h"

#define version(s,v) s ## _V_ ## v

/**
    Purpose
    -------
    SGEMM performs one of the matrix-matrix operations
    
        C = alpha*op( A )*op( B ) + beta*C,
    
    where op( X ) is one of
    
        op( X ) = X      or
        op( X ) = X**T   or
        op( X ) = X**H,
    
    alpha and beta are scalars, and A, B and C are matrices, with
    op( A ) an m by k matrix, op( B ) a k by n matrix and C an m by n matrix.
    
    Parameters
    ----------
    @param[in]
    transA  magma_trans_t.
            On entry, transA specifies the form of op( A ) to be used in
            the matrix multiplication as follows:
      -     = MagmaNoTrans:    op( A ) = A.
      -     = MagmaTrans:      op( A ) = A**T.
      -     = MagmaConjTrans:  op( A ) = A**H.
    
    @param[in]
    transB  magma_trans_t.
            On entry, transB specifies the form of op( B ) to be used in
            the matrix multiplication as follows:
      -     = MagmaNoTrans:    op( B ) = B.
      -     = MagmaTrans:      op( B ) = B**T.
      -     = MagmaConjTrans:  op( B ) = B**H.
    
    @param[in]
    m       INTEGER.
            On entry,  M  specifies  the number  of rows  of the  matrix
            op( A )  and of the  matrix C.  M  must  be at least  zero.
    
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
    alpha   REAL
            On entry, ALPHA specifies the scalar alpha.
    
    @param[in]
    dA_array      Array of pointers, dimension (batchCount). 
             Each is a REAL array A of DIMENSION ( ldda, ka ), where ka is
             k  when  transA = MagmaNoTrans,  and is  m  otherwise.
             Before entry with  transA = MagmaNoTrans,  the leading  m by k
             part of the array A must contain the matrix A, otherwise
             the leading  k by m  part of the array A must contain  the
             matrix A.
    
    @param[in]
    ldda    INTEGER.
            On entry, ldda specifies the first dimension of each array A as 
            declared in the calling (sub) program. When  transA = MagmaNoTrans then
            ldda must be at least  max( 1, m ), otherwise  ldda must be at
            least  max( 1, k ).
    
    @param[in]
    dB_array      Array of pointers, dimension (batchCount).
             Each is a REAL array B of DIMENSION ( lddb, kb ), where kb is
             n  when  transB = MagmaNoTrans,  and is  k  otherwise.
             Before entry with  transB = MagmaNoTrans,  the leading  k by n
             part of the array B must contain the matrix B, otherwise
             the leading  n by k  part of the array B must contain  the
             matrix B.
    
    @param[in]
    lddb    INTEGER.
            On entry, lddb specifies the first dimension of each array B as 
            declared in the calling (sub) program. When  transB = MagmaNoTrans then
            lddb must be at least  max( 1, k ), otherwise  lddb must be at
            least  max( 1, n ).
    
    @param[in]
    beta    REAL.
            On entry,  BETA  specifies the scalar  beta.  When  BETA  is
            supplied as zero then C need not be set on input.
    
    @param[in,out]
    dC_array      Array of pointers, dimension (batchCount).
             REAL array of DIMENSION ( lddc, n ).
             Before entry, the leading  m by n  part of the array  C must
             contain the matrix  C,  except when  beta  is zero, in which
             case C need not be set on entry.
             On exit, each array  C  is overwritten by the  m by n  matrix
             ( alpha*op( A )*op( B ) + beta*C ).
    
    @param[in]
    lddc    INTEGER.
            On entry, lddc specifies the first dimension of each array C as declared
            in  the  calling  (sub)  program.   lddc  must  be  at  least
            max( 1, m ).
    
    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_sblas3
    ********************************************************************/
void
magmablas_sgemm_batched(
    magma_trans_t transA, magma_trans_t transB, 
    magma_int_t m, magma_int_t n, magma_int_t k,
    float alpha,
    float const * const * dA_array, magma_int_t ldda,
    float const * const * dB_array, magma_int_t lddb,
    float beta,
    float **dC_array, magma_int_t lddc, 
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t info = 0;
    if      ( transA != MagmaNoTrans && transA != MagmaTrans && transA != MagmaConjTrans )
        info = -1;
    else if ( transB != MagmaNoTrans && transB != MagmaTrans && transB != MagmaConjTrans )
        info = -2;
    else if ( m < 0 )
        info = -3;
    else if ( n < 0 )
        info = -4;
    else if ( k < 0 )
        info = -5;
    else if ( transA == MagmaNoTrans ? ldda < m : ldda < k )
        info = -8;
    else if ( transB == MagmaNoTrans ? lddb < k : lddb < n )
        info = -10;
    else if ( lddc < m )
        info = -13;
    
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    magma_int_t arch = magma_getdevice_arch();
    if ( arch < 200  ) {
        fprintf( stderr, "%s: CUDA arch < 200 not supported\n", __func__ ); // TODO call cublas
        return;
    }
    
    if ( m <= 0 || n <= 0 || k <= 0 )
        return;

    magma_int_t shape = 0;
    if      (transA == MagmaNoTrans   && transB == MagmaNoTrans)   { shape = 0; } // nn
    else if (transA == MagmaNoTrans   && transB == MagmaTrans)     { shape = 1; } // nt
    else if (transA == MagmaNoTrans   && transB == MagmaConjTrans) { shape = 2; } // nc
    else if (transA == MagmaTrans     && transB == MagmaNoTrans)   { shape = 3; } // tn
    else if (transA == MagmaTrans     && transB == MagmaTrans)     { shape = 4; } // tt
    else if (transA == MagmaTrans     && transB == MagmaConjTrans) { shape = 5; } // tc
    else if (transA == MagmaConjTrans && transB == MagmaNoTrans)   { shape = 6; } // cn
    else if (transA == MagmaConjTrans && transB == MagmaTrans)     { shape = 7; } // ct
    else if (transA == MagmaConjTrans && transB == MagmaConjTrans) { shape = 8; } // cc
    
    //TODO: probably the texture init code should be placed here

    size_t offsetA = 0;
    size_t offsetB = 0;
    offsetA = offsetA/sizeof(float);
    offsetB = offsetB/sizeof(float);
    
    switch(shape)
    {
        case 0: // nn
            {
                if (k < 64)
                {
                    if (k == 8 && n == 24)
                        gemm_template_batched_nn<float, version(NN,512), 0, 0>
                        (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, offsetA, offsetB, batchCount, queue);
                    else if (n < 32)
                        gemm_template_batched_nn<float, version(NN,510), 0, 0>
                        (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, offsetA, offsetB, batchCount, queue);
                    else
                        gemm_template_batched_nn<float, version(NN,504), 0, 0>
                        (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, offsetA, offsetB, batchCount, queue);
                }
                else
                {
                    gemm_template_batched_nn<float, version(NN,518), 0, 0>
                    (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, offsetA, offsetB, batchCount, queue);
                }
            }
            break;
        case 1: // nt
            {
                gemm_template_batched_nt<float, version(NT,734), 0, 0>
                (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, offsetA, offsetB, batchCount, queue);
            }
            break;
        case 2: // nc
            {
                gemm_template_batched_nt<float, version(NT,734), 0, 1>
                (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, offsetA, offsetB, batchCount, queue);
            }
            break;
        case 3: // tn
            {
                if (k < 64)
                {
                    gemm_template_batched_tn<float, version(TN,654), 0, 0>
                    (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, offsetA, offsetB, batchCount, queue);
                }
                else
                {
                    gemm_template_batched_tn<float, version(TN,666), 0, 0>
                    (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, offsetA, offsetB, batchCount, queue);
                }
            }
            break;
        case 6: // cn
            {
                if (k < 64)
                {
                    gemm_template_batched_tn<float, version(TN,654), 1, 0>
                    (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, offsetA, offsetB, batchCount, queue);
                }
                else
                {
                    gemm_template_batched_tn<float, version(TN,666), 1, 0>
                    (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, offsetA, offsetB, batchCount, queue);
                }
            }
            break;
        case 4: // tt
            {
                if (k < 128)
                {
                    if (m < 128)
                    {
                        gemm_template_batched_tt<float, version(TT,275), 0, 0>
                        (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, offsetA, offsetB, batchCount, queue);
                    }
                    else
                    {
                        gemm_template_batched_tt<float, version(TT,312), 0, 0>
                        (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, offsetA, offsetB, batchCount, queue);
                    }
                }
                else
                {
                    gemm_template_batched_tt<float, version(TT,312), 0, 0>
                    (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, offsetA, offsetB, batchCount, queue);
                }
            }
            break;
        case 5: // tc
            {
                if (k < 128)
                {
                    if (m < 128)
                    {
                        gemm_template_batched_tt<float, version(TT,275), 0, 1>
                        (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, offsetA, offsetB, batchCount, queue);
                    }
                    else
                    {
                        gemm_template_batched_tt<float, version(TT,312), 0, 1>
                        (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, offsetA, offsetB, batchCount, queue);
                    }
                }
                else
                {
                    gemm_template_batched_tt<float, version(TT,312), 0, 1>
                    (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, offsetA, offsetB, batchCount, queue);
                }
            }
            break;
        case 7: // ct
            {
                if (k < 128)
                {
                    if (m < 128)
                    {
                        gemm_template_batched_tt<float, version(TT,275), 1, 0>
                        (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, offsetA, offsetB, batchCount, queue);
                    }
                    else
                    {
                        gemm_template_batched_tt<float, version(TT,312), 1, 0>
                        (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, offsetA, offsetB, batchCount, queue);
                    }
                }
                else
                {
                    gemm_template_batched_tt<float, version(TT,312), 1, 0>
                    (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, offsetA, offsetB, batchCount, queue);
                }
            }
            break;
        case 8: // cc
            {
                if (k < 128)
                {
                    if (m < 128)
                    {
                        gemm_template_batched_tt<float, version(TT,275), 1, 1>
                        (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, offsetA, offsetB, batchCount, queue);
                    }
                    else
                    {
                        gemm_template_batched_tt<float, version(TT,312), 1, 1>
                        (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, offsetA, offsetB, batchCount, queue);
                    }
                }
                else
                {
                    gemm_template_batched_tt<float, version(TT,312), 1, 1>
                    (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, offsetA, offsetB, batchCount, queue);
                }
            }
            break;
        default:; // propose something
    }
}
///////////////////////////////////////////////////////////////////////////////////////////////////
