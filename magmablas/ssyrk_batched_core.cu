/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal d

       @author Jakub Kurzak
       @author Stan Tomov
       @author Mark Gates
       @author Azzam Haidar
       @author Ahmad Abdelfattah
       
*/
#include "common_magma.h"
#define PRECISION_s

#include "herk_template_kernel_batched.cuh"
#include "gemm_config/sgemm_param_nn.h"
#include "gemm_config/sgemm_param_nt.h"
#include "gemm_config/sgemm_param_tn.h"
#include "gemm_config/sgemm_param_tt.h"
#define version(s,v) s ## _V_ ## v
/**
    Purpose
    -------
    SSYRK performs one of the symmetric rank k operations

    C := alpha*A*A**H + beta*C,

    or

    C := alpha*A**H*A + beta*C,

    where alpha and beta are real scalars, C is an n by n symmetric
    matrix and A is an n by k matrix in the first case and a k by n
    matrix in the second case.
    
    Parameters
    ----------

    @param[in]
    uplo    CHARACTER*1.
           On entry, uplo specifies whether the upper or lower
           triangular part of the array C is to be referenced as
           follows:

           uplo = 'U' or 'u' Only the upper triangular part of C
           is to be referenced.

           uplo = 'L' or 'l' Only the lower triangular part of C
           is to be referenced.
    
    @param[in]
    trans   CHARACTER*1.
            On entry, trans specifies the operation to be performed as
            follows:

            trans = 'N' or 'n' C := alpha*A*A**H + beta*C.

            trans = 'C' or 'c' C := alpha*A**H*A + beta*C.

    @param[in]
    n       INTEGER.
            On entry,  specifies the order of the matrix C. N must be
            at least zero.
    
    @param[in]
    k       INTEGER.
            On entry with trans = 'N' or 'n', k specifies the number
            of columns of the matrix A, and on entry with
            trans = 'C' or 'c', k specifies the number of rows of the
            matrix A. K must be at least zero.

    @param[in]
    alpha   REAL
            On entry, ALPHA specifies the scalar alpha.
    
    @param[in]
    dA      REAL array of DIMENSION ( ldda, ka ), where ka is
            k  when  trans = MagmaNoTrans,  and is  n  otherwise.
            Before entry with  trans = MagmaNoTrans,  the leading  m by k
            part of the array dA must contain the matrix dA, otherwise
            the leading  k by m  part of the array dA must contain  the
            matrix dA.
    
    @param[in]
    ldda    INTEGER.
            On entry, ldda specifies the first dimension of A as declared
            in the calling (sub) program. When  trans = MagmaNoTrans then
            ldda must be at least  max( 1, n ), otherwise  ldda must be at
            least  max( 1, k ).
    
    @param[in]
    beta    REAL.
            On entry,  BETA  specifies the scalar  beta.  When  BETA  is
            supplied as zero then dC need not be set on input.
    
    @param[in,out]
    dC      REAL array of DIMENSION ( lddc, n ).
            Before entry with uplo = 'U' or 'u', the leading n by n
            upper triangular part of the array C must contain the upper
            triangular part of the symmetric matrix and the strictly
            lower triangular part of C is not referenced. On exit, the
            upper triangular part of the array C is overwritten by the
            upper triangular part of the updated matrix.
            Before entry with uplo = 'L' or 'l', the leading n by n
            lower triangular part of the array C must contain the lower
            triangular part of the symmetric matrix and the strictly
            upper triangular part of C is not referenced. On exit, the
            lower triangular part of the array C is overwritten by the
            lower triangular part of the updated matrix.
            Note that the imaginary parts of the diagonal elements need
            not be set, they are assumed to be zero, and on exit they
            are set to zero.

    @param[in]
    lddc    INTEGER.
            On entry, lddc specifies the first dimension of dC as declared
            in  the  calling  (sub)  program.   lddc  must  be  at  least
            max( 1, m ).
    @ingroup magma_sblas3
    ********************************************************************/
void
magmablas_ssyrk_batched(
    magma_uplo_t uplo, magma_trans_t trans, 
    magma_int_t n, magma_int_t k,
    float alpha,
    float const * const * dA_array, magma_int_t ldda,
    float beta,
    float **dC_array, magma_int_t lddc, 
    magma_int_t batchCount, magma_queue_t queue )
{
    float cbeta  = MAGMA_S_MAKE( beta, 0. );
    float calpha = MAGMA_S_MAKE( alpha, 0. );

    magma_int_t info = 0;
    if      ( uplo != MagmaUpper && uplo != MagmaLower )
        info = -1;
    #if defined(PRECISION_c) || defined(PRECISION_z) 
    else if ( trans != MagmaNoTrans && trans != MagmaConjTrans )
    #else 
    else if ( trans != MagmaNoTrans && trans != MagmaTrans && trans != MagmaConjTrans )
    #endif
        info = -2;
    else if ( n < 0 )
        info = -3;
    else if ( k < 0 )
        info = -4;
    else if ( trans == MagmaNoTrans ? ldda < n : ldda < k )
        info = -7;
    else if ( lddc < n )
        info = -10;

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    magma_int_t arch = magma_getdevice_arch();
    if ( arch < 200  ) {
        printf("not supported \n"); // TODO call cublas
        return;
    }
    
    // --------------------
    // CUDA ARCH 2.x (Fermi) version
    if ( n <= 0 || k <= 0 )
        return;

    // we have two shapes only (nt or tn)
    magma_int_t shape;
    if      (trans == MagmaNoTrans)   { shape = 0; } // nt
    else                              { shape = 1; } // tn
    
    //TODO: probably the texture init code should be placed here

    size_t offsetA = 0;
    size_t offsetB = 0;
    offsetA = offsetA/sizeof(float);
    offsetB = offsetB/sizeof(float);
    
    switch(shape)
    {
        case 0: // nt
            {
                herk_template_batched_nt<float, version(NT,734), 0, 0>
                (uplo, n, k, dA_array, ldda, dC_array, lddc, calpha, cbeta, offsetA, offsetB, batchCount, queue);
            }
            break;
        case 1: // tn
            {
                if (k < 64)
                {
                    herk_template_batched_tn<float, version(TN,654), 0, 0>
                    (uplo, n, k, dA_array, ldda, dC_array, lddc, calpha, cbeta, offsetA, offsetB, batchCount, queue);
                }
                else
                {
                    herk_template_batched_tn<float, version(TN,666), 0, 0>
                    (uplo, n, k, dA_array, ldda, dC_array, lddc, calpha, cbeta, offsetA, offsetB, batchCount, queue);
                }
            }
            break;
        default:; // propose something
    }
}
///////////////////////////////////////////////////////////////////////////////////////////////////
