/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @author Tingxing Dong
       @author Azzam Haidar

       @precisions normal z -> s d c
*/

#include "common_magma.h"
#include "magma_templates.h"


#define PRECISION_z



#define NB 256  //NB is the 1st level blocking in recursive blocking, NUM_THREADS is the 2ed level, NB=256, NUM_THREADS=64 is optimal for batched

#define NUM_THREADS 128 //64 //128


#include "ztrsv_template_device.cuh"

#define A(i, j)  (A + (i) + (j)*lda)   // A(i, j) means at i row, j column

extern __shared__ magmaDoubleComplex shared_data[];




//==============================================================================

__global__ void
ztrsv_notrans_kernel_outplace(
    magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    int n, 
    const magmaDoubleComplex * __restrict__ A, int lda,
    magmaDoubleComplex *b, int incb, 
    magmaDoubleComplex *x,
    int flag=0)
{
    if(flag == 0 )
    {
        ztrsv_notrans_device<128, 128, 1, 1000000, 0>(uplo, transA, diag, n, A, lda, b, incb, x);
    }
    else
    {
        ztrsv_notrans_device<128, 128, 1, 1000000, 1>(uplo, transA, diag, n, A, lda, b, incb, x);
    }
}

//==============================================================================



//==============================================================================

__global__ void
ztrsv_trans_kernel_outplace(
    magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    int n, 
    const magmaDoubleComplex * __restrict__ A, int lda,
    magmaDoubleComplex *b, int incb, 
    magmaDoubleComplex *x,
    int flag=0)
{
   if(flag == 0 )
    {
        if(transA == MagmaConjTrans)
        {
            ztrsv_trans_device<32, 16, 8, 1000000, 1, 0>(uplo, transA, diag, n, A, lda, b, incb, x);
        }
        else
        {
            ztrsv_trans_device<32, 16, 8, 1000000, 0, 0>(uplo, transA, diag, n, A, lda, b, incb, x);
        }
    }
    else
    {
        if(transA == MagmaConjTrans)
        {
            ztrsv_trans_device<32, 16, 8, 1000000, 1, 1>(uplo, transA, diag, n, A, lda, b, incb, x);
        }
        else
        {
            ztrsv_trans_device<32, 16, 8, 1000000, 0, 1>(uplo, transA, diag, n, A, lda, b, incb, x);
        }
    }
}
 
//==============================================================================


//==============================================================================

extern "C" void
magmablas_ztrsv_outofplace(
    magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t n, 
    const magmaDoubleComplex * __restrict__ A, magma_int_t lda,
    magmaDoubleComplex *b, magma_int_t incb, 
    magmaDoubleComplex *x, magma_queue_t queue,
    magma_int_t flag=0)
{
    /* Check arguments */
    magma_int_t info = 0;
    if ( uplo != MagmaUpper && uplo != MagmaLower ) {
        info = -1;
    } else if ( transA != MagmaNoTrans && transA != MagmaTrans && transA != MagmaConjTrans ) {
        info = -2;
    } else if ( diag != MagmaUnit && diag != MagmaNonUnit ) {
        info = -3;
    } else if (n < 0) {
        info = -5;
    } else if (lda < max(1,n)) {
        info = -8;
    }

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;
    }

    // quick return if possible.
    if (n == 0)
        return;

    dim3 blocks(1, 1, 1);
    dim3 threads(NUM_THREADS);


    if (transA == MagmaNoTrans)
    {
        ztrsv_notrans_kernel_outplace<<<blocks, threads, sizeof(magmaDoubleComplex)*(n), queue>>>(uplo, transA, diag, n, A, lda, b, incb, x, flag);
    }
    else
    {
        ztrsv_trans_kernel_outplace<<<blocks, threads, sizeof(magmaDoubleComplex)*(n), queue>>>(uplo, transA, diag, n, A, lda, b, incb, x, flag); 
    }
}


/*
  README: flag decides if the ztrsv_outplace see an updated x or not. 0: No; other: Yes
  In recursive, flag must be nonzero except the 1st call
*/
extern "C" void
magmablas_ztrsv_recursive_outofplace(
    magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t n, 
    const magmaDoubleComplex * __restrict__ A, magma_int_t lda,
    magmaDoubleComplex *b, magma_int_t incb, 
    magmaDoubleComplex *x, magma_queue_t queue)
{
    /* Check arguments */
    magma_int_t info = 0;
    if ( uplo != MagmaUpper && uplo != MagmaLower ) {
        info = -1;
    } else if ( transA != MagmaNoTrans && transA != MagmaTrans && transA != MagmaConjTrans ) {
        info = -2;
    } else if ( diag != MagmaUnit && diag != MagmaNonUnit ) {
        info = -3;
    } else if (n < 0) {
        info = -5;
    } else if (lda < max(1,n)) {
        info = -8;
    }

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;
    }

    // quick return if possible.
    if (n == 0)
        return;

    //Init x with zero
    //magmablas_zlaset(MagmaFull, n, incb, MAGMA_Z_ZERO, MAGMA_Z_ZERO, x, n);

    magma_int_t col = n;

    if (transA == MagmaNoTrans)
    {
        for (magma_int_t i=0; i < n; i += NB)
        {    
            magma_int_t jb = min(NB, n-i);

            if (uplo == MagmaUpper)
            {
                col -= jb;
                //assume x_array contains zero elements, magmablas_zgemv will cause slow down
                magma_zgemv(MagmaNoTrans, jb, i, MAGMA_Z_ONE, A(col, col+jb), lda, 
                                x+col+jb, 1, MAGMA_Z_ONE, x+col, 1);
            }
            else
            {          
                col = i;
                magma_zgemv(MagmaNoTrans, jb, i, MAGMA_Z_ONE, A(col, 0), lda, 
                                x, 1, MAGMA_Z_ONE, x+col, 1);                  
            }

            magmablas_ztrsv_outofplace(uplo, transA, diag, jb, A(col, col), lda, b+col, incb, x+col, queue, i);
        }
    }
    else
    {
        for (magma_int_t i=0; i < n; i += NB)
        {    
            magma_int_t jb = min(NB, n-i);

            if (uplo == MagmaLower)
            {
                col -= jb;

                magma_zgemv(MagmaConjTrans, i, jb, MAGMA_Z_ONE, A(col+jb, col), lda, x+col+jb, 1, MAGMA_Z_ONE, x+col, 1);                  
            }
            else
            {
                col = i;
                
                magma_zgemv(MagmaConjTrans, i, jb, MAGMA_Z_ONE, A(0, col), lda, x, 1, MAGMA_Z_ONE, x+col, 1);                  
            }           
     
            magmablas_ztrsv_outofplace(uplo, transA, diag, jb, A(col, col), lda, b+col, incb, x+col, queue, i);
        }
    }
}



//==============================================================================

/**
    Purpose
    -------
    ztrsv solves one of the matrix equations on gpu

        op(A)*x = b,   or   x*op(A) = b,

    where alpha is a scalar, x and b are vectors, A is a unit, or
    non-unit, upper or lower triangular matrix and op(A) is one of

        op(A) = A,   or   op(A) = A^T,  or  op(A) = A^H.

    The vector x is overwritten on b.


    Arguments
    ----------

    @param[in]
    uplo    magma_uplo_t.
            On entry, uplo specifies whether the matrix A is an upper or
            lower triangular matrix as follows:
      -     = MagmaUpper:  A is an upper triangular matrix.
      -     = MagmaLower:  A is a  lower triangular matrix.

    @param[in]
    transA  magma_trans_t.
            On entry, transA specifies the form of op(A) to be used in
            the matrix multiplication as follows:
      -     = MagmaNoTrans:    op(A) = A.
      -     = MagmaTrans:      op(A) = A^T.
      -     = MagmaConjTrans:  op(A) = A^H.

    @param[in]
    diag    magma_diag_t.
            On entry, diag specifies whether or not A is unit triangular
            as follows:
      -     = MagmaUnit:     A is assumed to be unit triangular.
      -     = MagmaNonUnit:  A is not assumed to be unit triangular.

    @param[in]
    n       INTEGER.
            On entry, n N specifies the order of the matrix A. n >= 0.

    @param[in]
    A       COMPLEX_16 array of dimension ( lda, n )
            Before entry with uplo = MagmaUpper, the leading n by n
            upper triangular part of the array A must contain the upper
            triangular matrix and the strictly lower triangular part of
            A is not referenced.
            Before entry with uplo = MagmaLower, the leading n by n
            lower triangular part of the array A must contain the lower
            triangular matrix and the strictly upper triangular part of
            A is not referenced.
            Note that when diag = MagmaUnit, the diagonal elements of
            A are not referenced either, but are assumed to be unity.

    @param[in]
    lda     INTEGER.
            On entry, lda specifies the first dimension of A.
            lda >= max( 1, n ).

    @param[in]
    b       COMPLEX_16 array of dimension  n
            On exit, b is overwritten with the solution vector x.

    @param[in]
    incb    INTEGER.
            On entry,  incb specifies the increment for the elements of
            b. incb must not be zero.
            Unchanged on exit.


    @ingroup magma_zblas2
    ********************************************************************/
extern "C" void
magmablas_ztrsv(
    magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t n, 
    const magmaDoubleComplex * __restrict__ A, magma_int_t lda,
    magmaDoubleComplex *b, magma_int_t incb, 
    magma_queue_t queue)
{
    magma_int_t size_x = n * incb;

    magmaDoubleComplex *x=NULL;

    magma_zmalloc( &x, size_x);

    magmablas_zlaset(MagmaFull, n, 1, MAGMA_Z_ZERO, MAGMA_Z_ZERO, x, n);

    magmablas_ztrsv_recursive_outofplace(uplo, transA, diag, n, A, lda, b, incb, x, queue);

    magmablas_zlacpy( MagmaFull, n, 1, x, n, b, n);

    magma_free(x);
}

//==============================================================================
