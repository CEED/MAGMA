/*
    -- MAGMA (version 1.4) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s

       @author Peng Du
       @author Tingxing Dong
*/
#include "common_magma.h"

#define BLOCK_SIZE 16 // inner blocking size, <=32
#define NB 128        // outer blocking size, >BLOCK_SIZE

__global__ void
ztrsm_copy_kernel(int m, int n, magmaDoubleComplex *B, int ldb, magmaDoubleComplex *dX, int ldx)
{
    int by = blockIdx.y;
    int gx = blockIdx.x*blockDim.x + threadIdx.x;
    if (gx < m)
        B[by*ldb+gx] = dX[by*ldx+gx];
}


#define MAX_THREAD_PER_BLOCK 512
#define WARP_SIZE 32


#define ztrsm_copy() \
    do { \
        dim3 dimBlock( (m >= MAX_THREAD_PER_BLOCK) ? MAX_THREAD_PER_BLOCK : (WARP_SIZE*((m/WARP_SIZE)+(m % WARP_SIZE != 0))), 1 ); \
        dim3 dimGrid( (m - 1)/dimBlock.x + 1, n ); \
        ztrsm_copy_kernel<<< dimGrid, dimBlock, 0, magma_stream >>>(m, n, B, ldb, dX, m); \
        magma_device_sync(); \
    } while(0)


/*
 * magmablas_ztrsm
 */

extern "C"
void diag_ztrtri(
    magma_int_t m, magma_uplo_t uplo, magma_diag_t diag,
    const magmaDoubleComplex *dA, magmaDoubleComplex *d_invA, magma_int_t ldda);


/**
    Purpose
    -------
    ztrsm_work solves one of the matrix equations on gpu

        op( A )*X = alpha*B,   or   X*op( A ) = alpha*B,

    where alpha is a scalar, X and B are m by n matrices, A is a unit, or
    non-unit, upper or lower triangular matrix and op( A ) is one of

        op( A ) = A   or   op( A ) = A^T.

    The matrix X is overwritten on B.

    This is an asynchronous version of magmablas_ztrsm with flag,
    d_invA and dX workspaces as arguments.

    Arguments
    ----------
    @param[in]
    side    magma_side_t.
            On entry, side specifies whether op( A ) appears on the left
            or right of X as follows:
      -     = MagmaLeft:       op( A )*X = alpha*B.
      -     = MagmaRight:      X*op( A ) = alpha*B.

    @param[in]
    uplo    magma_uplo_t.
            On entry, uplo specifies whether the matrix A is an upper or
            lower triangular matrix as follows:
      -     = MagmaUpper:  A is an upper triangular matrix.
      -     = MagmaLower:  A is a  lower triangular matrix.

    @param[in]
    transA  magma_trans_t.
            On entry, transA specifies the form of op( A ) to be used in
            the matrix multiplication as follows:
      -     = MagmaNoTrans:    op( A ) = A.
      -     = MagmaTrans:      op( A ) = A^T.
      -     = MagmaConjTrans:  op( A ) = A^H.

    @param[in]
    diag    magma_diag_t.
            On entry, diag specifies whether or not A is unit triangular
            as follows:
      -     = MagmaUnit:     A is assumed to be unit triangular.
      -     = MagmaNonUnit:  A is not assumed to be unit triangular.

    @param[in]
    m       INTEGER.
            On entry, m specifies the number of rows of B. m must be at
            least zero.

    @param[in]
    n       INTEGER.
            On entry, n specifies the number of columns of B. n must be
            at least zero.

    @param[in]
    alpha   COMPLEX_16.
            On entry, alpha specifies the scalar alpha. When alpha is
            zero then A is not referenced and B need not be set before
            entry.

    @param[in]
    A       COMPLEX_16 array of DIMENSION ( lda, k ), where k is m
            when side = MagmaLeft and is n when side = MagmaRight.
            Before entry with uplo = MagmaUpper, the leading k by k
            upper triangular part of the array A must contain the upper
            triangular matrix and the strictly lower triangular part of
            A is not referenced.
            Before entry with uplo = MagmaLower, the leading k by k
            lower triangular part of the array A must contain the lower
            triangular matrix and the strictly upper triangular part of
            A is not referenced.
            Note that when diag = MagmaUnit, the diagonal elements of
            A are not referenced either, but are assumed to be unity.

    @param[in]
    lda     INTEGER.
            On entry, lda specifies the first dimension of A as declared
            in the calling (sub) program. When side = MagmaLeft then
            lda must be at least max( 1, m ), when side = MagmaRight
            then lda must be at least max( 1, n ).

    @param[in,out]
    B       COMPLEX_16 array of DIMENSION ( ldb, n ).
            Before entry, the leading m by n part of the array B must
            contain the right-hand side matrix B, and on exit is
            overwritten by the solution matrix X.

    @param[in]
    ldb     INTEGER.
            On entry, ldb specifies the first dimension of B as declared
            in the calling (sub) program. ldb must be at least
            max( 1, m ).

    @param[in]
    flag    BOOLEAN.
            If flag is true, invert diagonal blocks.
            If flag is false, assume diagonal blocks are already inverted.

    @param
    d_invA  (workspace) on device.
            If side == MagmaLeft,  d_invA size must be >= ((m+NB-1)/NB)*NB*NB,
            If side == MagmaRight, d_invA size must be >= ((n+NB-1)/NB)*NB*NB,
            where NB = 128.

    @param
    dX      (workspace) size m*n, on device.

    @ingroup magma_zblas3
    ********************************************************************/
extern "C"
void magmablas_ztrsm_work(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    const magmaDoubleComplex* A, magma_int_t lda,
    magmaDoubleComplex* B, magma_int_t ldb,
    magma_int_t flag,
    magmaDoubleComplex* d_invA, magmaDoubleComplex *dX)
{
    const magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    const magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    const magmaDoubleComplex c_zero    = MAGMA_Z_ZERO;

    magma_int_t i;
    magma_int_t nrowa = (side == MagmaLeft ? m : n);
    
    magma_int_t info = 0;
    if ( side != MagmaLeft && side != MagmaRight ) {
        info = -1;
    } else if ( uplo != MagmaUpper && uplo != MagmaLower ) {
        info = -2;
    } else if ( transA != MagmaNoTrans && transA != MagmaTrans && transA != Magma_ConjTrans ) {
        info = -3;
    } else if ( diag != MagmaUnit && diag != MagmaNonUnit ) {
        info = -4;
    } else if (m < 0) {
        info = -5;
    } else if (n < 0) {
        info = -6;
    } else if (lda < max(1,nrowa)) {
        info = -9;
    } else if (ldb < max(1,m)) {
        info = -11;
    }
    
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;
    }

    // quick return if possible.
    if (m == 0 || n == 0)
        return;

    if (side == MagmaLeft) {
        // side=L
        /* invert the diagonals
         */
        if (flag)
            diag_ztrtri(m, uplo, diag, A, d_invA, lda);

        if (transA == MagmaNoTrans) {
            /* the non-transpose case */
            if (uplo == MagmaLower) {

                /* the lower case */
                /* handle the first block seperately with alpha */
                int mm = min(NB, m);
                magma_zgemm( MagmaNoTrans, MagmaNoTrans, mm, n, mm, alpha, d_invA, NB, B, ldb, c_zero, dX, m);

            if (NB >= m) {
                    ztrsm_copy();
                    return;
                }

                magma_zgemm( MagmaNoTrans, MagmaNoTrans, m-NB, n, NB, c_neg_one, A+NB, lda, dX, m, alpha, B+NB, ldb);

                /* the rest blocks */
                for( i=NB; i < m; i += NB ) {
                    mm = min(m-i, NB);
                    magma_zgemm( MagmaNoTrans, MagmaNoTrans, mm, n, mm, c_one, d_invA+i*NB, NB, B+i, ldb, c_zero, dX+i, m);

                    if (i+NB >= m)
                        break;

                    magma_zgemm( MagmaNoTrans, MagmaNoTrans, m-i-NB, n, NB, c_neg_one, A+i*lda+i+NB, lda, dX+i, m, c_one, B+i+NB, ldb);
                }
            }
            else {
                /* the upper case */
                /* handle the first block seperately with alpha */
                int mm = (m % NB == 0) ? NB : (m % NB);
                i = m-mm;
                magma_zgemm( MagmaNoTrans, MagmaNoTrans, mm, n, mm, alpha, d_invA+i*NB, NB, B+i, ldb, c_zero, dX+i, m);

                if (i-NB < 0) {
                    ztrsm_copy();
                    return;
                }

                magma_zgemm( MagmaNoTrans, MagmaNoTrans, i, n, mm, c_neg_one, A+i*lda, lda, dX+i, m, alpha, B, ldb);

                /* the rest blocks */
                for( i=m-mm-NB; i >= 0; i -= NB ) {
                    magma_zgemm( MagmaNoTrans, MagmaNoTrans, NB, n, NB, c_one, d_invA+i*NB, NB, B+i, ldb, c_zero, dX+i, m);

                    if (i-NB < 0)
                        break;

                    magma_zgemm( MagmaNoTrans, MagmaNoTrans, i, n, NB, c_neg_one, A+i*lda, lda, dX+i, m, c_one, B, ldb);
                }
            }
        }
        else if( transA == MagmaTrans) {
            /* the transpose case */
            if (uplo == MagmaLower) {
                /* the lower case */
                /* handle the first block seperately with alpha */
                int mm = (m % NB == 0) ? NB : (m % NB);
                i = m-mm;
                magma_zgemm(MagmaTrans, MagmaNoTrans, mm, n, mm, alpha, d_invA+i*NB, NB, B+i, ldb, c_zero, dX+i, m);

                if (i-NB < 0) {
                    ztrsm_copy();
                    return;
                }

                magma_zgemm(MagmaTrans, MagmaNoTrans, i, n, mm, c_neg_one, A+i, lda, dX+i, m, alpha, B, ldb);

                /* the rest blocks */
                for( i=m-mm-NB; i >= 0; i -= NB ) {
                    magma_zgemm(MagmaTrans, MagmaNoTrans, NB, n, NB, c_one, d_invA+i*NB, NB, B+i, ldb, c_zero, dX+i, m);

                    if (i-NB < 0)
                        break;

                    magma_zgemm(MagmaTrans, MagmaNoTrans, i, n, NB, c_neg_one, A+i, lda, dX+i, m, c_one, B, ldb);
                }
            }
            else {
                /* the upper case */
                /* handle the first block seperately with alpha */
                int mm = min(NB, m);
                magma_zgemm(MagmaTrans, MagmaNoTrans, mm, n, mm, alpha, d_invA, NB, B, ldb, c_zero, dX, m);

                if (NB >= m) {
                    ztrsm_copy();
                    return;
                }

                magma_zgemm(MagmaTrans, MagmaNoTrans, m-NB, n, NB, c_neg_one, A+(NB)*lda, lda, dX, m, alpha, B+NB, ldb);

                /* the rest blocks */
                for( i=NB; i < m; i += NB ) {
                    mm = min(m-i, NB);
                    magma_zgemm(MagmaTrans, MagmaNoTrans, mm, n, mm, c_one, d_invA+i*NB, NB, B+i, ldb, c_zero, dX+i, m);

                    if (i+NB >= m)
                        break;

                    magma_zgemm(MagmaTrans, MagmaNoTrans, m-i-NB, n, NB, c_neg_one, A+(i+NB)*lda+i, lda, dX+i, m, c_one, B+i+NB, ldb);
                }
            }
        }
        else{
            /* the conj transpose case */
            if (uplo == MagmaLower) {
                /* the lower case */
                /* handle the first block seperately with alpha */
                int mm = (m % NB == 0) ? NB : (m % NB);
                i = m-mm;
                magma_zgemm(MagmaConjTrans, MagmaNoTrans, mm, n, mm, alpha, d_invA+i*NB, NB, B+i, ldb, c_zero, dX+i, m);

                if (i-NB < 0) {
                    ztrsm_copy();
                    return;
                }

                magma_zgemm(MagmaConjTrans, MagmaNoTrans, i, n, mm, c_neg_one, A+i, lda, dX+i, m, alpha, B, ldb);

                /* the rest blocks */
                for( i=m-mm-NB; i >= 0; i -= NB ) {
                    magma_zgemm(MagmaConjTrans, MagmaNoTrans, NB, n, NB, c_one, d_invA+i*NB, NB, B+i, ldb, c_zero, dX+i, m);

                    if (i-NB < 0)
                        break;

                    magma_zgemm(MagmaConjTrans, MagmaNoTrans, i, n, NB, c_neg_one, A+i, lda, dX+i, m, c_one, B, ldb);
                }
            }
            else {
                /* the upper case */
                /* handle the first block seperately with alpha */
                int mm = min(NB, m);
                magma_zgemm(MagmaConjTrans, MagmaNoTrans, mm, n, mm, alpha, d_invA, NB, B, ldb, c_zero, dX, m);

                if (NB >= m) {
                    ztrsm_copy();
                    return;
                }

                magma_zgemm(MagmaConjTrans, MagmaNoTrans, m-NB, n, NB, c_neg_one, A+(NB)*lda, lda, dX, m, alpha, B+NB, ldb);

                /* the rest blocks */
                for( i=NB; i < m; i += NB ) {
                    mm = min(m-i, NB);
                    magma_zgemm(MagmaConjTrans, MagmaNoTrans, mm, n, mm, c_one, d_invA+i*NB, NB, B+i, ldb, c_zero, dX+i, m);

                    if (i+NB >= m)
                        break;

                    magma_zgemm(MagmaConjTrans, MagmaNoTrans, m-i-NB, n, NB, c_neg_one, A+(i+NB)*lda+i, lda, dX+i, m, c_one, B+i+NB, ldb);
                }
            }
        }
    }
    else {
        // side=R
        /* invert the diagonals
         */

        if (flag)
           diag_ztrtri(n, uplo, diag, A, d_invA, lda);

        if (transA == MagmaNoTrans) {
            /* the non-transpose case */
            if (uplo == MagmaLower) {
                /* the lower case */
                /* handle the first block seperately with alpha */
                int nn = (n % NB == 0) ? NB : (n % NB);
                i = n-nn;
                magma_zgemm(MagmaNoTrans, MagmaNoTrans, m, nn, nn, alpha, B+ldb*i, ldb, d_invA+i*NB, NB, c_zero, dX+i*m, m);

                if (i-NB < 0) {
                    ztrsm_copy();
                    return;
                }

                magma_zgemm(MagmaNoTrans, MagmaNoTrans, m, i, nn, c_neg_one, dX+i*m, m, A+i, lda, alpha, B, ldb);

                /* the rest blocks */
                for( i=n-nn-NB; i >= 0; i -= NB ) {
                    magma_zgemm(MagmaNoTrans, MagmaNoTrans, m, NB, NB, c_one, B+ldb*i, ldb, d_invA+i*NB, NB, c_zero, dX+i*m, m);

                    if (i-NB < 0)
                        break;

                    magma_zgemm(MagmaNoTrans, MagmaNoTrans, m, i, NB, c_neg_one, dX+i*m, m, A+i, lda, c_one, B, ldb);
                }
            }
            else {
                /* the upper case */
                /* handle the first block seperately with alpha */
                int nn = min(NB, n);
                magma_zgemm(MagmaNoTrans, MagmaNoTrans, m, nn, nn, alpha, B, ldb, d_invA, NB, c_zero, dX, m);

                if (NB >= n) {
                    ztrsm_copy();
                    return;
                }

                magma_zgemm(MagmaNoTrans, MagmaNoTrans, m, n-NB, NB, c_neg_one, dX, m, A+NB*lda, lda, alpha, B+NB*ldb, ldb);

                /* the rest blocks */
                for( i=NB; i < n; i += NB ) {
                    nn = min(NB, n-i);
                    magma_zgemm(MagmaNoTrans, MagmaNoTrans, m, nn, nn, c_one, B+ldb*i, ldb, d_invA+i*NB, NB, c_zero, dX+i*m, m);

                    if (i+NB >= n)
                        break;

                    magma_zgemm(MagmaNoTrans, MagmaNoTrans, m, n-i-NB, NB, c_neg_one, dX+i*m, m,   A+(i+NB)*lda+i, lda, c_one, B+(i+NB)*ldb, ldb);
                }
            }
        }
        else if (transA == MagmaTrans) {
            /* the transpose case */
            if (uplo == MagmaLower) {
                /* the lower case */
                /* handle the first block seperately with alpha */
                int nn = min(NB, n);
                magma_zgemm(MagmaNoTrans, MagmaTrans, m, nn, nn, alpha, B, ldb, d_invA, NB, c_zero, dX, m);

                if (NB >= n) {
                    ztrsm_copy();
                    return;
                }

                magma_zgemm(MagmaNoTrans, MagmaTrans, m, n-NB, NB, c_neg_one, dX, m, A+NB, lda, alpha, B+NB*ldb, ldb);

                /* the rest blocks */
                for( i=NB; i < n; i += NB ) {
                    nn = min(NB, n-i);
                    magma_zgemm(MagmaNoTrans, MagmaTrans, m, nn, nn, c_one, B+ldb*i, ldb, d_invA+i*NB, NB, c_zero, dX+i*m, m);

                    if (i+NB >= n)
                        break;

                    magma_zgemm(MagmaNoTrans, MagmaTrans, m, n-i-NB, NB, c_neg_one, dX+i*m, m,   A+i*lda+NB+i, lda, c_one, B+(i+NB)*ldb, ldb);
                }
            }
            else {
                /* the upper case */
                /* handle the first block seperately with alpha */
                int nn = (n % NB == 0) ? NB : (n % NB);
                i = n-nn;
                magma_zgemm(MagmaNoTrans, MagmaTrans, m, nn, nn, alpha, B+ldb*i, ldb, d_invA+i*NB, NB, c_zero, dX+i*m, m);

                if (i-NB < 0) {
                    ztrsm_copy();
                    return;
                }

                magma_zgemm(MagmaNoTrans, MagmaTrans, m, i, nn, c_neg_one, dX+i*m, m, A+i*lda, lda, alpha, B, ldb);

                /* the rest blocks */
                for( i=n-nn-NB; i >= 0; i -= NB ) {
                    magma_zgemm(MagmaNoTrans, MagmaTrans, m, NB, NB, c_one, B+ldb*i, ldb, d_invA+i*NB, NB, c_zero, dX+i*m, m);

                    if (i-NB < 0)
                        break;

                    magma_zgemm(MagmaNoTrans, MagmaTrans, m, i, NB, c_neg_one, dX+i*m, m, A+i*lda, lda, c_one, B, ldb);
                }
            }
        }
        else{
            /* the Conj transpose case */
            if (uplo == MagmaLower) {
                /* the lower case */
                /* handle the first block seperately with alpha */
                int nn = min(NB, n);
                magma_zgemm(MagmaNoTrans, MagmaConjTrans, m, nn, nn, alpha, B, ldb, d_invA, NB, c_zero, dX, m);

                if (NB >= n) {
                    ztrsm_copy();
                    return;
                }

                magma_zgemm(MagmaNoTrans, MagmaConjTrans, m, n-NB, NB, c_neg_one, dX, m, A+NB, lda, alpha, B+NB*ldb, ldb);

                /* the rest blocks */
                for( i=NB; i < n; i += NB ) {
                    nn = min(NB, n-i);
                    magma_zgemm(MagmaNoTrans, MagmaConjTrans, m, nn, nn, c_one, B+ldb*i, ldb, d_invA+i*NB, NB, c_zero, dX+i*m, m);

                    if (i+NB >= n)
                        break;

                    magma_zgemm(MagmaNoTrans, MagmaConjTrans, m, n-i-NB, NB, c_neg_one, dX+i*m, m,
                                                A+i*lda+NB+i, lda, c_one, B+(i+NB)*ldb, ldb);
                }
            }
            else {
                /* the upper case */
                /* handle the first block seperately with alpha */
                int nn = (n % NB == 0) ? NB : (n % NB);
                i = n-nn;
                magma_zgemm(MagmaNoTrans, MagmaConjTrans, m, nn, nn, alpha, B+ldb*i, ldb, d_invA+i*NB, NB, c_zero, dX+i*m, m);

                if (i-NB < 0) {
                    ztrsm_copy();
                    return;
                }

                magma_zgemm(MagmaNoTrans, MagmaConjTrans, m, i, nn, c_neg_one, dX+i*m, m, A+i*lda, lda, alpha, B, ldb);

                /* the rest blocks */
                for( i=n-nn-NB; i >= 0; i -= NB ) {
                    magma_zgemm(MagmaNoTrans, MagmaConjTrans, m, NB, NB, c_one, B+ldb*i, ldb, d_invA+i*NB, NB, c_zero, dX+i*m, m);

                    if (i-NB < 0)
                        break;

                    magma_zgemm(MagmaNoTrans, MagmaConjTrans, m, i, NB, c_neg_one, dX+i*m, m, A+i*lda, lda, c_one, B, ldb);
                }
            }
        }

    }

    ztrsm_copy();
}


/**
    @see magmablas_ztrsm_work
    @ingroup magma_zblas3
    ********************************************************************/
extern "C"
void magmablas_ztrsm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    const magmaDoubleComplex* A, magma_int_t lda,
    magmaDoubleComplex* B, magma_int_t ldb )
{
    magma_int_t nrowa = (side == MagmaLeft ? m : n);
    
    magma_int_t info = 0;
    if ( side != MagmaLeft && side != MagmaRight ) {
        info = -1;
    } else if ( uplo != MagmaUpper && uplo != MagmaLower ) {
        info = -2;
    } else if ( transA != MagmaNoTrans && transA != MagmaTrans && transA != Magma_ConjTrans ) {
        info = -3;
    } else if ( diag != MagmaUnit && diag != MagmaNonUnit ) {
        info = -4;
    } else if (m < 0) {
        info = -5;
    } else if (n < 0) {
        info = -6;
    } else if (lda < max(1,nrowa)) {
        info = -9;
    } else if (ldb < max(1,m)) {
        info = -11;
    }
    
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;
    }
    
    magmaDoubleComplex *d_invA, *dX;
    magma_int_t size_invA;
    magma_int_t size_x = m*n;
    if ( side == MagmaLeft ) {
        size_invA = ((m+NB-1)/NB)*NB*NB;
    }
    else {
        size_invA = ((n+NB-1)/NB)*NB*NB;
    }
    
    magma_zmalloc( &d_invA, size_invA );
    magma_zmalloc( &dX,     size_x    );
    if ( d_invA == NULL || dX == NULL ) {
        info = MAGMA_ERR_DEVICE_ALLOC;
        magma_xerbla( __func__, -(info) );
        goto cleanup;
    }
    
    cudaMemset( d_invA, 0, size_invA*sizeof(magmaDoubleComplex) );
    cudaMemset( dX,     0, size_x   *sizeof(magmaDoubleComplex) );
    
    magmablas_ztrsm_work( side, uplo, transA, diag, m, n, alpha,
                          A, lda, B, ldb, 1, d_invA, dX );
    
cleanup:
    magma_free( d_invA );
    magma_free( dX );
}
