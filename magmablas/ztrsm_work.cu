/*
    -- MAGMA (version 1.4) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s

       @author Peng Du, Tingxing Dong
*/
#include "common_magma.h"

#define BLOCK_SIZE 16 // inner blocking size, <=32
#define NB 128        // outer blocking size, >BLOCK_SIZE

__global__ void
ztrsm_work_copy_kernel (int M, int N, magmaDoubleComplex *b, int ldb, magmaDoubleComplex *d_x, int ldx)
{
    int by = blockIdx.y;
    int gx = blockIdx.x*blockDim.x + threadIdx.x;
    if (gx < M)
        b[by*ldb+gx] = d_x[by*ldx+gx];
}


#define MAX_THREAD_PER_BLOCK 512
#define WARP_SIZE 32


#define ztrsm_work_copy() \
    do { \
        dim3 dimBlock( (M >= MAX_THREAD_PER_BLOCK) ? MAX_THREAD_PER_BLOCK : (WARP_SIZE*((M/WARP_SIZE)+(M % WARP_SIZE != 0))), 1 ); \
        dim3 dimGrid( (M - 1)/dimBlock.x + 1, N ); \
        ztrsm_work_copy_kernel<<< dimGrid, dimBlock, 0, magma_stream >>>(M, N, b, ldb, d_x, M); \
        magma_device_sync(); \
    } while(0)


/*
 * magmablas_ztrsm
 */

extern "C"
void diag_ztrtri (magma_int_t M, magma_uplo_t uplo, magma_diag_t diag, const magmaDoubleComplex *A, magmaDoubleComplex *d_dinvA, magma_int_t lda);

extern "C"
void magmablas_ztrsm_work(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, magma_int_t M, magma_int_t N,
    magmaDoubleComplex alpha,
    const magmaDoubleComplex* A, magma_int_t lda,
    magmaDoubleComplex* b, magma_int_t ldb,
    magma_int_t flag,
    magmaDoubleComplex* d_dinvA, magmaDoubleComplex *d_x)
{
/*  -- MAGMA (version 1.4) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

    Purpose
    =======

    ztrsm solves one of the matrix equations on gpu

        op( A )*x = alpha*b,   or   x*op( A ) = alpha*b,

    where alpha is a scalar, x and b are m by n matrices, A is a unit, or
    non-unit, upper or lower triangular matrix and op( A ) is one of

        op( A ) = A   or   op( A ) = A^T.

    The matrix X is overwritten on B.

    This is an asynchronous version of magmablas_dtrsm with "workspace" as an argument.

    Arguments
    ==========

    side    (input) CHARACTER*1.
            On entry, side specifies whether op( A ) appears on the left
            or right of X as follows:

                side = 'L' or 'l'   op( A )*X = alpha*B.

                side = 'R' or 'r'   X*op( A ) = alpha*B.

    uplo    (input) CHARACTER*1.
            On entry, uplo specifies whether the matrix A is an upper or
            lower triangular matrix as follows:

                uplo = 'U' or 'u'   A is an upper triangular matrix.

                uplo = 'L' or 'l'   A is a lower triangular matrix.

    transA  (input) CHARACTER*1.
            On entry, transA specifies the form of op( A ) to be used in
            the matrix multiplication as follows:

                transA = 'N' or 'n'   op( A ) = A.

                transA = 'T' or 't'   op( A ) = A^T.

                transA = 'C' or 'c'   op( A ) = A^T.

    diag    (input) CHARACTER*1.
            On entry, diag specifies whether or not A is unit triangular
            as follows:

                diag = 'U' or 'u'   A is assumed to be unit triangular.

                diag = 'N' or 'n'   A is not assumed to be unit triangular.

    m       (input) INTEGER.
            On entry, m specifies the number of rows of B. m must be at
            least zero.

    n       (input) INTEGER.
            On entry, n specifies the number of columns of B. n must be
            at least zero.

    alpha   (input) COMPLEX.
            On entry, alpha specifies the scalar alpha. When alpha is
            zero then A is not referenced and B need not be set before
            entry.

    A       (input) COMPLEX array of DIMENSION ( lda, k ), where k is m
            when side = 'L' or 'l' and is n when side = 'R' or 'r'.
            Before entry with uplo = 'U' or 'u', the leading k by k
            upper triangular part of the array A must contain the upper
            triangular matrix and the strictly lower triangular part of
            A is not referenced.
            Before entry with uplo = 'L' or 'l', the leading k by k
            lower triangular part of the array A must contain the lower
            triangular matrix and the strictly upper triangular part of
            A is not referenced.
            Note that when diag = 'U' or 'u', the diagonal elements of
            A are not referenced either, but are assumed to be unity.

    lda     (input) INTEGER.
            On entry, lda specifies the first dimension of A as declared
            in the calling (sub) program. When side = 'L' or 'l' then
            lda must be at least max( 1, m ), when side = 'R' or 'r'
            then lda must be at least max( 1, n ).

    b       (input,output) COMPLEX array of DIMENSION ( ldb, n ).
            Before entry, the leading m by n part of the array B must
            contain the right-hand side matrix B, and on exit is
            overwritten by the solution matrix X.

    ldb     (input) INTEGER.
            On entry, ldb specifies the first dimension of B as declared
            in the calling (sub) program. ldb must be at least
            max( 1, m ).

    flag    (input) BOOLEAN.
            If flag is true, invert diagonal blocks.
            If flag is false, assume diagonal blocks are already inverted. 

    d_dinvA (workspace) size NB*((M+NB-1)/NB))*NB, on device.

    d_x     (workspace) size N*M, on device.

    Level 3 Blas routine.
    ===================================================================== */

    int i;

    /* quick return on wrong size */
    if (M <= 0 || N <= 0)
        return;
    
    char Notrans = 'N';
    char Trans = 'T';
    char Conjtrans = 'C';
    magmaDoubleComplex neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex one = MAGMA_Z_ONE;
    magmaDoubleComplex zero = MAGMA_Z_ZERO;

    if (side == MagmaLeft) {
        // side=L
        /* invert the diagonals
         */
        if (flag == 1)
            diag_ztrtri (M, uplo, diag, A, d_dinvA, lda);

        if (transA == MagmaNoTrans) {
            /* the non-transpose case */
            if (uplo == MagmaLower) {

                /* the lower case */
                /* handle the first block seperately with alpha */
                int MM = min (NB, M);
                cublasZgemm(Notrans, Notrans, MM, N, MM, alpha, d_dinvA, NB, b, ldb, zero, d_x, M);

                if (NB >= M) {
                    ztrsm_work_copy();
                    return;
                }

                cublasZgemm(Notrans, Notrans, M-NB, N, NB, neg_one, A+NB, lda, d_x, M, alpha, b+NB, ldb);

                /* the rest blocks */
                for( i=NB; i < M; i += NB ) {
                    MM = min (M-i, NB);
                    cublasZgemm(Notrans, Notrans, MM, N, MM, one, d_dinvA+i*NB, NB, b+i, ldb, zero, d_x+i, M);

                    if (i+NB >= M)
                        break;

                    cublasZgemm(Notrans, Notrans, M-i-NB, N, NB, neg_one, A+i*lda+i+NB, lda, d_x+i, M, one, b+i+NB, ldb);
                }
            }
            else {
                /* the upper case */
                /* handle the first block seperately with alpha */
                int MM = (M % NB == 0) ? NB : (M % NB);
                i = M-MM;
                cublasZgemm(Notrans, Notrans, MM, N, MM, alpha, d_dinvA+i*NB, NB, b+i, ldb, zero, d_x+i, M);

                if (i-NB < 0) {
                    ztrsm_work_copy();
                    return;
                }

                cublasZgemm(Notrans, Notrans, i, N, MM, neg_one, A+i*lda, lda, d_x+i, M, alpha, b, ldb);

                /* the rest blocks */
                for( i=M-MM-NB; i >= 0; i -= NB ) {
                    cublasZgemm(Notrans, Notrans, NB, N, NB, one, d_dinvA+i*NB, NB, b+i, ldb, zero, d_x+i, M);

                    if (i-NB < 0)
                        break;

                    cublasZgemm(Notrans, Notrans, i, N, NB, neg_one, A+i*lda, lda, d_x+i, M, one, b, ldb);
                }
            }
        }
        else if( transA == MagmaTrans) {
            /* the transpose case */
            if (uplo == MagmaLower) {
                /* the lower case */
                /* handle the first block seperately with alpha */
                int MM = (M % NB == 0) ? NB : (M % NB);
                i = M-MM;
                cublasZgemm(Trans, Notrans, MM, N, MM, alpha, d_dinvA+i*NB, NB, b+i, ldb, zero, d_x+i, M);

                if (i-NB < 0) {
                    ztrsm_work_copy();
                    return;
                }

                cublasZgemm(Trans, Notrans, i, N, MM, neg_one, A+i, lda, d_x+i, M, alpha, b, ldb);

                /* the rest blocks */
                for( i=M-MM-NB; i >= 0; i -= NB ) {
                    cublasZgemm(Trans, Notrans, NB, N, NB, one, d_dinvA+i*NB, NB, b+i, ldb, zero, d_x+i, M);

                    if (i-NB < 0)
                        break;

                    cublasZgemm(Trans, Notrans, i, N, NB, neg_one, A+i, lda, d_x+i, M, one, b, ldb);
                }
            }
            else {
                /* the upper case */
                /* handle the first block seperately with alpha */
                int MM = min (NB, M);
                cublasZgemm(Trans, Notrans, MM, N, MM, alpha, d_dinvA, NB, b, ldb, zero, d_x, M);

                if (NB >= M) {
                    ztrsm_work_copy();
                    return;
                }

                cublasZgemm(Trans, Notrans, M-NB, N, NB, neg_one, A+(NB)*lda, lda, d_x, M, alpha, b+NB, ldb);

                /* the rest blocks */
                for( i=NB; i < M; i += NB ) {
                    MM = min (M-i, NB);
                    cublasZgemm(Trans, Notrans, MM, N, MM, one, d_dinvA+i*NB, NB, b+i, ldb, zero, d_x+i, M);

                    if (i+NB >= M)
                        break;

                    cublasZgemm(Trans, Notrans, M-i-NB, N, NB, neg_one, A+(i+NB)*lda+i, lda, d_x+i, M, one, b+i+NB, ldb);
                }
            }
        }
        else{
            /* the conj transpose case */
            if (uplo == MagmaLower) {
                /* the lower case */
                /* handle the first block seperately with alpha */
                int MM = (M % NB == 0) ? NB : (M % NB);
                i = M-MM;
                cublasZgemm(Conjtrans, Notrans, MM, N, MM, alpha, d_dinvA+i*NB, NB, b+i, ldb, zero, d_x+i, M);

                if (i-NB < 0) {
                    ztrsm_work_copy();
                    return;
                }

                cublasZgemm(Conjtrans, Notrans, i, N, MM, neg_one, A+i, lda, d_x+i, M, alpha, b, ldb);

                /* the rest blocks */
                for( i=M-MM-NB; i >= 0; i -= NB ) {
                    cublasZgemm(Conjtrans, Notrans, NB, N, NB, one, d_dinvA+i*NB, NB, b+i, ldb, zero, d_x+i, M);

                    if (i-NB < 0)
                        break;

                    cublasZgemm(Conjtrans, Notrans, i, N, NB, neg_one, A+i, lda, d_x+i, M, one, b, ldb);
                }
            }
            else {
                /* the upper case */
                /* handle the first block seperately with alpha */
                int MM = min (NB, M);
                cublasZgemm(Conjtrans, Notrans, MM, N, MM, alpha, d_dinvA, NB, b, ldb, zero, d_x, M);

                if (NB >= M) {
                    ztrsm_work_copy();
                    return;
                }

                cublasZgemm(Conjtrans, Notrans, M-NB, N, NB, neg_one, A+(NB)*lda, lda, d_x, M, alpha, b+NB, ldb);

                /* the rest blocks */
                for( i=NB; i < M; i += NB ) {
                    MM = min (M-i, NB);
                    cublasZgemm(Conjtrans, Notrans, MM, N, MM, one, d_dinvA+i*NB, NB, b+i, ldb, zero, d_x+i, M);

                    if (i+NB >= M)
                        break;

                    cublasZgemm(Conjtrans, Notrans, M-i-NB, N, NB, neg_one, A+(i+NB)*lda+i, lda, d_x+i, M, one, b+i+NB, ldb);
                }
            }
        }
    }
    else {
        // side=R
        /* invert the diagonals
         */

        if (flag == 1)
           diag_ztrtri (N, uplo, diag, A, d_dinvA, lda);

        if (transA == MagmaNoTrans) {
            /* the non-transpose case */
            if (uplo == MagmaLower) {
                /* the lower case */
                /* handle the first block seperately with alpha */
                int NN = (N % NB == 0) ? NB : (N % NB);
                i = N-NN;
                cublasZgemm(Notrans, Notrans, M, NN, NN, alpha, b+ldb*i, ldb, d_dinvA+i*NB, NB, zero, d_x+i*M, M);

                if (i-NB < 0) {
                    ztrsm_work_copy();
                    return;
                }

                cublasZgemm(Notrans, Notrans, M, i, NN, neg_one, d_x+i*M, M, A+i, lda, alpha, b, ldb);

                /* the rest blocks */
                for( i=N-NN-NB; i >= 0; i -= NB ) {
                    cublasZgemm(Notrans, Notrans, M, NB, NB, one, b+ldb*i, ldb, d_dinvA+i*NB, NB, zero, d_x+i*M, M);

                    if (i-NB < 0)
                        break;

                    cublasZgemm(Notrans, Notrans, M, i, NB, neg_one, d_x+i*M, M, A+i, lda, one, b, ldb);
                }
            }
            else {
                /* the upper case */
                /* handle the first block seperately with alpha */
                int NN = min(NB, N);
                cublasZgemm(Notrans, Notrans, M, NN, NN, alpha, b, ldb, d_dinvA, NB, zero, d_x, M);

                if (NB >= N) {
                    ztrsm_work_copy();
                    return;
                }

                cublasZgemm(Notrans, Notrans, M, N-NB, NB, neg_one, d_x, M, A+NB*lda, lda, alpha, b+NB*ldb, ldb);

                /* the rest blocks */
                for( i=NB; i < N; i += NB ) {
                    NN = min(NB, N-i);
                    cublasZgemm(Notrans, Notrans, M, NN, NN, one, b+ldb*i, ldb, d_dinvA+i*NB, NB, zero, d_x+i*M, M);

                    if (i+NB >= N)
                        break;

                    cublasZgemm(Notrans, Notrans, M, N-i-NB, NB, neg_one, d_x+i*M, M,   A+(i+NB)*lda+i, lda, one, b+(i+NB)*ldb, ldb);
                }
            }
        }
        else if (transA == MagmaTrans) {
            /* the transpose case */
            if (uplo == MagmaLower) {
                /* the lower case */
                /* handle the first block seperately with alpha */
                int NN = min(NB, N);
                cublasZgemm(Notrans, Trans, M, NN, NN, alpha, b, ldb, d_dinvA, NB, zero, d_x, M);

                if (NB >= N) {
                    ztrsm_work_copy();
                    return;
                }

                cublasZgemm(Notrans, Trans, M, N-NB, NB, neg_one, d_x, M, A+NB, lda, alpha, b+NB*ldb, ldb);

                /* the rest blocks */
                for( i=NB; i < N; i += NB ) {
                    NN = min(NB, N-i);
                    cublasZgemm(Notrans, Trans, M, NN, NN, one, b+ldb*i, ldb, d_dinvA+i*NB, NB, zero, d_x+i*M, M);

                    if (i+NB >= N)
                        break;

                    cublasZgemm(Notrans, Trans, M, N-i-NB, NB, neg_one, d_x+i*M, M,   A+i*lda+NB+i, lda, one, b+(i+NB)*ldb, ldb);
                }
            }
            else {
                /* the upper case */
                /* handle the first block seperately with alpha */
                int NN = (N % NB == 0) ? NB : (N % NB);
                i = N-NN;
                cublasZgemm(Notrans, Trans, M, NN, NN, alpha, b+ldb*i, ldb, d_dinvA+i*NB, NB, zero, d_x+i*M, M);

                if (i-NB < 0) {
                    ztrsm_work_copy();
                    return;
                }

                cublasZgemm(Notrans, Trans, M, i, NN, neg_one, d_x+i*M, M, A+i*lda, lda, alpha, b, ldb);

                /* the rest blocks */
                for( i=N-NN-NB; i >= 0; i -= NB ) {
                    cublasZgemm(Notrans, Trans, M, NB, NB, one, b+ldb*i, ldb, d_dinvA+i*NB, NB, zero, d_x+i*M, M);

                    if (i-NB < 0)
                        break;

                    cublasZgemm(Notrans, Trans, M, i, NB, neg_one, d_x+i*M, M, A+i*lda, lda, one, b, ldb);
                }
            }
        }
        else{
            /* the Conj transpose case */
            if (uplo == MagmaLower) {
                /* the lower case */
                /* handle the first block seperately with alpha */
                int NN = min(NB, N);
                cublasZgemm(Notrans, Conjtrans, M, NN, NN, alpha, b, ldb, d_dinvA, NB, zero, d_x, M);

                if (NB >= N) {
                    ztrsm_work_copy();
                    return;
                }

                cublasZgemm(Notrans, Conjtrans, M, N-NB, NB, neg_one, d_x, M, A+NB, lda, alpha, b+NB*ldb, ldb);

                /* the rest blocks */
                for( i=NB; i < N; i += NB ) {
                    NN = min(NB, N-i);
                    cublasZgemm(Notrans, Conjtrans, M, NN, NN, one, b+ldb*i, ldb, d_dinvA+i*NB, NB, zero, d_x+i*M, M);

                    if (i+NB >= N)
                        break;

                    cublasZgemm(Notrans, Conjtrans, M, N-i-NB, NB, neg_one, d_x+i*M, M,   
                                                A+i*lda+NB+i, lda, one, b+(i+NB)*ldb, ldb);
                }
            }
            else {
                /* the upper case */
                /* handle the first block seperately with alpha */
                int NN = (N % NB == 0) ? NB : (N % NB);
                i = N-NN;
                cublasZgemm(Notrans, Conjtrans, M, NN, NN, alpha, b+ldb*i, ldb, d_dinvA+i*NB, NB, zero, d_x+i*M, M);

                if (i-NB < 0) {
                    ztrsm_work_copy();
                    return;
                }

                cublasZgemm(Notrans, Conjtrans, M, i, NN, neg_one, d_x+i*M, M, A+i*lda, lda, alpha, b, ldb);

                /* the rest blocks */
                for( i=N-NN-NB; i >= 0; i -= NB ) {
                    cublasZgemm(Notrans, Conjtrans, M, NB, NB, one, b+ldb*i, ldb, d_dinvA+i*NB, NB, zero, d_x+i*M, M);

                    if (i-NB < 0)
                        break;

                    cublasZgemm(Notrans, Conjtrans, M, i, NB, neg_one, d_x+i*M, M, A+i*lda, lda, one, b, ldb);
                }
            }
        }

    }

    ztrsm_work_copy();

}
