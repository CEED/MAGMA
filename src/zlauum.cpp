/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

*/
#include "common_magma.h"

/**
    Purpose
    -------
    ZLAUUM computes the product U * U' or L' * L, where the triangular
    factor U or L is stored in the upper or lower triangular part of
    the array A.

    If UPLO = MagmaUpper then the upper triangle of the result is stored,
    overwriting the factor U in A.
    If UPLO = MagmaLower then the lower triangle of the result is stored,
    overwriting the factor L in A.
    This is the blocked form of the algorithm, calling Level 3 BLAS.

    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
            Specifies whether the triangular factor stored in the array A
            is upper or lower triangular:
      -     = MagmaUpper:  Upper triangular
      -     = MagmaLower:  Lower triangular

    @param[in]
    n       INTEGER
            The order of the triangular factor U or L.  N >= 0.

    @param[in,out]
    A       COPLEX_16 array, dimension (LDA,N)
            On entry, the triangular factor U or L.
            On exit, if UPLO = MagmaUpper, the upper triangle of A is
            overwritten with the upper triangle of the product U * U';
            if UPLO = MagmaLower, the lower triangle of A is overwritten with
            the lower triangle of the product L' * L.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    @param[out]
    info    INTEGER
      -     = 0: successful exit
      -     < 0: if INFO = -k, the k-th argument had an illegal value

    @ingroup magma_zposv_aux
    ***************************************************************************/
extern "C" magma_int_t
magma_zlauum(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    magma_int_t *info)
{
#define A(i, j)  (A  + (j)*lda  + (i))
#define dA(i, j) (dA + (j)*ldda + (i))

    /* Local variables */
    const char* uplo_ = lapack_uplo_const( uplo );
    magma_int_t     ldda, nb;
    magma_int_t i, ib;
    magmaDoubleComplex c_one = MAGMA_Z_ONE;
    double             d_one = MAGMA_D_ONE;
    magmaDoubleComplex    *dA;
    int upper = (uplo == MagmaUpper);

    *info = 0;
    if (! upper && uplo != MagmaLower)
        *info = -1;
    else if (n < 0)
        *info = -2;
    else if (lda < max(1,n))
        *info = -4;

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return */
    if ( n == 0 )
        return *info;

    ldda = magma_roundup( n, 32 );

    if (MAGMA_SUCCESS != magma_zmalloc( &dA, (n)*ldda )) {
        *info = MAGMA_ERR_DEVICE_ALLOC;
        return *info;
    }

    magma_queue_t stream[2];
    magma_queue_create( &stream[0] );
    magma_queue_create( &stream[1] );

    nb = magma_get_zpotrf_nb(n);

    if (nb <= 1 || nb >= n)
        lapackf77_zlauum(uplo_, &n, A, &lda, info);
    else {
        if (upper) {
            /* Compute the product U * U'. */
            for (i=0; i < n; i += nb) {
                ib=min(nb,n-i);

                magma_zsetmatrix_async( ib, ib,
                                        A(i,i),   lda,
                                        dA(i, i), ldda, stream[1] );

                magma_zsetmatrix_async( ib, (n-i-ib),
                                        A(i,i+ib),  lda,
                                        dA(i,i+ib), ldda, stream[0] );

                magma_queue_sync( stream[1] );

                magma_ztrmm( MagmaRight, MagmaUpper,
                             MagmaConjTrans, MagmaNonUnit, i, ib,
                             c_one, dA(i,i), ldda, dA(0, i),ldda);


                lapackf77_zlauum(MagmaUpperStr, &ib, A(i,i), &lda, info);

                magma_zsetmatrix_async( ib, ib,
                                        A(i, i),  lda,
                                        dA(i, i), ldda, stream[0] );

                if (i+ib < n) {
                    magma_zgemm( MagmaNoTrans, MagmaConjTrans,
                                 i, ib, (n-i-ib), c_one, dA(0,i+ib),
                                 ldda, dA(i, i+ib),ldda, c_one,
                                 dA(0,i), ldda);

                    magma_queue_sync( stream[0] );

                    magma_zherk( MagmaUpper, MagmaNoTrans, ib,(n-i-ib),
                                 d_one, dA(i, i+ib), ldda,
                                 d_one, dA(i, i), ldda);
                }

                magma_zgetmatrix( i+ib, ib,
                                  dA(0, i), ldda,
                                  A(0, i),  lda );
            }
        }
        else {
            /* Compute the product L' * L. */
            for (i=0; i < n; i += nb) {
                ib=min(nb,n-i);
                magma_zsetmatrix_async( ib, ib,
                                        A(i,i),   lda,
                                        dA(i, i), ldda, stream[1] );

                magma_zsetmatrix_async( (n-i-ib), ib,
                                        A(i+ib, i),  lda,
                                        dA(i+ib, i), ldda, stream[0] );

                magma_queue_sync( stream[1] );

                magma_ztrmm( MagmaLeft, MagmaLower,
                             MagmaConjTrans, MagmaNonUnit, ib,
                             i, c_one, dA(i,i), ldda,
                             dA(i, 0),ldda);


                lapackf77_zlauum(MagmaLowerStr, &ib, A(i,i), &lda, info);

                magma_zsetmatrix_async( ib, ib,
                                        A(i, i),  lda,
                                        dA(i, i), ldda, stream[0] );

                if (i+ib < n) {
                    magma_zgemm(MagmaConjTrans, MagmaNoTrans,
                                    ib, i, (n-i-ib), c_one, dA( i+ib,i),
                                    ldda, dA(i+ib, 0),ldda, c_one,
                                    dA(i,0), ldda);

                    magma_queue_sync( stream[0] );

                    magma_zherk(MagmaLower, MagmaConjTrans, ib, (n-i-ib),
                                    d_one, dA(i+ib, i), ldda,
                                    d_one, dA(i, i), ldda);
                }
                magma_zgetmatrix( ib, i+ib,
                                  dA(i, 0), ldda,
                                  A(i, 0),  lda );
            }
        }
    }
    magma_queue_destroy( stream[0] );
    magma_queue_destroy( stream[1] );

    magma_free( dA );

    return *info;
}
