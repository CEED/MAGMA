/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

*/
#include "common_magma.h"

#define PRECISION_z

/**
    Purpose
    =======
    ZGETF2_NOPIV computes an LU factorization of a general m-by-n
    matrix A without pivoting.

    The factorization has the form
       A = L * U
    where L is lower triangular with unit diagonal elements (lower
    trapezoidal if m > n), and U is upper triangular (upper
    trapezoidal if m < n).

    This is the right-looking Level 2 BLAS version of the algorithm.

    Arguments
    =========
    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in,out]
    A       COMPLEX_16 array, dimension (LDA,N)
            On entry, the m by n matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    @param[out]
    info    INTEGER
      -     = 0: successful exit
      -     < 0: if INFO = -k, the k-th argument had an illegal value
      -     > 0: if INFO = k, U(k,k) is exactly zero. The factorization
                 has been completed, but the factor U is exactly
                 singular, and division by zero will occur if it is used
                 to solve a system of equations.

    @ingroup magma_zgesv_aux
    ********************************************************************/
extern "C" magma_int_t
magma_zgetf2_nopiv(magma_int_t *m, magma_int_t *n, magmaDoubleComplex *a,
                   magma_int_t *lda, magma_int_t *info)
{
    magmaDoubleComplex c_one = MAGMA_Z_ONE;
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magma_int_t ione = 1;

    magma_int_t a_dim1, a_offset, i__1, i__2, i__3;
    magmaDoubleComplex z__1;
    magma_int_t i__, j;
    double sfmin;

    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;

    /* Function Body */
    *info = 0;
    if (*m < 0) {
        *info = -1;
    } else if (*n < 0) {
        *info = -2;
    } else if (*lda < max(1,*m)) {
        *info = -4;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return if possible */
    if (*m == 0 || *n == 0)
        return *info;

    /* Compute machine safe minimum */
    sfmin = lapackf77_dlamch("S");

    i__1 = min(*m,*n);
    for (j = 1; j <= i__1; ++j) {
        /* Test for singularity. */
        i__2 = j + j * a_dim1;
        if ( ! MAGMA_Z_EQUAL(a[i__2], c_zero)) {
            /* Compute elements J+1:M of J-th column. */
            if (j < *m) {
                if (MAGMA_Z_ABS(a[j + j * a_dim1]) >= sfmin) {
                    i__2 = *m - j;
                    z__1 = MAGMA_Z_DIV(c_one, a[j + j * a_dim1]);
                    blasf77_zscal(&i__2, &z__1, &a[j + 1 + j * a_dim1], &ione);
                }
                else {
                    i__2 = *m - j;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        i__3 = j + i__ + j * a_dim1;
                        a[i__3] = MAGMA_Z_DIV(a[j + i__ + j * a_dim1], a[j + j*a_dim1]);
                    }
                }
            }
        }
        else if (*info == 0) {
            *info = j;
        }

        if (j < min(*m,*n)) {
            /* Update trailing submatrix. */
            i__2 = *m - j;
            i__3 = *n - j;
            blasf77_zgeru( &i__2, &i__3, &c_neg_one,
                           &a[j + 1 + j * a_dim1], &ione,
                           &a[j + (j+1) * a_dim1], lda,
                           &a[j + 1 + (j+1) * a_dim1], lda);
        }
    }

    return *info;
} /* magma_zgetf2_nopiv */
