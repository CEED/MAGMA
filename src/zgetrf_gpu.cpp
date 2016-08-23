/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Stan Tomov
       @author Mark Gates
       
       @precisions normal z -> s d c

*/
#include "magma_internal.h"

/***************************************************************************//**
    Purpose
    -------
    ZGETRF computes an LU factorization of a general M-by-N matrix A
    using partial pivoting with row interchanges.

    The factorization has the form
        A = P * L * U
    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 3 BLAS version of the algorithm.
    
    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in,out]
    dA      COMPLEX_16 array on the GPU, dimension (LDDA,N).
            On entry, the M-by-N matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    @param[in]
    ldda     INTEGER
            The leading dimension of the array A.  LDDA >= max(1,M).

    @param[out]
    ipiv    INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.
      -     > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
                  has been completed, but the factor U is exactly
                  singular, and division by zero will occur if it is used
                  to solve a system of equations.

    @ingroup magma_getrf
*******************************************************************************/
extern "C" magma_int_t
magma_zgetrf_gpu(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t *ipiv,
    magma_int_t *info )
{
    #ifdef HAVE_clBLAS
    #define  dA(i_, j_) dA,  (dA_offset  + (i_)       + (j_)*ldda)
    #define dAT(i_, j_) dAT, (dAT_offset + (i_)*lddat + (j_))
    #define dAP(i_, j_) dAP, (             (i_)          + (j_)*maxm)
    #else
    #define  dA(i_, j_) (dA  + (i_)       + (j_)*ldda)
    #define dAT(i_, j_) (dAT + (i_)*lddat + (j_))
    #define dAP(i_, j_) (dAP + (i_)       + (j_)*maxm)
    #endif

    magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;

    magma_int_t iinfo, nb;
    magma_int_t maxm, maxn, minmn;
    magma_int_t i, j, jb, rows, lddat, ldwork;
    magmaDoubleComplex_ptr dAT=NULL, dAP=NULL;
    magmaDoubleComplex *work=NULL;

    /* Check arguments */
    *info = 0;
    if (m < 0)
        *info = -1;
    else if (n < 0)
        *info = -2;
    else if (ldda < max(1,m))
        *info = -4;

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0)
        return *info;

    /* Function Body */
    minmn = min( m, n );
    nb    = magma_get_zgetrf_nb( m, n );

    magma_queue_t queues[2] = { NULL };
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queues[0] );
    magma_queue_create( cdev, &queues[1] );

    if (nb <= 1 || nb >= min(m,n)) {
        /* Use CPU code. */
        if ( MAGMA_SUCCESS != magma_zmalloc_cpu( &work, m*n )) {
            *info = MAGMA_ERR_HOST_ALLOC;
            goto cleanup;
        }
        magma_zgetmatrix( m, n, dA(0,0), ldda, work, m, queues[0] );
        lapackf77_zgetrf( &m, &n, work, &m, ipiv, info );
        magma_zsetmatrix( m, n, work, m, dA(0,0), ldda, queues[0] );
        magma_free_cpu( work );  work=NULL;
    }
    else {
        /* Use hybrid blocked code. */
        maxm = magma_roundup( m, 32 );
        maxn = magma_roundup( n, 32 );

        if (MAGMA_SUCCESS != magma_zmalloc( &dAP, nb*maxm )) {
            *info = MAGMA_ERR_DEVICE_ALLOC;
            goto cleanup;
        }

        // square matrices can be done in place;
        // rectangular requires copy to transpose
        if ( m == n ) {
            dAT = dA;
            lddat = ldda;
            magmablas_ztranspose_inplace( m, dAT(0,0), lddat, queues[0] );
        }
        else {
            lddat = maxn;  // N-by-M
            if (MAGMA_SUCCESS != magma_zmalloc( &dAT, lddat*maxm )) {
                *info = MAGMA_ERR_DEVICE_ALLOC;
                goto cleanup;
            }
            magmablas_ztranspose( m, n, dA(0,0), ldda, dAT(0,0), lddat, queues[0] );
        }
        magma_queue_sync( queues[0] );  // finish transpose

        ldwork = maxm;
        if (MAGMA_SUCCESS != magma_zmalloc_pinned( &work, ldwork*nb )) {
            *info = MAGMA_ERR_HOST_ALLOC;
            goto cleanup;
        }

        for( j=0; j < minmn-nb; j += nb ) {
            // get j-th panel from device
            magmablas_ztranspose( nb, m-j, dAT(j,j), lddat, dAP(0,0), maxm, queues[1] );
            magma_queue_sync( queues[1] );  // wait for transpose
            magma_zgetmatrix_async( m-j, nb, dAP(0,0), maxm, work, ldwork, queues[0] );

            if ( j > 0 ) {
                magma_ztrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                             n-(j+nb), nb,
                             c_one, dAT(j-nb, j-nb), lddat,
                                    dAT(j-nb, j+nb), lddat, queues[1] );
                magma_zgemm( MagmaNoTrans, MagmaNoTrans,
                             n-(j+nb), m-j, nb,
                             c_neg_one, dAT(j-nb, j+nb), lddat,
                                        dAT(j,    j-nb), lddat,
                             c_one,     dAT(j,    j+nb), lddat, queues[1] );
            }

            // do the cpu part
            rows = m - j;
            magma_queue_sync( queues[0] );  // wait to get work
            lapackf77_zgetrf( &rows, &nb, work, &ldwork, ipiv+j, &iinfo );
            if ( *info == 0 && iinfo > 0 )
                *info = iinfo + j;

            // send j-th panel to device
            magma_zsetmatrix_async( m-j, nb, work, ldwork, dAP, maxm, queues[0] );

            for( i=j; i < j + nb; ++i ) {
                ipiv[i] += j;
            }
            magmablas_zlaswp( n, dAT(0,0), lddat, j + 1, j + nb, ipiv, 1, queues[1] );

            magma_queue_sync( queues[0] );  // wait to set dAP
            magmablas_ztranspose( m-j, nb, dAP(0,0), maxm, dAT(j,j), lddat, queues[1] );

            // do the small non-parallel computations (next panel update)
            if ( j + nb < minmn - nb ) {
                magma_ztrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                             nb, nb,
                             c_one, dAT(j, j   ), lddat,
                                    dAT(j, j+nb), lddat, queues[1] );
                magma_zgemm( MagmaNoTrans, MagmaNoTrans,
                             nb, m-(j+nb), nb,
                             c_neg_one, dAT(j,    j+nb), lddat,
                                        dAT(j+nb, j   ), lddat,
                             c_one,     dAT(j+nb, j+nb), lddat, queues[1] );
            }
            else {
                magma_ztrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                             n-(j+nb), nb,
                             c_one, dAT(j, j   ), lddat,
                                    dAT(j, j+nb), lddat, queues[1] );
                magma_zgemm( MagmaNoTrans, MagmaNoTrans,
                             n-(j+nb), m-(j+nb), nb,
                             c_neg_one, dAT(j,    j+nb), lddat,
                                        dAT(j+nb, j   ), lddat,
                             c_one,     dAT(j+nb, j+nb), lddat, queues[1] );
            }
        }

        jb = min( m-j, n-j );
        if ( jb > 0 ) {
            rows = m - j;
            
            magmablas_ztranspose( jb, rows, dAT(j,j), lddat, dAP(0,0), maxm, queues[1] );
            magma_zgetmatrix( rows, jb, dAP(0,0), maxm, work, ldwork, queues[1] );
            
            // do the cpu part
            lapackf77_zgetrf( &rows, &jb, work, &ldwork, ipiv+j, &iinfo );
            if ( *info == 0 && iinfo > 0 )
                *info = iinfo + j;
            
            for( i=j; i < j + jb; ++i ) {
                ipiv[i] += j;
            }
            magmablas_zlaswp( n, dAT(0,0), lddat, j + 1, j + jb, ipiv, 1, queues[1] );
            
            // send j-th panel to device
            magma_zsetmatrix( rows, jb, work, ldwork, dAP(0,0), maxm, queues[1] );
            magmablas_ztranspose( rows, jb, dAP(0,0), maxm, dAT(j,j), lddat, queues[1] );
            
            magma_ztrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                         n-j-jb, jb,
                         c_one, dAT(j,j),    lddat,
                                dAT(j,j+jb), lddat, queues[1] );
        }
        
        // undo transpose
        if ( m == n ) {
            magmablas_ztranspose_inplace( m, dAT(0,0), lddat, queues[1] );
        }
        else {
            magmablas_ztranspose( n, m, dAT(0,0), lddat, dA(0,0), ldda, queues[1] );
        }
    }
    
cleanup:
    magma_queue_destroy( queues[0] );
    magma_queue_destroy( queues[1] );
    
    magma_free( dAP );
    if (m != n) {
        magma_free( dAT );
    }
    magma_free_pinned( work );
    
    return *info;
} /* magma_zgetrf_gpu */
