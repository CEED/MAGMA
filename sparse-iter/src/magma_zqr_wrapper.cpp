/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Hartwig Anzt
       @author Eduardo Ponce

       @precisions normal z -> s d c
*/

#include "common_magmasparse.h"

#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define MAX(a,b) (((a) > (b)) ? (a) : (b))



/**
    Purpose
    -------

    This is a wrapper to call MAGMA QR on the data structure of sparse matrices.

    Arguments
    ---------

    @param[in]
    m           magma_int_t
                dimension m
                
    @param[in]
    n           magma_int_t
                dimension n
                
    @param[in]
    A           magma_z_matrix
                input matrix A
                
    @param[in,out]
    Q           magma_z_matrix*
                input matrix Q
                
    @param[in,out]
    R           magma_z_matrix*
                input matrix R

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zposv
    ********************************************************************/


extern "C" magma_int_t
magma_zqr(
    magma_int_t m, magma_int_t n,
    magma_z_matrix A, magma_z_matrix *Q, magma_z_matrix *R,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_int_t lwork;
    magma_int_t k = MIN(m,n);
    magma_int_t lda = MAX(1,m);
    magma_int_t inc = 1;
    magmaDoubleComplex *tau = NULL, *work = NULL, work1;
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO;
    magmaDoubleComplex *A1=NULL;

    // allocate CPU resources
    CHECK( magma_zmalloc_pinned(&tau, k) );
    CHECK( magma_zmalloc_pinned(&A1, m * n) );
    magma_zcopy(m * n, A.dval, inc, A1, inc);

    // query optimal dimension for QR work array and allocate
    lwork = -1;
    work1 = MAGMA_Z_MAKE(0.0, 0.0);
    magma_zgeqrf(m, n, NULL, lda, NULL, &work1, lwork, &info);
    lwork = MAX(1, (magma_int_t)MAGMA_Z_REAL(work1));
    CHECK( magma_zmalloc_pinned(&work, lwork) );

    // Note: need to sync here, if not the Q returned is I
    cudaDeviceSynchronize();

    // QR factorization (xgeqrf)
    magma_zgeqrf( m, n, A1, lda, tau, work, lwork, &info );

    // construct R matrix
    if ( R != NULL ) {
        CHECK( magma_zvinit( R, Magma_DEV, k, n, c_zero, queue ) );
        magma_zcopy( m * n, A1, inc, R->dval, inc );
    }

    // construct Q matrix (xorgqr)
    if ( Q != NULL ) {
        magma_zungqr( m, k, k, A1, lda, tau, work, lwork, &info );
        CHECK( magma_zvinit( Q, Magma_DEV, m, k, c_zero, queue ) );
        magma_zcopy( m * k, A1, inc, Q->dval, inc );
    }

cleanup:
    if( info != 0 ){
        magma_zmfree( Q, queue );
        magma_zmfree( R, queue );
    }
    // free CPU resources
    magma_free_pinned(tau);
    magma_free_pinned(work);
    magma_free_pinned(A1);

    return info;
}

