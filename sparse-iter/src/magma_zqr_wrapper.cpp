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
    Output matrices Q and R reside on the same memory location as matrix A. 


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

    // local constants
    const magmaDoubleComplex c_zero = MAGMA_Z_ZERO;

    // local variables
    magma_int_t inc = 1;
    magma_int_t ldda = MAX(1,m);    // multiple of 16 for coalesced accesses 
    magma_int_t k = MIN(m,n);
    magma_int_t ldt;
    magma_int_t nb;
    magmaDoubleComplex *tau = NULL;
    magmaDoubleComplex *dT = NULL;
    magmaDoubleComplex *dA = NULL;
    magma_z_matrix dR1 = {Magma_CSR};

    // allocate CPU resources
    CHECK( magma_zmalloc_pinned( &tau, k ) );

    // query number of blocks required for QR factorization
    nb = magma_get_zgeqrf_nb(m);
    ldt = (2 * k + magma_roundup(n, 32) * 32) * nb;
    CHECK( magma_zmalloc( &dT, ldt ) );

    // get copy of matrix array
    if ( A.memory_location == Magma_DEV ) {
        dA = A.dval;
    } else {
        CHECK( magma_zmalloc( &dA, ldda * n ) );
        magma_zsetvector( ldda * n, A.val, inc, dA, inc );
    }

    // QR factorization
    magma_zgeqrf_gpu( ldda, n, dA, ldda, tau, dT, &info );  

    // construct R matrix
    if ( R != NULL ) {
        if ( A.memory_location == Magma_DEV ) {
            CHECK( magma_zvinit( R, Magma_DEV, ldda, n, c_zero, queue ) );
            magmablas_zlacpy( MagmaUpper, k, n, dA, ldda, R->dval, ldda );
        } else {
            CHECK( magma_zvinit( &dR1, Magma_DEV, ldda, n, c_zero, queue ) );
            magmablas_zlacpy( MagmaUpper, k, n, dA, ldda, dR1.dval, ldda );
            CHECK( magma_zvinit( R, Magma_CPU, ldda, n, c_zero, queue ) );
            magma_zgetvector( ldda * n, dR1.dval, inc, R->val, inc );
        }
    }

    // construct Q matrix
    if ( Q != NULL ) {
        magma_zungqr_gpu( ldda, n, k, dA, ldda, tau, dT, nb, &info ); 

        if ( A.memory_location == Magma_DEV ) {
            CHECK( magma_zvinit( Q, Magma_DEV, ldda, n, c_zero, queue ) );
            magma_zcopy( ldda * n, dA, inc, Q->dval, inc );
        } else {
            CHECK( magma_zvinit( Q, Magma_CPU, ldda, n, c_zero, queue ) );
            magma_zgetvector( ldda * n, dA, inc, Q->val, inc );
        }
    }

cleanup:
    if( info != 0 ){
        magma_zmfree( Q, queue );
        magma_zmfree( R, queue );
        magma_zmfree( &dR1, queue );
    }

    // free resources
    magma_free_pinned( tau );
    magma_free( dT );
    if ( A.memory_location == Magma_CPU ) {
        magma_free( dA );
    }

    return info;
}

