/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Hartwig Anzt 

       @precisions normal z -> s d c
*/

#include "common_magma.h"
#include "magmasparse.h"

#include <assert.h>

#define RTOLERANCE     lapackf77_dlamch( "E" )
#define ATOLERANCE     lapackf77_dlamch( "E" )


/**
    Purpose
    -------

    Solves a system of linear equations
       A * X = B
    where A is a complex Hermitian N-by-N positive definite matrix A.
    This is a GPU implementation of the Jacobi method allowing for 
    domain overlap.

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                input matrix A

    @param[in]
    b           magma_z_matrix
                RHS b

    @param[in,out]
    x           magma_z_matrix*
                solution approximation

    @param[in,out]
    solver_par  magma_z_solver_par*
                solver parameters

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgesv
    ********************************************************************/

extern "C" magma_int_t
magma_zjacobidomainoverlap(
    magma_z_matrix A, 
    magma_z_matrix b, 
    magma_z_matrix *x,  
    magma_z_solver_par *solver_par,
    magma_queue_t queue )
{
    // set queue for old dense routines
    magma_queue_t orig_queue;
    magmablasGetKernelStream( &orig_queue );

    // prepare solver feedback
    solver_par->solver = Magma_JACOBI;
    solver_par->info = MAGMA_SUCCESS;

    real_Double_t tempo1, tempo2;
    double residual;
    magma_zresidual( A, b, *x, &residual, queue );
    solver_par->init_res = residual;
    solver_par->res_vec = NULL;
    solver_par->timing = NULL;

    // some useful variables
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO, c_one = MAGMA_Z_ONE, 
                                                c_mone = MAGMA_Z_NEG_ONE;
    magma_int_t dofs = A.num_rows*b.num_cols;
    double nom0 = 0.0;

    magma_z_matrix r, d;
    magma_zvinit( &r, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue );
    magma_z_spmv( c_one, A, *x, c_zero, r, queue );                  // r = A x
    magma_zaxpy(dofs,  c_mone, b.dval, 1, r.dval, 1);           // r = r - b
    nom0 = magma_dznrm2(dofs, r.dval, 1);                      // den = || r ||

    // Jacobi setup
    magma_zjacobisetup_diagscal( A, &d, queue );
    magma_z_solver_par jacobiiter_par;
    jacobiiter_par.maxiter = solver_par->maxiter;
    
    // generate the domain overlap
    magma_int_t stat_cpu=0, stat_dev=0,num_ind = 0;
    magma_index_t *indices, *hindices;
    stat_cpu += magma_index_malloc_cpu( &hindices, A.num_rows*10 );
        if( stat_cpu != 0 ){
        magma_free_cpu( indices );
        return MAGMA_ERR_HOST_ALLOC;
    } 
    stat_dev += magma_index_malloc( &indices, A.num_rows*10 );
        if( stat_cpu != 0 ){
        magma_free_cpu( hindices );
        return MAGMA_ERR_HOST_ALLOC;
    } 
    magma_z_matrix hA;
    magma_zmtransfer( A, &hA, Magma_DEV, Magma_CPU, queue );
    magma_zdomainoverlap( hA.num_rows, &num_ind, hA.row, hA.col, hindices, queue );
    magma_zmfree(&hA, queue );
    stat_dev += magma_index_malloc( &indices, num_ind );
        if( stat_dev != 0 ){
        magma_free( indices );
        return MAGMA_ERR_DEVICE_ALLOC;
    } 
    magma_index_setvector( num_ind, hindices, 1, indices, 1 );
    magma_free_cpu( hindices );
    
    
    
    tempo1 = magma_sync_wtime( queue );

    // Jacobi iterator    
    magma_zjacobispmvupdateselect(jacobiiter_par.maxiter, num_ind, indices, 
                                                        A, r, b, d, x, queue );
    
    //magma_zjacobispmvupdate(jacobiiter_par.maxiter, A, r, b, d, x, queue );

    tempo2 = magma_sync_wtime( queue );
    solver_par->runtime = (real_Double_t) tempo2-tempo1;
    magma_zresidual( A, b, *x, &residual, queue );
    solver_par->final_res = residual;
    solver_par->numiter = solver_par->maxiter;

    if ( solver_par->init_res > solver_par->final_res )
        solver_par->info = MAGMA_SUCCESS;
    else
        solver_par->info = MAGMA_DIVERGENCE;
    
    magma_zmfree( &r, queue );
    magma_zmfree( &d, queue );

    // domain overlap
    magma_free( indices );
    
    magmablasSetKernelStream( orig_queue );
    return MAGMA_SUCCESS;
}   /* magma_zjacobi */



