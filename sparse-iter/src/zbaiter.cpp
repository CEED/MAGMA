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
       A * x = b
    via the block asynchronous iteration method on GPU.

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
magma_zbaiter(
    magma_z_matrix A, 
    magma_z_matrix b, 
    magma_z_matrix *x,  
    magma_z_solver_par *solver_par,
    magma_queue_t queue )
{
    // prepare solver feedback
    solver_par->solver = Magma_BAITER;
    solver_par->info = MAGMA_SUCCESS;



    magma_z_matrix Ah, ACSR, A_d, D, R, D_d, R_d;

    magma_zmtransfer( A, &Ah, A.memory_location, Magma_CPU, queue );
    magma_zmconvert( Ah, &ACSR, Ah.storage_type, Magma_CSR, queue );

    magma_zmtransfer( ACSR, &A_d, Magma_CPU, Magma_DEV, queue );

    // initial residual
    real_Double_t tempo1, tempo2;
    double residual;
    magma_zresidual( A_d, b, *x, &residual, queue );
    solver_par->init_res = residual;
    solver_par->res_vec = NULL;
    solver_par->timing = NULL;


    // setup
    magma_zcsrsplit( 256, ACSR, &D, &R, queue );
    magma_zmtransfer( D, &D_d, Magma_CPU, Magma_DEV, queue );
    magma_zmtransfer( R, &R_d, Magma_CPU, Magma_DEV, queue );

    magma_int_t localiter = 1;

    tempo1 = magma_sync_wtime( queue );

    // block-asynchronous iteration iterator
    for( int iter=0; iter<solver_par->maxiter; iter++)
        magma_zbajac_csr( localiter, D_d, R_d, b, x, queue );

    tempo2 = magma_sync_wtime( queue );
    solver_par->runtime = (real_Double_t) tempo2-tempo1;
    magma_zresidual( A_d, b, *x, &residual, queue );
    solver_par->final_res = residual;
    solver_par->numiter = solver_par->maxiter;

    if ( solver_par->init_res > solver_par->final_res )
        solver_par->info = MAGMA_SUCCESS;
    else
        solver_par->info = MAGMA_DIVERGENCE;

    magma_zmfree(&D, queue );
    magma_zmfree(&R, queue );
    magma_zmfree(&D_d, queue );
    magma_zmfree(&R_d, queue );
    magma_zmfree(&A_d, queue );
    magma_zmfree(&ACSR, queue );
    magma_zmfree(&Ah, queue );

    return MAGMA_SUCCESS;
}   /* magma_zbaiter */

