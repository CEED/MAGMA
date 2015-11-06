/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Hartwig Anzt

       @precisions normal z -> s d c
*/

#include "common_magmasparse.h"

#define RTOLERANCE     lapackf77_dlamch( "E" )
#define ATOLERANCE     lapackf77_dlamch( "E" )


/**
    Purpose
    -------

    Solves a system of linear equations
       A * x = b
    via the block asynchronous iteration method on GPU.
    It used restricted additive Schwarz overlap in top-down direction.

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
    precond_par magma_z_preconditioner*
                preconditioner parameters

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgesv
    ********************************************************************/

extern "C" magma_int_t
magma_zbaiter_overlap(
    magma_z_matrix A,
    magma_z_matrix b,
    magma_z_matrix *x,
    magma_z_solver_par *solver_par,
    magma_z_preconditioner *precond_par,
    magma_queue_t queue )
{
    magma_int_t info = 0;
        
    // prepare solver feedback
    solver_par->solver = Magma_BAITER;
    solver_par->info = MAGMA_SUCCESS;
    
    // some useful variables 
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO;

    // initial residual
    real_Double_t tempo1, tempo2, runtime=0;
    double residual;
    magma_int_t localiter = precond_par->maxiter;
    
    magma_z_matrix Ah={Magma_CSR}, ACSR={Magma_CSR}, A_d={Magma_CSR}, r={Magma_CSR},
        D={Magma_CSR}, R={Magma_CSR};
        
    // setup
    magma_int_t matrices;
    magma_int_t overlap = precond_par->levels;
    magma_int_t blocksize = 256;
    if( overlap ){
        matrices = 100/overlap;
    }else{
        printf("error: overlap ratio not supported.\n");
        goto cleanup;
    }
    magma_z_matrix D_d[ matrices ];
    magma_z_matrix R_d[ matrices ];
    

    CHECK( magma_zmtransfer( A, &Ah, A.memory_location, Magma_CPU, queue ));
    CHECK( magma_zmconvert( Ah, &ACSR, Ah.storage_type, Magma_CSR, queue ));

    CHECK( magma_zmtransfer( ACSR, &A_d, Magma_CPU, Magma_DEV, queue ));
    
    CHECK( magma_zvinit( &r, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK(  magma_zresidualvec( A_d, b, *x, &r, &residual, queue));
    solver_par->init_res = residual;
    if ( solver_par->verbose > 0 ) {
        solver_par->res_vec[0] = (real_Double_t) residual;
    }
    

    
   // setup  
   for( int i=0; i<matrices; i++ ){
       CHECK( magma_zcsrsplit( overlap, 256, ACSR, &D1, &R1, queue ));
       
   }
    CHECK( magma_zmtransfer( D1, &D1_d, Magma_CPU, Magma_DEV, queue ));
    CHECK( magma_zmtransfer( R1, &R1_d, Magma_CPU, Magma_DEV, queue ));
    CHECK( magma_zcsrsplit( 128, 256, ACSR, &D2, &R2, queue ));
    CHECK( magma_zmtransfer( D2, &D2_d, Magma_CPU, Magma_DEV, queue ));
    CHECK( magma_zmtransfer( R2, &R2_d, Magma_CPU, Magma_DEV, queue ));
    
    solver_par->numiter = 0;
    printf("check before solver launch\n");
    // block-asynchronous iteration iterator
    do
    {
        tempo1 = magma_sync_wtime( queue );
        solver_par->numiter+= solver_par->verbose;
        for( int z=0; z<solver_par->verbose; z++){
            CHECK( magma_zbajac_csr_overlap( localiter, D1_d, R1_d, D2_d, R2_d, b, x, queue ));
        }
        tempo2 = magma_sync_wtime( queue );
        runtime += tempo2-tempo1;
        if ( solver_par->verbose > 0 ) {
        CHECK(  magma_zresidualvec( A_d, b, *x, &r, &residual, queue));
            solver_par->res_vec[(solver_par->numiter)/solver_par->verbose]
                = (real_Double_t) residual;
            solver_par->timing[(solver_par->numiter)/solver_par->verbose]
                = (real_Double_t) runtime;
        }
    }
    while ( solver_par->numiter+1 <= solver_par->maxiter );

    solver_par->runtime = runtime;
    CHECK(  magma_zresidual( A_d, b, *x, &residual, queue));
    solver_par->final_res = residual;
    solver_par->numiter = solver_par->maxiter;

    if ( solver_par->init_res > solver_par->final_res ){
        info = MAGMA_SUCCESS;
    }
    else {
        info = MAGMA_DIVERGENCE;
    }
    
cleanup:
    magma_zmfree(&r, queue );
    magma_zmfree(&D1, queue );
    magma_zmfree(&R1, queue );
    magma_zmfree(&D1_d, queue );
    magma_zmfree(&R1_d, queue );
    magma_zmfree(&D2, queue );
    magma_zmfree(&R2, queue );
    magma_zmfree(&D2_d, queue );
    magma_zmfree(&R2_d, queue );
    magma_zmfree(&A_d, queue );
    magma_zmfree(&ACSR, queue );
    magma_zmfree(&Ah, queue );

    solver_par->info = info;
    return info;
}   /* magma_zbaiter */
