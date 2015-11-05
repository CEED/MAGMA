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
       A * X = B
    where A is a complex matrix A.
    This is a GPU implementation of the Conjugate Gradient method in variant,
    where multiple operations are merged into one compute kernel.

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
                preconditioner
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgesv
    ********************************************************************/

extern "C" magma_int_t
magma_zpcg_merge(
    magma_z_matrix A, magma_z_matrix b, magma_z_matrix *x,
    magma_z_solver_par *solver_par,
    magma_z_preconditioner *precond_par,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    // set queue for old dense routines
    magma_queue_t orig_queue=NULL;
    magmablasGetKernelStream( &orig_queue );

    // prepare solver feedback
    solver_par->solver = Magma_PCGMERGE;
    solver_par->numiter = 0;
    solver_par->info = MAGMA_SUCCESS;
    
    // solver variables
    magmaDoubleComplex alpha, beta, gamma, rho, tmp1, *skp_h={0};
    double nom, nom0, r0,  res, nomb;
    magmaDoubleComplex den;

    // some useful variables
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO, c_one = MAGMA_Z_ONE;
    magma_int_t dofs = A.num_rows*b.num_cols;

    magma_z_matrix r={Magma_CSR}, d={Magma_CSR}, z={Magma_CSR}, h={Magma_CSR},
                    rt={Magma_CSR};
    magmaDoubleComplex *d1=NULL, *d2=NULL, *skp=NULL;

    // GPU stream
    magma_queue_t stream[2]={0};
    magma_event_t event[1]={0};
    magma_queue_create( &stream[0] );
    magma_queue_create( &stream[1] );
    magma_event_create( &event[0] );

    // GPU workspace
    CHECK( magma_zvinit( &r, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &d, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &z, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &rt, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &h, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    
    CHECK( magma_zmalloc( &d1, dofs*(1) ));
    CHECK( magma_zmalloc( &d2, dofs*(1) ));
    // array for the parameters
    CHECK( magma_zmalloc( &skp, 6 ));
    // skp = [alpha|beta|gamma|rho|tmp1|tmp2]

    // solver setup
    CHECK(  magma_zresidualvec( A, b, *x, &r, &nom0, queue));
    
    // preconditioner
    CHECK( magma_z_applyprecond_left( A, r, &rt, precond_par, queue ));
    CHECK( magma_z_applyprecond_right( A, rt, &h, precond_par, queue ));
    
    magma_zcopy( dofs, h.dval, 1, d.dval, 1 );  
    nom = MAGMA_Z_ABS( magma_zdotc(dofs, r.dval, 1, h.dval, 1) );
    CHECK( magma_z_spmv( c_one, A, d, c_zero, z, queue ));              // z = A d
    den = magma_zdotc(dofs, d.dval, 1, z.dval, 1); // den = d'* z
    solver_par->init_res = nom0;
    
    nomb = magma_dznrm2( dofs, b.dval, 1 );
    if ( nomb == 0.0 ){
        nomb=1.0;
    }       
    if ( (r0 = nomb * solver_par->rtol) < ATOLERANCE ){
        r0 = ATOLERANCE;
    }
    solver_par->final_res = solver_par->init_res;
    solver_par->iter_res = solver_par->init_res;
    if ( solver_par->verbose > 0 ) {
        solver_par->res_vec[0] = (real_Double_t)nom0;
        solver_par->timing[0] = 0.0;
    }
    if ( nom < r0 ) {
        goto cleanup;
    }
    // check positive definite
    if ( MAGMA_Z_ABS(den) <= 0.0 ) {
        info = MAGMA_NONSPD;
        goto cleanup;
    }    
    
    // array on host for the parameters
    CHECK( magma_zmalloc_cpu( &skp_h, 6 ));
    
    alpha = rho = gamma = tmp1 = c_one;
    beta =  magma_zdotc(dofs, r.dval, 1, r.dval, 1);
    skp_h[0]=alpha;
    skp_h[1]=beta;
    skp_h[2]=gamma;
    skp_h[3]=rho;
    skp_h[4]=tmp1;
    skp_h[5]=MAGMA_Z_MAKE(nom, 0.0);

    magma_zsetvector( 6, skp_h, 1, skp, 1 );

    //Chronometry
    real_Double_t tempo1, tempo2;
    tempo1 = magma_sync_wtime( queue );

    
    solver_par->numiter = 0;
    // start iteration
    do
    {
        solver_par->numiter++;

        magmablasSetKernelStream(stream[0]);
        
        // computes SpMV and dot product
        CHECK( magma_zcgmerge_spmv1(  A, d1, d2, d.dval, z.dval, skp, queue ));
            
        // updates x, r
        CHECK( magma_zpcgmerge_xrbeta1( dofs, x->dval, r.dval, d.dval, z.dval, skp, queue ));
        
        // preconditioner in between
        CHECK( magma_z_applyprecond_left( A, r, &rt, precond_par, queue ));
        CHECK( magma_z_applyprecond_right( A, rt, &h, precond_par, queue ));
        
        // computes scalars and updates d
        CHECK( magma_zpcgmerge_xrbeta2( dofs, d1, d2, h.dval, r.dval, d.dval, skp, queue ));
if( solver_par->numiter==1){
       magma_zcopy( dofs, h.dval, 1, d.dval, 1 );   
    }
        // check stopping criterion (asynchronous copy)
        magma_zgetvector_async( 1 , skp+1, 1,
                                                    skp_h+1, 1, stream[1] );
        res = sqrt(MAGMA_Z_ABS(skp_h[1]));

        if ( solver_par->verbose > 0 ) {
            tempo2 = magma_sync_wtime( queue );
            if ( (solver_par->numiter)%solver_par->verbose==0 ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) res;
                solver_par->timing[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) tempo2-tempo1;
            }
        }

        if (  res < solver_par->atol || 
              res/nomb < solver_par->rtol ) {
            break;
        }
    }
    while ( solver_par->numiter+1 <= solver_par->maxiter );
    
    tempo2 = magma_sync_wtime( queue );
    solver_par->runtime = (real_Double_t) tempo2-tempo1;
    double residual;
    CHECK(  magma_zresidualvec( A, b, *x, &r, &residual, NULL));
    solver_par->iter_res = res;
    solver_par->final_res = residual;

    if ( solver_par->numiter < solver_par->maxiter ) {
        solver_par->info = MAGMA_SUCCESS;
    } else if ( solver_par->init_res > solver_par->final_res ) {
        if ( solver_par->verbose > 0 ) {
            if ( (solver_par->numiter)%solver_par->verbose==0 ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) res;
                solver_par->timing[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) tempo2-tempo1;
            }
        }
        info = MAGMA_SLOW_CONVERGENCE;
        if( solver_par->iter_res < solver_par->atol ||
            solver_par->iter_res/solver_par->init_res < solver_par->rtol ){
            info = MAGMA_SUCCESS;
        }
    }
    else {
        if ( solver_par->verbose > 0 ) {
            if ( (solver_par->numiter)%solver_par->verbose==0 ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) res;
                solver_par->timing[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) tempo2-tempo1;
            }
        }
        solver_par->info = MAGMA_DIVERGENCE;
    }
    
cleanup:
    magma_zmfree(&r, queue );
    magma_zmfree(&z, queue );
    magma_zmfree(&d, queue );
    magma_zmfree(&rt, queue );
    magma_zmfree(&h, queue );

    magma_free( d1 );
    magma_free( d2 );
    magma_free( skp );
    magma_free_cpu( skp_h );

    magmablasSetKernelStream( orig_queue );
    solver_par->info = info;
    return info;
}   /* magma_zpcg_merge */
