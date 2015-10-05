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
    where A is a complex Hermitian N-by-N positive definite matrix A.
    This is a GPU implementation of the preconditioned transpose-free 
    Quasi-Minimal Residual method (TFQMR).

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

    @ingroup magmasparse_zposv
    ********************************************************************/

extern "C" magma_int_t
magma_zptfqmr(
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
    solver_par->solver = Magma_PTFQMR;
    solver_par->numiter = 0;
    solver_par->info = MAGMA_SUCCESS;
    
    // local variables
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO, c_one = MAGMA_Z_ONE;
    // solver variables
    double nom0, r0,  res, nomb;
    magmaDoubleComplex rho = c_one, rho_l = c_one, eta = c_zero , c = c_zero , 
                        theta = c_zero , tau = c_zero, alpha = c_one, beta = c_zero,
                        sigma = c_zero;
    
    magma_int_t dofs = A.num_rows* b.num_cols;

    // GPU workspace
    magma_z_matrix r={Magma_CSR}, r_tld={Magma_CSR}, pu_m={Magma_CSR},
                    d={Magma_CSR}, w={Magma_CSR}, v={Magma_CSR}, t={Magma_CSR},
                    u_mp1={Magma_CSR}, u_m={Magma_CSR}, Au={Magma_CSR}, 
                    Ad={Magma_CSR}, Au_new={Magma_CSR};
    CHECK( magma_zvinit( &r, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &t, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &u_mp1,Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &r_tld,Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &u_m, Magma_DEV, A.num_rows, b.num_cols, c_one, queue ));
    CHECK( magma_zvinit( &pu_m, Magma_DEV, A.num_rows, b.num_cols, c_one, queue ));
    CHECK( magma_zvinit( &v, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &d, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &w, Magma_DEV, A.num_rows, b.num_cols, c_one, queue ));
    CHECK( magma_zvinit( &Ad, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &Au_new, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &Au, Magma_DEV, A.num_rows, b.num_cols, c_one, queue ));
    
    // solver setup
    CHECK(  magma_zresidualvec( A, b, *x, &r, &nom0, queue));
    solver_par->init_res = nom0;
    magma_zcopy( dofs, r.dval, 1, r_tld.dval, 1 );   
    magma_zcopy( dofs, r.dval, 1, w.dval, 1 );   
    magma_zcopy( dofs, r.dval, 1, u_m.dval, 1 );  
    
    // preconditioner
    CHECK( magma_z_applyprecond_left( A, u_m, &t, precond_par, queue ));
    CHECK( magma_z_applyprecond_right( A, t, &pu_m, precond_par, queue ));
    
    CHECK( magma_z_spmv( c_one, A, pu_m, c_zero, v, queue ));   // v = A u
    magma_zcopy( dofs, v.dval, 1, Au.dval, 1 );  
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
    if ( nom0 < r0 ) {
        goto cleanup;
    }

    tau = magma_zsqrt( magma_zdotc(dofs, r.dval, 1, r_tld.dval, 1) );
    rho = magma_zdotc(dofs, r.dval, 1, r_tld.dval, 1);
    rho_l = rho;
    
    //Chronometry
    real_Double_t tempo1, tempo2;
    tempo1 = magma_sync_wtime( queue );
    
    solver_par->numiter = 0;
    // start iteration
    do
    {
        solver_par->numiter++;
        if( solver_par->numiter%2 == 1 ){
            alpha = rho / magma_zdotc(dofs, v.dval, 1, r_tld.dval, 1);
            magma_zcopy( dofs, u_m.dval, 1, u_mp1.dval, 1 );   
            magma_zaxpy(dofs,  -alpha, v.dval, 1, u_mp1.dval, 1);     // u_mp1 = u_m - alpha*v;
        }
        magma_zaxpy(dofs,  -alpha, Au.dval, 1, w.dval, 1);     // w = w - alpha*Au;
        sigma = theta * theta / alpha * eta;    
        magma_zscal(dofs, sigma, d.dval, 1);    
        magma_zaxpy(dofs, c_one, pu_m.dval, 1, d.dval, 1);     // d = pu_m + sigma*d;
        magma_zscal(dofs, sigma, Ad.dval, 1);         
        magma_zaxpy(dofs, c_one, Au.dval, 1, Ad.dval, 1);     // Ad = Au + sigma*Ad;

        
        theta = magma_zsqrt( magma_zdotc(dofs, w.dval, 1, w.dval, 1) ) / tau;
        c = c_one / magma_zsqrt( c_one + theta*theta );
        tau = tau * theta *c;
        eta = c * c * alpha;

        magma_zaxpy(dofs, eta, d.dval, 1, x->dval, 1);     // x = x + eta * d
        magma_zaxpy(dofs, -eta, Ad.dval, 1, r.dval, 1);     // r = r - eta * Ad
        res = magma_dznrm2( dofs, r.dval, 1 );
        
        if ( solver_par->verbose > 0 ) {
            tempo2 = magma_sync_wtime( queue );
            if ( (solver_par->numiter)%solver_par->verbose == 0 ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) res;
                solver_par->timing[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) tempo2-tempo1;
            }
        }

        if ( res/nomb <= solver_par->rtol || res <= solver_par->atol ){
            break;
        }
        if( solver_par->numiter%2 == 0 ){
            rho_l = rho;
            rho = magma_zdotc(dofs, w.dval, 1, r_tld.dval, 1);
            beta = rho / rho_l;
            magma_zcopy( dofs, w.dval, 1, u_mp1.dval, 1 );  
            magma_zaxpy(dofs, beta, u_m.dval, 1, u_mp1.dval, 1);     // u_mp1 = w + beta*u_m;
        }
              
        // preconditioner
        CHECK( magma_z_applyprecond_left( A, u_mp1, &t, precond_par, queue ));
        CHECK( magma_z_applyprecond_right( A, t, &pu_m, precond_par, queue ));
        
        CHECK( magma_z_spmv( c_one, A, pu_m, c_zero, Au_new, queue )); // Au_new = A pu_m
      
        if( solver_par->numiter%2 == 0 ){
            magma_zscal(dofs, beta*beta, v.dval, 1);                    
            magma_zaxpy(dofs, beta, Au.dval, 1, v.dval, 1);              
            magma_zaxpy(dofs, c_one, Au_new.dval, 1, v.dval, 1);      // v = Au_new + beta*(Au+beta*v);
        }
        magma_zcopy( dofs, Au_new.dval, 1, Au.dval, 1 );  
        magma_zcopy( dofs, u_mp1.dval, 1, u_m.dval, 1 );  
    }
    while ( solver_par->numiter+1 <= solver_par->maxiter );
    
    tempo2 = magma_sync_wtime( queue );
    solver_par->runtime = (real_Double_t) tempo2-tempo1;
    double residual;
    CHECK(  magma_zresidualvec( A, b, *x, &r, &residual, queue));
    solver_par->iter_res = res;
    solver_par->final_res = residual;

    if ( solver_par->numiter < solver_par->maxiter ) {
        info = MAGMA_SUCCESS;
    } else if ( solver_par->init_res > solver_par->final_res ) {
        if ( solver_par->verbose > 0 ) {
            if ( (solver_par->numiter)%solver_par->verbose == 0 ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) res;
                solver_par->timing[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) tempo2-tempo1;
            }
        }
        info = MAGMA_SLOW_CONVERGENCE;
        if( solver_par->iter_res < solver_par->rtol*solver_par->init_res ||
            solver_par->iter_res < solver_par->atol ) {
            info = MAGMA_SUCCESS;
        }
    }
    else {
        if ( solver_par->verbose > 0 ) {
            if ( (solver_par->numiter)%solver_par->verbose == 0 ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) res;
                solver_par->timing[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) tempo2-tempo1;
            }
        }
        info = MAGMA_DIVERGENCE;
    }
    
cleanup:
    magma_zmfree(&r, queue );
    magma_zmfree(&r_tld, queue );
    magma_zmfree(&d, queue );
    magma_zmfree(&w, queue );
    magma_zmfree(&v, queue );
    magma_zmfree(&pu_m, queue );
    magma_zmfree(&u_m, queue );
    magma_zmfree(&u_mp1, queue );
    magma_zmfree(&d, queue );
    magma_zmfree(&t, queue );
    magma_zmfree(&Au, queue );
    magma_zmfree(&Au_new, queue );
    magma_zmfree(&Ad, queue );
    
    magmablasSetKernelStream( orig_queue );
    solver_par->info = info;
    return info;
}   /* magma_zpfqmr */