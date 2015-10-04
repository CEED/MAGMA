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
    This is a GPU implementation of the transpose-free Quas-Minimal Residual 
    method (TFQMR).

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

    @ingroup magmasparse_zposv
    ********************************************************************/

extern "C" magma_int_t
magma_ztfqmr(
    magma_z_matrix A, magma_z_matrix b, magma_z_matrix *x,
    magma_z_solver_par *solver_par,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    // set queue for old dense routines
    magma_queue_t orig_queue=NULL;
    magmablasGetKernelStream( &orig_queue );

    // prepare solver feedback
    solver_par->solver = Magma_TFQMR;
    solver_par->numiter = 0;
    solver_par->info = MAGMA_SUCCESS;
    
    // local variables
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO, c_one = MAGMA_Z_ONE;
    // solver variables
    double nom0, r0,  res, nomb;
    magmaDoubleComplex rho = c_one, rho_l = c_one, nu = c_zero , c = c_zero , 
                        theta = c_zero , tau = c_zero, alpha = c_zero, beta = c_zero;
    
    magma_int_t dofs = A.num_rows* b.num_cols;

    // GPU workspace
    magma_z_matrix r={Magma_CSR}, rt={Magma_CSR}, r_tld={Magma_CSR}, t={Magma_CSR},
                    d={Magma_CSR}, w={Magma_CSR}, u={Magma_CSR}, v={Magma_CSR};
    CHECK( magma_zvinit( &r, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &rt,Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &r_tld,Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &d, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &u, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &v, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &w, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &t, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));

    // solver setup
    CHECK(  magma_zresidualvec( A, b, *x, &r, &nom0, queue));
    solver_par->init_res = nom0;
    magma_zcopy( dofs, r.dval, 1, r_tld.dval, 1 );   
    magma_zcopy( dofs, r.dval, 1, w.dval, 1 );   
    magma_zcopy( dofs, r.dval, 1, u.dval, 1 );   
    CHECK( magma_z_spmv( c_one, A, u, c_zero, v, queue ));   // v = A u
            
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
    rho = magma_zsqrt( magma_zdotc(dofs, r.dval, 1, r_tld.dval, 1) );
            printf("rho_0 = %.8e\n", rho);
    
    //Chronometry
    real_Double_t tempo1, tempo2;
    tempo1 = magma_sync_wtime( queue );
    
    solver_par->numiter = 0;
    // start iteration
    do
    {
        solver_par->numiter++;
                    printf("\n\n\n\niteration %d\n", solver_par->numiter);
        if( solver_par->numiter%2 == 0 ){
            alpha = rho / magma_zdotc(dofs, v.dval, 1, r_tld.dval, 1);
            printf("alpha = %.8e\n", alpha);
            magma_zaxpy(dofs,  -alpha, v.dval, 1, u.dval, 1);     // u = u - alpha v
        }
        
        CHECK( magma_z_spmv( c_one, A, u, c_zero, t, queue ));   // t = A u
        magma_zaxpy(dofs,  -alpha, t.dval, 1, w.dval, 1);     // w = w - alpha Au

        magma_zscal(dofs, theta * theta / alpha * nu, d.dval, 1);                 // p = beta*p
        magma_zaxpy(dofs, c_one, u.dval, 1, d.dval, 1);     // d = u + theta * theta / alpha * nu * d
        
        theta = magma_zsqrt( magma_zdotc(dofs, r.dval, 1, r_tld.dval, 1) ) / tau;
        c = c_one / magma_zsqrt( c_one + theta*theta );
        tau = tau * theta *c;
        nu = c * c * alpha;
printf("theta = %.8e\n",theta);
printf("c = %.8e\n",c);
printf("tau = %.8e\n",tau);
printf("nu = %.8e\n",nu);

        CHECK( magma_z_spmv( c_one, A, d, c_zero, rt, queue ));   // t = A d

        magma_zaxpy(dofs, nu, d.dval, 1, x->dval, 1);     // x = x + nu * d
        magma_zaxpy(dofs, -nu, rt.dval, 1, r.dval, 1);     // r = r - nu * Ad

        if( solver_par->numiter%2 == 1 ){
            rho_l = rho;
            rho = magma_zdotc(dofs, w.dval, 1, r_tld.dval, 1);
            beta = rho / rho_l;
printf("rho = %.8e\n",rho);
printf("beta = %.8e\n",beta);
            magma_zscal(dofs, beta, u.dval, 1);                 // u = beta*u
            magma_zaxpy(dofs, c_one, w.dval, 1, u.dval, 1);     // u = w + beta * u
            CHECK( magma_z_spmv( c_one, A, u, c_zero, rt, queue ));   // rt = A u
            magma_zscal(dofs, beta*beta, u.dval, 1);                    // v = beta*beta*v
            magma_zaxpy(dofs, beta, t.dval, 1, v.dval, 1);              // v = beta*beta*v + beta * t
            magma_zaxpy(dofs, c_one, rt.dval, 1, v.dval, 1);            // u = beta*beta*v + beta * t + rt
        }
               
        res = magma_dznrm2( dofs, r.dval, 1 );
printf("res = %.8e\n",res);
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
        rho_l = rho;
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
    magma_zmfree(&rt, queue );
    magma_zmfree(&r_tld, queue );
    magma_zmfree(&d, queue );
    magma_zmfree(&w, queue );
    magma_zmfree(&u, queue );
    magma_zmfree(&v, queue );
    magma_zmfree(&t, queue );
    
    magmablasSetKernelStream( orig_queue );
    solver_par->info = info;
    return info;
}   /* magma_zfqmr */