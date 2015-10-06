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
    This is a GPU implementation of the Conjugate
    Gradient Squared (CGS) method.

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
magma_zcgs(
    magma_z_matrix A, magma_z_matrix b, magma_z_matrix *x,
    magma_z_solver_par *solver_par,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    // set queue for old dense routines
    magma_queue_t orig_queue=NULL;
    magmablasGetKernelStream( &orig_queue );

    // prepare solver feedback
    solver_par->solver = Magma_CGS;
    solver_par->numiter = 0;
    solver_par->info = MAGMA_SUCCESS;
    
    // local variables
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO, c_one = MAGMA_Z_ONE,  c_mone = MAGMA_Z_MAKE(-1.0, 0.0);
    // solver variables
    double nom0, r0,  res, nomb;
    magmaDoubleComplex rho, rho_l = c_one, alpha, beta;
    
    magma_int_t dofs = A.num_rows* b.num_cols;

    // GPU workspace
    magma_z_matrix r={Magma_CSR}, rt={Magma_CSR}, r_tld={Magma_CSR},
                    p={Magma_CSR}, q={Magma_CSR}, u={Magma_CSR}, v={Magma_CSR},  t={Magma_CSR},
                    p_hat={Magma_CSR}, q_hat={Magma_CSR}, u_hat={Magma_CSR}, v_hat={Magma_CSR};
    CHECK( magma_zvinit( &r, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &rt,Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &r_tld,Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &p, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &p_hat, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &q, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &q_hat, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &u, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &u_hat, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &v, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &v_hat, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &t, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));

    // solver setup
    CHECK(  magma_zresidualvec( A, b, *x, &r, &nom0, queue));
    magma_zcopy( dofs, r.dval, 1, r_tld.dval, 1 );   

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
    if ( nom0 < r0 ) {
        goto cleanup;
    }

    //Chronometry
    real_Double_t tempo1, tempo2;
    tempo1 = magma_sync_wtime( queue );
    
    solver_par->numiter = 0;
    // start iteration
    do
    {
        solver_par->numiter++;
        
        rho = magma_zdotc(dofs, r.dval, 1, r_tld.dval, 1);
                                                            // rho = < r,r_tld>    
        if ( MAGMA_Z_ABS(rho) == 0.0 ) {
            goto cleanup;
        }
        
        if ( solver_par->numiter > 1 ) {                        // direction vectors
            beta = rho / rho_l;            
            magma_zcopy( dofs, r.dval, 1, u.dval, 1 );          // u = r
            magma_zaxpy(dofs,  beta, q.dval, 1, u.dval, 1);     // u = r + beta q
            magma_zscal(dofs, beta, p.dval, 1);                 // p = beta*p
            magma_zaxpy(dofs, c_one, q.dval, 1, p.dval, 1);      // p = q + beta*p
            magma_zscal(dofs, beta, p.dval, 1);                 // p = beta*(q + beta*p)
            magma_zaxpy(dofs, c_one, u.dval, 1, p.dval, 1);     // p = u + beta*(q + beta*p)
        //u = r + beta*q;
        //p = u + beta*( q + beta*p );
        }
        else{
            magma_zcopy( dofs, r.dval, 1, u.dval, 1 );          // u = r
            magma_zcopy( dofs, r.dval, 1, p.dval, 1 );          // p = r
        }
        
        CHECK( magma_z_spmv( c_one, A, p, c_zero, v_hat, queue ));   // v = A p
        alpha = rho / magma_zdotc(dofs, r_tld.dval, 1, v_hat.dval, 1);
        magma_zcopy( dofs, u.dval, 1, q.dval, 1 );              // q = u
        magma_zaxpy(dofs,  -alpha, v_hat.dval, 1, q.dval, 1);   // q = u - alpha v_hat
        
        magma_zcopy( dofs, u.dval, 1, t.dval, 1 );             // t = q
        magma_zaxpy(dofs,  c_one, q.dval, 1, t.dval, 1);       // t = u + q


        CHECK( magma_z_spmv( c_one, A, t, c_zero, rt, queue ));   // t = A u_hat
        magma_zaxpy(dofs,  c_mone*alpha, rt.dval, 1, r.dval, 1);       // r = r -alpha*A u_hat
        magma_zaxpy(dofs,  alpha, t.dval, 1, x->dval, 1);      // x = x + alpha u_hat

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
    magma_zmfree(&p, queue );
    magma_zmfree(&q, queue );
    magma_zmfree(&u, queue );
    magma_zmfree(&v, queue );
    magma_zmfree(&t, queue );
    magma_zmfree(&p_hat, queue );
    magma_zmfree(&q_hat, queue );
    magma_zmfree(&u_hat, queue );
    magma_zmfree(&v_hat, queue );

    magmablasSetKernelStream( orig_queue );
    solver_par->info = info;
    return info;
}   /* magma_zcgs */
