/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Hartwig Anzt

*/
#include "common_magma.h"
#include "../include/magmasparse.h"

#include <assert.h>


#define RTOLERANCE     lapackf77_dlamch( "E" )
#define ATOLERANCE     lapackf77_dlamch( "E" )


/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

    Purpose
    =======

    Solves a system of linear equations
       A * X = B
    where A is a complex Hermitian N-by-N positive definite matrix A.
    This is a GPU implementation of the preconditioned 
    Biconjugate Gradient Stabelized method.

    Arguments
    =========

    magma_z_sparse_matrix A                   input matrix A
    magma_z_vector b                          RHS b
    magma_z_vector *x                         solution approximation
    magma_z_solver_par *solver_par            solver parameters
    magma_z_preconditioner *precond_par       preconditioner parameters

    ========================================================================  */


magma_int_t
magma_zpbicgstab( magma_z_sparse_matrix A, magma_z_vector b, magma_z_vector *x,  
                    magma_z_solver_par *solver_par, 
                    magma_z_preconditioner *precond_par ){

    // prepare solver feedback
    solver_par->solver = Magma_PBICGSTAB;
    solver_par->numiter = 0;
    solver_par->info = 0;

    // some useful variables
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO, c_one = MAGMA_Z_ONE, 
                                            c_mone = MAGMA_Z_NEG_ONE;
    
    magma_int_t dofs = A.num_rows;

    // workspace
    magma_z_vector r,rr,p,v,s,t,ms,mt,y,z;
    magma_z_vinit( &r, Magma_DEV, dofs, c_zero );
    magma_z_vinit( &rr, Magma_DEV, dofs, c_zero );
    magma_z_vinit( &p, Magma_DEV, dofs, c_zero );
    magma_z_vinit( &v, Magma_DEV, dofs, c_zero );
    magma_z_vinit( &s, Magma_DEV, dofs, c_zero );
    magma_z_vinit( &t, Magma_DEV, dofs, c_zero );
    magma_z_vinit( &ms, Magma_DEV, dofs, c_zero );
    magma_z_vinit( &mt, Magma_DEV, dofs, c_zero );
    magma_z_vinit( &y, Magma_DEV, dofs, c_zero );
    magma_z_vinit( &z, Magma_DEV, dofs, c_zero );

    
    // solver variables
    magmaDoubleComplex alpha, beta, omega, rho_old, rho_new;
    double nom, betanom, nom0, r0, den;
    magma_int_t i;


    // solver setup
    magma_zscal( dofs, c_zero, x->val, 1) ;                    // x = 0
    magma_zcopy( dofs, b.val, 1, r.val, 1 );                   // r = b
    magma_zcopy( dofs, b.val, 1, rr.val, 1 );                  // rr = b
    nom0 = magma_dznrm2( dofs, r.val, 1 );                      // nom = || r ||
    nom = nom0*nom0;
    rho_old = omega = alpha = MAGMA_Z_MAKE( 1.0, 0. );
    solver_par->init_res = nom0;

    magma_z_spmv( c_one, A, r, c_zero, v );                      // z = A r
    den = MAGMA_Z_REAL( magma_zdotc(dofs, v.val, 1, r.val, 1) ); // den = z' * r

    if ( (r0 = nom * solver_par->epsilon) < ATOLERANCE ) 
        r0 = ATOLERANCE;
    if ( nom < r0 )
        return MAGMA_SUCCESS;
    // check positive definite  
    if (den <= 0.0) {
        printf("Operator A is not postive definite. (Ar,r) = %f\n", den);
        return -100;
    }

    //Chronometry
    real_Double_t tempo1, tempo2;
    magma_device_sync(); tempo1=magma_wtime();
    if( solver_par->verbose > 0 ){
        solver_par->res_vec[0] = nom0;
        solver_par->timing[0] = 0.0;
    }

    // start iteration
    for( solver_par->numiter= 1; solver_par->numiter<solver_par->maxiter; 
                                                    solver_par->numiter++ ){

        rho_new = magma_zdotc( dofs, rr.val, 1, r.val, 1 );  // rho=<rr,r>
        beta = rho_new/rho_old * alpha/omega;   // beta=rho/rho_old *alpha/omega
        magma_zscal( dofs, beta, p.val, 1 );                 // p = beta*p
        magma_zaxpy( dofs, c_mone * omega * beta, v.val, 1 , p.val, 1 );        
                                                        // p = p-omega*beta*v
        magma_zaxpy( dofs, c_one, r.val, 1, p.val, 1 );      // p = p+r

        // preconditioner
        //magma_z_applyprecond( A, p, &y, precond_par ); 
        magma_z_applyprecond_left( A, p, &mt, precond_par );      
        magma_z_applyprecond_right( A, mt, &y, precond_par );        

        magma_z_spmv( c_one, A, p, c_zero, v );              // v = Ap

        alpha = rho_new / magma_zdotc( dofs, rr.val, 1, v.val, 1 );
        magma_zcopy( dofs, r.val, 1 , s.val, 1 );            // s=r
        magma_zaxpy( dofs, c_mone * alpha, v.val, 1 , s.val, 1 ); // s=s-alpha*v

        // preconditioner
        //magma_z_applyprecond( A, s, &z, precond_par );
        magma_z_applyprecond_left( A, s, &ms, precond_par ); 
        magma_z_applyprecond_right( A, ms, &z, precond_par );      

        magma_z_spmv( c_one, A, z, c_zero, t );               // t=As

        // preconditioner
     //   magma_z_applyprecond_left( A, s, &ms, precond_par );      
     //   magma_z_applyprecond_left( A, t, &mt, precond_par );        

        // omega = <ms,mt>/<mt,mt>  
        omega = magma_zdotc( dofs, t.val, 1, s.val, 1 )
                   / magma_zdotc( dofs, t.val, 1, t.val, 1 );

        magma_zaxpy( dofs, alpha, y.val, 1 , x->val, 1 );     // x=x+alpha*p
        magma_zaxpy( dofs, omega, z.val, 1 , x->val, 1 );     // x=x+omega*s

        magma_zcopy( dofs, s.val, 1 , r.val, 1 );             // r=s
        magma_zaxpy( dofs, c_mone * omega, t.val, 1 , r.val, 1 ); // r=r-omega*t
        betanom = magma_dznrm2( dofs, r.val, 1 );

        nom = betanom*betanom;
        rho_old = rho_new;                                    // rho_old=rho

        if( solver_par->verbose > 0 ){
            magma_device_sync(); tempo2=magma_wtime();
            if( (solver_par->numiter)%solver_par->verbose==0 ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose] 
                        = (real_Double_t) betanom;
                solver_par->timing[(solver_par->numiter)/solver_par->verbose] 
                        = (real_Double_t) tempo2-tempo1;
            }
        }

        if ( betanom  < r0 ) {
            break;
        }
    }
    magma_device_sync(); tempo2=magma_wtime();
    solver_par->runtime = (real_Double_t) tempo2-tempo1;
    double residual;
    magma_zresidual( A, b, *x, &residual );
    solver_par->final_res = (real_Double_t) betanom;//residual;

    if( solver_par->numiter < solver_par->maxiter){
        solver_par->info = 0;
    }else if( solver_par->init_res < solver_par->final_res )
        solver_par->info = -2;
    else
        solver_par->info = -1;

    magma_z_vfree(&r);
    magma_z_vfree(&rr);
    magma_z_vfree(&p);
    magma_z_vfree(&v);
    magma_z_vfree(&s);
    magma_z_vfree(&t);
    magma_z_vfree(&ms);
    magma_z_vfree(&mt);
    magma_z_vfree(&y);
    magma_z_vfree(&z);


    return MAGMA_SUCCESS;
}   /* magma_zbicgstab */


