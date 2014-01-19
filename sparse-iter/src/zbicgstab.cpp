/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> s d c
       @author Hartwig Anzt

*/
#include "common_magma.h"
#include "../include/magmasparse.h"

#include <assert.h>


#define RTOLERANCE     1e-16
#define ATOLERANCE     1e-16

// uncomment for chronometry
#define ENABLE_TIMER
#define iterblock 1

/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======

    Solves a system of linear equations
       A * X = B
    where A is a complex Hermitian N-by-N positive definite matrix A.
    This is a GPU implementation of the Biconjugate Gradient Stabelized method.

    Arguments
    =========

    magma_z_sparse_matrix A                   input matrix A
    magma_z_vector b                          RHS b
    magma_z_vector *x                         solution approximation
    magma_solver_parameters *solver_par       solver parameters

    ========================================================================  */


magma_int_t
magma_zbicgstab( magma_z_sparse_matrix A, magma_z_vector b, magma_z_vector *x,  
                    magma_solver_parameters *solver_par )
{

    // some useful variables
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO, c_one = MAGMA_Z_ONE, 
                                            c_mone = MAGMA_Z_NEG_ONE;
    
    magma_int_t dofs = A.num_rows;

    // workspace
    magma_z_vector r,rr,p,v,s,t;
    magma_z_vinit( &r, Magma_DEV, dofs, c_zero );
    magma_z_vinit( &rr, Magma_DEV, dofs, c_zero );
    magma_z_vinit( &p, Magma_DEV, dofs, c_zero );
    magma_z_vinit( &v, Magma_DEV, dofs, c_zero );
    magma_z_vinit( &s, Magma_DEV, dofs, c_zero );
    magma_z_vinit( &t, Magma_DEV, dofs, c_zero );

    
    // solver variables
    magmaDoubleComplex alpha, beta, omega, rho_old, rho_new;
    double nom, betanom, nom0, r0, den;
    magma_int_t i;


    // solver setup
    magma_zscal( dofs, c_zero, x->val, 1) ;                    // x = 0
    magma_zcopy( dofs, b.val, 1, r.val, 1 );                   // r = b
    magma_zcopy( dofs, b.val, 1, rr.val, 1 );                  // rr = b
    nom = magma_dznrm2( dofs, r.val, 1 );                      // nom = || r ||
    nom0 = nom = nom*nom;
    rho_old = omega = alpha = MAGMA_Z_MAKE( 1.0, 0. );
    (solver_par->numiter) = 0;

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
    #ifdef ENABLE_TIMER
    double tempo1, tempo2;
    magma_device_sync(); tempo1=magma_wtime();
    printf("#=============================================================#\n");
    printf("# BiCGStab performance analysis every %d iteration\n", iterblock);
    printf("#   iter   ||   residual-nrm2    ||   runtime\n");
    printf("#=============================================================#\n");
    printf("      0    ||    %e    ||    0.0000      \n", nom);
    magma_device_sync(); tempo1=magma_wtime();
    #endif

    // start iteration
    for( solver_par->numiter= 1; solver_par->numiter<solver_par->maxiter; 
                                                    solver_par->numiter++ ){

        rho_new = magma_zdotc( dofs, rr.val, 1, r.val, 1 );  // rho=<rr,r>
        beta = rho_new/rho_old * alpha/omega;   // beta=rho/rho_old *alpha/omega
        magma_zscal( dofs, beta, p.val, 1 );                 // p = beta*p
        magma_zaxpy( dofs, c_mone * omega * beta, v.val, 1 , p.val, 1 );        
                                                        // p = p-omega*beta*v
        magma_zaxpy( dofs, c_one, r.val, 1, p.val, 1 );      // p = p+r
        magma_z_spmv( c_one, A, p, c_zero, v );              // v = Ap

        alpha = rho_new / magma_zdotc( dofs, rr.val, 1, v.val, 1 );
        magma_zcopy( dofs, r.val, 1 , s.val, 1 );            // s=r
        magma_zaxpy( dofs, c_mone * alpha, v.val, 1 , s.val, 1 ); // s=s-alpha*v

        magma_z_spmv( c_one, A, s, c_zero, t );               // t=As
        omega = magma_zdotc( dofs, t.val, 1, s.val, 1 )   // omega = <s,t>/<t,t>
                   / magma_zdotc( dofs, t.val, 1, t.val, 1 );

        magma_zaxpy( dofs, alpha, p.val, 1 , x->val, 1 );     // x=x+alpha*p
        magma_zaxpy( dofs, omega, s.val, 1 , x->val, 1 );     // x=x+omega*s

        magma_zcopy( dofs, s.val, 1 , r.val, 1 );             // r=s
        magma_zaxpy( dofs, c_mone * omega, t.val, 1 , r.val, 1 ); // r=r-omega*t
        betanom = magma_dznrm2( dofs, r.val, 1 );

        nom = betanom*betanom;
        rho_old = rho_new;                                    // rho_old=rho

        #ifdef ENABLE_TIMER
        //Chronometry  
        magma_device_sync(); tempo2=magma_wtime();
        if( solver_par->numiter%iterblock==0 ) {
            printf("   %4d    ||    %e    ||    %.4lf  \n", 
                (solver_par->numiter), betanom, tempo2-tempo1 );
        }
        #endif

        if ( betanom  < r0 ) {
            break;
        }
    }
    #ifdef ENABLE_TIMER
    double residual;
    magma_zresidual( A, b, *x, &residual );
    printf("#=============================================================#\n");
    printf("# BiCGStab solver summary:\n");
    printf("#    initial residual: %e\n", nom0 );
    printf("#    iterations: %4d\n#    iterative residual: %e\n",
            (solver_par->numiter), betanom );
    printf("#    exact relative residual: %e\n#    runtime: %.4lf sec\n", 
                residual, tempo2-tempo1);
    printf("#=============================================================#\n");
    #endif
        
    solver_par->residual = (double)(betanom);

    magma_z_vfree(&r);
    magma_z_vfree(&rr);
    magma_z_vfree(&p);
    magma_z_vfree(&v);
    magma_z_vfree(&s);
    magma_z_vfree(&t);

    return MAGMA_SUCCESS;
}   /* magma_zbicgstab */


