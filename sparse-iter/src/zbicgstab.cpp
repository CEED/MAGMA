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

#define RTOLERANCE     10e-16
#define ATOLERANCE     10e-16


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

    =====================================================================  */


magma_int_t
magma_zbicgstab( magma_z_sparse_matrix A, magma_z_vector b, magma_z_vector *x,  
           magma_solver_parameters *solver_par )
{

    // some useful variables
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO, c_one = MAGMA_Z_ONE, c_mone = MAGMA_Z_NEG_ONE;
    
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
    double nom, nom0, r0, den;
    magma_int_t i;


    // solver setup
    magma_zscal( dofs, c_zero, x->val, 1) ;                            // x = 0
    magma_zcopy( dofs, b.val, 1, r.val, 1 );                           // r = b
    magma_zcopy( dofs, b.val, 1, rr.val, 1 );                          // rr = b
    nom = magma_dznrm2( dofs, r.val, 1 );                              // nom = || r ||
    nom0 = nom = nom*nom;
    rho_old = omega = alpha = MAGMA_Z_MAKE( 1.0, 0. );
    (solver_par->numiter) = 0;

    magma_z_spmv( c_one, A, r, c_zero, v );                           // z = A r
    den = MAGMA_Z_REAL( magma_zdotc(dofs, v.val, 1, r.val, 1) );      // den = z dot r

    if ( (r0 = nom * solver_par->epsilon) < ATOLERANCE ) 
        r0 = ATOLERANCE;
    if ( nom < r0 )
        return MAGMA_SUCCESS;
    
    if (den <= 0.0) {
        printf("Operator A is not postive definite. (Ar,r) = %f\n", den);
        return -100;
    }

    printf("Iteration : %4d  Norm: %f\n", 0, nom );
    
    // start iteration
    while( nom > r0 && solver_par->numiter < solver_par->maxiter ){

        rho_new = magma_zdotc( dofs, rr.val, 1, r.val, 1 );
        beta = rho_new/rho_old * alpha/omega;
        magma_zscal( dofs, beta, p.val, 1 );                                    //p = beta*p
        magma_zaxpy( dofs, c_mone * omega * beta, v.val, 1 , p.val, 1 );        //p = p-omega*beta*v
        magma_zaxpy( dofs, c_one, r.val, 1, p.val, 1 );                         //p = p+r
        magma_z_spmv( c_one, A, p, c_zero, v );                                 //v = Ap
        alpha = rho_new / magma_zdotc( dofs, rr.val, 1, v.val, 1 );

        magma_zcopy( dofs, r.val, 1 , s.val, 1 );
        magma_zaxpy( dofs, c_mone * alpha, v.val, 1 , s.val, 1 );
        magma_z_spmv( c_one, A, s, c_zero, t );                                  //t=As
        omega = magma_zdotc( dofs, t.val, 1, s.val, 1 ) 
                   / magma_zdotc( dofs, t.val, 1, t.val, 1 );

        magma_zaxpy( dofs, alpha, p.val, 1 , x->val, 1 );
        magma_zaxpy( dofs, omega, s.val, 1 , x->val, 1 );

        magma_zcopy( dofs, s.val, 1 , r.val, 1 );
        magma_zaxpy( dofs, c_mone * omega, t.val, 1 , r.val, 1 );
        nom = magma_dznrm2( dofs, r.val, 1 );
        nom = nom*nom;
        rho_old = rho_new;

        (solver_par->numiter)++;

        printf("Iteration : %4d  Norm: %f\n", (solver_par->numiter), nom);
    
    }
  
    printf( "      (r_0, r_0) = %e\n", nom0 );
    printf( "      (r_N, r_N) = %e\n", nom );
    printf( "      Number of BICGSTAB iterations: %d\n", (solver_par->numiter));
    
    if (solver_par->epsilon == RTOLERANCE) {
        magma_z_spmv( c_one, A, *x, c_zero, r );                       // z = A d
        magma_zaxpy(dofs,  c_mone, b.val, 1, r.val, 1);                // r = r - b
        den = magma_dznrm2(dofs, r.val, 1);                            // den = || r ||
        printf( "      || r_N ||   = %f\n", den);
        solver_par->residual = (double)(den);
    }

    magma_z_vfree(&r);
    magma_z_vfree(&rr);
    magma_z_vfree(&p);
    magma_z_vfree(&v);
    magma_z_vfree(&s);
    magma_z_vfree(&t);
    
    return MAGMA_SUCCESS;
}


