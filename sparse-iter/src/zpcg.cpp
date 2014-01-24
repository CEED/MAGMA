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

#define RTOLERANCE     10e-16
#define ATOLERANCE     10e-16


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
    This is a GPU implementation of the Preconditioned Conjugate Gradient method.

    Arguments
    =========

    magma_z_sparse_matrix A                   input matrix A
    magma_z_vector b                          RHS b
    magma_z_vector *x                         solution approximation
    magma_solver_parameters *solver_par       solver parameters
    magma_precond_parameters *precond_par     preconditioner parameters

    =====================================================================  */


magma_int_t
magma_zpcg( magma_z_sparse_matrix A, magma_z_vector b, magma_z_vector *x,  
           magma_solver_parameters *solver_par, magma_precond_parameters *precond_par ){

    // some useful variables
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO, c_one = MAGMA_Z_ONE, c_mone = MAGMA_Z_NEG_ONE;
    
    magma_int_t dofs = A.num_rows;

    // workspace
    magma_z_vector r,p,q,z;
    magma_z_vinit( &r, Magma_DEV, dofs, c_zero );
    magma_z_vinit( &p, Magma_DEV, dofs, c_zero );
    magma_z_vinit( &q, Magma_DEV, dofs, c_zero );
    magma_z_vinit( &z, Magma_DEV, dofs, c_zero );
    
    // solver variables
    magmaDoubleComplex alpha, beta;
    double nom, nom0, r0, betanom, den;
    magma_int_t i;


    // solver setup
    magma_zscal( dofs, c_zero, x->val, 1) ;                           // x = 0
    magma_zcopy( dofs, b.val, 1, r.val, 1 );                           // r = b
    //magma_zcopy( dofs, r.val, 1, z.val, 1 );
    magma_z_precond( A, r, &z, *precond_par );                      // precond: M * z = r
    magma_zcopy( dofs, z.val, 1, p.val, 1 );                           // d = b
    nom = MAGMA_Z_REAL( magma_zdotc( dofs, r.val, 1, z.val, 1 ) );     // nom = r * z
    nom0 = nom;                                                        
    magma_z_spmv( c_one, A, p, c_zero, q );                           // q = A p
    den = MAGMA_Z_REAL( magma_zdotc(dofs, p.val, 1, q.val, 1) );      // den = p dot q
    
    if ( (r0 = nom * solver_par->epsilon) < ATOLERANCE ) 
        r0 = ATOLERANCE;
    if ( nom < r0 )
        return MAGMA_SUCCESS;
    
    if (den <= 0.0) {
        printf("Operator A is not postive definite. (Ar,r) = %f\n", den);
        return -100;
    }
    
    printf("Iteration : %4d  Norm: %f\n", 0, nom);
    
    // start iteration
    for( i= 1; i<solver_par->maxiter; i++ ) {
        alpha = MAGMA_Z_MAKE(nom/den, 0.);
        magma_zaxpy(dofs,  alpha, p.val, 1, x->val, 1);                    // x = x + alpha d
        magma_zaxpy(dofs, -alpha, q.val, 1, r.val, 1);                     // r = r - alpha z
        //magma_zcopy( dofs, r.val, 1, z.val, 1 );
        magma_z_precond( A, r, &z, *precond_par );                         // precond: M * z = r
        betanom = MAGMA_Z_REAL( magma_zdotc( dofs, r.val, 1, z.val, 1 ) ); // betanom = r * z
        printf("Iteration : %4d  Norm: %f\n", i, betanom);
        if ( betanom < r0 ) {
            solver_par->numiter = i;
            break;
        }
        beta = MAGMA_Z_MAKE(betanom/nom, 0.);                              // beta = betanom/nom
        magma_zscal(dofs, beta, p.val, 1);                                 // p = beta*p
        magma_zaxpy(dofs, c_one, r.val, 1, p.val, 1);                      // p = p + z 
        magma_z_spmv( c_one, A, p, c_zero, q );                            // z = A d
        den = MAGMA_Z_REAL(magma_zdotc(dofs, p.val, 1, q.val, 1));         // den = p dot q
        nom = betanom; 
    } 
    
    printf( "      (r_0, r_0) = %e\n", nom0);
    printf( "      (r_N, r_N) = %e\n", betanom);
    printf( "      Number of PCG iterations: %d\n", i);
    
    if (solver_par->epsilon == RTOLERANCE) {
        magma_z_spmv( c_one, A, *x, c_zero, r );                       // r = A x
        magma_zaxpy(dofs,  c_mone, b.val, 1, r.val, 1);                // r = r - b
        den = magma_dznrm2(dofs, r.val, 1);                            // den = || r ||
        printf( "      || r_N ||   = %f\n", den);
        solver_par->residual = (double)(den);
    }
    
    return MAGMA_SUCCESS;
}


