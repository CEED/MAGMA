/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @author Hartwig Anzt 

       @precisions normal z -> s d c
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
    This is a GPU implementation of the Iterative Refinement method.

    Arguments
    =========

    magma_z_sparse_matrix A                   input matrix A
    magma_z_vector b                          RHS b
    magma_z_vector *x                         solution approximation
    magma_solver_parameters *solver_par       solver parameters
    magma_precond_parameters *precond_par     parameters for inner solver

    =====================================================================  */


magma_int_t
magma_zir( magma_z_sparse_matrix A, magma_z_vector b, magma_z_vector *x,  
           magma_solver_parameters *solver_par, magma_precond_parameters *precond_par )
{

    // some useful variables
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO, c_one = MAGMA_Z_ONE, c_mone = MAGMA_Z_NEG_ONE;
    
    magma_int_t dofs = A.num_rows;

    // workspace
    magma_z_vector r,z;
    magma_z_vinit( &r, Magma_DEV, dofs, c_zero );
    magma_z_vinit( &z, Magma_DEV, dofs, c_zero );


    // solver variables
    double nom, nom0, r0, den;
    magma_int_t i;


    // solver setup
    magma_zscal( dofs, c_zero, x->val, 1) ;                           // x = 0

    magma_z_spmv( c_mone, A, *x, c_zero, r );                       // r = - A x
    magma_zaxpy(dofs,  c_one, b.val, 1, r.val, 1);                // r = r + b
    nom = nom0 = magma_dznrm2(dofs, r.val, 1);                            // nom0 = || r ||
    
    if ( (r0 = nom * solver_par->epsilon) < ATOLERANCE ) 
        r0 = ATOLERANCE;
    if ( nom < r0 )
        return MAGMA_SUCCESS;

    
    printf("Iteration : %4d  Norm: %f\n", 0, nom);
    

    // start iteration
    for( i= 1; i<solver_par->maxiter; i++ ) {

        magma_zscal( dofs, MAGMA_Z_MAKE(1./nom, 0.), r.val, 1) ;  // scale it
        magma_z_precond( A, r, &z, *precond_par );                         // inner solver:  A * z = r
        magma_zscal( dofs, MAGMA_Z_MAKE(nom, 0.), z.val, 1) ;  // scale it
        magma_zaxpy(dofs,  c_one, z.val, 1, x->val, 1);                    // x = x + z
        magma_z_spmv( c_mone, A, *x, c_zero, r );                       // r = - A x
        magma_zaxpy(dofs,  c_one, b.val, 1, r.val, 1);                // r = r + b
        nom = magma_dznrm2(dofs, r.val, 1);                            // nom = || r ||

        printf("Iteration : %4d  Norm: %f\n", i, nom);
        if ( nom < r0 ) {
            solver_par->numiter = i;
            break;
        }
    } 


    
    printf( "      (r_0, r_0) = %e\n", nom0);
    printf( "      (r_N, r_N) = %e\n", nom);
    printf( "      Number of Iterative Refinement iterations: %d\n", i);
    
    if (solver_par->epsilon == RTOLERANCE) {
        magma_z_spmv( c_one, A, *x, c_zero, r );                       // r = A x
        magma_zaxpy(dofs,  c_mone, b.val, 1, r.val, 1);                // r = r - b
        den = magma_dznrm2(dofs, r.val, 1);                            // den = || r ||
        printf( "      || r_N ||   = %f\n", den);
        solver_par->residual = (double)(den);
    }
    
    return MAGMA_SUCCESS;

    return MAGMA_SUCCESS;
}   /* magma_zir */






