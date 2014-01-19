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
    This is a GPU implementation of the Iterative Refinement method.
    The inner solver is passed via the preconditioner argument.

    Arguments
    =========

    magma_z_sparse_matrix A                   input matrix A
    magma_z_vector b                          RHS b
    magma_z_vector *x                         solution approximation
    magma_solver_parameters *solver_par       solver parameters
    magma_precond_parameters *precond_par     parameters for inner solver

    ========================================================================  */


magma_int_t
magma_ziterref( magma_z_sparse_matrix A, magma_z_vector b, magma_z_vector *x,  
   magma_solver_parameters *solver_par, magma_precond_parameters *precond_par ){

    // some useful variables
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO, c_one = MAGMA_Z_ONE, 
                                                c_mone = MAGMA_Z_NEG_ONE;
    
    magma_int_t dofs = A.num_rows;

    // workspace
    magma_z_vector r,z;
    magma_z_vinit( &r, Magma_DEV, dofs, c_zero );
    magma_z_vinit( &z, Magma_DEV, dofs, c_zero );

    // solver variables
    double nom, nom0, r0, den;

    // solver setup
    magma_zscal( dofs, c_zero, x->val, 1) ;                    // x = 0

    magma_z_spmv( c_mone, A, *x, c_zero, r );                  // r = - A x
    magma_zaxpy(dofs,  c_one, b.val, 1, r.val, 1);             // r = r + b
    nom = magma_dznrm2(dofs, r.val, 1);                 // nom0 = || r ||
    nom = nom0 = nom * nom;
    
    if ( (r0 = nom * solver_par->epsilon) < ATOLERANCE ) 
        r0 = ATOLERANCE;
    if ( nom < r0 )
        return MAGMA_SUCCESS;
    
    //Chronometry
    #ifdef ENABLE_TIMER
    double tempo1, tempo2;
    magma_device_sync(); tempo1=magma_wtime();
    printf("#=============================================================#\n");
    printf("#   Iterative Refinement performance analysis every %d iteration\n", 
                                                                    iterblock);
    printf("#   iter   ||   residual-nrm2    ||   runtime    \n");
    printf("#=============================================================#\n");
    printf("      0    ||    %e    ||    0.0000      \n", nom);
    magma_device_sync(); tempo1=magma_wtime();
    #endif
    
    // start iteration
    for( solver_par->numiter= 1; solver_par->numiter<solver_par->maxiter; 
                                                    solver_par->numiter++ ){

        magma_zscal( dofs, MAGMA_Z_MAKE(1./nom, 0.), r.val, 1) ;  // scale it
        magma_z_precond( A, r, &z, *precond_par );  // inner solver:  A * z = r
        magma_zscal( dofs, MAGMA_Z_MAKE(nom, 0.), z.val, 1) ;  // scale it
        magma_zaxpy(dofs,  c_one, z.val, 1, x->val, 1);        // x = x + z
        magma_z_spmv( c_mone, A, *x, c_zero, r );              // r = - A x
        magma_zaxpy(dofs,  c_one, b.val, 1, r.val, 1);         // r = r + b
        nom = magma_dznrm2(dofs, r.val, 1);                    // nom = || r || 
        #ifdef ENABLE_TIMER
        //Chronometry  
        magma_device_sync(); tempo2=magma_wtime();
        if( solver_par->numiter%iterblock==0 ) {
            printf("   %4d    ||    %e    ||    %.4lf  \n", 
                (solver_par->numiter), nom, tempo2-tempo1 );
        }
        #endif

        if (  nom  < r0 ) {
            break;
        }
    } 

    #ifdef ENABLE_TIMER
    double residual;
    magma_zresidual( A, b, *x, &residual );
    printf("#=============================================================#\n");
    printf("# Iterative Refinement solver summary:\n" );
    printf("#    initial residual: %e\n", nom0 );
    printf("#    iterations: %4d\n#    iterative residual: %e\n",
            (solver_par->numiter), nom );
    printf("#    exact relative residual: %e\n#    runtime: %.4lf sec\n", 
                residual, tempo2-tempo1);
    printf("#=============================================================#\n");
    #endif
        
    solver_par->residual = (double)(nom);
    
    magma_z_vfree(&r);
    magma_z_vfree(&z);


    return MAGMA_SUCCESS;
}   /* magma_ziterref */


