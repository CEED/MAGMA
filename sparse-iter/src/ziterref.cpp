/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Hartwig Anzt 

       @precisions normal z -> s d c
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
    This is a GPU implementation of the Iterative Refinement method.
    The inner solver is passed via the preconditioner argument.

    Arguments
    =========

    magma_z_sparse_matrix A                   input matrix A
    magma_z_vector b                          RHS b
    magma_z_vector *x                         solution approximation
    magma_solver_parameters *solver_par       solver parameters
    magma_z_preconditioner *precond_par       inner solver

    ========================================================================  */

magma_int_t
magma_ziterref( magma_z_sparse_matrix A, magma_z_vector b, magma_z_vector *x,  
   magma_solver_parameters *solver_par, magma_z_preconditioner *precond_par ){

    // prepare solver feedback
    solver_par->solver = Magma_ITERREF;
    solver_par->numiter = 0;
    solver_par->info = 0;
    magma_int_t iterblock = solver_par->verbose;
    if( solver_par->verbose > 0 ){
        magma_malloc_cpu( (void **)&solver_par->res_vec, sizeof(real_Double_t) 
                * ( (solver_par->maxiter)/(solver_par->verbose)+1) );
        magma_malloc_cpu( (void **)&solver_par->timing, sizeof(real_Double_t) 
                *( (solver_par->maxiter)/(solver_par->verbose)+1) );
    }else{
        solver_par->res_vec = NULL;
        solver_par->timing = NULL;
    }
    double residual;
    magma_zresidual( A, b, *x, &residual );
    solver_par->init_res = residual;

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
    nom0 = magma_dznrm2(dofs, r.val, 1);                       // nom0 = || r ||
    nom = nom0 * nom0;
    
    if ( (r0 = nom * solver_par->epsilon) < ATOLERANCE ) 
        r0 = ATOLERANCE;
    if ( nom < r0 )
        return MAGMA_SUCCESS;
    
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

        magma_zscal( dofs, MAGMA_Z_MAKE(1./nom, 0.), r.val, 1) ;  // scale it
        magma_z_precond( A, r, &z, *precond_par );  // inner solver:  A * z = r
        magma_zscal( dofs, MAGMA_Z_MAKE(nom, 0.), z.val, 1) ;  // scale it
        magma_zaxpy(dofs,  c_one, z.val, 1, x->val, 1);        // x = x + z
        magma_z_spmv( c_mone, A, *x, c_zero, r );              // r = - A x
        magma_zaxpy(dofs,  c_one, b.val, 1, r.val, 1);         // r = r + b
        nom = magma_dznrm2(dofs, r.val, 1);                    // nom = || r || 

        if( solver_par->verbose > 0 ){
            magma_device_sync(); tempo2=magma_wtime();
            if( (solver_par->numiter)%iterblock==0 ) {
                solver_par->res_vec[(solver_par->numiter)/iterblock] 
                        = (real_Double_t) nom;
                solver_par->timing[(solver_par->numiter)/iterblock] 
                        = (real_Double_t) tempo2-tempo1;
            }
        }

        if (  nom  < r0 ) {
            break;
        }
    } 
    magma_device_sync(); tempo2=magma_wtime();
    solver_par->runtime = (real_Double_t) tempo2-tempo1;
    magma_zresidual( A, b, *x, &residual );
    solver_par->final_res = residual;

    if( solver_par->numiter < solver_par->maxiter){
        solver_par->info = 0;
    }else if( solver_par->init_res > solver_par->final_res )
        solver_par->info = -2;
    else
        solver_par->info = -1;

    
    magma_z_vfree(&r);
    magma_z_vfree(&z);


    return MAGMA_SUCCESS;
}   /* magma_ziterref */


