/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @author Stan Tomov
       @author Hartwig Anzt 

       @precisions normal z -> s d c
*/

#include "common_magma.h"
#include "../include/magmasparse.h"

#include <assert.h>

#define RTOLERANCE     1e-16
#define ATOLERANCE     1e-16

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
    This is a GPU implementation of the Conjugate Gradient method.

    Arguments
    =========

    magma_z_sparse_matrix A                   input matrix A
    magma_z_vector b                          RHS b
    magma_z_vector *x                         solution approximation
    magma_solver_parameters *solver_par       solver parameters

    ========================================================================  */

magma_int_t
magma_zcg( magma_z_sparse_matrix A, magma_z_vector b, magma_z_vector *x,  
           magma_solver_parameters *solver_par ){

    // local variables
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO, c_one = MAGMA_Z_ONE, 
                                            c_mone = MAGMA_Z_NEG_ONE;
    
    magma_int_t dofs = A.num_rows;

    // GPU workspace
    magma_z_vector r, p, q;
    magma_z_vinit( &r, Magma_DEV, dofs, c_zero );
    magma_z_vinit( &p, Magma_DEV, dofs, c_zero );
    magma_z_vinit( &q, Magma_DEV, dofs, c_zero );
    
    // solver variables
    magmaDoubleComplex alpha, beta;
    double nom, nom0, r0, betanom, betanomsq, den;
    magma_int_t i;

    // solver setup
    magma_zscal( dofs, c_zero, x->val, 1) ;                     // x = 0
    magma_zcopy( dofs, b.val, 1, r.val, 1 );                    // r = b
    magma_zcopy( dofs, b.val, 1, p.val, 1 );                    // p = b
    nom = magma_dznrm2( dofs, r.val, 1 );                       // nom = || r ||
    nom = nom * nom;
    nom0 = nom;                                                 // nom = r dot r
    magma_z_spmv( c_one, A, p, c_zero, q );                     // q = A p
    den = MAGMA_Z_REAL( magma_zdotc(dofs, p.val, 1, q.val, 1) );// den = p dot q
    
    if ( (r0 = nom * solver_par->epsilon) < ATOLERANCE ) 
        r0 = ATOLERANCE;
    if ( nom < r0 )
        return MAGMA_SUCCESS;
    
    if (den <= 0.0) {
        printf("Operator A is not postive definite. (Ar,r) = %f\n", den);
        return -100;
    }
    
  //  printf("Iteration : %4d  Norm: %f\n", 0, nom);

    //Chronometry
    #define ENABLE_TIMER
    #ifdef ENABLE_TIMER
    double t_spmv1, t_spmv = 0.0;
    double tempo1, tempo2;
    int iterblock = 1;
    magma_device_sync(); tempo1=magma_wtime();
    printf("#===============================================#\n");
    printf("#   CG performance analysis ever %d iteration   #\n", iterblock);
    printf("#   iter   ||   residual-nrm2    ||   runtime    #\n");
    printf("#===============================================#\n");
    printf("      0    ||    %e    ||    0.0000      \n", nom);
    magma_device_sync(); tempo1=magma_wtime();
    #endif
    
    // start iteration
    for( solver_par->numiter= 1; i<solver_par->maxiter; solver_par->numiter++ ){
        alpha = MAGMA_Z_MAKE(nom/den, 0.);
        magma_zaxpy(dofs,  alpha, p.val, 1, x->val, 1);               // x = x + alpha p
        magma_zaxpy(dofs, -alpha, q.val, 1, r.val, 1);                // r = r - alpha q
        betanom = magma_dznrm2(dofs, r.val, 1);                       // betanom = || r ||
        betanomsq = betanom * betanom;                                // betanoms = r dot r
        if (  betanom  < r0 ) {
            break;
        }
        beta = MAGMA_Z_MAKE(betanomsq/nom, 0.);                       // beta = betanoms/nom
        magma_zscal(dofs, beta, p.val, 1);                            // p = beta*p
        magma_zaxpy(dofs, c_one, r.val, 1, p.val, 1);                 // p = p + r 
        magma_z_spmv( c_one, A, p, c_zero, q );                       // q = A p
        den = MAGMA_Z_REAL(magma_zdotc(dofs, p.val, 1, q.val, 1));    // den = p dot q
        nom = betanomsq;

        #ifdef ENABLE_TIMER
        //Chronometry  
        magma_device_sync(); tempo2=magma_wtime();
        if( solver_par->numiter%iterblock==0 ) {
            printf("   %4d    ||    %e    ||    %.4lf  \n", (solver_par->numiter), betanom, tempo2-tempo1 );
        }
        #endif
    } 

    #ifdef ENABLE_TIMER
    double residual;
    magma_zresidual( A, b, *x, &residual );
    printf("#===============================================#\n");
    printf("# CG solver summary:\n");
    printf("#    initial residual: %e\n", nom0 );
    printf("#    iterations: %4d\n#    iterative residual: %e\n",
            (solver_par->numiter), betanom );
    printf("#    exact relative residual: %e\n#    runtime: %.4lf sec\n", 
                residual, tempo2-tempo1);
    printf("#===============================================#\n");
    #endif
        
    solver_par->residual = (double)(betanom);
    magma_z_vfree(&r);
    magma_z_vfree(&p);
    magma_z_vfree(&q);

    return MAGMA_SUCCESS;
}   /* magma_zcg */


