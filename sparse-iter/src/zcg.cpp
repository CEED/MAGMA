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


magma_int_t
magma_zcg( magma_z_sparse_matrix A, magma_z_vector b, magma_z_vector *x,  
           magma_solver_parameters *solver_par )
{
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

    =====================================================================  */

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
    double nom, nom0, r0, betanom, den;
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
    magma_device_sync(); tempo1=magma_wtime();
   //     printf("Iteration: %4d  Norm: %e  Time: %.2lf  SpMV: %.2lf %.2lf%%  Rest: %.2lf\n", 
   //                 (solver_par->numiter), nom, 0.0, 0.0, 0.0 );
    #endif
    
    // start iteration
    for( solver_par->numiter= 1; i<solver_par->maxiter; solver_par->numiter++ ) {
        alpha = MAGMA_Z_MAKE(nom/den, 0.);
        magma_zaxpy(dofs,  alpha, p.val, 1, x->val, 1);               // x = x + alpha p
        magma_zaxpy(dofs, -alpha, q.val, 1, r.val, 1);                // r = r - alpha q
        betanom = magma_dznrm2(dofs, r.val, 1);                       // betanom = || r ||
        betanom = betanom * betanom;                                  // betanom = r dot r
      //  printf("Iteration : %4d  Norm: %f\n", i, betanom);
        if ( betanom < r0 ) {
            solver_par->numiter = i;
            break;
        }
        beta = MAGMA_Z_MAKE(betanom/nom, 0.);                         // beta = betanom/nom
        magma_zscal(dofs, beta, p.val, 1);                            // p = beta*p
        magma_zaxpy(dofs, c_one, r.val, 1, p.val, 1);                 // p = p + r 
        magma_z_spmv( c_one, A, p, c_zero, q );                       // q = A p
        den = MAGMA_Z_REAL(magma_zdotc(dofs, p.val, 1, q.val, 1));    // den = p dot q
        nom = betanom;


        #ifdef ENABLE_TIMER
        //Chronometry  
        magma_device_sync(); tempo2=magma_wtime();
        //cublasGetVector(1 , sizeof( magmaDoubleComplex ), skp+5, 1, skp_h+5, 1 );
        if( solver_par->numiter%1000==0 ) {
        printf("Iteration: %4d  Norm: %e  Time: %.2lf  SpMV: %.2lf %.2lf%%  Rest: %.2lf\n", 
                    (solver_par->numiter), nom, tempo2-tempo1, t_spmv, 100.0*t_spmv/(tempo2-tempo1), tempo2-tempo1-t_spmv);
        }
        #endif
    } 
    #ifndef ENABLE_TIMER
    printf("Iteration: %4d  Norm: %e  Time: %.2lf  SpMV: %.2lf %.2lf%%  Rest: %.2lf\n", 
                (solver_par->numiter), nom, tempo2-tempo1, t_spmv, 100.0*t_spmv/(tempo2-tempo1), tempo2-tempo1-t_spmv);
    #endif
    
    printf( "      (r_0, r_0) = %e\n", nom0);
    printf( "      (r_N, r_N) = %e\n", betanom);
    printf( "      Number of CG iterations: %d\n", i);
/*    
    if (solver_par->epsilon == RTOLERANCE) {
        magma_z_spmv( c_one, A, *x, c_zero, r );                       // r = A x
        magma_zaxpy(dofs,  c_mone, b.val, 1, r.val, 1);                // r = r - b
        den = magma_dznrm2(dofs, r.val, 1);                            // den = || r ||
        printf( "      || r_N ||   = %e\n", den);
        solver_par->residual = (double)(den);
    }
*/
    magma_z_vfree(&r);
    magma_z_vfree(&p);
    magma_z_vfree(&q);

    return MAGMA_SUCCESS;
}   /* magma_zcg */


