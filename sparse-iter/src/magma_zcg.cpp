/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> s d c

*/
#include "common_magma.h"
#include "../include/magmasparse.h"

#include <assert.h>

#define RTOLERANCE     10e-16
#define ATOLERANCE     10e-16



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

    ??? What should be the argument; order; what matrix format, etc ....

    =====================================================================  */

    magmaDoubleComplex c_zero = MAGMA_Z_ZERO, c_one = MAGMA_Z_ONE, c_mone = MAGMA_Z_NEG_ONE;

    magma_int_t dofs = A.num_rows;

    magma_z_vector r,d,z;
    magma_z_vinit( &r, Magma_DEV, dofs, c_zero );
    magma_z_vinit( &d, Magma_DEV, dofs, c_zero );
    magma_z_vinit( &z, Magma_DEV, dofs, c_zero );



    magmaDoubleComplex alpha, beta;
    double nom, nom0, r0, betanom, den;
    magma_int_t i;

    cublasZscal(dofs, c_zero, x->val, 1);     // x = 0
    cublasZcopy(dofs, b.val, 1, r.val, 1);       // r = b
    cublasZcopy(dofs, b.val, 1, d.val, 1);       // d = b
    nom = cublasDznrm2(dofs, r.val, 1);      // nom = || r ||
    nom = nom * nom;

    nom0 = nom;                          // nom = r dot r
    
    magma_z_spmv( c_one, A, r, c_zero, z ); // z = A r
    den = MAGMA_Z_REAL(cublasZdotc(dofs, z.val, 1, r.val, 1));  // den = z dot r
    
    if ( (r0 = nom * solver_par->epsilon) < ATOLERANCE) r0 = ATOLERANCE;
    if (nom < r0)
        return MAGMA_SUCCESS;
    
    if (den <= 0.0) {
        printf("Operator A is not postive definite. (Ar,r) = %f\n", den);
        return -100;
    }
    
    printf("Iteration : %4d  Norm: %f\n", 0, nom);
    
    // start iteration
    for( i= 1; i<solver_par->maxiter; i++ ) {
        alpha = MAGMA_Z_MAKE(nom/den, 0.);
        
        cublasZaxpy(dofs,  alpha, d.val, 1, x->val, 1);         // x = x + alpha d
        cublasZaxpy(dofs, -alpha, z.val, 1, r.val, 1);         // r = r - alpha z
        betanom = cublasDznrm2(dofs, r.val, 1);             // betanom = || r ||
        betanom = betanom * betanom;                   // betanom = r dot r
        
        printf("Iteration : %4d  Norm: %f\n", i, betanom);
        
        if ( betanom < r0 ) {
            solver_par->numiter = i;
            break;
        }

        beta = MAGMA_Z_MAKE(betanom/nom, 0.);         // beta = betanom/nom
        
        cublasZscal(dofs, beta, d.val, 1);                // d = beta*d
        cublasZaxpy(dofs, c_one, r.val, 1, d.val, 1);             // d = d + r 
        
        magma_z_spmv( c_one, A, d, c_zero, z );      // z = A d
        den = MAGMA_Z_REAL(cublasZdotc(dofs, d.val, 1, z.val, 1));// den = d dot z
        nom = betanom;
    } 
    
    printf( "      (r_0, r_0) = %e\n", nom0);
    printf( "      (r_N, r_N) = %e\n", betanom);
    printf( "      Number of CG iterations: %d\n", i);
    
    if (solver_par->epsilon == RTOLERANCE) {

        magma_z_spmv( c_one, A, *x, c_zero, r );      // z = A d
        
        cublasZaxpy(dofs,  c_mone, b.val, 1, r.val, 1);         // r = r - b
        den = cublasDznrm2(dofs, r.val, 1);                // den = || r ||
        
        printf( "      || r_N ||   = %f\n", den);
        solver_par->residual = (double)(den);
    }
    
    return MAGMA_SUCCESS;
}


