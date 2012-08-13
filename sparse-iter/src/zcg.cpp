/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> s d c

*/
#include "common_magma.h"

#include <assert.h>

#define RTOLERANCE     10e-16
#define ATOLERANCE     10e-16

magma_int_t
magma_zcg( magma_int_t dofs, magma_int_t & num_of_iter,  
           cuDoubleComplex *x, cuDoubleComplex *b,
           cuDoubleComplex *d_A, magma_int_t *d_I, magma_int_t *d_J, 
           cuDoubleComplex *dwork,
           double rtol = RTOLERANCE )
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

    cuDoubleComplex *r = dwork;
    cuDoubleComplex *d = dwork + dofs;
    cuDoubleComplex *z = dwork + 2*dofs;

    cuDoubleComplex r0, den, nom, nom0, betanom, alpha, beta;
    magma_int_t i;

    cublasZscal(dofs, 0.f, x, 1);        // x = 0
    cublasZcopy(dofs, b, 1, r, 1);       // r = b
    cublasZcopy(dofs, b, 1, d, 1);       // d = b
    nom = cublasZnrm2(dofs, r, 1);       // nom = || r ||
    nom = nom * nom;

    nom0 = nom;                          // nom = r dot r
    
    magma_zspmv(d_A, d_I, d_J, dofs, r, z); // z = A r
    den = cublasZdot(dofs, z, 1, r, 1);  // den = z dot r
    
    if ( (r0 = nom * rtol) < ATOLERANCE) r0 = ATOLERANCE;
    if (nom < r0)
        return;
    
    if (den <= 0.0) {
        printf("Operator A is not postive definite. (Ar,r) = %f\n", den);
        return;
    }
    
    printf("Iteration : %4d  Norm: %f\n", 0, nom);
    
    // start iteration
    for(i= 1; i<num_of_iter ;i++) {
        alpha = nom/den;
        
        cublasZaxpy(dofs,  alpha, d, 1, x, 1);         // x = x + alpha d
        cublasZaxpy(dofs, -alpha, z, 1, r, 1);         // r = r - alpha z
        betanom = cublasZnrm2(dofs, r, 1);             // betanom = || r ||
        betanom = betanom * betanom;                   // betanom = r dot r
        
        printf("Iteration : %4d  Norm: %f\n", i, betanom);
        
        if ( betanom < r0 ) {
            num_of_iter = i;
            break;
        }

        beta = betanom/nom;                           // beta = betanom/nom
        
        cublasZscal(dofs, beta, d, 1);                // d = beta*d
        cublasZaxpy(dofs, 1.f, r, 1, d, 1);           // d = d + r 
        
        magma_zspmv(d_A, d_I, d_J, dofs, d, z);       // z = A d
        den = cublasZdot(dofs, d, 1, z, 1);           // den = d dot z
        nom = betanom;
    } 
    
    printf( "      (r_0, r_0) = %e\n", nom0);
    printf( "      (r_N, r_N) = %e\n", betanom);
    printf( "      Number of CG iterations: %d\n", i);
    
    if (rtol == RTOLERANCE) {
        magma_zspmv(d_A, d_I, d_J, dofs, x, r);       // r = A x
        
        cublasZaxpy(dofs,  -1.f, b, 1, r, 1);         // r = r - b
        den = cublasZnrm2(dofs, r, 1);                // den = || r ||
        
        printf( "      || r_N ||   = %f\n", den);
    }
}


