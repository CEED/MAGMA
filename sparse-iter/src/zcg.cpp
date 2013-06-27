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
magma_zspmv(magmaDoubleComplex *d_A, magma_int_t *d_I, magma_int_t *d_J, magma_int_t dofs, 
            magmaDoubleComplex *r, magmaDoubleComplex *z);


magma_int_t
magma_zcg( magma_int_t dofs, magma_int_t & num_of_iter,  
           magmaDoubleComplex *x, magmaDoubleComplex *b,
           magmaDoubleComplex *d_A, magma_int_t *d_I, magma_int_t *d_J, 
           magmaDoubleComplex *dwork,
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

    magmaDoubleComplex *r = dwork;
    magmaDoubleComplex *d = dwork + dofs;
    magmaDoubleComplex *z = dwork + 2*dofs;

    magmaDoubleComplex c_zero = MAGMA_Z_ZERO, c_one = MAGMA_Z_ONE, c_mone = MAGMA_Z_NEG_ONE;

    magmaDoubleComplex alpha, beta;
    double nom, nom0, r0, betanom, den;
    magma_int_t i;

    cublasZscal(dofs, c_zero, x, 1);     // x = 0
    cublasZcopy(dofs, b, 1, r, 1);       // r = b
    cublasZcopy(dofs, b, 1, d, 1);       // d = b
    nom = cublasDznrm2(dofs, r, 1);      // nom = || r ||
    nom = nom * nom;

    nom0 = nom;                          // nom = r dot r
    
    magma_zspmv(d_A, d_I, d_J, dofs, r, z); // z = A r
    den = MAGMA_Z_REAL(cublasZdotc(dofs, z, 1, r, 1));  // den = z dot r
    
    if ( (r0 = nom * rtol) < ATOLERANCE) r0 = ATOLERANCE;
    if (nom < r0)
        return MAGMA_SUCCESS;
    
    if (den <= 0.0) {
        printf("Operator A is not postive definite. (Ar,r) = %f\n", den);
        return -100;
    }
    
    printf("Iteration : %4d  Norm: %f\n", 0, nom);
    
    // start iteration
    for(i= 1; i<num_of_iter ;i++) {
        alpha = MAGMA_Z_MAKE(nom/den, 0.);
        
        cublasZaxpy(dofs,  alpha, d, 1, x, 1);         // x = x + alpha d
        cublasZaxpy(dofs, -alpha, z, 1, r, 1);         // r = r - alpha z
        betanom = cublasDznrm2(dofs, r, 1);             // betanom = || r ||
        betanom = betanom * betanom;                   // betanom = r dot r
        
        printf("Iteration : %4d  Norm: %f\n", i, betanom);
        
        if ( betanom < r0 ) {
            num_of_iter = i;
            break;
        }

        beta = MAGMA_Z_MAKE(betanom/nom, 0.);         // beta = betanom/nom
        
        cublasZscal(dofs, beta, d, 1);                // d = beta*d
        cublasZaxpy(dofs, c_one, r, 1, d, 1);             // d = d + r 
        
        magma_zspmv(d_A, d_I, d_J, dofs, d, z);           // z = A d
        den = MAGMA_Z_REAL(cublasZdotc(dofs, d, 1, z, 1));// den = d dot z
        nom = betanom;
    } 
    
    printf( "      (r_0, r_0) = %e\n", nom0);
    printf( "      (r_N, r_N) = %e\n", betanom);
    printf( "      Number of CG iterations: %d\n", i);
    
    if (rtol == RTOLERANCE) {
        magma_zspmv(d_A, d_I, d_J, dofs, x, r);       // r = A x
        
        cublasZaxpy(dofs,  c_mone, b, 1, r, 1);         // r = r - b
        den = cublasDznrm2(dofs, r, 1);                // den = || r ||
        
        printf( "      || r_N ||   = %f\n", den);
    }

    return MAGMA_SUCCESS;
}


