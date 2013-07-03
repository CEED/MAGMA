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

#define RTOLERANCE     10e-10
#define ATOLERANCE     10e-10


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
    This is a GPU implementation of the GMRES method.

    Arguments
    =========

    magma_z_sparse_matrix A                   input matrix A
    magma_z_vector b                          RHS b
    magma_z_vector *x                         solution approximation
    magma_solver_parameters *solver_par       solver parameters

    =====================================================================  */

magma_int_t
magma_zgmres( magma_z_sparse_matrix A, magma_z_vector b, magma_z_vector *x,  
           magma_solver_parameters *solver_par ){

    // some useful variables
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO, c_one = MAGMA_Z_ONE, c_mone = MAGMA_Z_NEG_ONE;
    magma_int_t dofs = A.num_rows;
    magma_int_t i, j, k, m, iter;
    double rNorm, RNorm, den, nom0, r0 = 0.;

    // workspace
    magma_z_vector r;
    magma_z_vinit( &r, Magma_DEV, dofs, c_zero );
    cuDoubleComplex H[(solver_par->restart)+2][(solver_par->restart)+1], HH[(solver_par->restart)+1][(solver_par->restart)+1], v; 
    cuDoubleComplex y[(solver_par->restart)+1], h1[(solver_par->restart)+1];
    

    magma_z_vector q, q_t;
    magma_z_vinit( &q, Magma_DEV, dofs*((solver_par->restart)+1), c_zero );
    magma_z_vinit( &q_t, Magma_DEV, dofs, c_zero );

    
    magma_zscal( dofs, c_zero, x->val, 1 );              //  x = 0
    magma_zcopy( dofs, b.val, 1, r.val, 1 );             //  r = b

    r0 = magma_dznrm2( dofs, r.val, 1 );                 //  r0= || r||
    nom0 = r0*r0;
    H[1][0] = MAGMA_Z_MAKE( r0, 0. ); 

    if ((r0 *= solver_par->epsilon) < ATOLERANCE) r0 = ATOLERANCE;
    
    printf("Iteration : %4d  Norm: %f\n", 0, H[1][0]*H[1][0]);

    for (iter = 0; iter<solver_par->maxiter; iter++) {
        for(k=1; k<=(solver_par->restart); k++) {
            v =1./H[k][k-1];
            
            magma_zcopy(dofs, r.val, 1, q.val+k*dofs, 1);       //  q[k]    = 1.0/H[k][k-1] r
            magma_zscal(dofs, v, q.val+k*dofs, 1);          //  (to be fused)

            q_t.val = q.val+k*dofs;
            magma_z_spmv( c_one, A, q_t, c_zero, r ); //  r       = A q[k] 

            for (i=1; i<=k; i++) {
                H[i][k]=magma_zdotc(dofs,q.val+i*dofs,1,r.val,1);  //  H[i][k] = q[i] . r
                
                magma_zaxpy(dofs,-H[i][k],q.val+i*dofs,1,r.val,1);//  r       = r - H[i][k] q[i]
            }
            
            H[k+1][k]= MAGMA_Z_MAKE( magma_dznrm2(dofs, r.val, 1), 0. );       //  H[k+1][k] = sqrt(r . r) 

            //   Minimization of  || b-Ax ||  in K_k 
            for (i=1; i<=k; i++) {
                HH[k][i] = MAGMA_Z_MAKE( 0.0, 0. );
                for (j=1; j<=i+1; j++)
                    HH[k][i] +=  H[j][k] * H[j][i];
            } 
            
            h1[k] = H[1][k]*H[1][0];
            
            if (k != 1)
                for (i=1; i<k; i++) {
                    HH[k][i] = HH[k][i]/HH[i][i];
                    for (m=i+1; m<=k; m++)
                        HH[k][m] -= HH[k][i] * HH[m][i] * HH[i][i];
                    h1[k] -= h1[i] * HH[k][i];   
                }    
            y[k] = h1[k]/HH[k][k]; 
            if (k != 1)  
                for (i=k-1; i>=1; i--) {
                    y[i] = h1[i]/HH[i][i];
                    for (j=i+1; j<=k; j++)
                        y[i] -= y[j] * HH[j][i];
                }
            
            m = k;
            
            rNorm = fabs(MAGMA_Z_REAL(H[k+1][k]));
            //if (rNorm < r0) break;
        }
        
        //   Minimization Done      
        for (i=1; i<=m; i++)
            magma_zaxpy(dofs, y[i], q.val+i*dofs, 1, x->val, 1);  //  xNew += y[i]*q[i]
        

        magma_z_spmv( c_one, A, *x, c_zero, r );             //  r = A * x
        magma_zaxpy(dofs, c_mone, b.val, 1, r.val, 1);       //  r = r - b
        magma_zscal(dofs, c_mone, r.val, 1);             //  r = -r (to be fused)
        H[1][0] = MAGMA_Z_MAKE( magma_dznrm2(dofs, r.val, 1), 0. ); //  RNorm = H[1][0] = || r ||
        RNorm = MAGMA_Z_REAL( H[1][0] );
    
        printf("Iteration : %4d  Norm: %f\n", iter, RNorm*RNorm);
        
        if (fabs(RNorm*RNorm) < r0) break;    
        //if (rNorm < r0) break;
    }
    

    printf( "      (r_0, r_0) = %e\n", nom0 );
    printf( "      (r_N, r_N) = %e\n", RNorm*RNorm);
    printf( "      Number of GMRES restarts: %d\n", iter);
    
    if (solver_par->epsilon == RTOLERANCE) {
        magma_z_spmv( c_one, A, *x, c_zero, r );                       // q_t = A x
        magma_zaxpy(dofs,  c_mone, b.val, 1, r.val, 1);                // r = r - b
        den = magma_dznrm2(dofs, r.val, 1);                            // den = || r ||
        printf( "      || r_N ||   = %f\n", den);
        solver_par->residual = (double)(den);
    }
    solver_par->numiter = iter;

}

