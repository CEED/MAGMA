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

#define RTOLERANCE     10e-10
#define ATOLERANCE     10e-10

// This is the restart value
#define Kmax 30

magma_int_t
magma_gmres(magma_int_t n, magma_int_t &nit, cuDoubleComplex *x, cuDoubleComplex *b,
            cuDoubleComplex *d_A, magma_int_t *d_I, magma_int_t *d_J, 
            cuDoubleComplex *dwork)
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
    This is a GPU implementation of the GMRES method.

    Arguments
    =========

    ??? What should be the argument; order; what matrix format, etc ....

    =====================================================================  */

    magma_int_t i, j, k, m;  
    cuDoubleComplex H[Kmax+2][Kmax+1], HH[Kmax+1][Kmax+1], v; 
    cuDoubleComplex *r = dwork;
    magma_int_t iter;
    
    doubele rNorm, RNorm, r0 = 0.;
    cuDoubleComplex y[Kmax+1], h1[Kmax+1];
    
    // if (Kmax > n) Kmax=n; 
    cuDoubleComplex *q[Kmax+1];
    for (i=1; i<=Kmax; i++) 
        q[i] = dwork + i*n; 
    
    cublasZscal(n, 0.f, x, 1);              //  x = 0
    cublasZcopy(n, b, 1, r, 1);             //  r = b

    H[1][0] = r0 =cublasZnrm2(n, r, 1);     //  r0= || r||

    if ((r0 *= RTOLERANCE) < ATOLERANCE) r0 = ATOLERANCE;
    
    printf("Iteration %4d done!\n", 0);
    printf("The current residual RNorm = %f\n", H[1][0]);

    for (iter = 0; iter<nit; iter++) {
        
        for(k=1; k<=Kmax; k++) {
            v =1./H[k][k-1];
            
            cublasZcopy(n, r, 1, q[k], 1);       //  q[k]    = 1.0/H[k][k-1] r
            cublasZscal(n, v, q[k], 1);          //  (to be fused)
            
            ssmv_gpu(d_A, d_I, d_J, n, q[k], r); //  r       = A q[k] 
            for (i=1; i<=k; i++) {
                H[i][k]=cublasZdot(n,q[i],1,r,1);  //  H[i][k] = q[i] . r
                
                cublasZaxpy(n,-H[i][k],q[i],1,r,1);//  r       = r - H[i][k] q[i]
            }
            
            H[k+1][k]=cublasZnrm2(n, r, 1);       //  H[k+1][k] = sqrt(r . r) 
            
            //   Minimization of  || b-Ax ||  in K_k 
            for (i=1; i<=k; i++) {
                HH[k][i] = 0.0;
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
            
            rNorm = fabs(H[k+1][k]);
            //if (rNorm < r0) break;
        }
        
        //   Minimization Done      
        for (i=1; i<=m; i++)
            cublasZaxpy(n, y[i], q[i], 1, x, 1);  //  xNew += y[i]*q[i]
        
        ssmv_gpu(d_A, d_I, d_J, n, x, r);       //  r = Ax
        
        cublasZaxpy(n, -1.f, b, 1, r, 1);       //  r = r - b
        cublasZscal(n, -1.f, r, 1);             //  r = -r (to be fused)
        RNorm = H[1][0] = cublasZnrm2(n, r, 1); //  RNorm = H[1][0] = || r ||
        
        printf("Iteration %4d done!\n", iter);
        printf("The current residual RNorm = %f\n", RNorm);
        
        if (fabs(H[1][0]) < r0) break;    
        //if (rNorm < r0) break;
    }
    
    nit = iter;
    printf("\nThe final residual is %f\n\n", RNorm);
}

