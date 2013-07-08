/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @author Hartwig Anzt

       @precisions mixed z -> s d c
*/

#include "common_magma.h"
#include "../include/magmasparse.h"

#include <cblas.h>

#define PRECISION_z

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
    where A is a complex sparse matrix stored in the GPU memory.
    X and B are complex vectors stored on the GPU memory. 
    This is a GPU implementation of the GMRES method.

    Arguments
    =========

    magma_z_sparse_matrix A                   descriptor for matrix A
    magma_z_vector b                          RHS b vector
    magma_z_vector *x                         solution approximation
    magma_solver_parameters *solver_par       solver parameters
    magma_precond_parameters *precond_par     preconditioner parameters

    =====================================================================  */

magma_int_t
magma_zcpgmres( magma_z_sparse_matrix A, magma_z_vector b, magma_z_vector *x,  
              magma_solver_parameters *solver_par,  magma_precond_parameters *precond_par) 
{


#define  q(i)     (q.val + (i)*dofs)
#define  z(i)     (z.val + (i)*dofs)
#define  H(i,j)  H[(i)   + (j)*ldh]
#define HH(i,j) HH[(i)   + (j)*ldh]

    // local variables
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO, c_one = MAGMA_Z_ONE, c_mone = MAGMA_Z_NEG_ONE;
    magma_int_t dofs = A.num_rows;
    magma_int_t i, j, k, m, iter, ldh = solver_par->restart+1;
    double rNorm, RNorm, den, nom0, r0 = 0.;

    // CPU workspace
    magmaDoubleComplex H[(ldh+1)*ldh], HH[ldh*ldh]; 
    magmaDoubleComplex  y[ldh], h1[ldh];
    
    // GPU workspace
    magma_z_vector r, q, q_t, z, z_t;
    magma_z_vinit( &r  , Magma_DEV, dofs,     c_zero );
    magma_z_vinit( &q  , Magma_DEV, dofs*ldh, c_zero );
    magma_z_vinit( &q_t, Magma_DEV, dofs,     c_zero );
    magma_z_vinit( &z  , Magma_DEV, dofs*ldh, c_zero );
    magma_z_vinit( &z_t, Magma_DEV, dofs,     c_zero );
    // for mixed precision on GPU
    magma_c_vector qs_t, zs_t;
    magma_c_sparse_matrix AS;
    magma_sparse_matrix_zlag2c( A, &AS );


    magmaDoubleComplex *dy;
    if (MAGMA_SUCCESS != magma_zmalloc( &dy, ldh )) 
        return MAGMA_ERR_DEVICE_ALLOC;
    
    magma_zscal( dofs, c_zero, x->val, 1 );              //  x = 0
    magma_zcopy( dofs, b.val, 1, r.val, 1 );             //  r = b

    r0 = magma_dznrm2( dofs, r.val, 1 );                 //  r0= || r||
    nom0 = r0*r0;
    H(1,0) = MAGMA_Z_MAKE( r0, 0. ); 

    if ((r0 *= solver_par->epsilon) < ATOLERANCE) 
        r0 = ATOLERANCE;
    
    printf("Iteration : %4d  Norm: %f\n", 0, H(1,0)*H(1,0));

    for (iter = 0; iter<solver_par->maxiter; iter++) 
        {
            for(k=1; k<=(solver_par->restart); k++) 
                {
                    magma_zcopy(dofs, r.val, 1, q(k), 1);                        //  q[k]    = 1.0/H[k][k-1] r
                    magma_zscal(dofs, 1./H(k,k-1), q(k), 1);                     //  (to be fused)

                    q_t.val = q(k);
                    /*
                     * Convert to single precision
                     */
                    magma_vector_zlag2c(q_t, &qs_t);

                    magma_c_precond( AS, qs_t, &zs_t, *precond_par );  // preconditioner AS * zs =  qs[k]

                    /*
                     * Convert to double precision
                     */
                    magma_vector_clag2z(zs_t, &z_t);

                    z_t.val = z(k);
                    magma_z_spmv( c_one, A, z_t, c_zero, r );                    //  r       = A q[k] 
                    
                    for (i=1; i<=k; i++) {
                        H(i,k) =magma_zdotc(dofs, q(i), 1, r.val, 1);            //  H[i][k] = q[i] . r
                        magma_zaxpy(dofs,-H(i,k), q(i), 1, r.val, 1);            //  r       = r - H[i][k] q[i]
                    }
                    
                    H(k+1,k) = MAGMA_Z_MAKE( magma_dznrm2(dofs, r.val, 1), 0. ); //  H[k+1][k] = sqrt(r . r) 
                    
                    /*     Minimization of  || b-Ax ||  in K_k       */ 
                    for (i=1; i<=k; i++) {
                        #if defined(PRECISION_z) || defined(PRECISION_c)
                        cblas_zdotc_sub( i+1, &H(1,k), 1, &H(1,i), 1, &HH(k,i) );
                        #else
                        HH(k,i) = cblas_zdotc(i+1, &H(1,k), 1, &H(1,i), 1);
                        #endif
                    }
                    
                    h1[k] = H(1,k)*H(1,0);
                    
                    if (k != 1)
                        for (i=1; i<k; i++) {
                            for (m=i+1; m<k; m++)
                                HH(k,m) -= HH(k,i) * HH(m,i);
                           
                            HH(k,k) -= HH(k,i) * HH(k,i) / HH(i,i);
                            HH(k,i) = HH(k,i)/HH(i,i);
                            h1[k] -= h1[i] * HH(k,i);   
                        }    
                    y[k] = h1[k]/HH(k,k); 
                    if (k != 1)  
                        for (i=k-1; i>=1; i--) {
                            y[i] = h1[i]/HH(i,i);
                            for (j=i+1; j<=k; j++)
                                y[i] -= y[j] * HH(j,i);
                        }
                    
                    m = k;
                    
                    rNorm = fabs(MAGMA_Z_REAL(H(k+1,k)));
                    //if (rNorm < r0) break;
                }
            
            /*   Update the current approximation: x += Q y  */
            magma_zsetmatrix(m, 1, y+1, m, dy, m);
            magma_zgemv(MagmaNoTrans, dofs, m, c_one, z(1), dofs, dy, 1, c_one, x->val, 1); 

            magma_z_spmv( c_mone, A, *x, c_zero, r );                  //  r = - A * x
            magma_zaxpy(dofs, c_one, b.val, 1, r.val, 1);              //  r = r + b
            H(1,0) = MAGMA_Z_MAKE( magma_dznrm2(dofs, r.val, 1), 0. ); //  RNorm = H[1][0] = || r ||
            RNorm = MAGMA_Z_REAL( H(1,0) );
            
            printf("Iteration : %4d  Norm: %f\n", iter, RNorm*RNorm);
            
            if (fabs(RNorm*RNorm) < r0) break;    
            //if (rNorm < r0) break;
        }
    
    
    printf( "      (r_0, r_0) = %e\n", nom0 );
    printf( "      (r_N, r_N) = %e\n", RNorm*RNorm);
    printf( "      Number of GMRES restarts: %d\n", iter);
    
    if (solver_par->epsilon == RTOLERANCE) {
        magma_z_spmv( c_one, A, *x, c_zero, r );                       // r = A x
        magma_zaxpy(dofs,  c_mone, b.val, 1, r.val, 1);                // r = r - b
        den = magma_dznrm2(dofs, r.val, 1);                            // den = || r ||
        printf( "      || r_N ||   = %f\n", den);
        solver_par->residual = (double)(den);
    }
    solver_par->numiter = iter;

    magma_free(dy); 

    magma_z_vfree(&r);
    magma_z_vfree(&q);
    magma_z_vfree(&q_t);
    magma_z_vfree(&z);
    magma_z_vfree(&z_t);

    return MAGMA_SUCCESS;
}   /* magma_zgmres */

