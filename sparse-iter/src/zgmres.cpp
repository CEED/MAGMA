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
#include <sys/time.h>
#include <time.h>

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

    =====================================================================  */

magma_int_t
magma_zgmres( magma_z_sparse_matrix A, magma_z_vector b, magma_z_vector *x,  
              magma_solver_parameters *solver_par ){

#define  q(i)     (q.val + (i)*dofs)
#define  H(i,j)  H[(i)   + (j)*ldh]
#define HH(i,j) HH[(i)   + (j)*ldh]
#define dH(i,j)  (dH+ (i) + (j)*ldh)

    //Chronometry
    double tempo1, tempo2;

    // local variables
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO, c_one = MAGMA_Z_ONE, c_mone = MAGMA_Z_NEG_ONE;
    magma_int_t dofs = A.num_rows;
    magma_int_t i, j, k, m, iter, ldh = solver_par->restart+1;
    double rNorm, RNorm, den, nom0, r0 = 0.;

    // CPU workspace
    magmaDoubleComplex *H, *HH, *y, *h1;
    magma_zmalloc_cpu( &H, (ldh+1)*ldh );
    magma_zmalloc_cpu( &HH, ldh*ldh );
    magma_zmalloc_cpu( &y, ldh );
    magma_zmalloc_cpu( &h1, ldh );
    
    // GPU workspace
    magma_z_vector r, q, q_t;
    magma_z_vinit( &r  , Magma_DEV, dofs,     c_zero );
    magma_z_vinit( &q  , Magma_DEV, dofs*ldh, c_zero );
    magma_z_vinit( &q_t, Magma_DEV, dofs,     c_zero );

    magmaDoubleComplex *dy, *dH = NULL;
    if (MAGMA_SUCCESS != magma_zmalloc( &dy, ldh )) 
        return MAGMA_ERR_DEVICE_ALLOC;
    if (solver_par->ortho != MAGMA_MGS ) {
        if (MAGMA_SUCCESS != magma_zmalloc( &dH, (ldh+1)*ldh )) 
            return MAGMA_ERR_DEVICE_ALLOC;
    }

    magma_zscal( dofs, c_zero, x->val, 1 );              //  x = 0
    magma_zcopy( dofs, b.val, 1, r.val, 1 );             //  r = b

    r0 = magma_dznrm2( dofs, r.val, 1 );                 //  r0= || r||
    nom0 = r0*r0;
    H(1,0) = MAGMA_Z_MAKE( r0, 0. ); 

    if ((r0 *= solver_par->epsilon) < ATOLERANCE) 
        r0 = ATOLERANCE;
    
    //Chronometry 
    magma_device_sync(); tempo1=magma_wtime();
    //#define ENABLE_TIMER
    #ifdef ENABLE_TIMER
    printf("Iteration: %4d  Norm: %e  Time: %e\n", 0, MAGMA_Z_REAL(H(1,0)*H(1,0)), 0.0);
    double t_spmv1, t_spmv2, t_spmv = 0.0;
    double t_orth1, t_orth2, t_orth = 0.0;
    #endif

    for (iter = 0; iter<solver_par->maxiter; iter++) 
        {
            for(k=1; k<=(solver_par->restart); k++) 
                {
                    magma_zcopy(dofs, r.val, 1, q(k-1), 1);                        //  q[k]    = 1.0/H[k][k-1] r
                    magma_zscal(dofs, 1./H(k,k-1), q(k-1), 1);                     //  (to be fused)

                    #ifdef ENABLE_TIMER
                    magma_device_sync(); t_spmv1=magma_wtime();
                    #endif
                    q_t.val = q(k-1);
                    magma_z_spmv( c_one, A, q_t, c_zero, r );                      //  r       = A q[k] 
                    #ifdef ENABLE_TIMER
                    magma_device_sync(); t_spmv2=magma_wtime();
                    t_spmv += t_spmv2-t_spmv1;
                    #endif
                    
                    #ifdef ENABLE_TIMER
                    magma_device_sync(); t_orth1=magma_wtime();
                    #endif
                    if (solver_par->ortho == MAGMA_MGS ) {
                        // modified Gram-Schmidt
                        for (i=1; i<=k; i++) {
                            H(i,k) =magma_zdotc(dofs, q(i-1), 1, r.val, 1);            //  H[i][k] = q[i] . r
                            magma_zaxpy(dofs,-H(i,k), q(i-1), 1, r.val, 1);            //  r       = r - H[i][k] q[i]
                        }
                        H(k+1,k) = MAGMA_Z_MAKE( magma_dznrm2(dofs, r.val, 1), 0. );   //  H[k+1][k] = sqrt(r . r) 
                    } else if (solver_par->ortho == MAGMA_FUSED_CGS ) {
                        // fusing zgemv with dznrm2 in classical Gram-Schmidt
                        magma_zcopy(dofs, r.val, 1, q(k), 1);  
                        magma_zgemv(MagmaTrans,   dofs, k+1, c_one,  q(0), dofs, r.val,   1, c_zero, dH(1,k), 1);
                        magma_zgemv(MagmaNoTrans, dofs, k, c_mone, q(0), dofs, dH(1,k), 1, c_one,  r.val,   1);
                        magma_zgetvector(k+1, dH(1,k), 1, &H(1,k), 1);
                        for (i=1; i<=k; i++) H(k+1,k) -= H(i,k)*H(i,k);
                        H(k+1,k) = MAGMA_Z_MAKE( sqrt( MAGMA_Z_REAL(H(k+1,k))), 0 );
                    } else {
                        // classical Gram-Schmidt (default)
                        magma_zgemv(MagmaTrans,   dofs, k, c_one,  q(0), dofs, r.val,   1, c_zero, dH(1,k), 1);
                        magma_zgemv(MagmaNoTrans, dofs, k, c_mone, q(0), dofs, dH(1,k), 1, c_one,  r.val,   1);
                        magma_zgetvector(k, dH(1,k), 1, &H(1,k), 1);
                        H(k+1,k) = MAGMA_Z_MAKE( magma_dznrm2(dofs, r.val, 1), 0. );   //  H[k+1][k] = sqrt(r . r) 
                    }
                    #ifdef ENABLE_TIMER
                    magma_device_sync(); t_orth2=magma_wtime();
                    t_orth += t_orth2-t_orth1;
                    #endif
                }
            for(k=1; k<=(solver_par->restart); k++) 
                {
                    /*     Minimization of  || b-Ax ||  in H_k       */ 
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
            magma_zgemv(MagmaNoTrans, dofs, m, c_one, q(0), dofs, dy, 1, c_one, x->val, 1); 

            #ifdef ENABLE_TIMER
            magma_device_sync(); t_spmv1=magma_wtime();
            #endif
            magma_z_spmv( c_mone, A, *x, c_zero, r );                  //  r = - A * x
            #ifdef ENABLE_TIMER
            magma_device_sync(); t_spmv2=magma_wtime();
            t_spmv += t_spmv2 - t_spmv1;
            #endif

            magma_zaxpy(dofs, c_one, b.val, 1, r.val, 1);              //  r = r + b
            H(1,0) = MAGMA_Z_MAKE( magma_dznrm2(dofs, r.val, 1), 0. ); //  RNorm = H[1][0] = || r ||
            RNorm = MAGMA_Z_REAL( H(1,0) );
            
            #ifdef ENABLE_TIMER
            //Chronometry  
            magma_device_sync(); tempo2=magma_wtime();
            printf("Iteration: %4d  Norm: %e  Time: %.2lf, Orth: %.2lf (%.2lf%%) SpMV: %.2lf (%.2lf%%)\n", iter+1, RNorm*RNorm, tempo2-tempo1, 
                   t_orth, t_orth/(tempo2-tempo1), t_spmv,t_spmv/(tempo2-tempo1));
            #endif
            
            if (fabs(RNorm*RNorm) < r0) break;    
            //if (rNorm < r0) break;
        }
    
  
    printf( "\n" );
    printf( "      (r_0, r_0) = %e\n", nom0 );
    printf( "      (r_N, r_N) = %e\n", RNorm*RNorm);
    printf( "      Number of GMRES restarts: %d\n", iter);
    #ifndef ENABLE_TIMER
    magma_device_sync(); tempo2=magma_wtime();
    printf( "      Time: %.2lf\n",tempo2-tempo1 );
    #endif
    if (solver_par->epsilon == RTOLERANCE) {
        magma_z_spmv( c_one, A, *x, c_zero, r );                       // q_t = A x
        magma_zaxpy(dofs,  c_mone, b.val, 1, r.val, 1);                // r = r - b
        den = magma_dznrm2(dofs, r.val, 1);                            // den = || r ||
        printf( "      || r_N ||   = %f\n", den);
        solver_par->residual = (double)(den);
    }
    printf( "\n" );
    solver_par->numiter = iter;

    magma_free(dy); 
    if (dH != NULL ) magma_free(dH); 
    magma_z_vfree(&r);
    magma_z_vfree(&q);


    return MAGMA_SUCCESS;
}   /* magma_zgmres */

