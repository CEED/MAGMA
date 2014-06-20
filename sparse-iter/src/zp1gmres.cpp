/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

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

/**
    Purpose
    -------

    Solves a system of linear equations
       A * X = B
    where A is a complex sparse matrix stored in the GPU memory.
    X and B are complex vectors stored on the GPU memory. 
    This is a GPU implementation of the pipelined GMRES method
    proposed by P. Ghysels.

    Arguments
    ---------

    @param
    A           magma_z_sparse_matrix
                descriptor for matrix A

    @param
    b           magma_z_vector
                RHS b vector

    @param
    x           magma_z_vector*
                solution approximation

    @param
    solver_par  magma_z_solver_par*
                solver parameters

    @ingroup magmasparse_zgesv
    ********************************************************************/

magma_int_t
magma_zp1gmres( magma_z_sparse_matrix A, magma_z_vector b, magma_z_vector *x,  
              magma_z_solver_par *solver_par ){

#define  v(i)     (v.val + (i)*dofs)
#define  z(i)     (z.val + (i)*dofs)
#define  H(i,j)  H[(i)   + (j)*ldh]
#define HH(i,j) HH[(i)   + (j)*ldh]

    //Chronometry
    struct timeval inicio, fim;
    double tempo1, tempo2;
    double t_spmv1, t_spmv2, t_spmv = 0.0;
    double t_orth1, t_orth2, t_orth = 0.0;


    // local variables
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO, c_one = MAGMA_Z_ONE, c_mone = MAGMA_Z_NEG_ONE;
    magma_int_t dofs = A.num_rows;
    magma_int_t i, j, k, m, iter, ldh = solver_par->restart+1;
    double rNorm, RNorm, den, nom0, r0 = 0.;

    // CPU workspace
    magmaDoubleComplex *H, *HH, *y, *h1, *skp;
    magma_zmalloc_cpu( &H, (ldh+1)*ldh );
    magma_zmalloc_cpu( &HH, ldh*ldh );
    magma_zmalloc_cpu( &y, ldh );
    magma_zmalloc_cpu( &h1, ldh );
    magma_zmalloc_cpu( &skp, (ldh+1) ); 


    
    // GPU workspace
    magma_z_vector r, v, v_t, z, z_t, w;
    magma_z_vinit( &r  , Magma_DEV, dofs,     c_zero );
    magma_z_vinit( &v  , Magma_DEV, dofs*ldh, c_zero );
    magma_z_vinit( &v_t, Magma_DEV, dofs,     c_zero );
    magma_z_vinit( &z  , Magma_DEV, dofs*ldh, c_zero );
    magma_z_vinit( &z_t, Magma_DEV, dofs,     c_zero );
    magma_z_vinit( &w  , Magma_DEV, dofs,     c_zero );

    magmaDoubleComplex *dy, *skp_d;
    if (MAGMA_SUCCESS != magma_zmalloc( &dy, ldh )) 
        return MAGMA_ERR_DEVICE_ALLOC;
    if (MAGMA_SUCCESS != magma_zmalloc( &skp_d, ldh+1 )) 
        return MAGMA_ERR_DEVICE_ALLOC;
    
    magma_zscal( dofs, c_zero, x->val, 1 );              //  x = 0
    magma_zcopy( dofs, b.val, 1, r.val, 1 );             //  r = b

    r0 = magma_dznrm2( dofs, r.val, 1 );                 //  r0= || r||
    nom0 = r0*r0;
    H(1,0) = MAGMA_Z_MAKE( r0, 0. ); 

    magma_zcopy( dofs, r.val, 1, v(0), 1 );             //  r = b
    magma_zscal(dofs, 1./H(1,0), v(0), 1);                     //  (to be fused)

    magma_zcopy( dofs, r.val, 1, z(0), 1 );             //  r = b
    magma_zscal(dofs, 1./H(1,0), z(0), 1);                     //  (to be fused)

    if ((r0 *= solver_par->epsilon) < ATOLERANCE) 
        r0 = ATOLERANCE;
    
    //Chronometry 
    magma_device_sync();
    gettimeofday(&inicio, NULL);
    tempo1=inicio.tv_sec+(inicio.tv_usec/1000000.0);

    printf("Iteration: %4d  Norm: %e  Time: %e\n", 0, H(1,0)*H(1,0), 0.0);



    for (iter = 0; iter<solver_par->maxiter; iter++) 
        {


            for(k=1; k<=(solver_par->restart); k++) 
                {
                    magma_zcopy(dofs, r.val, 1, v(k-1), 1);                        //  q[k]    = 1.0/H[k][k-1] r
                    magma_zscal(dofs, 1./H(k,k-1), v(k-1), 1);                     //  (to be fused)

                    v_t.val = v(k-1);
                    magma_device_sync();
                    gettimeofday(&inicio, NULL);
                    t_spmv1=inicio.tv_sec+(inicio.tv_usec/1000000.0);
                    magma_z_spmv( c_one, A, v_t, c_zero, r );                    //  r       = A q[k] 
                    magma_device_sync();
                    gettimeofday(&inicio, NULL);
                    t_spmv2=inicio.tv_sec+(inicio.tv_usec/1000000.0);
                    t_spmv += t_spmv2-t_spmv1;

                    magma_device_sync();
                    gettimeofday(&inicio, NULL);
                    t_orth1=inicio.tv_sec+(inicio.tv_usec/1000000.0);

                    double tmp=0, zz=0;
                    for (i=1; i<=k; i++) {
                        skp[i-1] = H(i,k) =magma_zdotc(dofs, v(i-1), 1, r.val, 1);            //  H[i][k] = q[i] . r
                        tmp+=  MAGMA_Z_REAL( H(i,k)*H(i,k) );
                    }
                    zz = MAGMA_Z_REAL(magma_zdotc(dofs, r.val, 1, r.val, 1));  
                    skp[k] = H(k+1,k) = MAGMA_Z_MAKE( sqrt(zz-tmp),0.);

                    magma_zsetvector( k+1 , skp, 1, skp_d, 1 );

                    magma_zp1gmres_mgs(dofs, k, skp_d, v.val, r.val);       // block orthogonalization      
                    magma_device_sync();
                    gettimeofday(&inicio, NULL);
                    t_orth2=inicio.tv_sec+(inicio.tv_usec/1000000.0);
                    t_orth += t_orth2-t_orth1;       
            }

            for(k=1; k<(solver_par->restart)+1; k++){
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
            
            /*   Update the current approximation: x += V y  */
            magma_zsetmatrix(m, 1, y+1, m, dy, m);
            magma_zgemv(MagmaNoTrans, dofs, m, c_one, v(0), dofs, dy, 1, c_one, x->val, 1); 

            magma_device_sync();
            gettimeofday(&inicio, NULL);
            t_spmv1=inicio.tv_sec+(inicio.tv_usec/1000000.0);
            magma_z_spmv( c_mone, A, *x, c_zero, r );                  //  r = - A * x
            magma_device_sync();
            gettimeofday(&inicio, NULL);
            t_spmv2=inicio.tv_sec+(inicio.tv_usec/1000000.0);
            t_spmv += t_spmv2 - t_spmv1;

            magma_zaxpy(dofs, c_one, b.val, 1, r.val, 1);              //  r = r + b
            H(1,0) = MAGMA_Z_MAKE( magma_dznrm2(dofs, r.val, 1), 0. ); //  RNorm = H[0][0] = || v(k) ||
            RNorm = MAGMA_Z_REAL( H(1,0) );
            
            //Chronometry  
            magma_device_sync();
            gettimeofday(&fim, NULL);
            tempo2=fim.tv_sec+(fim.tv_usec/1000000.0);

            printf("Iteration: %4d  Norm: %e  Time: %.2lf, Orth: %.2lf (%.2lf\%) SpMV: %.2lf (%.2lf\%)\n", iter+1, RNorm*RNorm, tempo2-tempo1, 
                   t_orth, t_orth/(tempo2-tempo1), t_spmv,t_spmv/(tempo2-tempo1));

                
            if (fabs(RNorm*RNorm) < r0) break;    
            //if (rNorm < r0) break;
        }
    
    
    printf( "      (r_0, r_0) = %e\n", nom0 );
    printf( "      (r_N, r_N) = %e\n", RNorm*RNorm);
    printf( "      Number of GMRES restarts: %d\n", iter);
    
    if (solver_par->epsilon == RTOLERANCE) {
        magma_z_spmv( c_one, A, *x, c_zero, r );                       // v_t = A x
        magma_zaxpy(dofs,  c_mone, b.val, 1, r.val, 1);                // r = r - b
        den = magma_dznrm2(dofs, r.val, 1);                            // den = || r ||
        printf( "      || r_N ||   = %f\n", den);
        solver_par->residual = (double)(den);
    }
    solver_par->numiter = iter;

    magma_free(dy); 

    magma_z_vfree(&r);
    magma_z_vfree(&w);
    magma_z_vfree(&v);
    magma_z_vfree(&v_t);
    magma_z_vfree(&z);
    magma_z_vfree(&z_t);
    magma_free(skp_d);
    free(skp);

    return MAGMA_SUCCESS;
}   /* magma_zgmres */

