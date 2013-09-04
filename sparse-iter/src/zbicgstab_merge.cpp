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
#include <mkl_cblas.h>
#include <assert.h>

#define RTOLERANCE     10e-16
#define ATOLERANCE     10e-16

#define  q(i)     (q.val + (i)*dofs)

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
    This is a GPU implementation of the Biconjugate Gradient Stabelized method.
    The difference to magma_zbicgstab is that we use specifically designed kernels
    merging multiple operations into one kernel.

    Arguments
    =========

    magma_z_sparse_matrix A                   input matrix A
    magma_z_vector b                          RHS b
    magma_z_vector *x                         solution approximation
    magma_solver_parameters *solver_par       solver parameters

    =====================================================================  */


magma_int_t
magma_zbicgstab_merge( magma_z_sparse_matrix A, magma_z_vector b, magma_z_vector *x,  
           magma_solver_parameters *solver_par )
{

    // some useful variables
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO, c_one = MAGMA_Z_ONE, c_mone = MAGMA_Z_NEG_ONE;
    
    magma_int_t dofs = A.num_rows;

    // workspace
    magma_z_vector q, r,rr,p,v,s,t;
    magmaDoubleComplex *d1, *d2, *skp;
    magma_zmalloc( &d1, dofs*(2) );
    magma_zmalloc( &d2, dofs*(2) );

    magma_zmalloc( &skp, 18 );       // skp = [alpha|beta|omega|rho_old|rho|nom|tmp1|tmp2]
    magma_z_vinit( &q, Magma_DEV, dofs*6, c_zero );
    // q = rr|r|p|v|s|t

    rr.memory_location = Magma_DEV; rr.val = NULL; rr.num_rows = rr.nnz = dofs;
    r.memory_location = Magma_DEV; r.val = NULL; r.num_rows = r.nnz = dofs;
    p.memory_location = Magma_DEV; p.val = NULL; p.num_rows = p.nnz = dofs;
    v.memory_location = Magma_DEV; v.val = NULL; v.num_rows = v.nnz = dofs;
    s.memory_location = Magma_DEV; s.val = NULL; s.num_rows = s.nnz = dofs;
    t.memory_location = Magma_DEV; t.val = NULL; t.num_rows = t.nnz = dofs;

    rr.val = q(0);
    r.val = q(1);
    p.val = q(2);
    v.val = q(3);
    s.val = q(4);
    t.val = q(5);
    
    // solver variables
    magmaDoubleComplex alpha, beta, omega, rho_old, rho_new, *skp_h;
    double nom, nom0, r0, den;
    magma_int_t i;

    magma_zmalloc( &skp, 8 );       // skp = [alpha|beta|omega|rho_old|rho|nom|tmp1|tmp2]


    // solver setup
    magma_zscal( dofs, c_zero, x->val, 1) ;                            // x = 0
    magma_zcopy( dofs, b.val, 1, q(0), 1 );                            // rr = b
    magma_zcopy( dofs, b.val, 1, q(1), 1 );                            // r = b

    rho_new = magma_zdotc( dofs, r.val, 1, r.val, 1 );           // rho=<rr,r>
    nom0 = nom = MAGMA_Z_REAL(magma_zdotc( dofs, r.val, 1, r.val, 1 ));           // nom=<r,r>

    rho_old = omega = alpha = MAGMA_Z_MAKE( 1.0, 0. );
    beta = rho_new;
    (solver_par->numiter) = 0;
    
    magma_zmalloc_cpu( &skp_h, 8 );
    skp_h[0]=alpha; skp_h[1]=beta; skp_h[2]=omega; skp_h[3]=rho_old; skp_h[4]=rho_new; skp_h[5]=MAGMA_Z_MAKE(nom, 0.0);
    cudaMemcpy( skp, skp_h, 8*sizeof( magmaDoubleComplex ), cudaMemcpyHostToDevice );

    magma_z_spmv( c_one, A, r, c_zero, v );                           // z = A r
    den = MAGMA_Z_REAL( magma_zdotc(dofs, v.val, 1, r.val, 1) );      // den = z dot r

    if ( (r0 = nom * solver_par->epsilon) < ATOLERANCE ) 
        r0 = ATOLERANCE;
    if ( nom < r0 )
        return MAGMA_SUCCESS;
    
    if (den <= 0.0) {
        printf("Operator A is not postive definite. (Ar,r) = %f\n", den);
        return -100;
    }

    //Chronometry
    #define ENABLE_TIMER
    #ifdef ENABLE_TIMER
    double t_spmv1, t_spmv = 0.0;
    double tempo1, tempo2;
    magma_device_sync(); tempo1=magma_wtime();
        printf("Iteration: %4d  Norm: %e  Time: %.2lf  SpMV: %.2lf %.2lf%%  Rest: %.2lf\n", 
                    (solver_par->numiter), nom, 0.0, 0.0, 0.0 );
    #endif

    // start iteration
    while( solver_par->numiter < solver_par->maxiter ){
        magma_zbicgmerge1( dofs, skp, v.val, r.val, p.val );            // merge: p=r+beta*(p-omega*v)
            #ifdef ENABLE_TIMER
            magma_device_sync(); t_spmv1=magma_wtime();
            #endif
        magma_z_spmv( c_one, A, p, c_zero, v );                                 // v = Ap
            #ifdef ENABLE_TIMER
            magma_device_sync(); t_spmv += magma_wtime() - t_spmv1;
            #endif
        magma_zmdotc( dofs, 1, q.val, v.val, d1, d2, skp );                     
        magma_zbicgmerge4(  1, skp );
        magma_zbicgmerge2( dofs, skp, r.val, v.val, s.val );                  // s=r-alpha*v
            #ifdef ENABLE_TIMER
            magma_device_sync(); t_spmv1=magma_wtime();
            #endif
        magma_z_spmv( c_one, A, s, c_zero, t );                                  // t=As
            #ifdef ENABLE_TIMER
            magma_device_sync(); t_spmv += magma_wtime() - t_spmv1;
            #endif
        magma_zmdotc( dofs, 2, q.val+4*dofs, t.val, d1, d2, skp+6 );
        magma_zbicgmerge4(  2, skp );
        magma_zbicgmerge3( dofs, skp, p.val, s.val, t.val,              // x=x+alpha*p+omega*s
                            x->val, r.val );                                     // r=s-omega*t
        magma_zmdotc( dofs, 2, q.val, r.val, d1, d2, skp+4);
        magma_zbicgmerge4(  3, skp );

        (solver_par->numiter)++;

        #ifdef ENABLE_TIMER
        //Chronometry  
        magma_device_sync(); tempo2=magma_wtime();
        cublasGetVector(1 , sizeof( magmaDoubleComplex ), skp+5, 1, skp_h+5, 1 );
        nom = MAGMA_Z_REAL(skp_h[5]);
        if( solver_par->numiter==1000 ) 
        printf("Iteration: %4d  Norm: %e  Time: %.2lf  SpMV: %.2lf %.2lf%%  Rest: %.2lf\n", 
                    (solver_par->numiter), nom, tempo2-tempo1, t_spmv, 100.0*t_spmv/(tempo2-tempo1), tempo2-tempo1-t_spmv);
        #endif
    }

    printf( "      (r_0, r_0) = %e\n", nom0 );
    printf( "      (r_N, r_N) = %e\n", nom );
    printf( "      Number of BICGSTAB iterations: %d\n", (solver_par->numiter));

    
    if (solver_par->epsilon == RTOLERANCE) {
        magma_z_spmv( c_one, A, *x, c_zero, r );                       // z = A d
        magma_zaxpy(dofs,  c_mone, b.val, 1, r.val, 1);                // r = r - b
        den = magma_dznrm2(dofs, r.val, 1);                            // den = || r ||
        printf( "      || r_N ||   = %f\n", den);
        solver_par->residual = (double)(den);
    }

    magma_z_vfree(&q);
  
    magma_free(d1);
    magma_free(d2);

    return MAGMA_SUCCESS;
}


