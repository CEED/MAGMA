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

#define RTOLERANCE     10e-16
#define ATOLERANCE     10e-16


magma_int_t
magma_zcg_merge( magma_z_sparse_matrix A, magma_z_vector b, magma_z_vector *x,  
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

    // some useful variables
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO, c_one = MAGMA_Z_ONE, c_mone = MAGMA_Z_NEG_ONE;
    
    magma_int_t dofs = A.num_rows;

    // GPU stream
    magma_queue_t stream[3];
    magma_event_t event[1];
    magma_queue_create( &stream[0] );
    magma_queue_create( &stream[1] );
    magma_queue_create( &stream[2] );
    magma_event_create( &event[0] );


    // GPU workspace
    magma_z_vector r, d, z;
    magma_z_vinit( &r, Magma_DEV, dofs, c_zero );
    magma_z_vinit( &d, Magma_DEV, dofs, c_zero );
    magma_z_vinit( &z, Magma_DEV, dofs, c_zero );
    
    magmaDoubleComplex *d1, *d2, *skp;
    magma_zmalloc( &d1, dofs*(1) );
    magma_zmalloc( &d2, dofs*(1) );

    magma_zmalloc( &skp, 6 );       // skp = [alpha|beta|gamma|rho|tmp1|tmp2]


    // solver variables
    magmaDoubleComplex alpha, beta, gamma, rho, tmp1, *skp_h;
    double nom, nom0, r0, betanom, den;
    magma_int_t i;

    // solver setup
    magma_zscal( dofs, c_zero, x->val, 1) ;                         // x = 0
    magma_zcopy( dofs, b.val, 1, r.val, 1 );                          // r = b
    magma_zcopy( dofs, b.val, 1, d.val, 1 );                          // d = b
    nom = magma_dznrm2( dofs, r.val, 1 );                             // nom = || r ||
    nom = nom * nom;
    nom0 = nom;                                                       // nom = r dot r
    magma_z_spmv( c_one, A, d, c_zero, z );                           // z = A d
    den = MAGMA_Z_REAL( magma_zdotc(dofs, d.val, 1, z.val, 1) );      // den = d dot z

    magma_zmalloc_cpu( &skp_h, 6 );

    alpha = rho = gamma = tmp1 = c_one; 
    beta =  magma_zdotc(dofs, r.val, 1, r.val, 1);
    skp_h[0]=alpha; skp_h[1]=beta; skp_h[2]=gamma; skp_h[3]=rho; skp_h[4]=tmp1; skp_h[5]=MAGMA_Z_MAKE(nom, 0.0);
    cudaMemcpy( skp, skp_h, 6*sizeof( magmaDoubleComplex ), cudaMemcpyHostToDevice );

    
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
   //     printf("Iteration: %4d  Norm: %e  Time: %.2lf  SpMV: %.2lf %.2lf%%  Rest: %.2lf\n", 
   //                 (solver_par->numiter), nom, 0.0, 0.0, 0.0 );
    #endif
    
    // start iteration
    for( i= 1; i<solver_par->maxiter; i++ ) {

        magmablasSetKernelStream(stream[0]);

        magma_zcgmerge_spmv1(  A.storage_type, dofs, A.max_nnz_row, d1, d2, A.val, A.row, A.col, d.val, z.val, skp ); 


        magma_zcgmerge_xrbeta( dofs, d1, d2, x->val, r.val, d.val, z.val, skp ); 


        (solver_par->numiter)++;


        #ifdef ENABLE_TIMER
        //Chronometry  
        magma_device_sync(); tempo2=magma_wtime();
        //cublasGetVector(1 , sizeof( magmaDoubleComplex ), skp+5, 1, skp_h+5, 1 );
        cublasGetVectorAsync(1 , sizeof( magmaDoubleComplex ), skp+1, 1, skp_h+1, 1, stream[1] );
        nom = MAGMA_Z_REAL(skp_h[1]);
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
    printf( "      (r_N, r_N) = %e\n", nom);
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
    magma_z_vfree(&z);
    magma_z_vfree(&d);

    return MAGMA_SUCCESS;
}   /* magma_zcg_merge */


