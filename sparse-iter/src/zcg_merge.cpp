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

#define RTOLERANCE     1e-16
#define ATOLERANCE     1e-16

// uncomment for chronometry
#define ENABLE_TIMER
#define iterblock 1

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
    This is a GPU implementation of the Conjugate Gradient method in variant,
    where multiple operations are merged into one compute kernel.    

    Arguments
    =========

    magma_z_sparse_matrix A                   input matrix A
    magma_z_vector b                          RHS b
    magma_z_vector *x                         solution approximation
    magma_solver_parameters *solver_par       solver parameters

    ========================================================================  */

magma_int_t
magma_zcg_merge( magma_z_sparse_matrix A, magma_z_vector b, magma_z_vector *x,  
           magma_solver_parameters *solver_par ){

    // some useful variables
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO, c_one = MAGMA_Z_ONE, 
                                            c_mone = MAGMA_Z_NEG_ONE;
    magma_int_t dofs = A.num_rows;

    // GPU stream
    magma_queue_t stream[2];
    magma_event_t event[1];
    magma_queue_create( &stream[0] );
    magma_queue_create( &stream[1] );
    magma_event_create( &event[0] );

    // GPU workspace
    magma_z_vector r, d, z;
    magma_z_vinit( &r, Magma_DEV, dofs, c_zero );
    magma_z_vinit( &d, Magma_DEV, dofs, c_zero );
    magma_z_vinit( &z, Magma_DEV, dofs, c_zero );
    
    magmaDoubleComplex *d1, *d2, *skp;
    magma_zmalloc( &d1, dofs*(1) );
    magma_zmalloc( &d2, dofs*(1) );
    // array for the parameters
    magma_zmalloc( &skp, 6 );       // skp = [alpha|beta|gamma|rho|tmp1|tmp2]


    // solver variables
    magmaDoubleComplex alpha, beta, gamma, rho, tmp1, *skp_h;
    double nom, nom0, r0, betanom, den;
    magma_int_t i;

    // solver setup
    magma_zscal( dofs, c_zero, x->val, 1) ;                     // x = 0
    magma_zcopy( dofs, b.val, 1, r.val, 1 );                    // r = b
    magma_zcopy( dofs, b.val, 1, d.val, 1 );                    // d = b
    nom = magma_dznrm2( dofs, r.val, 1 );                       // nom = || r ||
    nom = nom * nom;
    nom0 = nom;                                                  // nom = r' * r
    magma_z_spmv( c_one, A, d, c_zero, z );                      // z = A d
    den = MAGMA_Z_REAL( magma_zdotc(dofs, d.val, 1, z.val, 1) ); // den = d'* z
    
    // array on host for the parameters
    magma_zmalloc_cpu( &skp_h, 6 );

    alpha = rho = gamma = tmp1 = c_one; 
    beta =  magma_zdotc(dofs, r.val, 1, r.val, 1);
    skp_h[0]=alpha; 
    skp_h[1]=beta; 
    skp_h[2]=gamma; 
    skp_h[3]=rho; 
    skp_h[4]=tmp1; 
    skp_h[5]=MAGMA_Z_MAKE(nom, 0.0);

    cudaMemcpy( skp, skp_h, 6*sizeof( magmaDoubleComplex ), 
                                            cudaMemcpyHostToDevice );
    
    if ( (r0 = nom * solver_par->epsilon) < ATOLERANCE ) 
        r0 = ATOLERANCE;
    if ( nom < r0 )
        return MAGMA_SUCCESS;
    // check positive definite
    if (den <= 0.0) {
        printf("Operator A is not postive definite. (Ar,r) = %f\n", den);
        return -100;
    }
    
    //Chronometry
    #ifdef ENABLE_TIMER
    double tempo1, tempo2;
    magma_device_sync(); tempo1=magma_wtime();
    printf("#===============================================#\n");
    printf("#   CG performance analysis ever %d iteration   #\n", iterblock);
    printf("#   iter   ||   residual-nrm2    ||   runtime    #\n");
    printf("#===============================================#\n");
    printf("      0    ||    %e    ||    0.0000      \n", nom);
    magma_device_sync(); tempo1=magma_wtime();
    #endif
    
    // start iteration
    for( solver_par->numiter= 1; i<solver_par->maxiter; solver_par->numiter++ ){

        magmablasSetKernelStream(stream[0]);
        
        // computes SpMV and dot product
        magma_zcgmerge_spmv1(  A.storage_type, dofs, A.max_nnz_row, d1, d2, 
                                    A.val, A.row, A.col, d.val, z.val, skp ); 
            
        // updates x, r, computes scalars and updates d
        magma_zcgmerge_xrbeta( dofs, d1, d2, x->val, r.val, d.val, z.val, skp ); 

        // check stopping criterion (asynchronous copy)
        cublasGetVectorAsync(1 , sizeof( magmaDoubleComplex ), skp+1, 1, 
                                                    skp_h+1, 1, stream[1] );
        betanom = sqrt(MAGMA_Z_REAL(skp_h[1]));
        if (  betanom  < r0 ) {
            break;
        }

        #ifdef ENABLE_TIMER
        //Chronometry  
        magma_device_sync(); tempo2=magma_wtime();
        if( solver_par->numiter%iterblock==0 ) {
            printf("   %4d    ||    %e    ||    %.4lf  \n", 
                (solver_par->numiter), betanom, tempo2-tempo1 );
        }
        #endif

    } 

    #ifdef ENABLE_TIMER
    double residual;
    magma_zresidual( A, b, *x, &residual );
    printf("#===============================================#\n");
    printf("# CG (merged) solver summary:\n");
    printf("#    initial residual: %e\n", nom0 );
    printf("#    iterations: %4d\n#    iterative residual: %e\n",
            (solver_par->numiter), betanom );
    printf("#    exact relative residual: %e\n#    runtime: %.4lf sec\n", 
                residual, tempo2-tempo1);
    printf("#===============================================#\n");
    #endif
        
    solver_par->residual = (double)(betanom);

    magma_z_vfree(&r);
    magma_z_vfree(&z);
    magma_z_vfree(&d);

    return MAGMA_SUCCESS;
}   /* magma_zcg_merge */


