/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Hartwig Anzt

*/
#include "common_magma.h"
#include "../include/magmasparse.h"
#include <mkl_cblas.h>
#include <assert.h>

#define RTOLERANCE     1e-16
#define ATOLERANCE     1e-16

#define  q(i)     (q.val + (i)*dofs)

// uncomment for chronometry
#define ENABLE_TIMER
#define iterblock 1

/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

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
magma_zbicgstab_merge2( magma_z_sparse_matrix A, magma_z_vector b, 
        magma_z_vector *x, magma_solver_parameters *solver_par ){

    // some useful variables
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO, c_one = MAGMA_Z_ONE;
    
    magma_int_t dofs = A.num_rows;

    // GPU stream
    magma_queue_t stream[2];
    magma_event_t event[1];
    magma_queue_create( &stream[0] );
    magma_queue_create( &stream[1] );
    magma_event_create( &event[0] );

    // workspace
    magma_z_vector q, r,rr,p,v,s,t;
    magmaDoubleComplex *d1, *d2, *skp;
    magma_zmalloc( &d1, dofs*(2) );
    magma_zmalloc( &d2, dofs*(2) );

    // array for the parameters
    magma_zmalloc( &skp, 8 );     
    // skp = [alpha|beta|omega|rho_old|rho|nom|tmp1|tmp2]  
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
    double nom, nom0, betanom, r0, den;

    // solver setup
    magma_zscal( dofs, c_zero, x->val, 1) ;                            // x = 0
    magma_zcopy( dofs, b.val, 1, q(0), 1 );                            // rr = b
    magma_zcopy( dofs, b.val, 1, q(1), 1 );                            // r = b

    rho_new = magma_zdotc( dofs, r.val, 1, r.val, 1 );           // rho=<rr,r>
    nom = MAGMA_Z_REAL(magma_zdotc( dofs, r.val, 1, r.val, 1 ));    
    nom0 = sqrt(nom);                                       // nom = || r ||   
    rho_old = omega = alpha = MAGMA_Z_MAKE( 1.0, 0. );
    beta = rho_new;
    (solver_par->numiter) = 0;
    // array on host for the parameters 
    magma_zmalloc_cpu( &skp_h, 8 );
    skp_h[0]=alpha; 
    skp_h[1]=beta; 
    skp_h[2]=omega; 
    skp_h[3]=rho_old; 
    skp_h[4]=rho_new; 
    skp_h[5]=MAGMA_Z_MAKE(nom, 0.0);
    cudaMemcpy( skp, skp_h, 8*sizeof( magmaDoubleComplex ), 
                                            cudaMemcpyHostToDevice );

    magma_z_spmv( c_one, A, r, c_zero, v );                     // z = A r
    den = MAGMA_Z_REAL( magma_zdotc(dofs, v.val, 1, r.val, 1) );// den = z dot r

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
    real_Double_t tempo1, tempo2;
    magma_device_sync(); tempo1=magma_wtime();
    printf("#=============================================================#\n");
    printf("#   BiCGStab (merged2) performance analysis every %d iteration\n",
                                                                     iterblock);
    printf("#   iter   ||   residual-nrm2    ||   runtime\n");
    printf("#=============================================================#\n");
    printf("      0    ||    %e    ||    0.0000      \n", nom0);
    magma_device_sync(); tempo1=magma_wtime();
    #endif

    // start iteration
    for( solver_par->numiter= 1; solver_par->numiter<solver_par->maxiter; 
                                                    solver_par->numiter++ ){

        magmablasSetKernelStream(stream[0]);

        // computes p=r+beta*(p-omega*v)
        magma_zbicgmerge1( dofs, skp, v.val, r.val, p.val );
        magma_zbicgmerge_spmv1(  dofs, d1, d2, A.val, A.row, A.col, 
                                                    q(2), q(0), q(3), skp );          
        magma_zbicgmerge2( dofs, skp, r.val, v.val, s.val );   // s=r-alpha*v
        magma_zbicgmerge_spmv2( dofs, d1, d2, A.val, A.row, A.col, 
                                                            q(4), q(5), skp); 
        magma_zbicgmerge_xrbeta( dofs, d1, d2, q(0), q(1), q(2), 
                                                    q(4), q(5), x->val, skp);  

        // check stopping criterion (asynchronous copy)
        cublasGetVectorAsync(1 , sizeof( magmaDoubleComplex ), skp+5, 1, 
                                                        skp_h+5, 1, stream[1] );

        #ifdef ENABLE_TIMER
        //Chronometry  
        magma_device_sync(); tempo2=magma_wtime();
        if( solver_par->numiter%iterblock==0 ) {
            printf("   %4d    ||    %e    ||    %.4lf  \n", 
                (solver_par->numiter), betanom, tempo2-tempo1 );
        }
        #endif

        betanom = sqrt(MAGMA_Z_REAL(skp_h[5]));
        if (  betanom  < r0 ) {
            break;
        }
    }

    #ifdef ENABLE_TIMER
    double residual;
    magma_zresidual( A, b, *x, &residual );
    printf("#=============================================================#\n");
    printf("# BiCGStab (merged2) solver summary:\n");
    printf("#    initial residual: %e\n", nom0 );
    printf("#    iterations: %4d\n#    iterative residual: %e\n",
            (solver_par->numiter), betanom );
    printf("#    exact relative residual: %e\n#    runtime: %.4lf sec\n", 
                residual, tempo2-tempo1);
    printf("#=============================================================#\n");
    #endif
        
    solver_par->residual = (double)(betanom);

    magma_z_vfree(&q);
  
    magma_free(d1);
    magma_free(d2);

    return MAGMA_SUCCESS;
}   /* zbicgstab_merge2 */


