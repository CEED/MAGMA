/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Hartwig Anzt 

       @precisions normal z -> s d c
*/

#include "common_magma.h"
#include "../include/magmasparse.h"

#include <assert.h>

#define RTOLERANCE     lapackf77_dlamch( "E" )
#define ATOLERANCE     lapackf77_dlamch( "E" )

#define  r(i)  r.val+i*dofs
#define  b(i)  b.val+i*dofs
#define  h(i)  h.val+i*dofs
#define  p(i)  p.val+i*dofs
#define  q(i)  q.val+i*dofs



/**
    Purpose
    -------

    Solves a system of linear equations
       A * X = B
    where A is a complex Hermitian N-by-N positive definite matrix A.
    This is a GPU implementation of the block preconditioned Conjugate 
    Gradient method.

    Arguments
    ---------

    @param
    A           magma_z_sparse_matrix
                input matrix A

    @param
    b           magma_z_vector
                RHS b - can be a block

    @param
    x           magma_z_vector*
                solution approximation

    @param
    solver_par  magma_z_solver_par*
                solver parameters

    @param
    precond_par magma_z_preconditioner*
                preconditioner

    @ingroup magmasparse_zhesv
    ********************************************************************/

magma_int_t
magma_zbpcg( magma_z_sparse_matrix A, magma_z_vector b, magma_z_vector *x,  
            magma_z_solver_par *solver_par, 
            magma_z_preconditioner *precond_par ){

    magma_int_t i, num_vecs = b.num_rows/A.num_rows;

    // prepare solver feedback
    solver_par->solver = Magma_PCG;
    solver_par->numiter = 0;
    solver_par->info = 0;

    // local variables
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO, c_one = MAGMA_Z_ONE;
    
    magma_int_t dofs = A.num_rows;

    // GPU workspace
    magma_z_vector r, rt, p, q, h;
    magma_z_vinit( &r, Magma_DEV, dofs*num_vecs, c_zero );
    magma_z_vinit( &rt, Magma_DEV, dofs*num_vecs, c_zero );
    magma_z_vinit( &p, Magma_DEV, dofs*num_vecs, c_zero );
    magma_z_vinit( &q, Magma_DEV, dofs*num_vecs, c_zero );
    magma_z_vinit( &h, Magma_DEV, dofs*num_vecs, c_zero );
    
    // solver variables
    magmaDoubleComplex *alpha, *beta;
    magma_zmalloc_cpu(&alpha, num_vecs);
    magma_zmalloc_cpu(&beta, num_vecs);

    double *nom, *nom0, *r0, *gammaold, *gammanew, *den, *res;
    magma_dmalloc_cpu(&nom, num_vecs);
    magma_dmalloc_cpu(&nom0, num_vecs);
    magma_dmalloc_cpu(&r0, num_vecs);
    magma_dmalloc_cpu(&gammaold, num_vecs);
    magma_dmalloc_cpu(&gammanew, num_vecs);
    magma_dmalloc_cpu(&den, num_vecs);
    magma_dmalloc_cpu(&res, num_vecs);

    // solver setup
    magma_zscal( dofs*num_vecs, c_zero, x->val, 1) ;                     // x = 0
    magma_zcopy( dofs*num_vecs, b.val, 1, r.val, 1 );                    // r = b

    // preconditioner
    magma_z_applyprecond_left( A, r, &rt, precond_par );
    magma_z_applyprecond_right( A, rt, &h, precond_par );

    magma_zcopy( dofs*num_vecs, h.val, 1, p.val, 1 );                 // p = h

    for( i=0; i<num_vecs; i++){
        nom[i] = MAGMA_Z_REAL( magma_zdotc(dofs, r(i), 1, h(i), 1) );     
        nom0[i] = magma_dznrm2( dofs, r(i), 1 );       
    }
                                          
    magma_z_spmv( c_one, A, p, c_zero, q );                     // q = A p

    for( i=0; i<num_vecs; i++)
        den[i] = MAGMA_Z_REAL( magma_zdotc(dofs, p(i), 1, q(i), 1) );  // den = p dot q

    solver_par->init_res = nom0[0];
    
    if ( (r0[0] = nom[0] * solver_par->epsilon) < ATOLERANCE ) 
        r0[0] = ATOLERANCE;
    if ( nom[0] < r0[0] )
        return MAGMA_SUCCESS;
    // check positive definite
    if (den[0] <= 0.0) {
        printf("Operator A is not postive definite. (Ar,r) = %f\n", den);
        return -100;
    }

    //Chronometry
    real_Double_t tempo1, tempo2;
    magma_device_sync(); tempo1=magma_wtime();
    if( solver_par->verbose > 0 ){
        solver_par->res_vec[0] = (real_Double_t)nom0[0];
        solver_par->timing[0] = 0.0;
    }
    
    // start iteration
    for( solver_par->numiter= 1; solver_par->numiter<solver_par->maxiter; 
                                                    solver_par->numiter++ ){
        // preconditioner
        magma_z_applyprecond_left( A, r, &rt, precond_par );
        magma_z_applyprecond_right( A, rt, &h, precond_par );


        for( i=0; i<num_vecs; i++)
            gammanew[i] = MAGMA_Z_REAL( magma_zdotc(dofs, r(i), 1, h(i), 1) );  // gn = < r,h>


        if( solver_par->numiter==1 ){
            magma_zcopy( dofs*num_vecs, h.val, 1, p.val, 1 );                    // p = h            
        }else{
            for( i=0; i<num_vecs; i++){
                beta[i] = MAGMA_Z_MAKE(gammanew[i]/gammaold[i], 0.);       // beta = gn/go
                magma_zscal(dofs, beta[i], p(i), 1);            // p = beta*p
                magma_zaxpy(dofs, c_one, h(i), 1, p(i), 1); // p = p + h 
            }
        }

        magma_z_spmv( c_one, A, p, c_zero, q );           // q = A p

     //   magma_z_bspmv_tuned( dofs, num_vecs, c_one, A, p.val, c_zero, q.val );


        for( i=0; i<num_vecs; i++){
            den[i] = MAGMA_Z_REAL(magma_zdotc(dofs, p(i), 1, q(i), 1));    
                // den = p dot q 

            alpha[i] = MAGMA_Z_MAKE(gammanew[i]/den[i], 0.);
            magma_zaxpy(dofs,  alpha[i], p(i), 1, x->val+dofs*i, 1); // x = x + alpha p
            magma_zaxpy(dofs, -alpha[i], q(i), 1, r(i), 1);      // r = r - alpha q
            gammaold[i] = gammanew[i];

            res[i] = magma_dznrm2( dofs, r(i), 1 );
        }

        if( solver_par->verbose > 0 ){
            magma_device_sync(); tempo2=magma_wtime();
            if( (solver_par->numiter)%solver_par->verbose==0 ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose] 
                        = (real_Double_t) res[0];
                solver_par->timing[(solver_par->numiter)/solver_par->verbose] 
                        = (real_Double_t) tempo2-tempo1;
            }
        }


        if (  res[0]/nom0[0]  < solver_par->epsilon ) {
            break;
        }
    } 
    magma_device_sync(); tempo2=magma_wtime();
    solver_par->runtime = (real_Double_t) tempo2-tempo1;
    double *residual;
    magma_dmalloc_cpu(&residual, num_vecs);
    magma_zresidual( A, b, *x, residual );
    solver_par->iter_res = res[0];
    solver_par->final_res = residual[0];

    if( solver_par->numiter < solver_par->maxiter){
        solver_par->info = 0;
    }else if( solver_par->init_res > solver_par->final_res ){
        if( solver_par->verbose > 0 ){
            if( (solver_par->numiter)%solver_par->verbose==0 ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose] 
                        = (real_Double_t) res[0];
                solver_par->timing[(solver_par->numiter)/solver_par->verbose] 
                        = (real_Double_t) tempo2-tempo1;
            }
        }
        solver_par->info = -2;
    }
    else{
        if( solver_par->verbose > 0 ){
            if( (solver_par->numiter)%solver_par->verbose==0 ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose] 
                        = (real_Double_t) res[0];
                solver_par->timing[(solver_par->numiter)/solver_par->verbose] 
                        = (real_Double_t) tempo2-tempo1;
            }
        }
        solver_par->info = -1;
    }
    for( i=0; i<num_vecs; i++){
        printf("%.4e  ",res[i]);
    }
    printf("\n");
    for( i=0; i<num_vecs; i++){
        printf("%.4e  ",residual[i]);
    }
    printf("\n");

    magma_z_vfree(&r);
    magma_z_vfree(&rt);
    magma_z_vfree(&p);
    magma_z_vfree(&q);
    magma_z_vfree(&h);

    magma_free_cpu(alpha);
    magma_free_cpu(beta);
    magma_free_cpu(nom);
    magma_free_cpu(nom0);
    magma_free_cpu(r0);
    magma_free_cpu(gammaold);
    magma_free_cpu(gammanew);
    magma_free_cpu(den);
    magma_free_cpu(res);

    return MAGMA_SUCCESS;
}   /* magma_zbpcg */


