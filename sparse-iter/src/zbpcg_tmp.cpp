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

#define  r(i)  r.dval+i*dofs
#define  b(i)  b.dval+i*dofs
#define  h(i)  h.dval+i*dofs
#define  p(i)  p.dval+i*dofs
#define  q(i)  q.dval+i*dofs



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

    @param[in]
    A           magma_z_sparse_matrix
                input matrix A

    @param[in]
    b           magma_z_vector
                RHS b - can be a block

    @param[in,out]
    x           magma_z_vector*
                solution approximation

    @param[in,out]
    solver_par  magma_z_solver_par*
                solver parameters

    @param[in]
    precond_par magma_z_preconditioner*
                preconditioner
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zhesv
    ********************************************************************/

extern "C" magma_int_t
magma_zbpcg(
    magma_z_sparse_matrix A, magma_z_vector b, magma_z_vector *x,  
    magma_z_solver_par *solver_par, 
    magma_z_preconditioner *precond_par,
    magma_queue_t queue )
{
    // set queue for old dense routines
    magma_queue_t orig_queue;
    magmablasGetKernelStream( &orig_queue );

    magma_int_t i, num_vecs = b.num_rows/A.num_rows;

    // Orthogonalization type (3:  MGS   4.  Cholesky QR)
    magma_int_t ikind = 4;

    // m - number of rows; n - number of vectors
    magma_int_t m = A.num_rows;
    magma_int_t n = num_vecs;

    // Work space for CPU and GPU
    magmaDoubleComplex *dwork, *h_work;

    magmaDoubleComplex *gramA, *gramB, *gramR, *gramRold, *gramT, *h_gram;

    // prepare solver feedback
    solver_par->solver = Magma_PCG;
    solver_par->numiter = 0;
    solver_par->info = 0;

    // local variables
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO, c_one = MAGMA_Z_ONE;
    magmaDoubleComplex c_mone = MAGMA_Z_MAKE( -1.0, 0.0 );

    magma_int_t dofs = A.num_rows;

    // GPU workspace
    magma_z_vector r, rt, p, pnew, q, h;
    magma_z_vinit( &r, Magma_DEV, dofs*num_vecs, c_zero, queue );
    magma_z_vinit( &rt, Magma_DEV, dofs*num_vecs, c_zero, queue );
    magma_z_vinit( &p, Magma_DEV, dofs*num_vecs, c_zero, queue );
    magma_z_vinit( &pnew, Magma_DEV, dofs*num_vecs, c_zero, queue );
    magma_z_vinit( &pnew, Magma_DEV, dofs*num_vecs, c_zero, queue );
    magma_z_vinit( &q, Magma_DEV, dofs*num_vecs, c_zero, queue );
    magma_z_vinit( &h, Magma_DEV, dofs*num_vecs, c_zero, queue );
    
    magma_int_t lwork = max( n*n, 2* 2*n* 2*n);
    magma_zmalloc(        &dwork     ,        m*n );
    magma_zmalloc_pinned( &h_work   ,        lwork );

    magma_zmalloc_pinned(&h_gram, n*n);
    magma_zmalloc(&gramA, n * n);
    magma_zmalloc(&gramB, n * n);
    magma_zmalloc(&gramR, n * n);
    magma_zmalloc(&gramRold, n * n);
    magma_zmalloc(&gramT, n * n);

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
    magma_zcopy( dofs*num_vecs, b.dval, 1, r.dval, 1 );                    // r = b

    // preconditioner
    magma_z_applyprecond_left( A, r, &rt, precond_par, queue );
    magma_z_applyprecond_right( A, rt, &h, precond_par, queue );

    //***missing: orthogonalize h here - after the preconditioner?
    magma_zgegqr_gpu(ikind, m, n, h.dval, m, dwork, h_work, &solver_par->info );

    magma_zcopy( m*n, h.dval, 1, p.dval, 1 );                 // p = h it should be -h?
    //magma_zaxpy( dofs*num_vecs, c_mone, h.dval, 1, p.dval, 1 );                 // p = - h

    /*
    for( i=0; i<num_vecs; i++) {
        //*** this should become GEMMS
        nom[i] = (MAGMA_Z_REAL( magma_zdotc(dofs, r(i), 1, h(i), 1) ));     
        nom0[i] = magma_dznrm2( dofs, r(i), 1 );       
    }
    */
    // === Compute the GramR matrix = R^T R ===
    magma_zgemm(MagmaConjTrans, MagmaNoTrans, n, n, m,
                c_one,  r.dval, m, r.dval, m, c_zero, gramR, n);
                        
    // === Compute the GramA matrix = R^T P ===
    magma_zgemm(MagmaConjTrans, MagmaNoTrans, n, n, m,
                c_one,  r.dval, m, p.dval, m, c_zero, gramA, n);
   
    magma_z_spmv( c_one, A, p, c_zero, q, queue );                     // q = A p

    for( i=0; i<num_vecs; i++)
        den[i] = MAGMA_Z_REAL( magma_zdotc(dofs, p(i), 1, q(i), 1) );  // den = p dot q
    // === Compute the GramR matrix = P^T Q ===
    magma_zgemm(MagmaConjTrans, MagmaNoTrans, n, n, m,
                c_one,  p.dval, m, q.dval, m, c_zero, gramB, n);

    // Get the residuals back
    magma_zgetmatrix(1, 1, gramR, n, h_work, n);
    solver_par->init_res = MAGMA_Z_REAL(h_work[0]);

    //  if ( (r0[0] = nom[0] * solver_par->epsilon) < ATOLERANCE ) 
    //    r0[0] = ATOLERANCE;
    // check positive definite

    // Get den back
    magma_zgetmatrix(1, 1, gramB, n, h_work, n);
    den[0] = MAGMA_Z_REAL(h_work[0]);

    // Get nom back
    magma_zgetmatrix(1, 1, gramA, n, h_work, n);
    nom[0] = MAGMA_Z_REAL(h_work[0]);

    if (den[0] <= 0.0) {
        printf("Operator A is not postive definite. (Ar,r) = %f\n", den);
        magmablasSetKernelStream( orig_queue );
        return -100;
    }
    if ( nom[0] < r0[0] ) {
        printf("### early breakdown: %.4e < %.4e \n",nom[0], r0[0]);
        magmablasSetKernelStream( orig_queue );
        return MAGMA_SUCCESS;
    }

    //Chronometry
    real_Double_t tempo1, tempo2;
    tempo1 = magma_sync_wtime( queue );
    if ( solver_par->verbose > 0 ) {
        solver_par->res_vec[0] = (real_Double_t)nom0[0];
        solver_par->timing[0] = 0.0;
    }
  
    // start iteration
    for( solver_par->numiter= 1; solver_par->numiter<solver_par->maxiter; 
                                                    solver_par->numiter++ ) {

        // preconditioner
        magma_z_applyprecond_left( A, r, &rt, precond_par, queue );
        magma_z_applyprecond_right( A, rt, &h, precond_par, queue );

        //***missing: orthogonalize h here - after the preconditioner?
        magma_zgegqr_gpu(ikind, m, n, h.dval, m, dwork, h_work, &solver_par->info ); 

        //*** this has to be changed - to: //GAMMANEW = R^TR
        // === Compute the GramR matrix = R^T R ===
        magma_zcopy( n*n, gramR, 1, gramRold, 1 );
        magma_zgemm(MagmaConjTrans, MagmaNoTrans, n, n, m,
                    c_one,  r.dval, m, r.dval, m, c_zero, gramR, n);
        /*
        for( i=0; i<num_vecs; i++)
            gammanew[i] = MAGMA_Z_REAL( magma_zdotc(dofs, r(i), 1, h(i), 1) );  // gn = < r,h>
        */

        if ( solver_par->numiter==1 ) {
            magma_zcopy( dofs*num_vecs, h.dval, 1, p.dval, 1 );                 // p = h it should be -h?
            //magma_zaxpy( dofs*num_vecs, c_mone, h.dval, 1, p.dval, 1 );                 // p = - h
         
        } else {
            //*** this llop should go away
      /*
            for( i=0; i<num_vecs; i++) {
                //*** this has to be changed - to: BETA = GAMMAOLD^-1 * GAMMANEW
                beta[i] = MAGMA_Z_MAKE(gammanew[i]/gammaold[i], 0.);       // beta = gn/go
                magma_zscal(dofs, beta[i], p(i), 1);            // p = beta*p
                magma_zaxpy(dofs, c_one, h(i), 1, p(i), 1); // p = p + h 
            }
      */
            magma_zcopy( n*n, gramR, 1, gramT, 1 ); 
            magma_zposv_gpu(MagmaUpper, n, n, gramRold, n, gramT, n, &solver_par->info);
            magma_zcopy( m*n, r.dval, 1, pnew.dval, 1 );
            magma_zgemm(MagmaNoTrans, MagmaNoTrans, m, n, n,
            c_one, p.dval, m, gramT, n, c_one, pnew.dval, m);
            magma_zcopy( m*n, pnew.dval, 1, p.dval, 1);
        }

        magma_z_spmv( c_one, A, p, c_zero, q, queue );           // q = A p

            //*** this llop should go away
    /*
        for( i=0; i<num_vecs; i++) {
            // den = p dot q 
            //*** this has to be changed - to: DEN = GEMM(P,Q)
            den[i] = MAGMA_Z_REAL(magma_zdotc(dofs, p(i), 1, q(i), 1));    

            //*** this has to be changed - to: ALPHA = DEN^-1 GAMMANEW
            alpha[i] = MAGMA_Z_MAKE(gammanew[i]/den[i], 0.);
            //*** this becomes GEMMS
            magma_zaxpy(dofs,  alpha[i], p(i), 1, x->val+dofs*i, 1); // x = x + alpha p
            magma_zaxpy(dofs, -alpha[i], q(i), 1, r(i), 1);      // r = r - alpha q
            gammaold[i] = gammanew[i];

            res[i] = magma_dznrm2( dofs, r(i), 1 );
         }
    */
        // === Compute the GramR matrix = P^T Q ===
        magma_zgemm(MagmaConjTrans, MagmaNoTrans, n, n, m,
            c_one,  p.dval, m, q.dval, m, c_zero, gramB, n);

    // Get den back
    magma_zgetmatrix(1, 1, gramB, n, h_work, n);
    den[0] = MAGMA_Z_REAL(h_work[0]);

        magma_zposv_gpu(MagmaUpper, n, n, gramR, n, gramB, n, &solver_par->info);
    magma_zgemm(MagmaNoTrans, MagmaNoTrans, m, n, n,
            c_one, p.dval, m, gramB, n, c_one, x->val, m);
        magma_zgemm(MagmaNoTrans, MagmaNoTrans, m, n, n,
                    c_mone, q.dval, m, gramB, n, c_one, r.dval, m);
        *gramRold = *gramR;

        // Get the residuals back
    magma_zgetmatrix(1, 1, gramRold, n, h_work, n);
    res[i] = MAGMA_Z_REAL(h_work[0]);


        if ( solver_par->verbose > 0 ) {
            tempo2 = magma_sync_wtime( queue );
            if ( (solver_par->numiter)%solver_par->verbose==0 ) {
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
    tempo2 = magma_sync_wtime( queue );
    solver_par->runtime = (real_Double_t) tempo2-tempo1;
    double *residual;
    magma_dmalloc_cpu(&residual, num_vecs);
    magma_zresidual( A, b, *x, residual, queue );
    solver_par->iter_res = res[0];
    solver_par->final_res = residual[0];

    if ( solver_par->numiter < solver_par->maxiter) {
        solver_par->info = 0;
    } else if ( solver_par->init_res > solver_par->final_res ) {
        if ( solver_par->verbose > 0 ) {
            if ( (solver_par->numiter)%solver_par->verbose==0 ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose] 
                        = (real_Double_t) res[0];
                solver_par->timing[(solver_par->numiter)/solver_par->verbose] 
                        = (real_Double_t) tempo2-tempo1;
            }
        }
        solver_par->info = -2;
    }
    else {
        if ( solver_par->verbose > 0 ) {
            if ( (solver_par->numiter)%solver_par->verbose==0 ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose] 
                        = (real_Double_t) res[0];
                solver_par->timing[(solver_par->numiter)/solver_par->verbose] 
                        = (real_Double_t) tempo2-tempo1;
            }
        }
        solver_par->info = -1;
    }
    for( i=0; i<num_vecs; i++) {
        printf("%.4e  ",res[i]);
    }
    printf("\n");
    for( i=0; i<num_vecs; i++) {
        printf("%.4e  ",residual[i]);
    }
    printf("\n");

    magma_z_vfree(&r, queue );
    magma_z_vfree(&rt, queue );
    magma_z_vfree(&p, queue );
    magma_z_vfree(&pnew, queue );
    magma_z_vfree(&q, queue );
    magma_z_vfree(&h, queue );

    magma_free_cpu(alpha);
    magma_free_cpu(beta);
    magma_free_cpu(nom);
    magma_free_cpu(nom0);
    magma_free_cpu(r0);
    magma_free_cpu(gammaold);
    magma_free_cpu(gammanew);
    magma_free_cpu(den);
    magma_free_cpu(res);

    magma_free( gramA );
    magma_free( gramB );
    magma_free( gramR );
    magma_free( gramRold );
    magma_free( gramT );
    magma_free( dwork );
    magma_free_pinned( h_work );
    magma_free_pinned( h_gram );

    magmablasSetKernelStream( orig_queue );
    return MAGMA_SUCCESS;
}   /* magma_zbpcg */


