/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions mixed zc -> ds
       @author Hartwig Anzt

*/
#include "common_magma.h"
#include "magmasparse.h"

#include <assert.h>

#define RTOLERANCE     10e-16
#define ATOLERANCE     10e-16


/**
    Purpose
    -------

    Solves a system of linear equations
       A * X = B
    where A is a complex Hermitian N-by-N positive definite matrix A.
    This is a GPU implementation of the Preconditioned 
    Biconjugate Gradient Stabelized method.

    Arguments
    ---------

    @param[in]
    A           magma_z_sparse_matrix
                input matrix A

    @param[in]
    b           magma_z_vector
                RHS b

    @param[in,out]
    x           magma_z_vector*
                solution approximation

    @param[in,out]
    solver_par  magma_z_solver_par*
                solver parameters

    magma_precond_parameters *precond_par     preconditioner parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_gesv
    ********************************************************************/

extern "C" magma_int_t
magma_zcpbicgstab(
    magma_z_sparse_matrix A, magma_z_vector b, magma_z_vector *x,  
    magma_z_solver_par *solver_par, magma_precond_parameters *precond_par,
    magma_queue_t queue )
{
    // set queue for old dense routines
    magma_queue_t orig_queue;
    magmablasGetKernelStream( &orig_queue );

    // some useful variables
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO, c_one = MAGMA_Z_ONE, c_mone = MAGMA_Z_NEG_ONE;
    
    magma_int_t dofs = A.num_rows;

    // workspace on GPU
    magma_z_vector r,rr,p,v,s,t,y,z;
    magma_z_vinit( &r, Magma_DEV, dofs, c_zero, queue );
    magma_z_vinit( &rr, Magma_DEV, dofs, c_zero, queue );
    magma_z_vinit( &p, Magma_DEV, dofs, c_zero, queue );
    magma_z_vinit( &v, Magma_DEV, dofs, c_zero, queue );
    magma_z_vinit( &s, Magma_DEV, dofs, c_zero, queue );
    magma_z_vinit( &t, Magma_DEV, dofs, c_zero, queue );
    magma_z_vinit( &y, Magma_DEV, dofs, c_zero, queue );
    magma_z_vinit( &z, Magma_DEV, dofs, c_zero, queue );

    // for mixed precision on GPU
    magma_c_vector ps, ys, ss, zs;
    magma_c_sparse_matrix AS;
    magma_sparse_matrix_zlag2c( A, &AS, queue );

    magma_c_vinit( &ys, Magma_DEV, dofs, MAGMA_C_ZERO, queue );
    magma_c_vinit( &zs, Magma_DEV, dofs, MAGMA_C_ZERO, queue );
    
    // solver variables
    magmaDoubleComplex alpha, beta, omega, rho_old, rho_new;
    double nom, nom0, r0, den;
    magma_int_t i;


    // solver setup
    magma_zscal( dofs, c_zero, x->val, 1) ;                            // x = 0
    magma_zcopy( dofs, b.dval, 1, r.dval, 1 );                           // r = b
    magma_zcopy( dofs, b.dval, 1, rr.dval, 1 );                          // rr = b
    nom = magma_dznrm2( dofs, r.dval, 1 );                              // nom = || r ||
    nom0 = nom = nom*nom;
    rho_old = omega = alpha = MAGMA_Z_MAKE( 1.0, 0. );
    (solver_par->numiter) = 0;

    magma_z_spmv( c_one, A, r, c_zero, v, queue );                           // z = A r
    den = MAGMA_Z_REAL( magma_zdotc(dofs, v.dval, 1, r.dval, 1) );      // den = z dot r

    if ( (r0 = nom * solver_par->epsilon) < ATOLERANCE ) 
        r0 = ATOLERANCE;
    if ( nom < r0 ) {
        magmablasSetKernelStream( orig_queue );
        return MAGMA_SUCCESS;
    }
    
    if (den <= 0.0) {
        printf("Operator A is not postive definite. (Ar,r) = %f\n", den);
        magmablasSetKernelStream( orig_queue );
        return -100;
    }

    printf("Iteration : %4d  Norm: %f\n", 0, nom );
    
    // start iteration
    while( nom > r0 && solver_par->numiter < solver_par->maxiter ) {

        rho_new = magma_zdotc( dofs, rr.dval, 1, r.dval, 1 );
        beta = rho_new/rho_old * alpha/omega;
        magma_zscal( dofs, beta, p.dval, 1 );                                    // p = beta*p
        magma_zaxpy( dofs, c_mone * omega * beta, v.dval, 1 , p.dval, 1 );        // p = p-omega*beta*v
        magma_zaxpy( dofs, c_one, r.dval, 1, p.dval, 1 );                         // p = p+r

        magma_vector_zlag2c(p, &ps, queue );                                            // conversion to single precision
        magma_c_precond( AS, ps, &ys, *precond_par, queue );                           // precond: MS * ys = ps
        magma_vector_clag2z(ys, &y, queue );                                            // conversion to double precision

        magma_z_spmv( c_one, A, y, c_zero, v, queue );                                 // v = Ay
        alpha = rho_new / magma_zdotc( dofs, rr.dval, 1, v.dval, 1 );

        magma_zcopy( dofs, r.dval, 1 , s.dval, 1 );
        magma_zaxpy( dofs, c_mone * alpha, v.dval, 1 , s.dval, 1 );

        magma_vector_zlag2c(s, &ss, queue );                                            // conversion to single precision
        magma_c_precond( AS, ss, &zs, *precond_par, queue );                           // precond: MS * zs = ss
        magma_vector_clag2z(zs, &z, queue );                                            // conversion to double precision

        magma_z_spmv( c_one, A, z, c_zero, t, queue );                                 //t=Az
        omega = magma_zdotc( dofs, t.dval, 1, s.dval, 1 ) 
                   / magma_zdotc( dofs, t.dval, 1, t.dval, 1 );

        magma_zaxpy( dofs, alpha, y.dval, 1 , x->val, 1 );
        magma_zaxpy( dofs, omega, z.dval, 1 , x->val, 1 );

        magma_zcopy( dofs, s.dval, 1 , r.dval, 1 );
        magma_zaxpy( dofs, c_mone * omega, t.dval, 1 , r.dval, 1 );
        nom = magma_dznrm2( dofs, r.dval, 1 );
        nom = nom*nom;
        rho_old = rho_new;

        (solver_par->numiter)++;

        printf("Iteration : %4d  Norm: %f\n", (solver_par->numiter), nom);
    
    }
  
    printf( "      (r_0, r_0) = %e\n", nom0 );
    printf( "      (r_N, r_N) = %e\n", nom );
    printf( "      Number of BICGSTAB iterations: %d\n", (solver_par->numiter));
    
    if (solver_par->epsilon == RTOLERANCE) {
        magma_z_spmv( c_one, A, *x, c_zero, r, queue );                       // r = A x
        magma_zaxpy(dofs,  c_mone, b.dval, 1, r.dval, 1);                // r = r - b
        den = magma_dznrm2(dofs, r.dval, 1);                            // den = || r ||
        printf( "      || r_N ||   = %f\n", den);
        solver_par->residual = (double)(den);
    }
/*
    magma_z_vfree(&r, queue );
    magma_z_vfree(&rr, queue );
    magma_z_vfree(&p, queue );
    magma_z_vfree(&v, queue );
    magma_z_vfree(&s, queue );
    magma_z_vfree(&t, queue );
    magma_z_vfree(&y, queue );
    magma_z_vfree(&z, queue );
    magma_c_vfree(&ps, queue );
    magma_c_vfree(&ys, queue );
    magma_c_vfree(&ss, queue );
    magma_c_vfree(&zs, queue );

    magma_c_mfree(&AS, queue );
  */  
    magmablasSetKernelStream( orig_queue );
    return MAGMA_SUCCESS;
}


