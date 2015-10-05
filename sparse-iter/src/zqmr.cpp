/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Hartwig Anzt

       @precisions normal z -> s d c
*/

#include "common_magmasparse.h"

#define RTOLERANCE     lapackf77_dlamch( "E" )
#define ATOLERANCE     lapackf77_dlamch( "E" )


/**
    Purpose
    -------

    Solves a system of linear equations
       A * X = B
    where A is a complex Hermitian N-by-N positive definite matrix A.
    This is a GPU implementation of the Quasi-Minimal Residual method (TFQMR).

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                input matrix A

    @param[in]
    b           magma_z_matrix
                RHS b

    @param[in,out]
    x           magma_z_matrix*
                solution approximation

    @param[in,out]
    solver_par  magma_z_solver_par*
                solver parameters

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zposv
    ********************************************************************/

extern "C" magma_int_t
magma_zqmr(
    magma_z_matrix A, magma_z_matrix b, magma_z_matrix *x,
    magma_z_solver_par *solver_par,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    // set queue for old dense routines
    magma_queue_t orig_queue=NULL;
    magmablasGetKernelStream( &orig_queue );

    // prepare solver feedback
    solver_par->solver = Magma_PTFQMR;
    solver_par->numiter = 0;
    solver_par->info = MAGMA_SUCCESS;
    
    // local variables
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO, c_one = MAGMA_Z_ONE;
    // solver variables
    double nom0, r0,  res, nomb;
    magmaDoubleComplex rho = c_one, rho1 = c_one, eta = -c_one , pds = c_one, 
                        thet = c_one, thet1 = c_one, epsilon = c_one, 
                        beta = c_one, delta = c_one, pde = c_one, rde = c_one,
                        gamm = c_one, gamm1 = c_one, psi = c_one;
    
    magma_int_t dofs = A.num_rows* b.num_cols;

    // GPU workspace
    magma_z_matrix r={Magma_CSR}, r_tld={Magma_CSR},
                    v={Magma_CSR}, w={Magma_CSR}, wt={Magma_CSR}, vt={Magma_CSR},
                    d={Magma_CSR}, s={Magma_CSR}, z={Magma_CSR}, zt={Magma_CSR}, q={Magma_CSR}, 
                    p={Magma_CSR}, pt={Magma_CSR}, y={Magma_CSR}, yt={Magma_CSR};
    CHECK( magma_zvinit( &r, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &r_tld, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &v, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &w, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &wt,Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &vt,Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &d, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &s, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &z, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &zt, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &q, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &p, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &pt,Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &y, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &yt,Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));

    
    // solver setup
    CHECK(  magma_zresidualvec( A, b, *x, &r, &nom0, queue));
    solver_par->init_res = nom0;
    magma_zcopy( dofs, r.dval, 1, r_tld.dval, 1 );   
    magma_zcopy( dofs, r.dval, 1, y.dval, 1 );   
    magma_zcopy( dofs, r.dval, 1, vt.dval, 1 );  
    magma_zcopy( dofs, r.dval, 1, wt.dval, 1 );   
    magma_zcopy( dofs, r.dval, 1, z.dval, 1 );  
    

    
    nomb = magma_dznrm2( dofs, b.dval, 1 );
    if ( nomb == 0.0 ){
        nomb=1.0;
    }       
    if ( (r0 = nomb * solver_par->rtol) < ATOLERANCE ){
        r0 = ATOLERANCE;
    }
    solver_par->final_res = solver_par->init_res;
    solver_par->iter_res = solver_par->init_res;
    if ( solver_par->verbose > 0 ) {
        solver_par->res_vec[0] = (real_Double_t)nom0;
        solver_par->timing[0] = 0.0;
    }
    if ( nom0 < r0 ) {
        goto cleanup;
    }

    psi = magma_zsqrt( magma_zdotc(dofs, z.dval, 1, z.dval, 1) );
    rho = magma_zsqrt( magma_zdotc(dofs, y.dval, 1, y.dval, 1) );
    
    //Chronometry
    real_Double_t tempo1, tempo2;
    tempo1 = magma_sync_wtime( queue );
    
    solver_par->numiter = 0;
    // start iteration
    do
    {
        solver_par->numiter++;
        if( rho == c_zero || rho == 'NaN' || psi == c_zero || psi == 'NaN' ){
            printf("break 1"\n);
            goto cleanup;
        }

            // v = vt / rho;
        magma_zcopy( dofs, vt.dval, 1, v.dval, 1 );  
        magma_zscal(dofs, c_one / rho, v.dval, 1);  
            // y = y / rho;
        magma_zscal(dofs, c_one / rho, y.dval, 1);  
            // w = wt / psi;
        magma_zcopy( dofs, wt.dval, 1, w.dval, 1 );  
        magma_zscal(dofs, c_one / psi, w.dval, 1); 
            // z = z / psi;
        magma_zscal(dofs, c_one / psi, z.dval, 1); 
            // delta = z' * y;
        delta = magma_zdotc(dofs, z.dval, 1, y.dval, 1);
        if( delta == c_zero || delta == 'NaN' ){
            printf("break 2"\n);
            break;
        }
        
        // no precond: yt = y, zt = z
        magma_zcopy( dofs, y.dval, 1, y.dval, 1 );
        magma_zcopy( dofs, z.dval, 1, zt.dval, 1 );
        
        if( solver_par->numiter == 1 ){
                // p = yt;
                // q = zt;
            magma_zcopy( dofs, yt.dval, 1, p.dval, 1 );
            magma_zcopy( dofs, zt.dval, 1, q.dval, 1 );
        }
        else{
            pde = psi * delta / epsilon;
            rde = rho * MAGMA_Z_CNJG(delta/epsilon);

                // p = yt - pde * p;
            magma_zscal(dofs, -pde, p.dval, 1);    
            magma_zaxpy(dofs, c_one, yt.dval, 1, p.dval, 1);
                // q = zt - rde * q;
            magma_zscal(dofs, -rde, q.dval, 1);    
            magma_zaxpy(dofs, c_one, zt.dval, 1, q.dval, 1);
        }
        if( rho == c_zero || rho == 'NaN' || psi == c_zero || psi == 'NaN' ){
            printf("break 3"\n);
            break;
        }
        
        CHECK( magma_z_spmv( c_one, A, p, c_zero, pt, queue ));
            // epsilon = q' * pt;
        epsilon = magma_zdotc(dofs, q.dval, 1, pt.dval, 1);
        beta = epsilon / delta;
        if( epsilon == c_zero || epsilon == 'NaN' || beta == c_zero || beta == 'NaN' ){
                        printf("break 4"\n);
            break;
        }
        
            // vt = pt - beta * v;
        magma_zcopy( dofs, pt.dval, 1, vt.dval, 1 ); 
        magma_zaxpy(dofs, -beta, v.dval, 1, vt.dval, 1);   
        
            // no precond: y = vt
        magma_zcopy( dofs, vt.dval, 1, y.dval, 1 );
        
        rho1 = rho;      
            // rho = norm(y);
        rho = magma_zsqrt( magma_zdotc(dofs, y.dval, 1, y.dval, 1) );
        
            // wt = A' * q - beta' * w;
        CHECK( magma_z_spmv( c_one, A, q, c_zero, wt, queue ));
        magma_zaxpy(dofs, - MAGMA_Z_CNJG( beta ), w.dval, 1, wt.dval, 1);  
        
                    // no precond: z = wt
        magma_zcopy( dofs, wt.dval, 1, z.dval, 1 );
        
                    // psi = norm(z);
        psi = magma_zsqrt( magma_zdotc(dofs, z.dval, 1, z.dval, 1) );
        
        thet1 = thet;        
        thet = rho / (gamm * MAGMA_Z_MAKE( MAGMA_Z_ABS(beta), 0.0 ));
        gamm1 = gamm;
        gamm = c_one / magma_zsqrt(c_one + thet*thet);
        eta = - eta * rho1 * gamm * gamm / (beta * gamm1 * gamm1);
        if( thet == c_zero || thet == 'NaN' || gamm == c_zero || gamm == 'NaN' || eta == c_zero || eta == 'NaN' ){
                        printf("break 1\n");
            break;
        }
        
        if( solver_par->numiter == 1 ){
                // d = eta * p;
                // s = eta * pt;
            magma_zcopy( dofs, p.dval, 1, d.dval, 1 );
            magma_zscal( dofs, eta, d.dval, 1);
            magma_zcopy( dofs, pt.dval, 1, s.dval, 1 );
            magma_zscal( dofs, eta, s.dval, 1);
        }
        else{
                // d = eta * p + (thet1 * gamm)^2 * d;
                // s = eta * pt + (thet1 * gamm)^2 * s;
            pds = (thet1 * gamm) * (thet1 * gamm);
            magma_zscal(dofs, pds, d.dval, 1);    
            magma_zaxpy(dofs, eta, p.dval, 1, d.dval, 1);
            magma_zscal(dofs, pds, s.dval, 1);    
            magma_zaxpy(dofs, eta, pt.dval, 1, s.dval, 1);
        }
        
            // x = x + d;                    
        magma_zaxpy(dofs, c_one, d.dval, 1, x->dval, 1);
            // r = r - s;
        magma_zaxpy(dofs, -c_one, d.dval, 1, r.dval, 1);
        
        res = magma_dznrm2( dofs, r.dval, 1 );
        
        if ( solver_par->verbose > 0 ) {
            tempo2 = magma_sync_wtime( queue );
            if ( (solver_par->numiter)%solver_par->verbose == c_zero ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) res;
                solver_par->timing[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) tempo2-tempo1;
            }
        }

        if ( res/nomb <= solver_par->rtol || res <= solver_par->atol ){
            break;
        }
 
    }
    while ( solver_par->numiter+1 <= solver_par->maxiter );
    
    tempo2 = magma_sync_wtime( queue );
    solver_par->runtime = (real_Double_t) tempo2-tempo1;
    double residual;
    CHECK(  magma_zresidualvec( A, b, *x, &r, &residual, queue));
    solver_par->iter_res = res;
    solver_par->final_res = residual;

    if ( solver_par->numiter < solver_par->maxiter ) {
        info = MAGMA_SUCCESS;
    } else if ( solver_par->init_res > solver_par->final_res ) {
        if ( solver_par->verbose > 0 ) {
            if ( (solver_par->numiter)%solver_par->verbose == c_zero ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) res;
                solver_par->timing[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) tempo2-tempo1;
            }
        }
        info = MAGMA_SLOW_CONVERGENCE;
        if( solver_par->iter_res < solver_par->rtol*solver_par->init_res ||
            solver_par->iter_res < solver_par->atol ) {
            info = MAGMA_SUCCESS;
        }
    }
    else {
        if ( solver_par->verbose > 0 ) {
            if ( (solver_par->numiter)%solver_par->verbose == c_zero ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) res;
                solver_par->timing[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) tempo2-tempo1;
            }
        }
        info = MAGMA_DIVERGENCE;
    }
    
cleanup:
    magma_zmfree(&r, queue );
    magma_zmfree(&r_tld, queue );
    magma_zmfree(&v,  queue );
    magma_zmfree(&w,  queue );
    magma_zmfree(&wt, queue );
    magma_zmfree(&vt, queue );
    magma_zmfree(&d,  queue );
    magma_zmfree(&s,  queue );
    magma_zmfree(&z,  queue );
    magma_zmfree(&zt,  queue );
    magma_zmfree(&q,  queue );
    magma_zmfree(&p,  queue );
    magma_zmfree(&pt, queue );
    magma_zmfree(&y,  queue );
    magma_zmfree(&yt, queue );

    
    magmablasSetKernelStream( orig_queue );
    solver_par->info = info;
    return info;
}   /* magma_zqmr */