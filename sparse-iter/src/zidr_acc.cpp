/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Hartwig Anzt
       @author Eduardo Ponce
       @author Moritz Kreutzer

       @precisions normal z -> s d c
*/

#include "magmasparse_internal.h"
#include <cuda_profiler_api.h>

#define RTOLERANCE     lapackf77_dlamch( "E" )
#define ATOLERANCE     lapackf77_dlamch( "E" )


/**
    Purpose
    -------

    Solves a system of linear equations
       A * X = B
    where A is a complex Hermitian N-by-N positive definite matrix A.
    This is a GPU implementation of the Induced Dimension Reduction method applying kernel fusion.

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

// TODO: probably need to sync after 'magma_zgetvector' calls for scaling scalars.
// TODO: probably need to sync after 'magma_znrm' calls.

extern "C" magma_int_t
magma_zidr_acc(
    magma_z_matrix A, magma_z_matrix b, magma_z_matrix *x,
    magma_z_solver_par *solver_par,
    magma_queue_t queue )
{

    // prepare solver feedback
    solver_par->solver = Magma_IDR;
    solver_par->numiter = 0;
    solver_par->info = MAGMA_SUCCESS;

    // constants
    const magmaDoubleComplex c_zero = MAGMA_Z_ZERO;
    const magmaDoubleComplex c_one = MAGMA_Z_ONE;
    const magmaDoubleComplex c_n_one = MAGMA_Z_NEG_ONE;

    // internal user parameters 
    const magma_int_t smoothing = 1;   // 0 = disable, 1 = enable
    const double angle = 0.7;          // [0-1]

    // local variables
    magma_int_t info = 0;
    magma_int_t iseed[4] = { 0, 0, 0, 1 };
    magma_int_t dofx = x->num_rows * x->num_cols;
    magma_int_t dofb = b.num_rows * b.num_cols;
    magma_int_t dofr = A.num_rows * b.num_cols;
    magma_int_t dofM; 
    magma_int_t dofP;
    magma_int_t doft;
    magma_int_t inc = 1;
    magma_int_t s;
    magma_int_t distr;
    magma_int_t k, i, sk;
    magma_int_t innerflag;
    magma_int_t ldd;
    double residual;
    double nrm;
    double nrmb;
    double nrmr;
    double nrmt;
    double rho;
    magmaDoubleComplex om;
    magmaDoubleComplex tr;
    magmaDoubleComplex gamma;
    magmaDoubleComplex mkk;
    magmaDoubleComplex fk;
    magmaDoubleComplex_ptr tmp;

    // matrices and vectors
    magma_z_matrix dxs = {Magma_CSR};
    magma_z_matrix dP1 = {Magma_CSR}, dP = {Magma_CSR};
    magma_z_matrix dr = {Magma_CSR}, drs = {Magma_CSR};
    magma_z_matrix dG = {Magma_CSR}, dGcol = {Magma_CSR};
    magma_z_matrix dU = {Magma_CSR};
    magma_z_matrix dM1 = {Magma_CSR}, dM = {Magma_CSR}, hMdiag = {Magma_CSR};
    magma_z_matrix df = {Magma_CSR};
    magma_z_matrix dt = {Magma_CSR}, dtt = {Magma_CSR};
    magma_z_matrix dc = {Magma_CSR};
    magma_z_matrix dv1 = {Magma_CSR}, dv = {Magma_CSR};
    magma_z_matrix dskp = {Magma_CSR}, skp = {Magma_CSR};
    magma_z_matrix dalpha = {Magma_CSR}, alpha = {Magma_CSR};
    magma_z_matrix dbeta = {Magma_CSR}, beta = {Magma_CSR};
    magmaDoubleComplex *d1 = NULL, *d2 = NULL;
    
    // chronometry
    real_Double_t tempo1, tempo2;

    // initial s space
    // TODO: add option for 's' (shadow space number)
    // Hack: uses '--restart' option as the shadow space number.
    //       This is not a good idea because the default value of restart option is used to detect
    //       if the user provided a custom restart. This means that if the default restart value
    //       is changed then the code will think it was the user (unless the default value is
    //       also updated in the 'if' statement below.
    s = 1;
    if ( solver_par->restart != 50 ) {
        if ( solver_par->restart > A.num_cols ) {
            s = A.num_cols;
        } else {
            s = solver_par->restart;
        }
    }
    solver_par->restart = s;

    // set max iterations
    solver_par->maxiter = min( 2 * A.num_cols, solver_par->maxiter );

    // check if matrix A is square
    if ( A.num_rows != A.num_cols ) {
        //printf("Matrix A is not square.\n");
        info = MAGMA_ERR_NOT_SUPPORTED;
        goto cleanup;
    }

    // |b|
    nrmb = magma_dznrm2( b.num_rows, b.dval, inc, queue );

    // check for |b| == 0
    if ( nrmb == 0.0 ) {
        magma_zscal( dofx, MAGMA_Z_ZERO, x->dval, 1, queue );
        solver_par->init_res = 0.0;
        solver_par->final_res = 0.0;
        solver_par->iter_res = 0.0;
        solver_par->runtime = 0.0;
        goto cleanup;
    }

    // t = 0
    // make t twice as large to contain both, dt and dr
    ldd = magma_roundup( b.num_rows, 32 );
    CHECK( magma_zvinit( &dt, Magma_DEV, ldd, 2 * b.num_cols, c_zero, queue ));
    dt.num_rows = b.num_rows;
    dt.num_cols = b.num_cols;
    dt.nnz = dt.num_rows * dt.num_cols;
    doft = dt.ld * dt.num_cols;

    // redirect the dr.dval to the second part of dt
    CHECK( magma_zvinit( &dr, Magma_DEV, b.num_rows, b.num_cols, c_zero, queue ));
    magma_free( dr.dval );
    dr.dval = dt.dval + ldd * b.num_cols;

    // r = b - A x
    CHECK( magma_zresidualvec( A, b, *x, &dr, &nrmr, queue ));
    
    // |r|
    solver_par->init_res = nrmr;
    solver_par->final_res = solver_par->init_res;
    solver_par->iter_res = solver_par->init_res;
    if ( solver_par->verbose > 0 ) {
        solver_par->res_vec[0] = (real_Double_t) nrmr;
    }

    // check if initial is guess good enough
    if ( nrmr <= solver_par->atol ||
        nrmr/nrmb <= solver_par->rtol ) {
        goto cleanup;
    }

    // P = randn(n, s)
    // P = ortho(P)
//---------------------------------------
    // P = 0.0
    CHECK( magma_zvinit( &dP, Magma_CPU, A.num_cols, s, c_zero, queue ));

    // P = randn(n, s)
    distr = 3;        // 1 = unif (0,1), 2 = unif (-1,1), 3 = normal (0,1) 
    dofP = dP.num_rows * dP.num_cols;
    lapackf77_zlarnv( &distr, iseed, &dofP, dP.val );

    // transfer P to device
    CHECK( magma_zmtransfer( dP, &dP1, Magma_CPU, Magma_DEV, queue ));
    magma_zmfree( &dP, queue );

    // P = ortho(P1)
    if ( dP1.num_cols > 1 ) {
        // P = magma_zqr(P1), QR factorization
        CHECK( magma_zqr( dP1.num_rows, dP1.num_cols, dP1, dP1.ld, &dP, NULL, queue ));
    } else {
        // P = P1 / |P1|
        nrm = magma_dznrm2( dofP, dP1.dval, inc, queue );
        nrm = 1.0 / nrm;
        magma_zdscal( dofP, nrm, dP1.dval, 1, queue );
        CHECK( magma_zmtransfer( dP1, &dP, Magma_DEV, Magma_DEV, queue ));
    }
    magma_zmfree( &dP1, queue );
//---------------------------------------

    // allocate memory for the scalar products
    CHECK( magma_zvinit( &skp, Magma_CPU, 4, 1, c_zero, queue ));
    CHECK( magma_zvinit( &dskp, Magma_DEV, 4, 1, c_zero, queue ));

    CHECK( magma_zvinit( &alpha, Magma_CPU, s, 1, c_zero, queue ));
    CHECK( magma_zvinit( &dalpha, Magma_DEV, s, 1, c_zero, queue ));

    CHECK( magma_zvinit( &beta, Magma_CPU, s, 1, c_zero, queue ));
    CHECK( magma_zvinit( &dbeta, Magma_DEV, s, 1, c_zero, queue ));
    
    // workspace for merged dot product
    CHECK( magma_zmalloc( &d1, max(2,s) * dofb ));
    CHECK( magma_zmalloc( &d2, max(2,s) * dofb ));

    // smoothing enabled
    if ( smoothing == 1 ) {
        // set smoothing solution vector
        CHECK( magma_zmtransfer( *x, &dxs, Magma_DEV, Magma_DEV, queue ));

        // tt = 0
        // make tt twice as large to contain both, dtt and drs
        ldd = magma_roundup( b.num_rows, 32 );
        CHECK( magma_zvinit( &dtt, Magma_DEV, ldd, 2 * b.num_cols, c_zero, queue ));
        dtt.num_rows = b.num_rows;
        dtt.num_cols = b.num_cols;
        dtt.nnz = dtt.num_rows * dtt.num_cols;

        // redirect the drs.dval to the second part of dtt
        CHECK( magma_zvinit( &drs, Magma_DEV, b.num_rows, b.num_cols, c_zero, queue ));
        magma_free( drs.dval );
        drs.dval = dtt.dval + ldd * b.num_cols;

        // set smoothing residual vector
        magma_zcopy( dofr, dr.dval, 1, drs.dval, 1, queue );
    }

    // G(n,s) = 0
    if ( s > 1 ) {
        ldd = magma_roundup( A.num_rows, 32 );
        CHECK( magma_zvinit( &dG, Magma_DEV, ldd, s, c_zero, queue ));
        dG.num_rows = A.num_rows;
    } else {
        CHECK( magma_zvinit( &dG, Magma_DEV, A.num_rows, s, c_zero, queue ));
    }
    CHECK( magma_zvinit( &dGcol, Magma_DEV, A.num_rows, 1, c_zero, queue ));
    magma_free(dGcol.dval);

    // U(n,s) = 0
    if ( s > 1 ) {
        ldd = magma_roundup( A.num_cols, 32 );
        CHECK( magma_zvinit( &dU, Magma_DEV, ldd, s, c_zero, queue ));
        dU.num_rows = A.num_cols;
    } else {
        CHECK( magma_zvinit( &dU, Magma_DEV, A.num_cols, s, c_zero, queue ));
    }

    // M1 = 0
    // M(s,s) = I
    CHECK( magma_zvinit( &dM1, Magma_DEV, s, s, c_zero, queue ));
    CHECK( magma_zvinit( &dM, Magma_DEV, s, s, c_zero, queue ));
    CHECK( magma_zvinit( &hMdiag, Magma_CPU, s, 1, c_zero, queue ));
    dofM = dM.num_rows * dM.num_cols;
    magmablas_zlaset( MagmaFull, dM.num_rows, dM.num_cols, c_zero, c_one, dM.dval, dM.ld, queue );

    // f = 0
    CHECK( magma_zvinit( &df, Magma_DEV, dP.num_cols, dr.num_cols, c_zero, queue ));

    // c = 0
    CHECK( magma_zvinit( &dc, Magma_DEV, dM.num_cols, dr.num_cols, c_zero, queue ));

    // v1 = 0
    // v = 0
    CHECK( magma_zvinit( &dv1, Magma_DEV, dr.num_rows, dr.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &dv, Magma_DEV, dr.num_rows, dr.num_cols, c_zero, queue ));

    //--------------START TIME---------------
    // chronometry
    tempo1 = magma_sync_wtime( queue );
    if ( solver_par->verbose > 0 ) {
        solver_par->timing[0] = 0.0;
    }
    
cudaProfilerStart();

    om = MAGMA_Z_ONE;
    innerflag = 0;
    solver_par->numiter = 0;
    solver_par->spmv_count = 0;

    // start iteration
    do
    {
        solver_par->numiter++;
    
        // new RHS for small systems
        // f = (r' P)' = P' r
        magma_zgemvmdot_shfl( dP.num_rows, dP.num_cols, dP.dval, dr.dval, d1, d2, df.dval, queue );
        //magmablas_zgemv( MagmaConjTrans, dP.num_rows, dP.num_cols, c_one, dP.dval, dP.ld, dr.dval, 1, c_zero, df.dval, 1, queue );

        // shadow space loop
        for ( k = 0; k < s; k++ ) {
            sk = s - k;
    
            // solve small system and make v orthogonal to P
            // f(k:s) = M(k:s,k:s) c(k:s)
            magma_zcopy( sk, &df.dval[k], 1, &dc.dval[k], 1, queue );
            magma_ztrsv( MagmaLower, MagmaNoTrans, MagmaNonUnit, sk, &dM.dval[k*dM1.ld+k], dM.ld, &dc.dval[k], 1, queue );

            // v1 = r - G(:,k:s) c(k:s)
            magma_zcopy( dofr, dr.dval, 1, dv1.dval, 1, queue );
            magmablas_zgemv( MagmaNoTrans, dG.num_rows, sk, c_n_one, &dG.dval[k*dG.ld], dG.ld, &dc.dval[k], 1, c_one, dv1.dval, 1, queue );

            // compute new U
            // U(:,k) = om * v1 + U(:,k:s) c(k:s)
            magmablas_zgemv( MagmaNoTrans, dU.num_rows, sk, c_one, &dU.dval[k*dU.ld], dU.ld, &dc.dval[k], 1, om, dv1.dval, 1, queue );
            magma_zcopy( dU.num_rows, dv1.dval, 1, &dU.dval[k*dU.ld], 1, queue );

            // compute new G
            // G(:,k) = A U(:,k)
            dGcol.dval = &dG.dval[k*dG.ld];
            CHECK( magma_z_spmv( c_one, A, dv1, c_zero, dGcol, queue ));
            solver_par->spmv_count++;

            // bi-orthogonalize the new basis vectors
            for ( i = 0; i < k; ++i ) {
                // alpha = P(:,i)' G(:,k) / M(i,i)
                alpha.val[i] = magma_zdotc( dP.num_rows, &dP.dval[i*dP.ld], 1, &dG.dval[k*dG.ld], 1, queue );
                alpha.val[i] = alpha.val[i] / hMdiag.val[i];
                
                // G(:,k) = G(:,k) - alpha * G(:,i)
                magma_zaxpy( dG.num_rows, -alpha.val[i], &dG.dval[i*dG.ld], 1, &dG.dval[k*dG.ld], 1, queue );
            }

            // U update outside of loop using GEMV
            if ( k > 0 ) {
                // U(:,k) = U(:,k) - alpha * U(:,i)
                // copy scalars alpha needed for gemv to device
                magma_zsetvector( k, alpha.val, 1, dalpha.dval, 1, queue );
                magmablas_zgemv( MagmaNoTrans, dU.num_rows, k, c_n_one, &dU.dval[0], dU.ld, &dalpha.dval[0], 1, c_one, &dU.dval[k*dU.ld], 1, queue );
            }
            
            // new column of M = P'G, first k-1 entries are zero
            // M(k:s,k) = (G(:,k)' P(:,k:s))' = P(:,k:s)' G(:,k)
            magma_zgemvmdot_shfl( dP.num_rows, sk, &dP.dval[k*dP.ld], &dG.dval[k*dG.ld], d1, d2, &dM.dval[k*dM.ld+k], queue );
            magma_zgetvector( k+1, dM.dval, dM.ld+1, hMdiag.val, 1, queue );

            // check M(k,k) == 0
            if ( MAGMA_Z_EQUAL(hMdiag.val[k], MAGMA_Z_ZERO) ) {
                s = k; // for the x-update outside the loop
                info = MAGMA_DIVERGENCE;
                innerflag = 1;
                break;
            }

            // beta = f(k) / M(k,k)
            magma_zgetvector( 1, &df.dval[k], 1, &fk, 1, queue );
            beta.val[k] = fk / hMdiag.val[k];

            // make r orthogonal to q_i, i = 1..k
            // r = r - beta * G(:,k)
            magma_zaxpy( dr.num_rows, -beta.val[k], &dG.dval[k*dG.ld], 1, dr.dval, 1, queue );

            // smoothing disabled 
            if ( smoothing == 0 ) {
                // |r|
                nrmr = magma_dznrm2( dofb, dr.dval, inc, queue );

            // smoothing enabled 
            } else {
                // x = x + beta * U(:,k)
                magma_zaxpy( x->num_rows, beta.val[k], &dU.dval[k*dU.ld], 1, x->dval, 1, queue );

                // smoothing operation
//---------------------------------------
                // t = rs - r
                magma_zidr_smoothing_1( drs.num_rows, drs.num_cols, drs.dval, dr.dval, dtt.dval, queue );

                // gamma = (t' * rs) / (|t| * |t|)
                CHECK( magma_zgemvmdot_shfl( doft, 2, dtt.dval, dtt.dval, d1, d2, dskp.dval+2, queue ));
                magma_zgetvector( 2 , dskp.dval+2, 1, skp.val+2, 1, queue );
                gamma = skp.val[3] / skp.val[2];
                
                // rs = rs - gamma * (rs - r) 
                magma_zaxpy( drs.num_rows, -gamma, dtt.dval, inc, drs.dval, inc, queue );

                // xs = xs - gamma * (xs - x) 
                magma_zidr_smoothing_2( dxs.num_rows, dxs.num_cols, -gamma, x->dval, dxs.dval, queue );

                // |rs|
                nrmr = magma_dznrm2( dofb, drs.dval, inc, queue );       
//---------------------------------------
            }

            // store current timing and residual
            if ( solver_par->verbose > 0 ) {
                tempo2 = magma_sync_wtime( queue );
                if ( (solver_par->numiter) % solver_par->verbose == 0 ) {
                    solver_par->res_vec[(solver_par->numiter) / solver_par->verbose]
                            = (real_Double_t) nrmr;
                    solver_par->timing[(solver_par->numiter) / solver_par->verbose]
                            = (real_Double_t) tempo2 - tempo1;
                }
            }

            // check convergence or iteration limit
            if ( nrmr <= solver_par->atol ||
                nrmr/nrmb <= solver_par->rtol || 
                solver_par->numiter >= solver_par->maxiter ) {
                s = k; // for the x-update outside the loop
                innerflag = 1;
                break;
            }

            // new f = P' r (first k components are zero)
            if ( (k + 1) < s ) {
                // f(k+1:s) = f(k+1:s) - beta * M(k+1:s,k)
                magma_zaxpy( sk - 1, -beta.val[k], &dM.dval[k*dM.ld+(k+1)], 1, &df.dval[k+1], 1, queue );
            }
        }

        // update solution approximation x
        if ( smoothing == 0 ) {
            magma_zsetvector( s, beta.val, 1, dbeta.dval, 1, queue );
            magmablas_zgemv( MagmaNoTrans, dU.num_rows, s, c_one, &dU.dval[0], dU.ld, &dbeta.dval[0], 1, c_one, &x->dval[0], 1, queue );
        }
 
        // check convergence or iteration limit or invalid of inner loop
        if ( innerflag == 1 ) {
            break;
        }

        // v = r
        magma_zcopy( dofr, dr.dval, 1, dv.dval, 1, queue );

        // t = A v
        CHECK( magma_z_spmv( c_one, A, dv, c_zero, dt, queue ));
        solver_par->spmv_count++;

        // computation of a new omega
//---------------------------------------
        // |t|
        CHECK( magma_zgemvmdot_shfl( doft, 2, dt.dval, dt.dval, d1, d2, dskp.dval, queue ));
        magma_zgetvector( 2 , dskp.dval, 1, skp.val, 1, queue );

        // tr = t' * r
        nrmt = magma_dsqrt( MAGMA_Z_REAL(skp.val[0]) );
        tr = skp.val[1];
        
        // rho = abs((t' * r) / (|t| * |r|))
        rho = MAGMA_D_ABS( MAGMA_Z_REAL(tr) / (nrmt * nrmr) );

        // om = (t' * r) / (|t| * |t|)
        om = tr / (nrmt * nrmt);
        if ( rho < angle )
            om = (om * angle) / rho;
//---------------------------------------
        if ( MAGMA_Z_EQUAL(om, MAGMA_Z_ZERO) ) {
            info = MAGMA_DIVERGENCE;
            break;
        }

        // update approximation vector
        // x = x + om * v
        magma_zaxpy( x->num_rows, om, dv.dval, 1, x->dval, 1, queue );

        // update residual vector
        // r = r - om * t
        magma_zaxpy( dr.num_rows, -om, dt.dval, 1, dr.dval, 1, queue );

        // smoothing disabled
        if ( smoothing == 0 ) {
            // residual norm
            nrmr = magma_dznrm2( dofb, dr.dval, inc, queue );

        // smoothing enabled
        } else {
            // smoothing operation
//---------------------------------------
            // t = rs - r
            magma_zidr_smoothing_1( drs.num_rows, drs.num_cols, drs.dval, dr.dval, dtt.dval, queue );

            // gamma = (t' * rs) / (|t| * |t|)
            CHECK( magma_zgemvmdot_shfl( doft, 2, dtt.dval, dtt.dval, d1, d2, dskp.dval+2, queue ));
            magma_zgetvector( 2 , dskp.dval+2, 1, skp.val+2, 1, queue );
            gamma = skp.val[3] / skp.val[2];

            // rs = rs - gamma * (rs - r) 
            magma_zaxpy( drs.num_rows, -gamma, dtt.dval, 1, drs.dval, 1, queue );

            // xs = xs - gamma * (xs - x) 
            magma_zidr_smoothing_2( dxs.num_rows, dxs.num_cols, -gamma, x->dval, dxs.dval, queue );

            // |rs|
            nrmr = magma_dznrm2( dofb, drs.dval, inc, queue );           
//---------------------------------------
        }

        // store current timing and residual
        if ( solver_par->verbose > 0 ) {
            tempo2 = magma_sync_wtime( queue );
            if ( (solver_par->numiter) % solver_par->verbose == 0 ) {
                solver_par->res_vec[(solver_par->numiter) / solver_par->verbose]
                        = (real_Double_t) nrmr;
                solver_par->timing[(solver_par->numiter) / solver_par->verbose]
                        = (real_Double_t) tempo2 - tempo1;
            }
        }

        // check convergence or iteration limit
        if ( nrmr <= solver_par->atol ||
            nrmr/nrmb <= solver_par->rtol || 
            solver_par->numiter >= solver_par->maxiter ) {
            break;
        }
    }
    while ( solver_par->numiter + 1 <= solver_par->maxiter );
            
    if ( smoothing == 1 ) {
        magma_zcopy( dofr, drs.dval, 1, dr.dval, 1, queue );
        magma_zcopy( dofx, dxs.dval, 1, x->dval, 1, queue );
    }

cudaProfilerStop();

    // get last iteration timing
    tempo2 = magma_sync_wtime( queue );
    solver_par->runtime = (real_Double_t) tempo2 - tempo1;
//--------------STOP TIME----------------

    // get final stats
    solver_par->iter_res = nrmr;
    CHECK( magma_zresidualvec( A, b, *x, &dr, &residual, queue ));
    solver_par->final_res = residual;

    // set solver conclusion
    if ( info != MAGMA_SUCCESS && info != MAGMA_NOTCONVERGED ) {
        if ( solver_par->init_res > solver_par->final_res ) {
            info = MAGMA_SLOW_CONVERGENCE;
        }
    }

    
cleanup:
    // free resources
    dr.dval = NULL;   // needed because its pointer is redirected to dt
    drs.dval = NULL;  // needed because its pointer is redirected to dtt
    magma_zmfree( &dP1, queue );
    magma_zmfree( &dP, queue );
    magma_zmfree( &dr, queue );
    magma_zmfree( &dG, queue );
    magma_zmfree( &dU, queue );
    magma_zmfree( &dM1, queue );
    magma_zmfree( &dM, queue );
    magma_zmfree( &df, queue );
    magma_zmfree( &dt, queue );
    magma_zmfree( &dc, queue );
    magma_zmfree( &dv1, queue );
    magma_zmfree( &dv, queue );
    magma_zmfree( &dxs, queue );
    magma_zmfree( &drs, queue ); 
    magma_zmfree( &dtt, queue );
    magma_zmfree( &dskp, queue );
    magma_zmfree( &dalpha, queue );
    magma_zmfree( &dbeta, queue );
    magma_zmfree( &skp, queue );
    magma_zmfree( &alpha, queue );
    magma_zmfree( &beta, queue );
    magma_free( d1 );
    magma_free( d2 );

    solver_par->info = info;
    return info;
    /* magma_zidr_acc */
}
