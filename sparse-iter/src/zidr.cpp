/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Hartwig Anzt
       @author Eduardo Ponce

       @precisions normal z -> s d c
*/

#include "common_magmasparse.h"
#include <cuda_profiler_api.h>

#define RTOLERANCE     lapackf77_dlamch( "E" )
#define ATOLERANCE     lapackf77_dlamch( "E" )


/**
    Purpose
    -------

    Solves a system of linear equations
       A * X = B
    where A is a complex Hermitian N-by-N positive definite matrix A.
    This is a GPU implementation of the Induced Dimension Reduction method.

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

// -1 = no print but write iniital p
//  0 = no printing
//  1 = print scalars
//  2 = print all (few iters)
// >2 = prints all (all iters)
#define MYDEBUG 0

#if MYDEBUG <= 0
#define printD(...)
#define printMatrix(s,m)
#elif MYDEBUG == 1
#define printD(...) printf("%% " __VA_ARGS__)
#define printMatrix(s,m)
#elif MYDEBUG == 2
#define printD(...) printf("%% " __VA_ARGS__)
#define printMatrix(s,m) magma_zmatrixInfo(s,m)
#else
#define printD(...) printf("%% " __VA_ARGS__)
#define printMatrix(s,m) magma_zmatrixInfo(s,m)
#endif

#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define MAX(a,b) (((a) > (b)) ? (a) : (b))


extern "C" void
magma_zmatrixInfo(
    const char *s,
    magma_z_matrix A ) {

    printD(" %s dims = %d x %d\n", s, A.num_rows, A.num_cols);
    printD(" %s location = %d = %s\n", s, A.memory_location, (A.memory_location == Magma_CPU) ? "CPU" : "DEV");
    printD(" %s storage = %d = %s\n", s, A.storage_type, (A.storage_type == Magma_CSR) ? "CSR" : "DENSE");
    printD(" %s major = %d = %s\n", s, A.major, (A.major == MagmaRowMajor) ? "row" : "column");
    printD(" %s nnz = %d\n", s, A.nnz);
    if (A.memory_location == Magma_DEV)
        magma_zprint_gpu( A.num_rows, A.num_cols, A.dval, A.num_rows );
    else
        magma_zprint( A.num_rows, A.num_cols, A.val, A.num_rows );
}


extern "C" magma_int_t
magma_zidr(
    magma_z_matrix A, magma_z_matrix b, magma_z_matrix *x,
    magma_z_solver_par *solver_par,
    magma_queue_t queue )
{
    // set queue for old dense routines
    magma_queue_t orig_queue = NULL;
    magmablasGetKernelStream( &orig_queue );

    // prepare solver feedback
    solver_par->solver = Magma_IDR;
    solver_par->numiter = 0;
    solver_par->info = MAGMA_SUCCESS;

    // constants
    const magmaDoubleComplex c_zero = MAGMA_Z_ZERO;
    const magmaDoubleComplex c_one = MAGMA_Z_ONE;
    const magmaDoubleComplex c_n_one = MAGMA_Z_NEG_ONE;

    // internal user parameters
    const magma_int_t smoothing = 1;   // 1 = enable, 0 = disabled, -1 = disabled with delayed x update
    const double angle = 0.7;          // [0-1]

    // local variables
    magma_int_t info = 0;
    magma_int_t iseed[4] = { 0, 0, 0, 1 };
    magma_int_t dofb = b.num_rows * b.num_cols;
    magma_int_t inc = 1;
    magma_int_t s;
    magma_int_t dof;
    magma_int_t distr;
    magma_int_t k, i, sk;
    magma_int_t *piv = NULL;
    magma_int_t innerflag;
    double residual;
    double nrm;
    double nrmb;
    double nrmr;
    double nrmt;
    double rho;
    double tolb;
    magmaDoubleComplex om;
    magmaDoubleComplex tr;
    magmaDoubleComplex gamma;
    magmaDoubleComplex alpha;
    magmaDoubleComplex mkk;
    magmaDoubleComplex fk;

    // matrices and vectors
    magma_z_matrix P1 = {Magma_CSR}, dP1 = {Magma_CSR}, dP = {Magma_CSR};
    magma_z_matrix dr = {Magma_CSR};
    magma_z_matrix dG = {Magma_CSR};
    magma_z_matrix dU = {Magma_CSR};
    magma_z_matrix dM1 = {Magma_CSR}, dM = {Magma_CSR};
    magma_z_matrix df = {Magma_CSR};
    magma_z_matrix dt = {Magma_CSR};
    magma_z_matrix dc = {Magma_CSR};
    magma_z_matrix dv1 = {Magma_CSR}, dv = {Magma_CSR};
    magma_z_matrix dxs = {Magma_CSR};
    magma_z_matrix drs = {Magma_CSR};
    magma_z_matrix dbeta = {Magma_CSR}, beta = {Magma_CSR};

    // queue variables
    const magma_queue_t squeue = 0;    // synchronous kernel queues

    // performance variables
    long long int gpumem = 0;

    // chronometry
    real_Double_t tempo1, tempo2;
   
    // set synchrounous kernel queues
    queue = squeue;
    printD("Kernel queues: (orig, queue) = (%p, %p)\n", (void *)orig_queue, (void *)queue);

// Set to Q
    magmablasSetKernelStream( queue );

    // initial s space
    // hack --> use "--restart" option as the shadow space number
    s = 1;
    if ( solver_par->restart != 30 ) {
        if ( solver_par->restart > A.num_cols )
            s = A.num_cols;
        else
            s = solver_par->restart;
    }
    solver_par->restart = s;

    // set max iterations
    solver_par->maxiter = MIN( 2 * A.num_cols, solver_par->maxiter );

    // check if matrix A is square
    if ( A.num_rows != A.num_cols ) {
        printD("Error! matrix must be square.\n");
        info = MAGMA_ERR;
        goto cleanup;
    }
    gpumem += (A.nnz * sizeof(magmaDoubleComplex)) + (A.nnz * sizeof(magma_index_t)) + ((A.num_rows + 1) * sizeof(magma_index_t));

    // initial solution vector
    // x = 0
    //magmablas_zlaset( MagmaFull, x->num_rows, x->num_cols, c_zero, c_zero, x->dval, x->num_rows );
    printMatrix("X", *x);
    gpumem += x->nnz * sizeof(magmaDoubleComplex);

    // initial RHS
    // b = 1
    //magmablas_zlaset( MagmaFull, b.num_rows, b.num_cols, c_one, c_one, b.dval, b.num_rows );
    printMatrix("B", b);
    gpumem += b.nnz * sizeof(magmaDoubleComplex);

    // |b|
    nrmb = magma_dznrm2( b.num_rows, b.dval, inc );
    printD("init norm(b) ..........%lg\n", nrmb);

    // check for |b| == 0
    if ( nrmb == 0.0 ) {
        printD("RHS is zero, exiting...\n");
        magma_zscal( x->num_rows * x->num_cols, MAGMA_Z_ZERO, x->dval, inc );
        solver_par->init_res = 0.0;
        solver_par->final_res = 0.0;
        solver_par->iter_res = 0.0;
        solver_par->runtime = 0.0;
        goto cleanup;
    }

    // relative tolerance
    tolb = nrmb * solver_par->epsilon;
    if ( tolb < ATOLERANCE ) {
        tolb = ATOLERANCE;
    }

    CHECK( magma_zvinit( &dr, Magma_DEV, b.num_rows, b.num_cols, c_zero, queue ));
    gpumem += dr.nnz * sizeof(magmaDoubleComplex);

    // r = b - A x
    CHECK(  magma_zresidualvec( A, b, *x, &dr, &nrmr, queue ));
    printMatrix("R", dr);
    
    // |r|
    solver_par->init_res = nrmr;
    if ( solver_par->verbose > 0 ) {
        solver_par->res_vec[0] = (real_Double_t)nrmr;
    }
   
    // check if initial is guess good enough
    if ( nrmr <= tolb ) {
        solver_par->final_res = solver_par->init_res;
        solver_par->iter_res = solver_par->init_res;
        goto cleanup;
    }

    // P = randn(n, s)
    // P = ortho(P)
//---------------------------------------
    // P1 = 0.0
    CHECK( magma_zvinit( &P1, Magma_CPU, A.num_cols, s, c_zero, queue ));

    // P1 = randn(n, s)
    distr = 3;        // 1 = unif (0,1), 2 = unif (-1,1), 3 = normal (0,1) 
    dof = P1.num_rows * P1.num_cols;
    lapackf77_zlarnv( &distr, iseed, &dof, P1.val );
    printMatrix("P1", P1);

    // transfer P1 to device
    CHECK( magma_zmtransfer( P1, &dP1, Magma_CPU, Magma_DEV, queue ));
    dP1.major = P1.major;
    magma_zmfree( &P1, queue );

    // P = ortho(P1)
    if ( dP1.num_cols > 1 ) {
        // P = magma_zqr(P1), QR factorization
        CHECK( magma_zqr( dP1.num_rows, dP1.num_cols, dP1, dP1.ld, &dP, NULL, queue ));
    } else {
        // P = P1 / |P1|
        dof = dP1.num_rows * dP1.num_cols;        // can remove
        nrm = magma_dznrm2( dof, dP1.dval, inc );
        nrm = 1.0 / nrm;
        magma_zdscal( dof, nrm, dP1.dval, inc );
        CHECK( magma_zmtransfer( dP1, &dP, Magma_DEV, Magma_DEV, queue ));
        dP.major = dP1.major;
    }
    magma_zmfree(&dP1, queue );
//---------------------------------------
    printMatrix("P", dP);
    gpumem += dP.nnz * sizeof(magmaDoubleComplex);

#if MYDEBUG == -1
    // Note: write P matrix to file to use in MATLAB for validation
    printf("P = ");
    magma_zprint_gpu( dP.num_rows, dP.num_cols, dP.dval, dP.num_rows );
#endif

    // allocate memory for the scalar products
    CHECK( magma_zvinit( &beta, Magma_CPU, s, 1, c_zero, queue ));
    CHECK( magma_zvinit( &dbeta, Magma_DEV, s, 1, c_zero, queue ));
    gpumem += dbeta.nnz * sizeof(magmaDoubleComplex);

    if ( smoothing > 0 ) {
        // set smoothing solution vector
        CHECK( magma_zmtransfer( *x, &dxs, Magma_DEV, Magma_DEV, queue ));
        dxs.major = x->major;
        gpumem += dxs.nnz * sizeof(magmaDoubleComplex);

        // set smoothing residual vector
        CHECK( magma_zmtransfer( dr, &drs, Magma_DEV, Magma_DEV, queue ));
        drs.major = dr.major;
        gpumem += drs.nnz * sizeof(magmaDoubleComplex);
    }

    // G(n,s) = 0
    CHECK( magma_zvinit( &dG, Magma_DEV, A.num_cols, s, c_zero, queue ));
    gpumem += dG.nnz * sizeof(magmaDoubleComplex);

    // U(n,s) = 0
    CHECK( magma_zvinit( &dU, Magma_DEV, A.num_cols, s, c_zero, queue ));
    gpumem += dU.nnz * sizeof(magmaDoubleComplex);

    // M1 = 0
    // M(s,s) = I
    CHECK( magma_zvinit( &dM1, Magma_DEV, s, s, c_zero, queue ));
    CHECK( magma_zvinit( &dM, Magma_DEV, s, s, c_zero, queue ));
    magmablas_zlaset( MagmaFull, s, s, c_zero, c_one, dM.dval, s );
    gpumem += (dM1.nnz + dM.nnz) * sizeof(magmaDoubleComplex);

    // f = 0
    CHECK( magma_zvinit( &df, Magma_DEV, dP.num_cols, dr.num_cols, c_zero, queue ));
    gpumem += df.nnz * sizeof(magmaDoubleComplex);

    // t = 0
    CHECK( magma_zvinit( &dt, Magma_DEV, A.num_rows, dr.num_cols, c_zero, queue ));
    gpumem += dt.nnz * sizeof(magmaDoubleComplex);

    // c = 0
    CHECK( magma_zvinit( &dc, Magma_DEV, dM.num_cols, dr.num_cols, c_zero, queue ));
    gpumem += dc.nnz * sizeof(magmaDoubleComplex);

    // v1 = 0
    // v = 0
    CHECK( magma_zvinit( &dv1, Magma_DEV, dr.num_rows, dr.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &dv, Magma_DEV, dr.num_rows, dr.num_cols, c_zero, queue ));
    gpumem += (dv1.nnz + dv.nnz) * sizeof(magmaDoubleComplex);

    // piv = 0
    CHECK( magma_imalloc_pinned(&piv, s));

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

    // start iteration
    do
    {
        solver_par->numiter++;
    
        // new RHS for small systems
        // f = (r' P)' = P' r
        magmablas_zgemv( MagmaConjTrans, dP.num_rows, dP.num_cols, c_one, dP.dval, dP.num_rows, dr.dval, inc, c_zero, df.dval, inc );
        printMatrix("F", df);

        // shadow space loop
        for ( k = 0; k < s; ++k ) {
            sk = s - k;
    
            // solve small system and make v orthogonal to P
            // f(k:s) = M(k:s,k:s) c(k:s)
//---------------------------------------
            // c(k:s) = f(k:s)
            //magma_zcopy( sk, &df.dval[k], inc, &dc.dval[k], inc );
            magma_zcopyvector( sk, &df.dval[k], inc, &dc.dval[k], inc );

            // M1 = M
            //magma_zcopy( dM.num_rows * dM.num_cols, dM.dval, inc, dM1.dval, inc );
            magma_zcopyvector( dM.num_rows * dM.num_cols, dM.dval, inc, dM1.dval, inc );

            // c(k:s) = M1(k:s,k:s) \ c(k:s)
            CHECK( magma_zgesv_gpu( sk, dc.num_cols, &dM1.dval[k*dM1.num_rows+k], dM1.num_rows, piv, &dc.dval[k], dc.num_rows, &info ) );
//---------------------------------------
            printMatrix("C", dc);

            // v1 = r - G(:,k:s) c(k:s)
//---------------------------------------
            // v1 = r
            //magma_zcopy( dr.num_rows * dr.num_cols, dr.dval, inc, dv1.dval, inc );
            magma_zcopyvector( dr.num_rows * dr.num_cols, dr.dval, inc, dv1.dval, inc );

            // v1 = v1 - G(:,k:s) c(k:s)
            magmablas_zgemv( MagmaNoTrans, dG.num_rows, sk, c_n_one, &dG.dval[k*dG.num_rows], dG.num_rows, &dc.dval[k], inc, c_one, dv1.dval, inc );
//---------------------------------------
            printMatrix("V", dv1);

            // compute new U
            // U(:,k) = om * v1 + U(:,k:s) c(k:s)
//---------------------------------------
            // v1 = om * v1 + U(:,k:s) c(k:s)
            magmablas_zgemv( MagmaNoTrans, dU.num_rows, sk, c_one, &dU.dval[k*dU.num_rows], dU.num_rows, &dc.dval[k], inc, om, dv1.dval, inc );

            // U(:,k) = v1
            //magma_zcopy( dU.num_rows, dv1.dval, inc, &dU.dval[k*dU.num_rows], inc );
            magma_zcopyvector( dU.num_rows, dv1.dval, inc, &dU.dval[k*dU.num_rows], inc );
//---------------------------------------
            printMatrix("U", dU);

            // compute new G
            // G(:,k) = A U(:,k)
//---------------------------------------
            // v = A v1
            CHECK( magma_z_spmv( c_one, A, dv1, c_zero, dv, queue ));

            // G(:,k) = v
            //magma_zcopy( dG.num_rows, dv.dval, inc, &dG.dval[k*dG.num_rows], inc );
            magma_zcopyvector( dG.num_rows, dv.dval, inc, &dG.dval[k*dG.num_rows], inc );
//---------------------------------------
            printMatrix("G", dG);


            // bi-orthogonalize the new basis vectors
            for ( i = 0; i < k; ++i ) {

                // alpha = P(:,i)' G(:,k) / M(i,i)
//---------------------------------------
                // alpha = P(:,i)' G(:,k)
                alpha = magma_zdotc( dP.num_rows, &dP.dval[i*dP.num_rows], inc, &dG.dval[k*dG.num_rows], inc );
                
                // alpha = alpha / M(i,i)
                magma_zgetvector( 1, &dM.dval[i*dM.num_rows+i], inc, &mkk, inc );
                alpha = alpha / mkk;
//---------------------------------------
                printD("bi-ortho: i, k, alpha ...................%d, %d, (%lg, %lg)\n", i, k, MAGMA_Z_REAL(alpha), MAGMA_Z_IMAG(alpha));

                // G(:,k) = G(:,k) - alpha * G(:,i)
                magma_zaxpy( dG.num_rows, -alpha, &dG.dval[i*dG.num_rows], inc, &dG.dval[k*dG.num_rows], inc );
                printMatrix("G", dG);

                // U(:,k) = U(:,k) - alpha * U(:,i)
                magma_zaxpy( dU.num_rows, -alpha, &dU.dval[i*dU.num_rows], inc, &dU.dval[k*dU.num_rows], inc );
            }
#if MYDEBUG > 0
            if ( k > 0 ) {
                printMatrix("U", dU);
            }
#endif

            // new column of M = P'G, first k-1 entries are zero
            // M(k:s,k) = (G(:,k)' P(:,k:s))' = P(:,k:s)' G(:,k)
            magmablas_zgemv( MagmaConjTrans, dP.num_rows, sk, c_one, &dP.dval[k*dP.num_rows], dP.num_rows, &dG.dval[k*dG.num_rows], inc, c_zero, &dM.dval[k*dM.num_rows+k], inc );
            printMatrix("M", dM);

            // check M(k,k) == 0
            magma_zgetvector( 1, &dM.dval[k*dM.num_rows+k], inc, &mkk, inc );
            if ( MAGMA_Z_EQUAL(mkk, MAGMA_Z_ZERO) ) {
                info = MAGMA_DIVERGENCE;
                innerflag = 1;
                break;
            }

            // beta = f(k) / M(k,k)
            magma_zgetvector( 1, &df.dval[k], inc, &fk, inc );
            beta.val[k] = fk / mkk;
            printD("beta: k ...................%d, (%lg, %lg)\n", k, MAGMA_Z_REAL(beta.val[k]), MAGMA_Z_IMAG(beta.val[k]));

            // make r orthogonal to q_i, i = 1..k
            // r = r - beta * G(:,k)
            magma_zaxpy( dr.num_rows, -beta.val[k], &dG.dval[k*dG.num_rows], 1, dr.dval, 1 );
            printMatrix("R", dr);

            if ( smoothing < 0 ) {
                // |r|
                nrmr = magma_dznrm2( dofb, dr.dval, inc );
                printD("norm(r): k ...................%d, %lg\n", k, nrmr);

            } else if ( smoothing == 0 ) {
                // x = x + beta * U(:,k)
                magma_zaxpy( x->num_rows, beta.val[k], &dU.dval[k*dU.num_rows], 1, x->dval, 1 );
                printMatrix("X", *x);

                // |r|
                nrmr = magma_dznrm2( dofb, dr.dval, inc );
                printD("norm(r): k ...................%d, %lg\n", k, nrmr);

            } else {
                // x = x + beta * U(:,k)
                magma_zaxpy( x->num_rows, beta.val[k], &dU.dval[k*dU.num_rows], 1, x->dval, 1 );
                printMatrix("X", *x);

                // smoothing operation
//---------------------------------------
                // t = rs - r
                //magma_zcopy( drs.num_rows * drs.num_cols, drs.dval, inc, dt.dval, inc );
                magma_zcopyvector( drs.num_rows * drs.num_cols, drs.dval, inc, dt.dval, inc );
                magma_zaxpy( dt.num_rows, c_n_one, dr.dval, inc, dt.dval, inc );

                // |t|
                dof = dt.num_rows * dt.num_cols;
                nrmt = magma_dznrm2( dof, dt.dval, inc );

                // gamma = t' rs
                gamma = magma_zdotc( dt.num_rows, dt.dval, inc, drs.dval, inc );

                // gamma = gamma / (|t| * |t|)
                gamma = gamma / (nrmt * nrmt);

                // rs = rs - gamma * t
                magma_zaxpy( drs.num_rows, -gamma, dt.dval, inc, drs.dval, inc );
                printMatrix("RS", drs);

                // t = xs - x
                //magma_zcopy( dxs.num_rows * dxs.num_cols, dxs.dval, inc, dt.dval, inc );
                magma_zcopyvector( dxs.num_rows * dxs.num_cols, dxs.dval, inc, dt.dval, inc );
                magma_zaxpy( dt.num_rows, c_n_one, x->dval, inc, dt.dval, inc );

                // xs = xs - gamma * t
                magma_zaxpy( dxs.num_rows, -gamma, dt.dval, inc, dxs.dval, inc );
                printMatrix("XS", dxs);

                // |rs|
                nrmr = magma_dznrm2( dofb, drs.dval, inc );           
                printD("norm(rs): k ...................%d, %lg\n", k, nrmr);
            }
//---------------------------------------

            // store current timing and residual
            if ( solver_par->verbose > 0 ) {
                tempo2 = magma_sync_wtime( queue );
                if ( (solver_par->numiter)%solver_par->verbose==0 ) {
                    solver_par->res_vec[(solver_par->numiter)/solver_par->verbose]
                            = (real_Double_t) nrmr;
                    solver_par->timing[(solver_par->numiter)/solver_par->verbose]
                            = (real_Double_t) tempo2-tempo1;
                }
            }

            // check convergence or iteration limit
            if ( nrmr <= tolb || solver_par->numiter >= solver_par->maxiter ) {
                s = k; // for the x-update outside the loop
                innerflag = 1;
                break;
            }

            // new f = P' r (first k components are zero)
            if ( (k + 1) < s ) {
                // f(k+1:s) = f(k+1:s) - beta * M(k+1:s,k)
                magma_zaxpy( sk - 1, -beta.val[k], &dM.dval[k*dM.num_rows+(k+1)], inc, &df.dval[k+1], inc );
                printMatrix("F", df);
            }

            // iter = iter + 1
            solver_par->numiter++;
        }

        // update solution approximation x
        if ( smoothing < 0 ) {
            magma_zsetvector( s, beta.val, 1, dbeta.dval, 1 );
            magmablas_zgemv( MagmaNoTrans, dU.num_rows, s, c_one, &dU.dval[0], dU.num_rows, &dbeta.dval[0], 1, c_one, &x->dval[0], 1 );
            printMatrix("X", *x);
        }

        // check convergence or iteration limit or invalid of inner loop
        if ( innerflag == 1 ) {
            break;
        }

        // v = r
        //magma_zcopy( dr.num_rows * dr.num_cols, dr.dval, inc, dv.dval, inc );
        magma_zcopyvector( dr.num_rows * dr.num_cols, dr.dval, inc, dv.dval, inc );

        // t = A v
        CHECK( magma_z_spmv( c_one, A, dv, c_zero, dt, queue ));
        printMatrix("T", dt);

        // computation of a new omega
        // om = omega(t, r, angle);
//---------------------------------------
        // |t|
        dof = dt.num_rows * dt.num_cols;
        nrmt = magma_dznrm2( dof, dt.dval, inc );

        // tr = t' r
        tr = magma_zdotc( dr.num_rows, dt.dval, inc, dr.dval, inc );
        printD("tr, norm(t): k .................... %d, (%lg, %lg), %lg\n", k, MAGMA_Z_REAL(tr), MAGMA_Z_IMAG(tr), nrmt);

        // rho = abs(tr / (|t| * |r|))
        rho = fabs( MAGMA_Z_REAL(tr) / (nrmt * nrmr) );

        // om = tr / (|t| * |t|)
        om = tr / (nrmt * nrmt);
        if ( rho < angle )
            om = om * angle / rho;
//---------------------------------------
        printD("omega: k .................... %d, (%lg, %lg)\n", k, MAGMA_Z_REAL(om), MAGMA_Z_IMAG(om));
        if ( MAGMA_Z_EQUAL(om, MAGMA_Z_ZERO) ) {
            info = MAGMA_DIVERGENCE;
            break;
        }

        // update approximation vector
        // x = x + om * v
        magma_zaxpy(x->num_rows, om, dv.dval, inc, x->dval, inc);
        printMatrix("X", *x);

        // update residual vector
        // r = r - om * t
        magma_zaxpy(dr.num_rows, -om, dt.dval, inc, dr.dval, inc);
        printMatrix("R", dr);

        if ( smoothing < 1 ) {
            // residual norm
            nrmr = magma_dznrm2( dofb, dr.dval, inc );
            printD("norm(r): k ...................%d, %lg\n", k, nrmr);
        } else {
            // smoothing operation
//---------------------------------------
            // t = rs - r
            //magma_zcopy( drs.num_rows * drs.num_cols, drs.dval, inc, dt.dval, inc );
            magma_zcopyvector( drs.num_rows * drs.num_cols, drs.dval, inc, dt.dval, inc );
            magma_zaxpy( dt.num_rows, c_n_one, dr.dval, inc, dt.dval, inc );

            // |t|
            dof = dt.num_rows * dt.num_cols;
            nrmt = magma_dznrm2( dof, dt.dval, inc );

            // gamma = t' rs
            gamma = magma_zdotc( dt.num_rows, dt.dval, inc, drs.dval, inc );

            // gamma = gamma / (|t| * |t|)
            gamma = gamma / (nrmt * nrmt);

            // rs = rs - gamma * t
            magma_zaxpy( drs.num_rows, -gamma, dt.dval, inc, drs.dval, inc );
            printMatrix("RS", drs);

            // t = xs - x
            //magma_zcopy( dxs.num_rows * dxs.num_cols, dxs.dval, inc, dt.dval, inc );
            magma_zcopyvector( dxs.num_rows * dxs.num_cols, dxs.dval, inc, dt.dval, inc );
            magma_zaxpy( dt.num_rows, c_n_one, x->dval, inc, dt.dval, inc );

            // xs = xs - gamma * t
            magma_zaxpy( dxs.num_rows, -gamma, dt.dval, inc, dxs.dval, inc );
            printMatrix("XS", dxs);

            // |rs|
            nrmr = magma_dznrm2( dofb, drs.dval, inc );           
            printD("norm(rs): k ...................%d, %lg\n", k, nrmr);
        }
//---------------------------------------

        // store current timing and residual
        if ( solver_par->verbose > 0 ) {
            tempo2 = magma_sync_wtime( queue );
            if ( (solver_par->numiter)%solver_par->verbose==0 ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) nrmr;
                solver_par->timing[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) tempo2-tempo1;
            }
        }

        // check convergence or iteration limit
        if ( nrmr <= tolb || solver_par->numiter >= solver_par->maxiter ) {
            break;
        }

#if MYDEBUG == 2
        // Note: exit loop after a few iterations
        if ( solver_par->numiter + 1 >= (2 * (s + 1)) ) {
            break;
        }
#endif
    }
    while ( solver_par->numiter + 1 <= solver_par->maxiter );

    if ( smoothing > 0 ) {
        //magma_zcopy( dr.num_rows * dr.num_cols, drs.dval, 1, dr.dval, 1 );
        magma_zcopyvector( dr.num_rows * dr.num_cols, drs.dval, 1, dr.dval, 1 );
        //magma_zcopy( x->num_rows * x->num_cols, dxs.dval, 1, x->dval, 1 );
        magma_zcopyvector( x->num_rows * x->num_cols, dxs.dval, 1, x->dval, 1 );
    }

cudaProfilerStop();

    // get last iteration timing
    tempo2 = magma_sync_wtime( queue );
    solver_par->runtime = (real_Double_t) tempo2-tempo1;
//--------------STOP TIME----------------

    // last stats
    printMatrix("B_last", b);
    printMatrix("X_last", *x);
    printMatrix("R_last", dr);
    printD("last norm(r): ................. %lg\n", nrmr);
    solver_par->iter_res = nrmr;
    CHECK( magma_zresidualvec( A, b, *x, &dr, &residual, queue ));
    solver_par->final_res = residual;
    printD("last residual: ................. %lg\n", residual);

    // set solver conclusion
    if ( info != MAGMA_SUCCESS ) {
        if ( solver_par->init_res > solver_par->final_res ) {
            info = MAGMA_SLOW_CONVERGENCE;
        }
        else {
            info = MAGMA_DIVERGENCE;
        }
    }
//---------------------------------------

    // print local stats
    printD("GPU memory = %f MB\n", (real_Double_t)gpumem / (1<<20));
    
cleanup:
    // free resources
    magma_zmfree( &P1, queue );
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
    magma_zmfree( &dbeta, queue );
    magma_zmfree( &beta, queue );
    magma_free_pinned( piv );

    magmablasSetKernelStream( orig_queue );
    solver_par->info = info;
    return info;
    /* magma_zidr */
}
