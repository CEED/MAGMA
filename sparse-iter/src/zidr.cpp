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

#include <assert.h>

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

#define MYDEBUG 0
#define WRITEP 0

#if MYDEBUG == 1
#define printD(...) printf(__VA_ARGS__)
#define printMatrix(s,m)
#elif MYDEBUG == 2
#define printD(...) printf(__VA_ARGS__)
#define printMatrix(s,m) magma_zmatrixInfo(s,m)
#else
#define printD(...)
#define printMatrix(s,m)
#endif

#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define MAX(a,b) (((a) > (b)) ? (a) : (b))

// Notes:
// 1. Currently, scaling factors in cuBLAS call reside in CPU, change to GPU variables? (use cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_(HOST|DEVICE)).
//    For CPU scaling factors, put them in array in pinned memory.
//
// 2. Overlap kernels using cuBLAS streams
//
// 3. Build dependency graph of IDR(s)-biortho
//
// 4. Apparently, some precision is lost using MAGMA when compared to MATLAB, probably is that matrices are not displayed with full precision on screen.
//
// 5. Optimize: merge kernels, reuse arrays, concurrent kernels
//
// 6. What other methods instead of QR for forming othornormal basis P? LU?


extern "C" void
magma_zmatrixInfo(
    const char *s,
    magma_z_matrix A ) {

    printf(" %s dims = %d x %d\n", s, A.num_rows, A.num_cols);
    printf(" %s location = %d = %s\n", s, A.memory_location, (A.memory_location == 471) ? "CPU" : "DEV");
    printf(" %s storage = %d = %s\n", s, A.storage_type, (A.storage_type == 411) ? "CSR" : "DENSE");
    printf(" %s major = %d = %s\n", s, A.major, (A.major == 101) ? "row" : "column");
    printf(" %s nnz = %d\n", s, A.nnz);
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
    magma_queue_t orig_queue=NULL;
    magmablasGetKernelStream( &orig_queue );

    // prepare solver feedback
    solver_par->solver = Magma_IDR;
    solver_par->numiter = 0;
    solver_par->info = MAGMA_SUCCESS;

    // local constants
    const magmaDoubleComplex c_zero = MAGMA_Z_ZERO;
    const magmaDoubleComplex c_one = MAGMA_Z_ONE;
    const magmaDoubleComplex c_n_one = MAGMA_Z_NEG_ONE;

    // local variables
    magma_int_t info = 0;
    magma_int_t iseed[4] = { 0, 0, 0, 1 };
    magma_int_t dofb = b.num_rows * b.num_cols;
    magma_int_t inc = 1;
    magma_int_t s;
    magma_int_t dof;
    magma_int_t distr;
    magma_int_t k, i, sk;
    magma_int_t *piv=NULL;
    magma_int_t innerflag;
    double residual;
    double nrm;
    double nrmb;
    double nrmr;
    double nrmt;
    double rho;
    double tolb;
    double angle;
    magmaDoubleComplex om;
    magmaDoubleComplex tr;
    magmaDoubleComplex alpha;
    magmaDoubleComplex beta;
    magmaDoubleComplex mkk;
    magmaDoubleComplex fk;
    //magmaDoubleComplex_ptr scalars;    // scaling factors in CPU pinned memory
                                        // [alpha, beta, mkk, fk, om, tr]

    // local matrices and vectors
    magma_z_matrix P1={Magma_CSR}, P2={Magma_CSR}, P={Magma_CSR};
    magma_z_matrix r={Magma_CSR};
    magma_z_matrix G={Magma_CSR};
    magma_z_matrix U={Magma_CSR};
    magma_z_matrix M1={Magma_CSR}, M={Magma_CSR};
    magma_z_matrix f={Magma_CSR};
    magma_z_matrix t={Magma_CSR};
    magma_z_matrix c={Magma_CSR};
    magma_z_matrix v1={Magma_CSR}, v={Magma_CSR};

    // local performance variables
    long long int gpumem = 0;

    gpumem += (A.nnz * sizeof(magmaDoubleComplex)) + (A.nnz * sizeof(magma_index_t)) + ((A.num_rows + 1) * sizeof(magma_index_t));

    // check if matrix A is square
    if ( A.num_rows != A.num_cols ) {
        printD("Error! matrix must be square.\n");
        info = MAGMA_ERR;
        goto cleanup;
    }

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
    solver_par->maxiter = MIN(2 * A.num_cols, solver_par->maxiter);

    // initial angle
    angle = 0.7;

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

    // chronometry
    real_Double_t tempo1, tempo2;

    // |b|
    nrmb = magma_dznrm2( b.num_rows, b.dval, inc );

    // check for |b| == 0
    printD("init norm(b) ..........%f\n", nrmb);
    if ( nrmb == 0.0 ) {
        printD("RHS is zero, exiting...\n");
        magma_zscal( x->num_rows*x->num_cols, MAGMA_Z_ZERO, x->dval, inc );
        solver_par->init_res = 0.0;
        solver_par->final_res = 0.0;
        solver_par->iter_res = 0.0;
        solver_par->runtime = 0.0;
        goto cleanup;
    }

    // relative tolerance
    //tolb = solver_par->epsilon * nrmb;
    tolb = solver_par->epsilon;

    // P = randn(n, s)
    // P = ortho(P)
//---------------------------------------
    // P1 = 0.0
    CHECK( magma_zvinit( &P1, Magma_CPU, A.num_cols, s, c_zero, queue ));


    // P1 = randn(n, s)
    distr = 3;        // 3 = normal distribution
    dof = P1.num_rows * P1.num_cols;
    lapackf77_zlarnv( &distr, iseed, &dof, P1.val );
    printMatrix("P1", P1);

    // transfer P to device
    // P2 = P1
    CHECK( magma_zmtransfer( P1, &P2, Magma_CPU, Magma_DEV, queue ));
    magma_zmfree( &P1, queue );

    // P = ortho(P2)
    if ( P2.num_cols > 1 ) {
        // P =magma_zqr(P2), QR factorization
        CHECK( magma_zqr( P2.num_rows, P2.num_cols, P2, &P, NULL, queue ) );
    } else {
        // P = P2 / |P2|
        dof = P2.num_rows * P2.num_cols;        // can remove
        nrm = magma_dznrm2( dof, P2.dval, inc );
        nrm = 1.0 / nrm;
        magma_zdscal( dof, nrm, P2.dval, inc );
        CHECK( magma_zmtransfer( P2, &P, Magma_DEV, Magma_DEV, queue ));
    }
    magma_zmfree(&P2, queue );
//---------------------------------------
    printMatrix("P", P);
    gpumem += P.nnz * sizeof(magmaDoubleComplex);

#if WRITEP == 1
    // Note: write P matrix to file to use in MATLAB for validation
    magma_zprint_gpu( P.num_rows, P.num_cols, P.dval, P.num_rows );
#endif

    // initial residual
    // r = b - A x
    CHECK( magma_zvinit( &r, Magma_DEV, b.num_rows, b.num_cols, c_zero, queue ));
    CHECK(  magma_zresidualvec( A, b, *x, &r, &nrmr, queue));

    printMatrix("R" , r);
    gpumem += r.nnz * sizeof(magmaDoubleComplex);

    // |r|
    solver_par->init_res = nrmr;
    if ( solver_par->verbose > 0 ) {
        solver_par->res_vec[0] = (real_Double_t)nrmr;
    }
    
    tolb = nrmr * solver_par->epsilon;
    if ( tolb < ATOLERANCE )
        tolb = ATOLERANCE;
    // check if initial is guess good enough
    if ( nrmr < tolb ) {
        solver_par->final_res = solver_par->init_res;
        solver_par->iter_res = solver_par->init_res;
        goto cleanup;
    }

    // G(n,s) = 0
    CHECK( magma_zvinit( &G, Magma_DEV, A.num_cols, s, c_zero, queue ));
    gpumem += G.nnz * sizeof(magmaDoubleComplex);

    // U(n,s) = 0
    CHECK( magma_zvinit( &U, Magma_DEV, A.num_cols, s, c_zero, queue ));
    gpumem += U.nnz * sizeof(magmaDoubleComplex);

    // M1 = 0
    // M(s,s) = I
    CHECK( magma_zvinit( &M1, Magma_DEV, s, s, c_zero, queue ));
    CHECK( magma_zvinit( &M, Magma_DEV, s, s, c_zero, queue ));
    magmablas_zlaset( MagmaFull, s, s, c_zero, c_one, M.dval, s );
    gpumem += 2 * M.nnz * sizeof(magmaDoubleComplex);

    // f = 0
    CHECK( magma_zvinit( &f, Magma_DEV, P.num_cols, r.num_cols, c_zero, queue ));
    gpumem += f.nnz * sizeof(magmaDoubleComplex);

    // t = 0
    CHECK( magma_zvinit( &t, Magma_DEV, A.num_rows, r.num_cols, c_zero, queue ));
    gpumem += t.nnz * sizeof(magmaDoubleComplex);

    // c = 0
    CHECK( magma_zvinit( &c, Magma_DEV, M.num_cols, f.num_cols, c_zero, queue ));
    gpumem += c.nnz * sizeof(magmaDoubleComplex);

    // v1 = 0
    // v = 0
    CHECK( magma_zvinit( &v1, Magma_DEV, r.num_rows, r.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &v, Magma_DEV, r.num_rows, r.num_cols, c_zero, queue ));
    gpumem += 2 * v.nnz * sizeof(magmaDoubleComplex);

    // piv = 0
    CHECK( magma_imalloc_pinned(&piv, s));

    // om = 1
    om = MAGMA_Z_ONE;

    
    //--------------START TIME---------------
    // chronometry
    tempo1 = magma_sync_wtime( queue );
    if ( solver_par->verbose > 0 ) {
        solver_par->timing[0] = 0.0;
    }
    
    // main iteration loop (begins at index 1 to store residuals correctly in array)
    innerflag = 0;
    solver_par->numiter = 0;
    // start iteration
    do
    {
        solver_par->numiter++;
    //for( solver_par->numiter = 1; solver_par->numiter <= solver_par->maxiter; ++solver_par->numiter ) {
    
        // new RHS for small systems
        // f = (r' P)' = P' r
        magmablas_zgemv( MagmaConjTrans, P.num_rows, P.num_cols, c_one, P.dval, P.num_rows, r.dval, inc, c_zero, f.dval, inc );
        printMatrix("F", f);

        // shadow space loop
        for ( k = 0; k < s; ++k ) {
            sk = s - k;
    
            // solve small system and make v orthogonal to P
            // f(k:s) = M(k:s,k:s) c(k:s)
//---------------------------------------
            // c(k:s) = f(k:s)
            magma_zcopy( sk, &f.dval[k], inc, &c.dval[k], inc );

            // M1 = M
            magma_zcopy( M.num_rows * M.num_cols, M.dval, inc, M1.dval, inc );

            // c(k:s) = M1(k:s,k:s) \ c(k:s)
            CHECK( magma_zgesv_gpu( sk, c.num_cols, &M1.dval[k*M1.num_rows+k], M1.num_rows, piv, &c.dval[k], c.num_rows, &info ) );
//---------------------------------------
            printMatrix("C", c);

            // v1 = r - G(:,k:s) c(k:s)
//---------------------------------------
            // v1 = r
            magma_zcopy( r.num_rows * r.num_cols, r.dval, inc, v1.dval, inc );

            // v1 = v1 - G(:,k:s) c(k:s)
            magmablas_zgemv( MagmaNoTrans, G.num_rows, sk, c_n_one, &G.dval[k*G.num_rows], G.num_rows, &c.dval[k], inc, c_one, v1.dval, inc );
//---------------------------------------
            printMatrix("V1", v1);

            // compute new U
            // U(:,k) = om * v1 + U(:,k:s) c(k:s)
//---------------------------------------
            // v1 = om * v1 + U(:,k:s) c(k:s)
            magmablas_zgemv( MagmaNoTrans, U.num_rows, sk, c_one, &U.dval[k*U.num_rows], U.num_rows, &c.dval[k], inc, om, v1.dval, inc );

            // U(:,k) = v1
            magma_zcopy( U.num_rows, v1.dval, inc, &U.dval[k*U.num_rows], inc );
//---------------------------------------
            printMatrix("U", U);

            // compute new U and G
            // G(:,k) = A U(:,k)
//---------------------------------------
            // v = A v1
            CHECK( magma_z_spmv( c_one, A, v1, c_zero, v, queue ));

            // G(:,k) = v
            magma_zcopy( G.num_rows, v.dval, inc, &G.dval[k*G.num_rows], inc );
//---------------------------------------
            printMatrix("G", G);


            // bi-orthogonalize the new basis vectors
            for ( i = 0; i < k; ++i ) {

                // alpha = P(:,i)' G(:,k) / M(i,i)
//---------------------------------------
                // alpha = P(:,i)' G(:,k)
                alpha = magma_zdotc( P.num_rows, &P.dval[i*P.num_rows], inc, &G.dval[k*G.num_rows], inc );
                
                // alpha = alpha / M(i,i)
                magma_zgetvector( 1, &M.dval[i*M.num_rows+i], inc, &mkk, inc );
                alpha = alpha / mkk;
//---------------------------------------
                printD("bi-ortho: i, k, alpha ...................%d, %d, (%f, %f)\n", i, k, MAGMA_Z_REAL(alpha), MAGMA_Z_IMAG(alpha));

                // G(:,k) = G(:,k) - alpha * G(:,i)
                magma_zaxpy( G.num_rows, -alpha, &G.dval[i*G.num_rows], inc, &G.dval[k*G.num_rows], inc );
                printMatrix("G", G);

                // U(:,k) = U(:,k) - alpha * U(:,i)
                magma_zaxpy( U.num_rows, -alpha, &U.dval[i*U.num_rows], inc, &U.dval[k*U.num_rows], inc );
                printMatrix("U", U);
            }

            // new column of M = P'G, first k-1 entries are zero
            // Note: need to verify that first k-1 entries are zero

            // M(k:s,k) = (G(:,k)' P(:,k:s))' = P(:,k:s)' G(:,k)
            magmablas_zgemv( MagmaConjTrans, P.num_rows, sk, c_one, &P.dval[k*P.num_rows], P.num_rows, &G.dval[k*G.num_rows], inc, c_zero, &M.dval[k*M.num_rows+k], inc );
            printMatrix("M", M);

            // check M(k,k) == 0
            magma_zgetvector( 1, &M.dval[k*M.num_rows+k], inc, &mkk, inc );
            if ( MAGMA_Z_EQUAL(mkk, MAGMA_Z_ZERO) ) {
                info = MAGMA_DIVERGENCE;
                innerflag = 1;
                break;
            }

            // beta = f(k) / M(k,k)
            magma_zgetvector( 1, &f.dval[k], inc, &fk, inc );
            beta = fk / mkk;
            printD("beta: k ...................%d, (%f, %f)\n", k, MAGMA_Z_REAL(beta), MAGMA_Z_IMAG(beta));

            // x = x + beta * U(:,k)
            magma_zaxpy( x->num_rows, beta, &U.dval[k*U.num_rows], inc, x->dval, inc );
            printMatrix("X", *x);

            // make r orthogonal to q_i, i = 1..k
            // r = r - beta * G(:,k)
            magma_zaxpy( r.num_rows, -beta, &G.dval[k*G.num_rows], inc, r.dval, inc );
            printMatrix("R", r);

            // |r|
            nrmr = magma_dznrm2( dofb, r.dval, inc );
            printD("norm(r): k ...................%d, %f\n", k, nrmr);

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
                innerflag = 1;
                break;
            }

            // new f = P' r (first k components are zero)
            if ( (k + 1) < s ) {
                // f(k+1:s) = f(k+1:s) - beta * M(k+1:s,k)
                magma_zaxpy( sk - 1, -beta, &M.dval[k*M.num_rows+(k+1)], inc, &f.dval[k+1], inc );
                printMatrix("F", f);
            }

            // iter = iter + 1
            solver_par->numiter++;
        }

        // check convergence or iteration limit or invalid of inner loop
        if ( innerflag == 1 ) {
            break;
        }

        // v = r
        magma_zcopy( r.num_rows * r.num_cols, r.dval, inc, v.dval, inc );
        printMatrix("V", v);

        // t = A v
        CHECK( magma_z_spmv( c_one, A, v, c_zero, t, queue ));
        printMatrix("T", t);

        // computation of a new omega
        // om = omega(t, r, angle);
//---------------------------------------
        // |t|
        dof = t.num_rows * t.num_cols;
        nrmt = magma_dznrm2( dof, t.dval, inc );

        // tr = t' r
        tr = magma_zdotc( r.num_rows, t.dval, inc, r.dval, inc );

        // rho = abs(tr / (|t| * |r|))
        rho = fabs( MAGMA_Z_REAL(tr) / (nrmt * nrmr) );

        // om = tr / (|t| * |t|)
        om = tr / (nrmt * nrmt);
        if ( rho < angle )
            om = om * angle / rho;
//---------------------------------------
        printD("omega: k .................... %d, (%f, %f)\n", k, MAGMA_Z_REAL(om), MAGMA_Z_IMAG(om));
        if ( MAGMA_Z_EQUAL(om, MAGMA_Z_ZERO) ) {
            info = MAGMA_DIVERGENCE;
            break;
        }

        // update approximation vector
        // x = x + om * v
        magma_zaxpy(x->num_rows, om, v.dval, inc, x->dval, inc);
        printMatrix("X", *x);

        // update residual vector
        // r = r - om * t
        magma_zaxpy(r.num_rows, -om, t.dval, inc, r.dval, inc);
        printMatrix("R", r);

        // residual norm
        nrmr = magma_dznrm2( dofb, r.dval, inc );
        printD("norm(r): k ...................%d, %f\n", k, nrmr);

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
        if ( solver_par->numiter >= (2 * (s + 1)) ) {
            break;
        }
#endif
    }
    while ( solver_par->numiter+1 <= solver_par->maxiter );

    // get last iteration timing
    tempo2 = magma_sync_wtime( queue );
    solver_par->runtime = (real_Double_t) tempo2-tempo1;
//--------------STOP TIME----------------

    // last stats
    solver_par->iter_res = nrmr;
    CHECK( magma_zresidualvec( A, b, *x, &r, &residual, queue ));
    solver_par->final_res = residual;

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
#if WRITEP == 1
    printf("GPU memory = %f KB\n", (real_Double_t)gpumem / 1024);
#endif
   
    
    
cleanup:
    // free resources
    magma_zmfree(&P, queue);
    magma_zmfree(&P1, queue);
    magma_zmfree(&P2, queue);
    magma_zmfree(&r, queue);
    magma_zmfree(&G, queue);
    magma_zmfree(&U, queue);
    magma_zmfree(&M1, queue);
    magma_zmfree(&M, queue);
    magma_zmfree(&f, queue);
    magma_zmfree(&t, queue);
    magma_zmfree(&c, queue);
    magma_zmfree(&v1, queue);
    magma_zmfree(&v, queue);
    magma_free_pinned(piv);

    magmablasSetKernelStream( orig_queue );
    solver_par->info = info;
    return info;
    /* magma_zidr */
}
