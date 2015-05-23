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

#define MYDEBUG 0   // 0 = no printing, 1 = print scalars, 2 = print all (few iters), >2 = prints all (all iters)
#define WRITEP 1

#if MYDEBUG == 0
#define printD(...)
#define printMatrix(s,m)
#elif MYDEBUG == 1
#define printD(...) printf(__VA_ARGS__)
#define printMatrix(s,m)
#elif MYDEBUG == 2
#define printD(...) printf(__VA_ARGS__)
#define printMatrix(s,m) magma_zmatrixInfo_acc(s,m)
#else
#define printD(...) printf(__VA_ARGS__)
#define printMatrix(s,m) magma_zmatrixInfo_acc(s,m)
#endif

#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define MAX(a,b) (((a) > (b)) ? (a) : (b))


extern "C" void
magma_zmatrixInfo_acc(
    const char *s,
    magma_z_matrix A ) {

    printf(" %s dims = %d x %d\n", s, A.num_rows, A.num_cols);
    printf(" %s location = %d = %s\n", s, A.memory_location, (A.memory_location == Magma_CPU) ? "CPU" : "DEV");
    printf(" %s storage = %d = %s\n", s, A.storage_type, (A.storage_type == Magma_CSR) ? "CSR" : "DENSE");
    printf(" %s major = %d = %s\n", s, A.major, (A.major == MagmaRowMajor) ? "row" : "column");
    printf(" %s nnz = %d\n", s, A.nnz);
    
    magma_int_t ldd = magma_roundup( A.num_rows, 32 );
    if (A.ld != ldd) {
        A.ld = A.num_rows;
    }
    if (A.memory_location == Magma_DEV)
        magma_zprint_gpu( A.num_rows, A.num_cols, A.dval, A.ld );
    else
        magma_zprint( A.num_rows, A.num_cols, A.val, A.ld );
}


extern "C" magma_int_t
magma_zidr_acc(
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

    // local constants
    const magmaDoubleComplex c_zero = MAGMA_Z_ZERO;
    const magmaDoubleComplex c_one = MAGMA_Z_ONE;
    const magmaDoubleComplex c_n_one = MAGMA_Z_NEG_ONE;

    // internal user options
    const magma_int_t smoothing = 1;   // 1 = enable, 0 = disabled, -1 = disabled with delayed x update
    const double angle = 0.7;          // [0-1]

    // local variables
    magma_int_t info = 0;
    magma_int_t iseed[4] = { 0, 0, 0, 1 };
    magma_int_t dofx = x->num_rows * x->num_cols;
    magma_int_t dofb = b.num_rows * b.num_cols;
    magma_int_t dofr = A.num_rows * b.num_cols;
    magma_int_t dofM; 
    magma_int_t dofP;
    magma_int_t inc = 1;

    magma_int_t s;
    magma_int_t distr;
    magma_int_t k, i, sk;
    magma_int_t *piv = NULL;
    magma_int_t innerflag;
    magma_int_t ldd;
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
    magmaDoubleComplex *mkk = NULL;
    magmaDoubleComplex fk;

    // local matrices and vectors
    magma_z_matrix dP1 = {Magma_CSR}, dP = {Magma_CSR};
    magma_z_matrix dr = {Magma_CSR};
    magma_z_matrix dg = {Magma_CSR}, dG = {Magma_CSR};
    magma_z_matrix du = {Magma_CSR}, dU = {Magma_CSR};
    magma_z_matrix dM1 = {Magma_CSR}, dM = {Magma_CSR};
    magma_z_matrix df = {Magma_CSR};
    magma_z_matrix dt = {Magma_CSR};
    magma_z_matrix dtt = {Magma_CSR};
    magma_z_matrix dc = {Magma_CSR};
    magma_z_matrix dv = {Magma_CSR};
    magma_z_matrix dxs = {Magma_CSR};
    magma_z_matrix drs = {Magma_CSR};

    // arrays for scalar products
    magma_z_matrix dskp = {Magma_CSR};
    magma_z_matrix dalpha = {Magma_CSR};
    magma_z_matrix dbeta = {Magma_CSR};
    magmaDoubleComplex *skp = NULL;
    magmaDoubleComplex *alpha = NULL;
    magmaDoubleComplex *beta = NULL;
    
    //workspace for merged dot product
    magmaDoubleComplex *d1 = NULL, *d2 = NULL;
    
    // local performance variables
    long long int gpumem = 0;

    // chronometry
    real_Double_t tempo1, tempo2;

    // local stream variables
    magma_queue_t queues[2];    
    magma_int_t nqueues = sizeof(queues)/sizeof(queues[0]);

    // check if matrix A is square
    if ( A.num_rows != A.num_cols ) {
        printD("Error! matrix must be square.\n");
        info = MAGMA_ERR;
        goto cleanup;
    }
    gpumem += (A.nnz * sizeof(magmaDoubleComplex)) + (A.nnz * sizeof(magma_index_t)) + ((A.num_rows + 1) * sizeof(magma_index_t));

    // create kernel streams
    for ( i = 0; i < nqueues; ++i ) {
        magma_queue_create( &queues[i] );
    }

    // initial s space
    // hack --> use "--restart" option as the shadow space number.
    // At the time of writing, 30 was the default value for "--restart" option.
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

    // check for |b| == 0
    printD("init norm(b) ..........%lg\n", nrmb);
    if ( nrmb == 0.0 ) {
        printD("RHS is zero, exiting...\n");
        magma_zscal( dofx, MAGMA_Z_ZERO, x->dval, 1 );
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

    // P = randn(n, s)
    // P = ortho(P)
//---------------------------------------
    // P = 0.0
    CHECK( magma_zvinit( &dP, Magma_CPU, A.num_cols, s, c_zero, queue ));

    // P = randn(n, s)
    distr = 3;        // 1 = unif (0,1), 2 = unif (-1,1), 3 = normal (0,1) 
    dofP = dP.num_rows * dP.num_cols;
    lapackf77_zlarnv( &distr, iseed, &dofP, dP.val );
    printMatrix("P1", dP);

    // transfer P to device
    CHECK( magma_zmtransfer( dP, &dP1, Magma_CPU, Magma_DEV, queue ));
    dP1.major = dP.major;
    magma_zmfree( &dP, queue );

    // P = ortho(P1)
    if ( dP1.num_cols > 1 ) {
        // P = magma_zqr(P1), QR factorization
        CHECK( magma_zqr( dP1.num_rows, dP1.num_cols, dP1, dP1.ld, &dP, NULL, queue ));
    } else {
        // P = P1 / |P1|
        nrm = magma_dznrm2( dofP, dP1.dval, inc );
        nrm = 1.0 / nrm;
        magma_zdscal( dofP, nrm, dP1.dval, 1 );
        CHECK( magma_zmtransfer( dP1, &dP, Magma_DEV, Magma_DEV, queue ));
        dP.major = dP1.major;
    }
    magma_zmfree(&dP1, queue );
//---------------------------------------
    printMatrix("P", dP);
    gpumem += dP.nnz * sizeof(magmaDoubleComplex);

#if WRITEP == 1
    // Note: write P matrix to file to use in MATLAB for validation
    printf("P = ");
    magma_zprint_gpu( dP.num_rows, dP.num_cols, dP.dval, dP.ld );
#endif

    // t = 0
    // make t twice as large to contain both, dt and dr
    CHECK( magma_zvinit( &dt, Magma_DEV, A.num_rows, 2 * b.num_cols, c_zero, queue ));
    dt.num_cols = b.num_cols;

    // redirect the dr.dval to the second part of dt
    CHECK( magma_zvinit( &dr, Magma_DEV, b.num_rows, b.num_cols, c_zero, queue ));
    magma_free( dr.dval );
    dr.dval = dt.dval + dofb;
    gpumem += dt.nnz * sizeof(magmaDoubleComplex);

    // initial residual
    // r = b - A x
    CHECK( magma_zresidualvec( A, b, *x, &dr, &nrmr, queue ));
    printMatrix("R" , dr);

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
    
    // allocate memory for the scalar products
    CHECK( magma_zvinit( &dskp, Magma_DEV, 4, 1, c_zero, queue ));
    gpumem += dskp.nnz * sizeof(magmaDoubleComplex);
    CHECK( magma_zvinit( &dalpha, Magma_DEV, s, 1, c_zero, queue ));
    gpumem += dalpha.nnz * sizeof(magmaDoubleComplex);
    CHECK( magma_zvinit( &dbeta, Magma_DEV, s, 1, c_zero, queue ));
    gpumem += dbeta.nnz * sizeof(magmaDoubleComplex);
    
    // also on CPU
    CHECK( magma_zmalloc_pinned( &skp, 4 ));
    CHECK( magma_zmalloc_pinned( &alpha, s ));
    CHECK( magma_zmalloc_pinned( &beta, s ));
    
    // workspace for merged dot product
    CHECK( magma_zmalloc( &d1, 2 * dofb ));
    CHECK( magma_zmalloc( &d2, 2 * dofb ));
    gpumem += 4 * dofb * sizeof(magmaDoubleComplex);

    if ( smoothing > 0 ) {
        // set smoothing solution vector
        CHECK( magma_zmtransfer( *x, &dxs, Magma_DEV, Magma_DEV, queue ));
        dxs.major = x->major;
        gpumem += dxs.nnz * sizeof(magmaDoubleComplex);

        
        CHECK( magma_zvinit( &dtt, Magma_DEV, dr.num_rows, 2 * dr.num_cols, c_zero, queue ));
        dtt.num_cols = b.num_cols;
        dtt.nnz = dtt.num_rows * dtt.num_cols;

        // redirect the dr.dval to the second part of dt
        CHECK( magma_zvinit( &drs, Magma_DEV, dr.num_rows, dr.num_cols, c_zero, queue ));
        magma_free( drs.dval );
        drs.dval = dtt.dval + dofr;
        gpumem += dtt.nnz * sizeof(magmaDoubleComplex);
        magma_zcopy( dofr, dr.dval, 1, drs.dval, 1 );
    }

    // G(n,s) = 0
    if ( s > 1 ) {
        ldd = magma_roundup( A.num_rows, 32 );
        CHECK( magma_zvinit( &dG, Magma_DEV, ldd, s, c_zero, queue ));
        dG.num_rows = A.num_rows;
        dG.ld = ldd;
    } else {
        ldd = A.num_rows;
        CHECK( magma_zvinit( &dG, Magma_DEV, ldd, s, c_zero, queue ));
        dG.ld = ldd;
    }
    gpumem += dG.nnz * sizeof(magmaDoubleComplex);
        
    // U(n,s) = 0
    if ( s > 1 ) {
        ldd = magma_roundup( A.num_cols, 32 );
        CHECK( magma_zvinit( &dU, Magma_DEV, ldd, s, c_zero, queue ));
        dU.num_rows = A.num_cols;
        dU.ld = ldd;
    } else {
        ldd = A.num_cols;
        CHECK( magma_zvinit( &dU, Magma_DEV, ldd, s, c_zero, queue ));
        dU.ld = ldd;
    }
    gpumem += dU.nnz * sizeof(magmaDoubleComplex);

    // M1 = 0
    // M(s,s) = I
    ldd = s;
    CHECK( magma_zvinit( &dM1, Magma_DEV, ldd, s, c_zero, queue ));
    dM1.ld = ldd;
    CHECK( magma_zvinit( &dM, Magma_DEV, ldd, s, c_zero, queue ));
    dM.ld = ldd;
    dofM = dM.ld * dM.num_cols;
    magmablas_zlaset( MagmaFull, dM.num_rows, dM.num_cols, c_zero, c_one, dM.dval, dM.ld );
    gpumem += 2 * dM.nnz * sizeof(magmaDoubleComplex);

    // f = 0
    CHECK( magma_zvinit( &df, Magma_DEV, dP.num_cols, dr.num_cols, c_zero, queue ));
    gpumem += df.nnz * sizeof(magmaDoubleComplex);

    // c = 0
    CHECK( magma_zvinit( &dc, Magma_DEV, dM.num_cols, dr.num_cols, c_zero, queue ));
    gpumem += dc.nnz * sizeof(magmaDoubleComplex);

    // v = 0
    CHECK( magma_zvinit( &dv, Magma_DEV, dr.num_rows, dr.num_cols, c_zero, queue ));
    gpumem += dv.nnz * sizeof(magmaDoubleComplex);

    // piv = 0
    CHECK( magma_imalloc_pinned( &piv, s ));

    // om = 1
    om = MAGMA_Z_ONE;

//------------extra arrays for concurrent support-----------
    // dg represents a single column of dG
    CHECK( magma_zvinit( &dg, Magma_DEV, dG.num_rows, 1, c_zero, queue ));
magma_device_sync();
    magma_free( dg.dval );

    // du represents a column of dU 
    CHECK( magma_zvinit( &du, Magma_DEV, dU.num_rows, 1, c_zero, queue ));
    gpumem += du.nnz * sizeof(magmaDoubleComplex);

    CHECK( magma_zmalloc_pinned( &mkk, s ));
//------------extra arrays for concurrent support-----------

    // synchronize all queues
    for ( i = 0; i < nqueues; ++i ) {
        magma_queue_sync( queues[i] );
    }
    
// Queue 2 starts
magmablasSetKernelStream( queues[1] );

    //--------------START TIME---------------
    // chronometry
    tempo1 = magma_sync_wtime( queue );
    if ( solver_par->verbose > 0 ) {
        solver_par->timing[0] = 0.0;
    }
    
    innerflag = 0;
    solver_par->numiter = 0;

cudaProfilerStart();

//------------------concurrent support-----------
    // v = r
    // Q0
    magma_zcopyvector_async( dofr, dr.dval, 1, dv.dval, 1, queues[0] );
//------------------concurrent support-----------
    
    // start iteration
    do
    {
        solver_par->numiter++;
    
        // new RHS for small systems
        // f = (r' P)' = P' r
        // Q1
        magmablas_zgemv( MagmaConjTrans, dP.num_rows, dP.num_cols, c_one, dP.dval, dP.ld, dr.dval, 1, c_zero, df.dval, 1 );
        printMatrix("F", df);

        // fk = f(0)
        // Q1
        magma_zgetvector_async( 1, &df.dval[0], 1, &fk, 1, queues[1] );

        // shadow space loop
        for ( k = 0; k < s; ++k ) {
            sk = s - k;

            // solve small system and make v orthogonal to P
            // f(k:s) = M(k:s,k:s) c(k:s)
//---------------------------------------
            // M1 = M
            // Q1
            magma_zcopyvector_async( dofM, dM.dval, 1, dM1.dval, 1, queues[1] );

            // c(k:s) = f(k:s)
            // Q1
            magma_zcopyvector_async( sk, &df.dval[k], 1, &dc.dval[k], 1, queues[1] );

            // c(k:s) = M1(k:s,k:s) \ c(k:s)
            // Q1
            CHECK( magma_zgesv_gpu( sk, dc.num_cols, &dM1.dval[k*dM1.ld+k], dM1.ld, piv, &dc.dval[k], dc.num_rows, &info ));
//---------------------------------------
            printMatrix("C", dc);

// sync Q0
// U(:,k) = U(:,k) - alpha U(:,1:k-1)
//magma_queue_sync( queues[0] );

// sync Q1
// c(k:s) = M1(k:s,k:s) \ c(k:s)
magma_queue_sync( queues[1] );

// Set to Q0
magmablasSetKernelStream( queues[0] );

            // u = U(:,k:s) c(k:s)
            // Q0
            magmablas_zgemv( MagmaNoTrans, dU.num_rows, sk, c_one, &dU.dval[k*dU.ld], dU.ld, &dc.dval[k], 1, c_zero, du.dval, 1 );

// sync Q0
// v = r
magma_queue_sync( queues[0] );

// Set to Q1
magmablasSetKernelStream( queues[1] );

            // v = r - G(:,k:s) c(k:s)
            // Q1
            magmablas_zgemv( MagmaNoTrans, dG.num_rows, sk, c_n_one, &dG.dval[k*dG.ld], dG.ld, &dc.dval[k], 1, c_one, dv.dval, 1 );
            printMatrix("V", dv);

// sync Q0
// u = U(:,k:s) c(k:s)
magma_queue_sync( queues[0] );

            // compute new U
            // u = u + om * v 
            // Q1
            magma_zaxpy( dU.num_rows, om, dv.dval, 1, du.dval, 1 );

// sync Q1
// u = u + om * v 
magma_queue_sync( queues[1] );

            // U(:,k) = u
            // Q0
            magma_zcopyvector_async( dU.num_rows, du.dval, 1, &dU.dval[k*dU.ld], 1, queues[0] );
//---------------------------------------
            printMatrix("U", dU);

            // compute new G
            // G(:,k) = A u 
            // Q1
            dg.dval = dG.dval + k * dG.ld;
            CHECK( magma_z_spmv( c_one, A, du, c_zero, dg, queues[1] ));
            //CHECK( magma_z_spmv( c_one, A, du, c_zero, dg, queue ));
            printMatrix("G", dG);

            // bi-orthogonalize the new basis vectors
            for ( i = 0; i < k; ++i ) {

                // alpha = P(:,i)' G(:,k) / M(i,i)
//---------------------------------------
                // alpha = P(:,i)' G(:,k)
                // Q1
                //CHECK( magma_zmdotc( dP.num_rows, 1, &dP.dval[i*dP.ld], &dG.dval[k*dG.ld], d1, d2, dalpha.dval+i, queue ));
                CHECK( magma_zmdotc( dP.num_rows, 1, &dP.dval[i*dP.ld], &dG.dval[k*dG.ld], d1, d2, dalpha.dval+i, queues[1] ));

                // Note: why do we not need a sync here?
                // Q
                //magma_zgetvector_async( 1, dalpha.dval+i, 1, alpha+i, 1, queues[1] );
                magma_zgetvector( 1, dalpha.dval+i, 1, alpha+i, 1 );
                
// sync Q1
//magma_queue_sync( queues[1] );

                // alpha = alpha / M(i,i)
                alpha[i] = alpha[i] / mkk[i];
                
//---------------------------------------
                printD("bi-ortho: i, k, alpha ...................%d, %d, (%lg, %lg)\n", i, k, MAGMA_Z_REAL(alpha[i]), MAGMA_Z_IMAG(alpha[i]));

                // G(:,k) = G(:,k) - alpha * G(:,i)
                // Q1
                magma_zaxpy( dG.num_rows, -alpha[i], &dG.dval[i*dG.ld], 1, &dG.dval[k*dG.ld], 1 );
                printMatrix("G", dG);
            }

            if ( k > 0 ) {
// Set to Q0
magmablasSetKernelStream( queues[0] );

                // copy scalars alpha needed for gemv to device
                // Q0
                magma_zsetvector_async( k, alpha, 1, dalpha.dval, 1, queues[0] );

                // U(:,k) = U(:,k) - alpha * U(:,i) outside the loop using GEMV
                // Q0
                magmablas_zgemv( MagmaNoTrans, dU.num_rows, k, c_n_one, &dU.dval[0], dU.ld, &dalpha.dval[0], 1, c_one, &dU.dval[k*dU.ld], 1 );
                printMatrix("U", dU);

// Set to Q1
magmablasSetKernelStream( queues[1] );
            }
            
            // new column of M = P'G, first k-1 entries are zero
            // M(k:s,k) = (G(:,k)' P(:,k:s))' = P(:,k:s)' G(:,k)
            // Q1
            magmablas_zgemv( MagmaConjTrans, dP.num_rows, sk, c_one, &dP.dval[k*dP.ld], dP.ld, &dG.dval[k*dG.ld], 1, c_zero, &dM.dval[k*dM.ld+k], 1 );
            printMatrix("M", dM);

            // Note: why do we not need a sync here?
            // mkk = M(k,k)
            // Q
            //magma_zgetvector_async( 1, &dM.dval[k*dM.ld+k], 1, &mkk[k], 1, queues[1] );
            magma_zgetvector( 1, &dM.dval[k*dM.ld+k], 1, &mkk[k], 1 );

// sync Q1
// mkk = M(k,k)
//magma_queue_sync( queues[1] );

            if ( MAGMA_Z_EQUAL(mkk[k], MAGMA_Z_ZERO) ) {
                s = k; // for the x-update outside the loop
                info = MAGMA_DIVERGENCE;
                innerflag = 1;
                break;
            }

            beta[k] = fk / mkk[k];
            printD("beta: k ...................%d, (%lg, %lg)\n", k, MAGMA_Z_REAL(beta[k]), MAGMA_Z_IMAG(beta[k]));

            // make r orthogonal to q_i, i = 1..k
            // r = r - beta * G(:,k)
            // Q1
            magma_zaxpy( dr.num_rows, -beta[k], &dG.dval[k*dG.ld], 1, dr.dval, 1 );
            printMatrix("R", dr);

            if ( smoothing < 0 ) {
                // |r|
                // Q1
                nrmr = magma_dznrm2( dofb, dr.dval, inc );
                printD("norm(r): k ...................%d, %lg\n", k, nrmr);

            } else if ( smoothing == 0 ) {

// Set to Q0
magmablasSetKernelStream( queues[0] );

                // x = x + beta * U(:,k)
                // Q0
                magma_zaxpy( x->num_rows, beta[k], &dU.dval[k*dU.ld], inc, x->dval, inc );
                printMatrix("X", *x);

// Set to Q1
magmablasSetKernelStream( queues[1] );

                // |r|
                // Q1
                nrmr = magma_dznrm2( dofb, dr.dval, inc );
                printD("norm(r): k ...................%d, %lg\n", k, nrmr);
            } else {

// Set to Q0
magmablasSetKernelStream( queues[0] );

                // x = x + beta * U(:,k)
                // Q0
                magma_zaxpy( x->num_rows, beta[k], &dU.dval[k*dU.ld], 1, x->dval, 1 );
                printMatrix("X", *x);

// Set to Q1
magmablasSetKernelStream( queues[1] );

                // smoothing operation
//---------------------------------------
                // t = rs - r
                // Q1
                magma_zidr_smoothing_1( drs.num_rows, drs.num_cols, drs.dval, dr.dval, dtt.dval, queues[1] );

                // t't
                // t'r
                // Q1
                CHECK( magma_zmdotc( dofr, 2, dtt.dval, dtt.dval, d1, d2, dskp.dval+2, queues[1] ));

                // Note: why do we not need a sync here?
                // Q
                //magma_zgetvector_async( 2 , dskp.dval+2, 1, skp+2, 1, queues[1] );
                magma_zgetvector( 2 , dskp.dval+2, 1, skp+2, 1 );

// sync Q1
// gamma = t'rs/t't
//magma_queue_sync( queues[1] );

                // gamma = gamma / (|t| * |t|)
                gamma = skp[3] / skp[2];
                
                // rs = rs - gamma * t
                // Q1
                magma_zaxpy( drs.num_rows, -gamma, dtt.dval, inc, drs.dval, inc );
                printMatrix("RS", drs);
               
                // xs = xs - gamma * (xs - x) 
                // Q0
                magma_zidr_smoothing_2( dxs.num_rows, dxs.num_cols, gamma, x->dval, dxs.dval, queues[0] );
                // t = xs - x
                //magma_zcopy( dofx, dxs.dval, inc, dtt.dval, inc );
                //magma_zcopyvector_async( dofx, dxs.dval, inc, dtt.dval, inc, queues[1] );
                //magma_zaxpy( dtt.num_rows, c_n_one, x->dval, inc, dtt.dval, inc );
                // xs = xs - gamma * t
                //magma_zaxpy( dxs.num_rows, -gamma, dtt.dval, inc, dxs.dval, inc );
                printMatrix("XS", dxs);

                // |rs|
                // Q1
                nrmr = magma_dznrm2( dofb, drs.dval, inc );           
                printD("norm(rs): k ...................%d, %lg\n", k, nrmr);
            }
//---------------------------------------

// sync Q1
// |r|
magma_queue_sync( queues[1] );

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
                // Q1
                magma_zaxpy( sk - 1, -beta[k], &dM.dval[k*dM.ld+(k+1)], 1, &df.dval[k+1], 1 );
                printMatrix("F", df);

                // fk = f(k+1)
                // Q1
                magma_zgetvector_async( 1, &df.dval[k+1], 1, &fk, 1, queues[1] );

            } else {
                // t = A v = A r
                // Q0
                //CHECK( magma_z_spmv( c_one, A, dv, c_zero, dt, queues[0] ));
                CHECK( magma_z_spmv( c_one, A, dr, c_zero, dt, queues[0] ));
            }

            // v = r
            // Q0
            magma_zcopyvector_async( dofr, dr.dval, 1, dv.dval, 1, queues[0] );

            // iter = iter + 1
            solver_par->numiter++;
        }

        // update solution approximation x
        if ( smoothing < 0 ) {
            // Q1
            magma_zsetvector_async( s, beta, 1, dbeta.dval, 1, queues[1] );
 
            // x = x + beta * U(:,1:s)
            // Q1
            magmablas_zgemv( MagmaNoTrans, dU.num_rows, s, c_one, &dU.dval[0], dU.ld, &dbeta.dval[0], 1, c_one, &x->dval[0], 1 );
            printMatrix("X", *x);
        }
 
        // check convergence or iteration limit or invalid of inner loop
        if ( innerflag == 1 ) {
            break;
        }

        // print here to match output order of original version
        printMatrix("T", dt);

// sync Q0
// t = A v
magma_queue_sync( queues[0] ); 

        // computation of a new omega
        // om = omega(t, r, angle);
//---------------------------------------
        // Q1
        CHECK( magma_zmdotc( dofr, 2, dt.dval, dt.dval, d1, d2, dskp.dval, queues[1] ));

        // Q
        //magma_zgetvector_async( 2 , dskp.dval, 1, skp, 1, queues[1] );
        magma_zgetvector( 2 , dskp.dval, 1, skp, 1 );
       
// sync Q1
// (t' * r) / (|t| * |t|)
//magma_queue_sync( queues[1] ); 

        nrmt = sqrt(MAGMA_Z_REAL(skp[0]));
        tr = skp[1];
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

// Set to Q0
magmablasSetKernelStream( queues[0] );

        // update approximation vector
        // x = x + om * v
        // Q0
        magma_zaxpy(x->num_rows, om, dv.dval, 1, x->dval, 1);
        printMatrix("X", *x);

// Set to Q1
magmablasSetKernelStream( queues[1] );

        // update residual vector
        // r = r - om * t
        // Q1
        magma_zaxpy(dr.num_rows, -om, dt.dval, 1, dr.dval, 1);
        printMatrix("R", dr);

        if ( smoothing < 1 ) {
            // residual norm
            // Q1
            nrmr = magma_dznrm2( dofb, dr.dval, inc );
            printD("norm(r): k ...................%d, %lg\n", k, nrmr);

        } else {
            // smoothing operation
//---------------------------------------
            // t = rs - r
            // Q1
            magma_zidr_smoothing_1( drs.num_rows, drs.num_cols, drs.dval, dr.dval, dtt.dval, queues[1] );

            // t't
            // t'r
            // Q1
            CHECK( magma_zmdotc( dofr, 2, dtt.dval, dtt.dval, d1, d2, dskp.dval+2, queues[1] ));
 
            // Note: why do we not need a sync here?
            // Q
            //magma_zgetvector_async( 2 , dskp.dval+2, 1, skp+2, 1, queues[1] );
            magma_zgetvector( 2 , dskp.dval+2, 1, skp+2, 1 );

// sync Q1
// gamma = t'rs/t't
//magma_queue_sync( queues[1] );

            // gamma = gamma / (|t| * |t|)
            gamma = skp[3] / skp[2];

            // rs = rs - gamma * t
            // Q1
            magma_zaxpy( drs.num_rows, -gamma, dtt.dval, inc, drs.dval, inc );
            printMatrix("RS", drs);

            // xs = xs - gamma * (xs - x) 
            // Q0
            magma_zidr_smoothing_2( dxs.num_rows, dxs.num_cols, gamma, x->dval, dxs.dval, queues[0] );
            //printMatrix("XS", dxs);
            // t = xs - x
            //magma_zcopy( dofx, dxs.dval, inc, dtt.dval, inc );
            //magma_zcopyvector_async( dofx, dxs.dval, inc, dtt.dval, inc, queues[1] );
            //magma_zaxpy( dtt.num_rows, c_n_one, x->dval, inc, dtt.dval, inc );
            // xs = xs - gamma * t
            //magma_zaxpy( dxs.num_rows, -gamma, dtt.dval, inc, dxs.dval, inc );
            printMatrix("XS", dxs);

            // |rs|
            // Q1
            nrmr = magma_dznrm2( dofb, drs.dval, inc );           
            printD("norm(rs): k ...................%d, %lg\n", k, nrmr);
        }
//---------------------------------------

// sync Q1
// |r|
magma_queue_sync( queues[1] );

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

        // v = r
        // Q0
        magma_zcopyvector_async( dr.num_rows * dr.num_cols, dr.dval, 1, dv.dval, 1, queues[0] );

#if MYDEBUG == 2
        // Note: exit loop after a few iterations
        if ( solver_par->numiter + 1 >= (2 * (s + 1)) ) {
            break;
        }
#endif
    }
    while ( solver_par->numiter + 1 <= solver_par->maxiter );
           
    // sync all queues 
    for ( i = 0; i < nqueues; ++i ) {
        magma_queue_sync( queues[i] );
    }
 
    if ( smoothing > 0 ) {
        // Q
        magma_zcopy( dofx, dxs.dval, 1, x->dval, 1 );

        // Q
        magma_zcopy( dofr, drs.dval, 1, dr.dval, 1 );
    }


cudaProfilerStop();

// set to given queue 
magmablasSetKernelStream( queue );

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

#if MYDEBUG > 0 || WRITEP == 1
    // print local stats
    printf("%%GPU memory = %f MB\n", (real_Double_t)gpumem / (1<<20));
#endif
    
cleanup:

// set to given queue 
magmablasSetKernelStream( queue );

    // sync and destroy queues
    for ( i = 0; i < nqueues; ++i ) {
        magma_queue_sync( queues[i] );
        magma_queue_destroy( queues[i] );
    }

    // free resources
    magma_zmfree( &dP1, queue );
    magma_zmfree( &dP, queue );
    magma_zmfree( &dG, queue );
    magma_zmfree( &dU, queue );
    magma_zmfree( &dM1, queue );
    magma_zmfree( &dM, queue );
    magma_zmfree( &df, queue );
    magma_zmfree( &dt, queue );
    magma_zmfree( &dc, queue );
    magma_zmfree( &dv, queue );
    magma_zmfree( &dxs, queue );
    magma_zmfree( &dtt, queue );
    magma_zmfree( &dskp, queue );
    magma_zmfree( &dalpha, queue );
    magma_zmfree( &dbeta, queue );
    magma_free_pinned( skp );
    magma_free_pinned( alpha );
    magma_free_pinned( beta );
    magma_free_pinned( piv );
    magma_free_pinned( mkk );
    magma_free( d1 );
    magma_free( d2 );

    magmablasSetKernelStream( orig_queue );
    solver_par->info = info;
    return info;
    /* magma_zidr_acc */
}
