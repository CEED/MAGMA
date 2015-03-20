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
#include "magmasparse.h"

#include <assert.h>

#define RTOLERANCE     lapackf77_dlamch( "E" )
#define ATOLERANCE     lapackf77_dlamch( "E" )


/**
    Purpose
    -------

    Solves a system of linear equations
       A * X = B
    where A is a complex Hermitian N-by-N positive definite matrix A.
    This is a GPU implementation of the Jacobi method.

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

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgesv
    ********************************************************************/

extern "C" magma_int_t
magma_zjacobi(
    magma_z_sparse_matrix A, 
    magma_z_vector b, 
    magma_z_vector *x,  
    magma_z_solver_par *solver_par,
    magma_queue_t queue )
{
    // set queue for old dense routines
    magma_queue_t orig_queue;
    magmablasGetKernelStream( &orig_queue );

    // prepare solver feedback
    solver_par->solver = Magma_JACOBI;
    solver_par->info = MAGMA_SUCCESS;

    real_Double_t tempo1, tempo2;
    double residual;
    magma_zresidual( A, b, *x, &residual, queue );
    solver_par->init_res = residual;
    solver_par->res_vec = NULL;
    solver_par->timing = NULL;

    // some useful variables
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO, c_one = MAGMA_Z_ONE, 
                                                c_mone = MAGMA_Z_NEG_ONE;
    magma_int_t dofs = A.num_rows;
    double nom0 = 0.0;


    magma_z_sparse_matrix M;
    magma_z_vector c, r;
    magma_zvinit( &r, Magma_DEV, dofs, c_zero, queue );
    magma_z_spmv( c_one, A, *x, c_zero, r, queue );                  // r = A x
    magma_zaxpy(dofs,  c_mone, b.dval, 1, r.dval, 1);           // r = r - b
    nom0 = magma_dznrm2(dofs, r.dval, 1);                      // den = || r ||

    // Jacobi setup
    magma_zjacobisetup( A, b, &M, &c, queue );
    magma_z_solver_par jacobiiter_par;
    jacobiiter_par.maxiter = solver_par->maxiter;

    tempo1 = magma_sync_wtime( queue );

    // Jacobi iterator
    magma_zjacobiiter( M, c, x, &jacobiiter_par, queue ); 

    tempo2 = magma_sync_wtime( queue );
    solver_par->runtime = (real_Double_t) tempo2-tempo1;
    magma_zresidual( A, b, *x, &residual, queue );
    solver_par->final_res = residual;
    solver_par->numiter = solver_par->maxiter;

    if ( solver_par->init_res > solver_par->final_res )
        solver_par->info = MAGMA_SUCCESS;
    else
        solver_par->info = MAGMA_DIVERGENCE;

    magma_zmfree( &M, queue );
    magma_z_vfree( &c, queue );
    magma_z_vfree( &r, queue );

    magmablasSetKernelStream( orig_queue );
    return MAGMA_SUCCESS;
}   /* magma_zjacobi */






/**
    Purpose
    -------

    Prepares the Matrix M for the Jacobi Iteration according to
       x^(k+1) = D^(-1) * b - D^(-1) * (L+U) * x^k
       x^(k+1) =      c     -       M        * x^k.

    It returns the preconditioner Matrix M and a vector d
    containing the diagonal elements. 

    Arguments
    ---------

    @param[in]
    A           magma_z_sparse_matrix
                input matrix A

    @param[in]
    M           magma_z_sparse_matrix*
                M = D^(-1) * (L+U)

    @param[in,out]
    d           magma_z_vector*
                vector with diagonal elements of A
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_z
    ********************************************************************/

extern "C" magma_int_t
magma_zjacobisetup_matrix(
    magma_z_sparse_matrix A, 
    magma_z_sparse_matrix *M, magma_z_vector *d,
    magma_queue_t queue )
{
    magma_int_t i;

    magma_z_sparse_matrix A_h1, A_h2, B, C;
    magma_z_vector diag;
    magma_zvinit( &diag, Magma_CPU, A.num_rows, MAGMA_Z_ZERO, queue );

    if ( A.storage_type != Magma_CSR) {
        magma_zmtransfer( A, &A_h1, A.memory_location, Magma_CPU, queue );
        magma_zmconvert( A_h1, &B, A_h1.storage_type, Magma_CSR, queue );
    }
    else {
        magma_zmtransfer( A, &B, A.memory_location, Magma_CPU, queue );
    }
    for( magma_int_t rowindex=0; rowindex<B.num_rows; rowindex++ ) {
        magma_int_t start = (B.drow[rowindex]);
        magma_int_t end = (B.drow[rowindex+1]);
        for( i=start; i<end; i++ ) {
            if ( B.dcol[i]==rowindex ) {
                diag.val[rowindex] = B.val[i];
                if ( MAGMA_Z_REAL( diag.val[rowindex]) == 0 )
                    printf(" error: zero diagonal element in row %d!\n", 
                                                                (int) rowindex);
            }
        }
        for( i=start; i<end; i++ ) {
            B.val[i] = B.val[i] / diag.val[rowindex];
            if ( B.dcol[i]==rowindex ) {
                B.val[i] = MAGMA_Z_MAKE( 0., 0. );
            }
        }
    }
    magma_z_csr_compressor(&B.val, &B.drow, &B.dcol, 
                           &C.val, &C.drow, &C.dcol, &B.num_rows, queue );  
    C.num_rows = B.num_rows;
    C.num_cols = B.num_cols;
    C.memory_location = B.memory_location;
    C.nnz = C.drow[B.num_rows];
    C.storage_type = B.storage_type;
    C.memory_location = B.memory_location;
    if ( A.storage_type != Magma_CSR) {
        magma_zmconvert( C, &A_h2, Magma_CSR, A_h1.storage_type, queue );
        magma_zmtransfer( A_h2, M, Magma_CPU, A.memory_location, queue );
    }
    else {
        magma_zmtransfer( C, M, Magma_CPU, A.memory_location, queue );
    }    
    magma_zvtransfer( diag, d, Magma_CPU, A.memory_location, queue );

    if ( A.storage_type != Magma_CSR) {
        magma_zmfree( &A_h1, queue );
        magma_zmfree( &A_h2, queue );   
    }
    magma_zmfree( &B, queue );
    magma_zmfree( &C, queue ); 

    magma_z_vfree( &diag, queue );
 
    return MAGMA_SUCCESS;
}


/**
    Purpose
    -------


    It returns a vector d
    containing the inverse diagonal elements. 

    Arguments
    ---------

    @param[in]
    A           magma_z_sparse_matrix
                input matrix A

    @param[in,out]
    d           magma_z_vector*
                vector with diagonal elements
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_z
    ********************************************************************/

extern "C" magma_int_t
magma_zjacobisetup_diagscal(
    magma_z_sparse_matrix A, magma_z_vector *d,
    magma_queue_t queue )
{
    magma_int_t i;

    magma_z_sparse_matrix A_h1, B;
    magma_z_vector diag;
    magma_zvinit( &diag, Magma_CPU, A.num_rows, MAGMA_Z_ZERO, queue );

    if ( A.storage_type != Magma_CSR) {
        magma_zmtransfer( A, &A_h1, A.memory_location, Magma_CPU, queue );
        magma_zmconvert( A_h1, &B, A_h1.storage_type, Magma_CSR, queue );
    }
    else {
        magma_zmtransfer( A, &B, A.memory_location, Magma_CPU, queue );
    }
    for( magma_int_t rowindex=0; rowindex<B.num_rows; rowindex++ ) {
        magma_int_t start = (B.drow[rowindex]);
        magma_int_t end = (B.drow[rowindex+1]);
        for( i=start; i<end; i++ ) {
            if ( B.dcol[i]==rowindex ) {
                diag.val[rowindex] = 1.0/B.val[i];
                if ( MAGMA_Z_REAL( diag.val[rowindex]) == 0 )
                    printf(" error: zero diagonal element in row %d!\n", 
                                                                (int) rowindex);
            }
        }
    }
    magma_zvtransfer( diag, d, Magma_CPU, A.memory_location, queue );

    if ( A.storage_type != Magma_CSR) {
        magma_zmfree( &A_h1, queue );
    }
    magma_zmfree( &B, queue );
    magma_z_vfree( &diag, queue );
 
    return MAGMA_SUCCESS;
}



/**
    Purpose
    -------

    Prepares the Jacobi Iteration according to
       x^(k+1) = D^(-1) * b - D^(-1) * (L+U) * x^k
       x^(k+1) =      c     -       M        * x^k.

    Returns the vector c

    Arguments
    ---------

    @param[in]
    b           magma_z_vector
                RHS b

    @param[in]
    d           magma_z_vector
                vector with diagonal entries

    @param[in]
    c           magma_z_vector*
                c = D^(-1) * b
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_z
    ********************************************************************/

extern "C" magma_int_t
magma_zjacobisetup_vector(
    magma_z_vector b, magma_z_vector d, 
    magma_z_vector *c,
    magma_queue_t queue )
{
    if ( b.memory_location == Magma_CPU ) {
        magma_z_vector diag, c_t, b_h;
        magma_zvinit( &c_t, Magma_CPU, b.num_rows, MAGMA_Z_ZERO, queue );

        magma_zvtransfer( b, &b_h, b.memory_location, Magma_CPU, queue );
        magma_zvtransfer( d, &diag, b.memory_location, Magma_CPU, queue );

        for( magma_int_t rowindex=0; rowindex<b.num_rows; rowindex++ ) {   
            c_t.val[rowindex] = b_h.val[rowindex] / diag.val[rowindex];

        }  
        magma_zvtransfer( c_t, c, Magma_CPU, b.memory_location, queue ); 

        magma_z_vfree( &diag, queue );
        magma_z_vfree( &c_t, queue );
        magma_z_vfree( &b_h, queue );

        return MAGMA_SUCCESS;
    }
    else if ( b.memory_location == Magma_DEV ) {
        // fill vector
        magma_z_vector tmp;
        magma_zvinit( &tmp, Magma_DEV, b.num_rows, MAGMA_Z_ZERO, queue );
        magma_zjacobisetup_vector_gpu( 
                    b.num_rows, b, d, *c, &tmp, queue );
        magma_z_vfree( &tmp, queue );
        return MAGMA_SUCCESS;
    }

    return MAGMA_SUCCESS;
}


/**
    Purpose
    -------

    Prepares the Jacobi Iteration according to
       x^(k+1) = D^(-1) * b - D^(-1) * (L+U) * x^k
       x^(k+1) =      c     -       M        * x^k.

    Arguments
    ---------

    @param[in]
    A           magma_z_sparse_matrix
                input matrix A

    @param[in]
    b           magma_z_vector
                RHS b

    @param[in]
    M           magma_z_sparse_matrix*
                M = D^(-1) * (L+U)

    @param[in]
    c           magma_z_vector*
                c = D^(-1) * b
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_z
    ********************************************************************/

extern "C" magma_int_t
magma_zjacobisetup(
    magma_z_sparse_matrix A, magma_z_vector b, 
    magma_z_sparse_matrix *M, magma_z_vector *c,
    magma_queue_t queue )
{
    magma_int_t i;

    magma_z_sparse_matrix A_h1, A_h2, B, C;
    magma_z_vector diag, c_t, b_h;
    magma_zvinit( &c_t, Magma_CPU, A.num_rows, MAGMA_Z_ZERO, queue );
    magma_zvinit( &diag, Magma_CPU, A.num_rows, MAGMA_Z_ZERO, queue );
    magma_zvtransfer( b, &b_h, A.memory_location, Magma_CPU, queue );

    if ( A.storage_type != Magma_CSR ) {
        magma_zmtransfer( A, &A_h1, A.memory_location, Magma_CPU, queue );
        magma_zmconvert( A_h1, &B, A_h1.storage_type, Magma_CSR, queue );
    }
    else {
        magma_zmtransfer( A, &B, A.memory_location, Magma_CPU, queue );
    }
    for( magma_int_t rowindex=0; rowindex<B.num_rows; rowindex++ ) {
        magma_int_t start = (B.drow[rowindex]);
        magma_int_t end = (B.drow[rowindex+1]);
        for( i=start; i<end; i++ ) {
            if ( B.dcol[i]==rowindex ) {
                diag.val[rowindex] = B.val[i];
                if ( MAGMA_Z_REAL( diag.val[rowindex]) == 0 )
                    printf(" error: zero diagonal element in row %d!\n", 
                                                               (int) rowindex);
            }
        }
        for( i=start; i<end; i++ ) {
            B.val[i] = B.val[i] / diag.val[rowindex];
            if ( B.dcol[i]==rowindex ) {
                B.val[i] = MAGMA_Z_MAKE( 0., 0. );
            }
        }
        c_t.val[rowindex] = b_h.val[rowindex] / diag.val[rowindex];

    }

    magma_z_csr_compressor(&B.val, &B.drow, &B.dcol, 
                           &C.val, &C.drow, &C.dcol, &B.num_rows, queue );  

    C.num_rows = B.num_rows;
    C.num_cols = B.num_cols;
    C.memory_location = B.memory_location;
    C.nnz = C.drow[B.num_rows];
    C.storage_type = B.storage_type;
    C.memory_location = B.memory_location;
    if ( A.storage_type != Magma_CSR) {
        A_h2.alignment = A.alignment;
        A_h2.blocksize = A.blocksize;
        magma_zmconvert( C, &A_h2, Magma_CSR, A_h1.storage_type, queue );
        magma_zmtransfer( A_h2, M, Magma_CPU, A.memory_location, queue );
    }
    else {
        magma_zmtransfer( C, M, Magma_CPU, A.memory_location, queue );
    }     
    magma_zvtransfer( c_t, c, Magma_CPU, A.memory_location, queue );

    if ( A.storage_type != Magma_CSR) {
        magma_zmfree( &A_h1, queue );
        magma_zmfree( &A_h2, queue );   
    }   
    magma_zmfree( &B, queue );
    magma_zmfree( &C, queue );  
    magma_z_vfree( &diag, queue );
    magma_z_vfree( &c_t, queue );
    magma_z_vfree( &b_h, queue );

    return MAGMA_SUCCESS;
}



/**
    Purpose
    -------

    Iterates the solution approximation according to
       x^(k+1) = D^(-1) * b - D^(-1) * (L+U) * x^k
       x^(k+1) =      c     -       M        * x^k.

    Arguments
    ---------

    @param[in]
    M           magma_z_sparse_matrix
                input matrix M = D^(-1) * (L+U)

    @param[in]
    c           magma_z_vector
                c = D^(-1) * b

    @param[in,out]
    x           magma_z_vector*
                iteration vector x

    @param[in,out]
    solver_par  magma_z_solver_par*
                solver parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_z
    ********************************************************************/

extern "C" magma_int_t
magma_zjacobiiter(
    magma_z_sparse_matrix M, magma_z_vector c, magma_z_vector *x,  
    magma_z_solver_par *solver_par,
    magma_queue_t queue )
{
    // set queue for old dense routines
    magma_queue_t orig_queue;
    magmablasGetKernelStream( &orig_queue );

    // local variables
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO, c_one = MAGMA_Z_ONE, 
                                            c_mone = MAGMA_Z_NEG_ONE;
    magma_int_t dofs = M.num_rows;
    magma_z_vector t, swap;
    magma_zvinit( &t, Magma_DEV, dofs, c_zero, queue );


    for( magma_int_t i=0; i<solver_par->maxiter; i++ ) {
        magma_z_spmv( c_mone, M, *x, c_zero, t, queue );                // t = - M * x
        magma_zaxpy( dofs, c_one , c.dval, 1 , t.dval, 1 );        // t = t + c

        // swap so that x again contains solution, and y is ready to be used
        swap = *x;
        *x = t;
        t = swap;        
        //magma_zcopy( dofs, t.dval, 1 , x->dval, 1 );               // x = t
    }

    magma_z_vfree( &t, queue );

    magmablasSetKernelStream( orig_queue );
    return MAGMA_SUCCESS;
}   /* magma_zjacobiiter */



/**
    Purpose
    -------

    Iterates the solution approximation according to
       x^(k+1) = D^(-1) * b - D^(-1) * (L+U) * x^k
       x^(k+1) =      c     -       M        * x^k.

    Arguments
    ---------

    @param[in]
    M           magma_z_sparse_matrix
                input matrix M = D^(-1) * (L+U)

    @param[in]
    c           magma_z_vector
                c = D^(-1) * b

    @param[in,out]
    x           magma_z_vector*
                iteration vector x

    @param[in,out]
    solver_par  magma_z_solver_par*
                solver parameters

    @param[in]
    solver_par  magma_z_precond_par*
                precond parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_z
    ********************************************************************/

extern "C" magma_int_t
magma_zjacobiiter_precond(
    magma_z_sparse_matrix M, magma_z_vector *x, 
    magma_z_solver_par *solver_par, magma_z_preconditioner *precond,
    magma_queue_t queue )
{
    // set queue for old dense routines
    magma_queue_t orig_queue;
    magmablasGetKernelStream( &orig_queue );

    // local variables
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO, c_one = MAGMA_Z_ONE, 
                                            c_mone = MAGMA_Z_NEG_ONE;
    magma_int_t dofs = M.num_rows;
    magma_int_t num_vecs = x->num_rows / dofs;
    magma_z_vector swap;

    for( magma_int_t i=0; i<solver_par->maxiter; i++ ) {
        magma_z_spmv( c_mone, M, *x, c_zero, precond->work2, queue );   // t = - M * x

        magma_zaxpy( num_vecs*dofs, c_one , 
                precond->work1.dval, 1 , precond->work2.dval, 1 ); // t = t + c

        // swap so that x again contains solution, and y is ready to be used
        swap = *x;
        *x = precond->work2;
        precond->work2 = swap;        
        //magma_zcopy( dofs, t.dval, 1 , x->dval, 1 );               // x = t
    }

    magmablasSetKernelStream( orig_queue );
    return MAGMA_SUCCESS;
}   /* magma_zjacobiiter */
