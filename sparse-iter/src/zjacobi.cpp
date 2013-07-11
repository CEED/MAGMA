/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @author Hartwig Anzt 

       @precisions normal z -> s d c
*/

#include "common_magma.h"
#include "../include/magmasparse.h"

#include <assert.h>

#define RTOLERANCE     10e-16
#define ATOLERANCE     10e-16


/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======

    Solves a system of linear equations
       A * X = B
    where A is a complex Hermitian N-by-N positive definite matrix A.
    This is a GPU implementation of the Conjugate Gradient method.

    Arguments
    =========

    magma_z_sparse_matrix A                   input matrix A
    magma_z_vector b                          RHS b
    magma_z_vector *x                         solution approximation
    magma_solver_parameters *solver_par       solver parameters

    =====================================================================  */


magma_int_t
magma_zjacobi( magma_z_sparse_matrix A, magma_z_vector b, magma_z_vector *x,  
           magma_solver_parameters *solver_par )
{

    // some useful variables
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO, c_one = MAGMA_Z_ONE, c_mone = MAGMA_Z_NEG_ONE;
    magma_int_t dofs = A.num_rows;
    double nom0, nom1;


    magma_z_sparse_matrix M;
    magma_z_vector c, r;
    magma_z_vinit( &r, Magma_DEV, dofs, c_zero );

    magma_z_spmv( c_one, A, *x, c_zero, r );                       // r = A x
    magma_zaxpy(dofs,  c_mone, b.val, 1, r.val, 1);                // r = r - b
    nom0 = magma_dznrm2(dofs, r.val, 1);                            // den = || r ||
    

    // Jacobi setup
    magma_zjacobisetup( A, b, &M, &c );
    magma_solver_parameters jacobiiter_par;
    jacobiiter_par.maxiter = solver_par->maxiter;

    // Jacobi iterator
    magma_zjacobiiter( M, c, x, &jacobiiter_par ); 



    
    magma_z_spmv( c_one, A, *x, c_zero, r );                       // r = A x
    magma_zaxpy(dofs,  c_mone, b.val, 1, r.val, 1);                // r = r - b
    nom1 = magma_dznrm2(dofs, r.val, 1);                            // den = || r ||
    printf( "      (r_0, r_0) = %e\n", nom0);
    printf( "      (r_N, r_N) = %e\n", nom1);
    printf( "      Number of Jacobi iterations: %d\n", solver_par->maxiter);
    printf( "      || r_N ||   = %f\n", nom1);
    solver_par->residual = (double)(nom1);

    return MAGMA_SUCCESS;
}   /* magma_zjacobi */






/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======

    Prepares the Jacobi Iteration according to
       x^(k+1) = D^(-1) * b - D^(-1) * (L+U) * x^k
       x^(k+1) =      c     -       M        * x^k.

    Arguments
    =========

    magma_z_sparse_matrix A                   input matrix A
    magma_z_vector b                          RHS b
    magma_z_sparse_matrix *M                  M = D^(-1) * (L+U)
    magma_z_vector *c                         c = D^(-1) * b

    =====================================================================  */

magma_int_t
magma_zjacobisetup( magma_z_sparse_matrix A, magma_z_vector b, 
                    magma_z_sparse_matrix *M, magma_z_vector *c ){
    magma_int_t i;

    

    magma_z_sparse_matrix A_h1, A_h2, B, C;
    magma_z_vector diag, c_t, b_h;
    magma_z_vinit( &c_t, Magma_CPU, A.num_rows, MAGMA_Z_ZERO );
    magma_z_vinit( &diag, Magma_CPU, A.num_rows, MAGMA_Z_ZERO );

    magma_z_mtransfer( A, &A_h1, A.memory_location, Magma_CPU);
    magma_z_vtransfer( b, &b_h, A.memory_location, Magma_CPU);
    magma_z_mconvert( A_h1, &B, A_h1.storage_type, Magma_CSR);

    for( magma_int_t rowindex=0; rowindex<B.num_rows; rowindex++ ){
        magma_int_t start = (B.row[rowindex]);
        magma_int_t end = (B.row[rowindex+1]);
        for( i=start; i<end; i++ ){
            if( B.col[i]==rowindex ){
                diag.val[rowindex] = B.val[i];
                if( MAGMA_Z_REAL( diag.val[rowindex]) == 0 )
                    printf(" error: zero diagonal element in row %d!\n", rowindex);
            }
        }
        for( i=start; i<end; i++ ){
            B.val[i] = B.val[i] / diag.val[rowindex];
            if( B.col[i]==rowindex ){
                B.val[i] = MAGMA_Z_MAKE( 0., 0. );
            }
        }
        c_t.val[rowindex] = b_h.val[rowindex] / diag.val[rowindex];

    }
    magma_z_csr_compressor(&B.val, &B.row, &B.col, 
                           &C.val, &C.row, &C.col, &B.num_rows);  
    C.num_rows = B.num_rows;
    C.num_cols = B.num_cols;
    C.memory_location = B.memory_location;
    C.nnz = C.row[B.num_rows];
    C.storage_type = B.storage_type;
    C.memory_location = B.memory_location;
    magma_z_mconvert( C, &A_h2, Magma_CSR, A_h1.storage_type);
    magma_z_mtransfer( A_h2, M, Magma_CPU, A.memory_location);    
    magma_z_vtransfer( c_t, c, Magma_CPU, A.memory_location);
    //magma_z_mfree( &H1 );
    //magma_z_mfree( &H2 );   
    //magma_z_mfree( &B );
    //magma_z_mfree( &C );  
    return MAGMA_SUCCESS;


}



/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======

    Iterates the solution approximation according to
       x^(k+1) = D^(-1) * b - D^(-1) * (L+U) * x^k
       x^(k+1) =      c     -       M        * x^k.

    Arguments
    =========

    magma_z_sparse_matrix M                   input matrix M = D^(-1) * (L+U)
    magma_z_vector c                          c = D^(-1) * b
    magma_z_vector *x                         iteration vector x
    magma_solver_parameters *solver_par       solver parameters

    =====================================================================  */


magma_int_t
magma_zjacobiiter( magma_z_sparse_matrix M, magma_z_vector c, magma_z_vector *x,  
           magma_solver_parameters *solver_par ){

    // local variables
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO, c_one = MAGMA_Z_ONE, c_mone = MAGMA_Z_NEG_ONE;
    magma_int_t dofs = M.num_rows;
    magma_z_vector t;
    magma_z_vinit( &t, Magma_DEV, dofs, c_zero );

    for( magma_int_t i=0; i<solver_par->maxiter; i++ ){
        magma_z_spmv( c_mone, M, *x, c_zero, t );                // t = - M * x
        magma_zaxpy( dofs, c_one , c.val, 1 , t.val, 1 );        // t = t + c
        magma_zcopy( dofs, t.val, 1 , x->val, 1 );               // x = t
    }

    //magma_z_vfree(&t);

    return MAGMA_SUCCESS;
}   /* magma_zjacobiiter */

