/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> c d s
       @author Hartwig Anzt
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"
magma_int_t
magma_zgmres0( magma_z_sparse_matrix A, magma_z_vector b, magma_z_vector *x,
               magma_solver_parameters *solver_par );

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing magma_zgmres
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    magma_solver_parameters solver_par;
    solver_par.epsilon = 10e-8;
    solver_par.maxiter = 1000;
    solver_par.restart = 30;
    solver_par.ortho = MAGMA_CGS;

    int ortho_int = 1;
    for( int i = 1; i < argc; ++i ) {
        if ( strcmp("--ortho", argv[i]) == 0 ) {
            ortho_int = atoi( argv[++i] );
            switch( ortho_int ) {
                case 0: solver_par.ortho = MAGMA_MGS; break;
                case 1: solver_par.ortho = MAGMA_CGS; break;
                case 2: solver_par.ortho = MAGMA_FUSED_CGS; break;
            }
        } else if ( strcmp("--restart", argv[i]) == 0 ) {
            solver_par.restart = atoi( argv[++i] );
        } else if ( strcmp("--maxiter", argv[i]) == 0 ) {
            solver_par.maxiter = atoi( argv[++i] );
        }
    }
    printf( "\n    Usage: ./testing_zgmres --restart %d --maxiter %d --ortho %d (0=MGS, 1=CGS, 2=FUSED_CGS)\n\n",
            solver_par.restart, solver_par.maxiter, ortho_int );

    const char *filename[] =
    {
     "test_matrices/G3_circuit.mtx",
     "test_matrices/Trefethen_20.mtx",
     "test_matrices/test.mtx",
     "test_matrices/bcsstk01.mtx",
     "test_matrices/Trefethen_200.mtx",
     "test_matrices/Trefethen_2000.mtx",
     "test_matrices/Pres_Poisson.mtx",
     "test_matrices/bloweybq.mtx",
     "test_matrices/ecology2.mtx",
     "test_matrices/apache2.mtx",
     "test_matrices/crankseg_2.mtx",
     "test_matrices/bmwcra_1.mtx",
     "test_matrices/F1.mtx",
     "test_matrices/audikw_1.mtx",
     "test_matrices/circuit5M.mtx",
     "test_matrices/boneS10.mtx",
     "test_matrices/parabolic_fem.mtx",
     "test_matrices/inline_1.mtx",
     "test_matrices/ldoor.mtx"
    };

    magma_z_sparse_matrix A, C, D;
    magma_z_vector x, b;

    magmaDoubleComplex one = MAGMA_Z_MAKE(1.0, 0.0);
    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);

 
    //magma_z_csr_mtx( &A, filename[0] ); // G3_circuit
    magma_z_csr_mtx( &A, filename[5] ); // Trefethen_2000
    //magma_z_csr_mtx( &A, filename[8] ); // ecology_2
    //magma_z_csr_mtx( &A, filename[11] ); // bmwcra_1.mtx
    //magma_z_csr_mtx( &A, filename[13] ); // audidx_1.mtx
    //magma_z_csr_mtx( &A, filename[9] );
    //print_z_csr_matrix( A.num_rows, A.num_cols, A.nnz, &A.val, &A.row, &A.col );
    printf( "\nmatrix read: %d-by-%d with %d nonzeros\n\n",A.num_rows,A.num_cols,A.nnz );

    magma_z_vinit( &b, Magma_DEV, A.num_cols, one );
    magma_z_vinit( &x, Magma_DEV, A.num_cols, zero );

    //magma_z_vvisu( x, 0,10);


    //magma_z_mconvert( A, &B, Magma_CSR, Magma_ELLPACK);
    magma_z_mconvert( A, &C, Magma_CSR, Magma_ELLPACKT);
    magma_z_mtransfer( C, &D, Magma_CPU, Magma_DEV);


    //magma_zgmres0( D, b, &x, &solver_par );
    magma_zgmres( D, b, &x, &solver_par );

    //magma_z_vvisu( x, 0,10);

    magma_z_vfree(&x);
    magma_z_vfree(&b);
    magma_z_mfree(&A);
   // magma_z_mfree(&B);
    magma_z_mfree(&C);
    magma_z_mfree(&D);

    TESTING_FINALIZE();
    return 0;
}
