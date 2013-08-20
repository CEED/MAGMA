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
    solver_par.gmres_ortho = MAGMA_CGS;

    int i, id = -1, ortho_int = 1;
    for( int i = 1; i < argc; ++i ) {
        if ( strcmp("--ortho", argv[i]) == 0 ) {
            ortho_int = atoi( argv[++i] );
            switch( ortho_int ) {
                case 0: solver_par.gmres_ortho = MAGMA_MGS; break;
                case 1: solver_par.gmres_ortho = MAGMA_CGS; break;
                case 2: solver_par.gmres_ortho = MAGMA_FUSED_CGS; break;
            }
        } else if ( strcmp("--restart", argv[i]) == 0 ) {
            solver_par.restart = atoi( argv[++i] );
        } else if ( strcmp("--maxiter", argv[i]) == 0 ) {
            solver_par.maxiter = atoi( argv[++i] );
        } else if ( strcmp("--tol", argv[i]) == 0 ) {
            sscanf( argv[++i], "%lf", &solver_par.epsilon );
        } else if ( strcmp("--id", argv[i]) == 0 ) {
            id = atoi( argv[++i] );
        }
    }
    printf( "\n    Usage: ./testing_zgmres --restart %d --maxiter %d --tol %.2e --ortho %d (0=MGS, 1=CGS, 2=FUSED_CGS)\n\n",
            solver_par.restart, solver_par.maxiter, solver_par.epsilon, ortho_int );

    const char *filename[] =
    {
     "test_matrices/Trefethen_20.mtx",       // 0
     "test_matrices/Trefethen_200.mtx",      // 1
     "test_matrices/Trefethen_2000.mtx",     // 2
     "test_matrices/G3_circuit.mtx",         // 3
     "test_matrices/test.mtx",               // 4
     "test_matrices/bcsstk01.mtx",           // 5
     "test_matrices/Pres_Poisson.mtx",       // 6
     "test_matrices/bloweybq.mtx",           // 7
     "test_matrices/ecology2.mtx",           // 8
     "test_matrices/apache2.mtx",            // 9
     "test_matrices/crankseg_2.mtx",         // 10
     "test_matrices/bmwcra_1.mtx",           // 11
     "test_matrices/F1.mtx",                 // 12
     "test_matrices/audikw_1.mtx",           // 13
     "test_matrices/boneS10.mtx",            // 14
     "test_matrices/parabolic_fem.mtx",      // 15
    };

    magma_z_sparse_matrix A, C, D;
    magma_z_vector x, b;

    magmaDoubleComplex one = MAGMA_Z_MAKE(1.0, 0.0);
    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);

    for( i=0; i<16; i++ ) 
    {
        if (id > -1) i = id;
        magma_z_csr_mtx( &A, filename[i] ); 
        printf( "\nmatrix read: %d-by-%d with %d nonzeros\n\n",A.num_rows,A.num_cols,A.nnz );

        magma_z_vinit( &b, Magma_DEV, A.num_cols, one );
        magma_z_vinit( &x, Magma_DEV, A.num_cols, zero );

        magma_z_mconvert( A, &C, Magma_CSR, Magma_ELLPACKT);
        magma_z_mtransfer( C, &D, Magma_CPU, Magma_DEV);

        magma_zgmres( D, b, &x, &solver_par );

        magma_z_vfree(&x);
        magma_z_vfree(&b);
        magma_z_mfree(&A); 
        magma_z_mfree(&C);
        magma_z_mfree(&D);

        if (id > -1) break;
    }

    TESTING_FINALIZE();
    return 0;
}
