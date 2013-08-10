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
        }
    }
    printf( "\n    Usage: ./testing_zgmres --ortho %d (0=MGS, 1=CGS, 2=FUSED_CGS)\n\n",ortho_int );

    const char *filename[] =
    {
     "test_matrices/LF10.mtx",
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

 
    magma_z_csr_mtx( &A, filename[5] ); // Trefethen_2000
    //magma_z_csr_mtx( &A, filename[9] );
    //print_z_csr_matrix( A.num_rows, A.num_cols, A.nnz, &A.val, &A.row, &A.col );


    magma_z_vinit( &b, Magma_DEV, A.num_cols, one );
    magma_z_vinit( &x, Magma_DEV, A.num_cols, zero );

    //magma_z_vvisu( x, 0,10);


    //magma_z_mconvert( A, &B, Magma_CSR, Magma_ELLPACK);
    magma_z_mconvert( A, &C, Magma_CSR, Magma_ELLPACKT);
    magma_z_mtransfer( C, &D, Magma_CPU, Magma_DEV);


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
