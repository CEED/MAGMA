/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

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
magma_zgmres_pipe( magma_z_sparse_matrix A, magma_z_vector b, magma_z_vector *x,
                   magma_z_solver_par *solver_par );

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing magma_zgmres
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    magma_z_solver_par solver_par;
    solver_par.epsilon = 10e-8;
    solver_par.maxiter = 10;
    solver_par.restart = 30;
    for( int i = 1; i < argc; ++i ) {
        if ( strcmp("--restart", argv[i]) == 0 ) {
            solver_par.restart = atoi( argv[++i] );
        } else if ( strcmp("--maxiter", argv[i]) == 0 ) {
            solver_par.maxiter = atoi( argv[++i] );
        }
    }
    printf( "\n    Usage: ./testing_zgmres_pipe --restart %d --maxiter %d\n\n",
            solver_par.restart, solver_par.maxiter );

    const char *filename[] =
    {
     "test_matrices/Trefethen_2000.mtx",
     "test_matrices/Pres_Poisson.mtx",
     "test_matrices/bloweybq.mtx",
     "test_matrices/ecology2.mtx",
     "test_matrices/apache2.mtx",
     "test_matrices/crankseg_2.mtx",
     "test_matrices/bmwcra_1.mtx",
     "test_matrices/F1.mtx",

     "test_matrices/audikw_1.mtx",
     //"test_matrices/inline_1.mtx",
     "test_matrices/ldoor.mtx",
     "test_matrices/kkt_power.mtx",
     "test_matrices/Fault_639.mtx",
     "test_matrices/thermomech_dK.mtx",
     "test_matrices/thermal2.mtx",
     "test_matrices/G3_circuit.mtx",
     "test_matrices/tmt_sym.mtx",
     "test_matrices/offshore.mtx",
     "test_matrices/bmw3_2.mtx",
     "test_matrices/airfoil_2d.mtx",
     "test_matrices/cyl6.mtx",
     "test_matrices/poisson3Da.mtx",
     "test_matrices/stokes64.mtx",
     "test_matrices/circuit_3.mtx",
     "test_matrices/cage10.mtx",
     "test_matrices/boneS10.mtx",
     "test_matrices/parabolic_fem.mtx",
     "test_matrices/circuit5M.mtx"
    };

for(magma_int_t matrix=9; matrix<10; matrix++){

    magma_z_sparse_matrix A, C, D;
    magma_z_vector x, b;

    magmaDoubleComplex one = MAGMA_Z_MAKE(1.0, 0.0);
    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);

  
    magma_z_csr_mtx( &A, filename[matrix] ); 

    magma_z_vinit( &b, Magma_DEV, A.num_cols, one );
    magma_z_vinit( &x, Magma_DEV, A.num_cols, zero );



    magma_z_mconvert( A, &C, Magma_CSR, Magma_ELLPACKT);
    magma_z_mtransfer( C, &D, Magma_CPU, Magma_DEV);

    magma_zgmres_pipe( D, b, &x, &solver_par );



    //magma_z_vfree(&x);
    //magma_z_vfree(&b);
    magma_z_mfree(&A);
    magma_z_mfree(&C);
    magma_z_mfree(&D);

}

    TESTING_FINALIZE();
    return 0;
}
