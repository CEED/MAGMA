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
#include <sys/time.h>
#include <time.h>
#include <cuda_runtime_api.h>
#include <cublas.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing magma_zcg
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    magma_opts opts;
    parse_opts( argc, argv, &opts );

    //Chronometry
    struct timeval inicio, fim;
    double tempo1, tempo2;
    
  const char *filename[] =
    {
     "test_matrices/fv1.mtx",
     "test_matrices/Trefethen_2000.mtx",
     "test_matrices/parabolic_fem.mtx",
     "test_matrices/shallow_water1.mtx",
     "test_matrices/G3_circuit.mtx",
     "test_matrices/thermal1.mtx",
     "test_matrices/nasa1824.mtx",
     "test_matrices/m_t1.mtx",
     "test_matrices/bloweybq.mtx",
     "test_matrices/ecology2.mtx",
     "test_matrices/apache2.mtx",
     "test_matrices/crankseg_2.mtx",
     "test_matrices/bmwcra_1.mtx",
     "test_matrices/F1.mtx",
     "test_matrices/boneS10.mtx",
     "test_matrices/inline_1.mtx",
     "test_matrices/ldoor.mtx",
     "test_matrices/audikw_1.mtx",
     "test_matrices/circuit5M.mtx"
    };
for(magma_int_t matrix=4; matrix<5; matrix++){
for(magma_int_t iters=1000; iters<20000; iters+=1000){
    magma_z_sparse_matrix A, C, D;
    magma_z_vector x, b;

    magmaDoubleComplex one = MAGMA_Z_MAKE(1.0, 0.0);
    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
    const char *N="N";

  
    magma_z_csr_mtx( &A, filename[matrix] );
    //print_z_csr_matrix( A.num_rows, A.num_cols, A.nnz, &A.val, &A.row, &A.col );


    magma_z_vinit( &b, Magma_DEV, A.num_cols, one );
    magma_z_vinit( &x, Magma_DEV, A.num_cols, zero );

    //magma_z_vvisu( x, 0,10);




    //magma_z_mconvert( A, &C, Magma_CSR, Magma_ELLPACKT);
    magma_z_mtransfer( A, &D, Magma_CPU, Magma_DEV);


    magma_solver_parameters solver_par;

    solver_par.epsilon = 10e-20;
    solver_par.maxiter = iters;

    //Chronometry 
    gettimeofday(&inicio, NULL);
    tempo1=inicio.tv_sec+(inicio.tv_usec/1000000.0);

    magma_zcg( D, b, &x, &solver_par );

    //Chronometry  
    gettimeofday(&fim, NULL);
    tempo2=fim.tv_sec+(fim.tv_usec/1000000.0);
    printf("runtime: %f\n", tempo2-tempo1);

    //magma_z_vvisu( x, 0,10);

    magma_z_vfree(&b);
    magma_z_vfree(&x);
    magma_z_mfree(&A);
    //magma_z_mfree(&C);
    magma_z_mfree(&D);

}
}





    TESTING_FINALIZE();
    return 0;
}
