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

     "test_matrices/boneS10.mtx",            // 14
     "test_matrices/parabolic_fem.mtx",      // 15

     "test_matrices/jj_StocF-1465.mtx",      // 16
     "test_matrices/jj_Geo_1438.mtx",
     "test_matrices/jj_Emilia_923.mtx",
     "test_matrices/jj_ML_Geer.mtx",
     "test_matrices/jj_Flan_1565.mtx",
     "test_matrices/jj_Hook_1498.mtx",
     "test_matrices/jj_Long_Coup_dt0.mtx",
     "test_matrices/jj_Long_Coup_dt6.mtx",
     "test_matrices/jj_Cube_Coup_dt6.mtx",
     "test_matrices/jj_Cube_Coup_dt0.mtx",
     "test_matrices/jj_CoupCons3D.mtx",
     "test_matrices/jj_ML_Laplace.mtx",      // 27
     "test_matrices/crankseg_2.mtx",         // 
     "test_matrices/bmwcra_1.mtx",           // 
     "test_matrices/F1.mtx",                 // 
     "test_matrices/inline_1.mtx",           // 
     "test_matrices/ldoor.mtx",           // 
     "test_matrices/audikw_1.mtx",           // 33
    };

    for(magma_int_t matrix=2; matrix<3; matrix++){

    magma_z_sparse_matrix A, C, D;
    magma_z_vector x, b;

    magmaDoubleComplex one = MAGMA_Z_MAKE(1.0, 0.0);
    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
    const char *N="N";
  
    magma_z_csr_mtx( &A, filename[matrix] );

    magma_z_vinit( &b, Magma_DEV, A.num_cols, one );
    magma_z_vinit( &x, Magma_DEV, A.num_cols, zero );

    //magma_z_mconvert( A, &C, Magma_CSR, Magma_ELLPACKT);
    magma_z_mtransfer( A, &D, Magma_CPU, Magma_DEV);

    magma_solver_parameters solver_par;
    solver_par.epsilon = 10e-16;
    solver_par.maxiter = 1000;

    //Chronometry 
    gettimeofday(&inicio, NULL);
    tempo1=inicio.tv_sec+(inicio.tv_usec/1000000.0);

    magma_zcg_merge( D, b, &x, &solver_par );

    //Chronometry  
    gettimeofday(&fim, NULL);
    tempo2=fim.tv_sec+(fim.tv_usec/1000000.0);
    printf("runtime: %f\n", tempo2-tempo1);

    //magma_z_vvisu( x, 0,10);

    magma_z_vfree(&b);
    magma_z_vfree(&x);
    magma_z_mfree(&A);
    magma_z_mfree(&D);
}





    TESTING_FINALIZE();
    return 0;
}
