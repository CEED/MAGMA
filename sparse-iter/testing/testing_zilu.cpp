/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
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
   -- Debugging file
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    //Chronometry
    struct timeval inicio, fim;
    double tempo1, tempo2;

    magma_opts opts;
    parse_opts( argc, argv, &opts );
    
    printf("If running lapack (option --lapack), MAGMA and CUBLAS error are both computed\n"
           "relative to CPU BLAS result. Else, MAGMA error is computed relative to CUBLAS result.\n\n"
           "transA = %c, transB = %c\n", opts.transA, opts.transB );
    printf("    M     N     K   MAGMA Gflop/s (ms)  CUBLAS Gflop/s (ms)   CPU Gflop/s (ms)  MAGMA error  CUBLAS error\n");
    printf("=========================================================================================================\n");


    const char *filename[] =
    {
     "test_matrices/fv1.mtx",
     "test_matrices/lap_25.mtx",
     "test_matrices/diag.mtx",
     "test_matrices/tridiag_2.mtx",
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
for(magma_int_t matrix=8; matrix<10; matrix++){


    magma_z_sparse_matrix A, B, C, D, E, F, G, H, I, J, K, Z;
    magma_z_vector a,b,c,x, y, z;

    magmaDoubleComplex one = MAGMA_Z_MAKE(1.0, 0.0);
    magmaDoubleComplex mone = MAGMA_Z_MAKE(-1.0, 0.0);
    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
    const char *N="N";



    
    magma_z_csr_mtx( &A, filename[matrix] );
    B.blocksize = 20;
    magma_z_mconvert( A, &B, Magma_CSR, Magma_BCSR);
    //printf("A:\n");
    //magma_z_mvisu( A );
    //printf("B:\n");
    //magma_z_mvisu( B );
    magma_z_mtransfer( B, &C, Magma_CPU, Magma_DEV);
    
    // prepare pivoting vector
    magma_int_t *ipiv;
    magma_imalloc_cpu( &ipiv, C.blocksize*(ceil( (float)C.num_rows / (float)C.blocksize )+1) );
    // compute LU factorization
    magma_zilusetup( C, &D, ipiv);



    //printf("D:\n");

    //magma_z_mvisu( D );
    // setup vector
    magma_z_vinit( &b, Magma_DEV, A.num_cols, mone );
    magma_z_vinit( &x, Magma_DEV, A.num_cols, one );
    magma_z_vvisu( x, 0, 5 );

    magma_solver_parameters solver_par;
    solver_par.epsilon = 10e-8;
    solver_par.maxiter = 1000;

    magma_zilu( D, b, &x, &solver_par, ipiv );


    magma_z_vvisu( x, 0, 5 );

    magma_z_mfree(&A);
    magma_z_mfree(&B);
    magma_z_mfree(&C);
    magma_z_mfree(&D);
    magma_z_vfree(&b);
    magma_z_vfree(&x);

}




    TESTING_FINALIZE();
    return 0;
}
