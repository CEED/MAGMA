/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> s d c
       @author Hartwig Anzt
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <cublas.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing matrix IO
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    magma_opts opts;
    parse_opts( argc, argv, &opts );
    
    printf("If running lapack (option --lapack), MAGMA and CUBLAS error are both computed\n"
           "relative to CPU BLAS result. Else, MAGMA error is computed relative to CUBLAS result.\n\n"
           "transA = %c, transB = %c\n", opts.transA, opts.transB );
    printf("    M     N     K   MAGMA Gflop/s (ms)  CUBLAS Gflop/s (ms)   CPU Gflop/s (ms)  MAGMA error  CUBLAS error\n");
    printf("=========================================================================================================\n");


    const char *filename[] =
    {
     "test_matrices/Trefethen_20.mtx",
     "test_matrices/Trefethen_2000.mtx",
     "test_matrices/Trefethen_20_new.mtx",
     "test_matrices/Trefethen_20_new2.mtx",
     "test_matrices/Trefethen_20_new3.mtx"
    };

    magma_z_sparse_matrix A, B, C, D, E, Z;
    magma_z_vector x, y, z;

    magmaDoubleComplex one = MAGMA_Z_MAKE(1.0, 0.0);
    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
    const char *N="N";

  
    magma_z_csr_mtx( &A, filename[1] );
    //print_z_csr_matrix( A.num_rows, A.num_cols, A.nnz, &A.val, &A.row, &A.col );


    magma_z_vinit( &x, Magma_CPU, A.num_cols, one );
    magma_z_vinit( &y, Magma_DEV, A.num_cols, one );
    magma_z_vtransfer( x, &z, Magma_CPU, Magma_DEV);
    magma_z_vvisu( y, 0,10);


    magma_z_mtransfer( A, &B, Magma_CPU, Magma_DEV);

    magma_z_mconvert( A, &C, Magma_CSR, Magma_ELLPACK);
    magma_z_mtransfer( C, &D, Magma_CPU, Magma_DEV);




   //magma_zgecsrmv( N, B.num_rows, B.num_cols, one, B.val, B.row, B.col, y.val, one, z.val);
    magma_z_spmv( one, D, y, one, z);

    magma_z_vvisu( z, 0,10);






    TESTING_FINALIZE();
    return 0;
}
