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
    {//"test_matrices/FullChip.mtx",-
     //"test_matrices/kkt_power.mtx",
     //"test_matrices/pre2.mtx",
     //"test_matrices/kron_g500-logn21.mtx",-
     "test_matrices/diag.mtx",
     "test_matrices/diag.mtx",
     "test_matrices/Trefethen_20.mtx",
     "test_matrices/Trefethen_20_b.mtx",
     "test_matrices/Trefethen_20_c.mtx",
    };
for(magma_int_t matrix=0; matrix<1; matrix++){

    magma_z_sparse_matrix A, B;

    //read matrix from file
    magma_z_csr_mtx( &A, filename[matrix] );

    // magma_z_mvisu( A );

    magma_z_csr_compressor( &A.val, &A.row, &A.col, 
                            &B.val, &B.row, &B.col,
                            &A.num_rows);
    B.memory_location = Magma_CPU;
    B.num_rows = A.num_rows;
    B.num_cols = A.num_cols;
    B.storage_type = A.storage_type;

    // magma_z_mvisu( B );

    write_z_csr_mtx( B.num_rows, B.num_cols, B.nnz, &B.val, &B.row, &B.col, 
                     MagmaColMajor, filename[matrix+1] );

    magma_z_mfree(&A);
    magma_z_mfree(&B);

  

}
    TESTING_FINALIZE();
    return 0;
}
