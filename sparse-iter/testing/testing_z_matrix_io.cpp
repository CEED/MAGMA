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
    magma_int_t n, nnz, n_col;
  
    //create vectors for sparse matrix
    magmaDoubleComplex *val_h;
    magma_int_t *row_h;
    magma_int_t *col_h;
    //read matrix from file
    read_z_csr_from_mtx( &n,&n_col,&nnz, &val_h, &row_h, &col_h, filename[0] );

    print_z_csr_mtx( n, n, nnz, &val_h, &row_h, &col_h, 0 );
    write_z_csr_mtx( n, n, nnz, &val_h, &row_h, &col_h, 0, filename[2] );
    write_z_csr_mtx( n, n, nnz, &val_h, &row_h, &col_h, 1, filename[3] );


    TESTING_FINALIZE();
    return 0;
}
