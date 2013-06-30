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

    magma_z_sparse_matrix A, B, C, D, E, F, Z;

    magma_int_t n, nnz, n_col;
  
    //create vectors for sparse matrix
    magmaDoubleComplex *val_h;
    magma_int_t *row_h;
    magma_int_t *col_h;
    //read matrix from file
    //read_z_csr_from_mtx( &A.storage_type, &A.memory_location, &A.num_rows, &A.num_cols, &A.nnz, &A.val, &A.row, &A.col, filename[0] );
    magma_z_csr_mtx( &A, filename[0] );

    magma_z_mtransfer( A, &B, Magma_CPU, Magma_DEV);
    magma_z_mtransfer( B, &C, Magma_DEV, Magma_DEV);
    magma_z_mtransfer( C, &D, Magma_DEV, Magma_CPU);
    magma_z_mconvert( D, &E, Magma_CSR, Magma_ELLPACK, Magma_RowMajor, Magma_RowMajor);
    magma_z_mconvert( E, &Z, Magma_ELLPACK, Magma_CSR, Magma_RowMajor, Magma_RowMajor);
/*
    printf("\n now to ELLPACK \n");
    for(magma_int_t i=0; i<E.num_rows*E.max_nnz_row; i++)
      printf(" %f, %d \n", E.val[i],E.col[i]);

    printf("E    num_rows:%d, nnz:%d\n", E.num_rows, E.nnz);
    printf("\n now back CSR \n");
    printf("Z    num_rows:%d, nnz:%d\n", Z.num_rows, Z.nnz);


    for(magma_int_t i=0; i<Z.num_rows; i++)
      printf(" %f, %d, %d \n", Z.val[i],Z.col[i],Z.row[i]);
    printf("\n now back CSR \n");
    printf("Z    num_rows:%d, nnz:%d\n", Z.num_rows, Z.nnz);
    for(magma_int_t i=0; i<Z.nnz; i++)
      printf(" %f, %d \n", Z.val[i],Z.col[i]);
*/




    //print_z_csr_mtx( A.num_rows, A.num_cols, A.nnz, &A.val, &A.row, &A.col, MagmaRowMajor );
    print_z_csr_matrix( A.num_rows, A.num_cols, A.nnz, &A.val, &A.row, &A.col );
 printf("\n now back CSR \n");
    print_z_csr_matrix( Z.num_rows, Z.num_cols, Z.nnz, &Z.val, &Z.row, &Z.col );
    write_z_csr_mtx( A.num_rows, A.num_cols, A.nnz, &A.val, &A.row, &A.col, MagmaRowMajor, filename[2] );
    write_z_csr_mtx( A.num_rows, A.num_cols, A.nnz, &A.val, &A.row, &A.col, MagmaRowMajor, filename[3] );


    TESTING_FINALIZE();
    return 0;
}
