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
     "test_matrices/ecology2.mtx",
     "test_matrices/apache2.mtx",
     "test_matrices/audikw_1.mtx",
     "test_matrices/boneS10.mtx",
     "test_matrices/inline_1.mtx",
     "test_matrices/ldoor.mtx"
     "test_matrices/bmwcra_1.mtx",
     "test_matrices/F1.mtx",
     "test_matrices/circuit5M.mtx",
     "test_matrices/parabolic_fem.mtx",
     "test_matrices/crankseg_2.mtx",
    };

    magma_z_sparse_matrix A, B, C, D, E, F, G, Z;
    magma_z_sparse_matrix A_D, B_D, C_D, D_D, E_D, F_D, G_D, Z_D;

    magma_int_t n, nnz, n_col;
  

    //read matrix from file
    magma_z_csr_mtx( &A, filename[0] );


    magma_z_mvisu( A );
    write_z_csr_mtx( A.num_rows, A.num_cols, A.nnz, &A.val, &A.row, &A.col, MagmaRowMajor, filename[2] );
    magma_z_mfree( &A );
    magma_z_csr_mtx( &A, filename[2] );
    magma_z_mvisu( A );

    magma_z_mconvert( A, &B, Magma_CSR, Magma_BCSR);
    magma_z_mconvert( B, &C, Magma_BCSR, Magma_CSR);

    magma_z_mvisu( C );
    /*

    magma_z_mconvert( A, &B, Magma_CSR, Magma_ELLPACK);
    magma_z_mconvert( B, &C, Magma_ELLPACK, Magma_CSR);
    magma_z_mconvert( C, &D, Magma_CSR, Magma_ELLPACKT);
    magma_z_mconvert( D, &E, Magma_ELLPACKT, Magma_CSR);
    magma_z_mconvert( A, &F, Magma_CSR, Magma_DENSE);
    magma_z_mconvert( F, &G, Magma_DENSE, Magma_CSR);

    magma_z_mvisu( G );
   
    magma_z_mtransfer( A, &A_D, Magma_CPU, Magma_DEV);
    magma_z_mtransfer( B, &B_D, Magma_CPU, Magma_DEV);
    magma_z_mtransfer( C, &C_D, Magma_CPU, Magma_DEV);
    magma_z_mtransfer( D, &D_D, Magma_CPU, Magma_DEV);
    magma_z_mtransfer( E, &E_D, Magma_CPU, Magma_DEV);
    magma_z_mtransfer( F, &F_D, Magma_CPU, Magma_DEV);





printf("A:\n");
    magma_z_mfree(&A);
printf("B:\n");
    magma_z_mfree(&B);
printf("C:\n");
    magma_z_mfree(&C);
printf("D:\n");
    magma_z_mfree(&D);
printf("E:\n");
    magma_z_mfree(&E);
printf("F:\n");
    magma_z_mfree(&F);
printf("G:\n");
    magma_z_mfree(&G);

printf("A_D:\n");
    magma_z_mfree(&A_D);
printf("B_D:\n");
    magma_z_mfree(&B_D);
printf("C_D:\n");
    magma_z_mfree(&C_D);
printf("D_D:\n");
    magma_z_mfree(&D_D);
printf("E_D:\n");
    magma_z_mfree(&E_D);
printf("F_D:\n");
    magma_z_mfree(&F_D);



   // magma_z_mconvert( G, &Z, Magma_CSR, Magma_BCSR);

/*
    magma_int_t size_b=3;

    printf("row=\n");
    for( magma_int_t i=0; i<ceil(A.num_rows/size_b)+1; i++){
        printf("%d  ", Z.row[i]);
    } 
    printf("\n");
*/


    //write_z_csr_mtx( A.num_rows, A.num_cols, A.nnz, &A.val, &A.row, &A.col, MagmaRowMajor, filename[3] );
/*
    magma_z_vector x, y, z;

    magmaDoubleComplex one = MAGMA_Z_MAKE(1.0, 0.0);
    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);

    magma_z_vinit( &x, Magma_CPU, A.num_cols, one );
    magma_z_vinit( &y, Magma_DEV, A.num_cols, one );
    magma_z_vinit( &z, Magma_DEV, A.num_cols, zero );

    magma_z_vvisu( z, 0,10);
    magma_z_vfree(&z);
    magma_z_vfree(&x);
    magma_z_vfree(&y);
*/
    TESTING_FINALIZE();
    return 0;
}
