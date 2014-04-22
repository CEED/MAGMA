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
#include "../include/magmasparse.h"
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
/*
"/home/hanzt/sparse_matrices/mtx/audikw_1.mtx",
     "/home/hanzt/sparse_matrices/mtx/bone010.mtx",
     "/home/hanzt/sparse_matrices/mtx/bmwcra_1.mtx",
     "/home/hanzt/sparse_matrices/mtx/crankseg_2.mtx",
     "/home/hanzt/sparse_matrices/mtx/F1.mtx",   
     "/home/hanzt/sparse_matrices/mtx/inline_1.mtx",
     "/home/hanzt/sparse_matrices/mtx/ldoor.mtx",
*/
     "/home/hanzt/sparse_matrices/mtx/webbase-1M.mtx",
            "/home/hanzt/sparse_matrices/mtx/thermal1.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
            "/home/hanzt/sparse_matrices/mtx/apache1.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
            "/home/hanzt/sparse_matrices/mtx/shallow_water1.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
            "/home/hanzt/sparse_matrices/mtx/shallow_water2.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
            "/home/hanzt/sparse_matrices/mtx/minsurfo.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
       //     "/home/hanzt/sparse_matrices/mtx/bloweybq.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
            "/home/hanzt/sparse_matrices/mtx/Pres_Poisson.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0

            "/home/hanzt/sparse_matrices/mtx/A010.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
            "/home/hanzt/sparse_matrices/mtx/A013.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
            "/home/hanzt/sparse_matrices/mtx/A016.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
            "/home/hanzt/sparse_matrices/mtx/A020.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0      
            "/home/hanzt/sparse_matrices/mtx/A025.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0            
            "/home/hanzt/sparse_matrices/mtx/A032.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
            "/home/hanzt/sparse_matrices/mtx/A040.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
            "/home/hanzt/sparse_matrices/mtx/A050.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
            "/home/hanzt/sparse_matrices/mtx/A063.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
            "/home/hanzt/sparse_matrices/mtx/A080.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
            "/home/hanzt/sparse_matrices/mtx/A100.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0            
            "/home/hanzt/sparse_matrices/mtx/A126.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0


     "test_matrices/Trefethen_20.mtx",       // 0
     "/home/hanzt/sparse_matrices/mtx/Trefethen_200.mtx",
     "/home/hanzt/sparse_matrices/mtx/Trefethen_2000.mtx",
     "/home/hanzt/sparse_matrices/mtx/Trefethen_20000.mtx",
     "/home/hanzt/sparse_matrices/mtx/apache2.mtx", //4
     "/home/hanzt/sparse_matrices/mtx/ecology2.mtx",
     "/home/hanzt/sparse_matrices/mtx/parabolic_fem.mtx",
     "/home/hanzt/sparse_matrices/mtx/G3_circuit.mtx",
     "/home/hanzt/sparse_matrices/mtx/af_shell3.mtx",
     "/home/hanzt/sparse_matrices/mtx/offshore.mtx",
     "/home/hanzt/sparse_matrices/mtx/thermal2.mtx",
     "/home/hanzt/sparse_matrices/mtx/audikw_1.mtx",    //11
     "/home/hanzt/sparse_matrices/mtx/bone010.mtx",
     "/home/hanzt/sparse_matrices/mtx/boneS10.mtx",
     "/home/hanzt/sparse_matrices/mtx/bmw3_2.mtx",
     "/home/hanzt/sparse_matrices/mtx/bmwcra_1.mtx",
     "/home/hanzt/sparse_matrices/mtx/cage10.mtx",
     "/home/hanzt/sparse_matrices/mtx/cant.mtx",
     "/home/hanzt/sparse_matrices/mtx/crankseg_2.mtx",
     "/home/hanzt/sparse_matrices/mtx/jj_Cube_Coup_dt0.mtx",
     "/home/hanzt/sparse_matrices/mtx/dielFilterV2real.mtx",
     "/home/hanzt/sparse_matrices/mtx/F1.mtx",   
     "/home/hanzt/sparse_matrices/mtx/Fault_639.mtx",
     "/home/hanzt/sparse_matrices/mtx/Hook_1498.mtx",
     "/home/hanzt/sparse_matrices/mtx/inline_1.mtx",
     "/home/hanzt/sparse_matrices/mtx/ldoor.mtx",
     "/home/hanzt/sparse_matrices/mtx/m_t1.mtx",
     "/home/hanzt/sparse_matrices/mtx/jj_ML_Geer.mtx",
     "/home/hanzt/sparse_matrices/mtx/pwtk.mtx",
     "/home/hanzt/sparse_matrices/mtx/shipsec1.mtx",
     "/home/hanzt/sparse_matrices/mtx/Trefethen_20000.mtx",
     "/home/hanzt/sparse_matrices/mtx/StocF-1465.mtx",
     "/home/hanzt/sparse_matrices/mtx/kkt_power.mtx",
     "/home/hanzt/sparse_matrices/mtx/bmwcra_1.mtx",
     "/home/hanzt/sparse_matrices/mtx/thermal2.mtx",
     "/home/hanzt/sparse_matrices/mtx/pre2.mtx",
     "/home/hanzt/sparse_matrices/mtx/xenon2.mtx",
     "/home/hanzt/sparse_matrices/mtx/stomach.mtx",

    };
for(magma_int_t matrix=0; matrix<100; matrix++){

    magma_z_sparse_matrix A, B, C, D, E, F, G, Z;
    magma_z_sparse_matrix A_D, B_D, C_D, D_D, E_D, F_D, G_D, Z_D;

    magma_int_t n, nnz, n_col;
  

    //read matrix from file
    magma_z_csr_mtx( &A, filename[matrix] );
    //magma_z_mvisu(A);
/*   
    printf("convert to CSRCSC:");
    magma_z_mconvert( A, &F, Magma_CSR, Magma_CSRCSC);
    printf("done.\n");
    printf("convert to CSR:");
    magma_z_mconvert( F, &G, Magma_CSRCSC, Magma_CSR);
    printf("done.\n");
for(int z=A.nnz; z<2*A.nnz; z++)
        F.val[z] = MAGMA_Z_MAKE(55.0, 0.0);

    magma_z_mtransfer( F, &D, Magma_CPU, Magma_DEV);
    //magma_z_mvisu(D);

    //magma_z_mvisu(F);
    //magma_z_mfree(&F);*/

    B.blocksize = 2;
    B.alignment = 1;
    magma_z_mconvert( A, &B, Magma_CSR, Magma_SELLP);
printf("conversion to SELLP successful.\n");
  //  magma_z_mconvert( A, &C, Magma_CSR, Magma_ELL);
printf("conversion to ELL successful.\n");

    printf("n:%d nnz:%d nnz/n:%d max_nnz_row:%d SELLP-storage:%d overhead:  %.4e\n", A.num_rows, A.nnz, A.nnz/A.num_rows, B.max_nnz_row, B.nnz,
((double)B.nnz-(double)A.nnz)/((double)B.nnz)*100.0);
   printf("%d  &  %d & %.2e\n", A.nnz, A.num_rows, (double)A.nnz/(double)A.num_rows );
int nnzellpack = 0;//A.num_rows * C.max_nnz_row;
int nnzsellp = B.nnz;
double ratioellpack = (double) (nnzellpack-A.nnz)/((double)nnzellpack)*100.0;
double ratiosellp = (double) (nnzsellp-A.nnz)/((double)nnzsellp)*100.0;

   printf("%d  &  %.6e%%  &  %d  &  %.6e%%\n", nnzellpack, ratioellpack, nnzsellp, ratiosellp);

    magma_z_mfree(&A);
    magma_z_mfree(&B);
    magma_z_mfree(&C);

    //magma_z_mvisu( A );
    //write_z_csr_mtx( A.num_rows, A.num_cols, A.nnz, &A.val, &A.row, &A.col, MagmaRowMajor, filename[2] );
    //magma_z_mfree( &A );
    //magma_z_csr_mtx( &A, filename[2] );
    //magma_z_mvisu( A );

    //magma_z_mconvert( A, &B, Magma_CSR, Magma_BCSR);
    //magma_z_mconvert( B, &C, Magma_BCSR, Magma_CSR);

    //magma_z_mvisu( C );
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

}
    TESTING_FINALIZE();
    return 0;
}
