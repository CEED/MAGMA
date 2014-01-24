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
     "test_matrices/diag.mtx",
     "test_matrices/tridiag.mtx",
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
for(magma_int_t matrix=2; matrix<4; matrix++){


    magma_z_sparse_matrix A, B, C, D, E, F, G, H, I, J, K, Z;
    magma_z_vector a,b,c,x, y, z;
    magma_int_t k=30;

    magmaDoubleComplex one = MAGMA_Z_MAKE(1.0, 0.0);
    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
    const char *N="N";




    magma_z_csr_mtx( &A, filename[matrix] );
    magma_z_mconvert( A, &B, Magma_CSR, Magma_ELLPACK);
    magma_z_mvisu( A );
    //magma_zlra( B, &C, 15 );
    magma_z_mconvert( C, &D, Magma_ELLPACK, Magma_CSR);
    magma_z_mvisu( D );






/*
    magma_z_vinit( &x, Magma_DEV, A.num_cols, one );
    magma_z_vinit( &y, Magma_DEV, A.num_cols*(k+1), one );
    printf("A.num_cols: %d  x.num_rows: %d\n", A.num_cols, x.num_rows);



    magma_zdiameter( &A );
    magma_zrowentries( &A );
    magma_z_mconvert( A, &B, Magma_CSR, Magma_ELLPACKT);
    magma_z_mtransfer( B, &C, Magma_CPU, Magma_DEV);
    printf("A max row: %d  A diamter: %d\n", A.max_nnz_row, A.diameter);
    printf("B max row: %d  B diamter: %d\n", B.max_nnz_row, B.diameter);

/*

    //Chronometry 
    gettimeofday(&inicio, NULL);
    tempo1=inicio.tv_sec+(inicio.tv_usec/1000000.0);
    magma_z_mpk( one, C, x, zero, y, k);
    magma_device_sync();
    //Chronometry  
    gettimeofday(&fim, NULL);
    tempo2=fim.tv_sec+(fim.tv_usec/1000000.0);

    printf("matrix powers: %4d  runtime: %e\n", k, tempo2-tempo1);

    //for( magma_int_t i=0; i<k; i++ )
       // magma_z_vvisu( y, A.num_rows*(i),10);

    //for( magma_int_t i=0; i<k; i++ )
      //  magma_z_vvisu( y, A.num_rows*(i+1)-10,5);

    magma_z_vvisu( y, A.num_rows*(k-1),10);

    magma_z_vfree(&x);
    magma_z_vfree(&y);
    magma_z_vinit( &b, Magma_DEV, A.num_cols*k, zero );
    magma_z_vinit( &a, Magma_DEV, A.num_cols, one );
    magma_z_vinit( &c, Magma_DEV, A.num_cols, zero );
*/
/*
    //Chronometry 
    gettimeofday(&inicio, NULL);
    tempo1=inicio.tv_sec+(inicio.tv_usec/1000000.0);
    for(magma_int_t i=0; i<k; i++){
        magma_z_spmv( one, C, a, zero, c);
        magma_device_sync();
        magma_zcopy(A.num_rows, c.val, 1, b.val+i*A.num_rows, 1);  
        magma_device_sync();
        a.val = b.val+i*A.num_rows;
        magma_device_sync();
        magma_z_vvisu( a, 0,10);
        magma_device_sync();
    }
    //Chronometry  
    gettimeofday(&fim, NULL);
    tempo2=fim.tv_sec+(fim.tv_usec/1000000.0);

    printf("matrix powers: %4d  runtime: %e\n", k, tempo2-tempo1);
    magma_z_vvisu( a, 0,10);

*/
/*
    magma_z_vfree(&a);
    magma_z_vfree(&b);
    magma_z_vfree(&c);
*/
    magma_z_mfree(&A);
    magma_z_mfree(&B);
    magma_z_mfree(&C);

}




    TESTING_FINALIZE();
    return 0;
}
