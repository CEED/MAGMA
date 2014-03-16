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
#include <omp.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "../include/magmasparse.h"
#include "magma_lapack.h"
#include "testings.h"



/* ////////////////////////////////////////////////////////////////////////////
   -- running magma_zcg magma_zcg_merge 
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

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
     "test_matrices/audikw_1.mtx",           // 13
     "test_matrices/boneS10.mtx",            // 14
     "test_matrices/parabolic_fem.mtx",      // 15
     "test_matrices/airfoil_2d.mtx",         // 16
     "test_matrices/bmw3_2.mtx",             // 17
     "test_matrices/cage10.mtx",             // 18
     "test_matrices/fv1.mtx",                // 19
     "test_matrices/poisson3Da.mtx",         // 20
     "test_matrices/Trefethen_20000.mtx",    // 21
     "test_matrices/inline_1.mtx",           // 22
     "test_matrices/ldoor.mtx",              // 23
     "test_matrices/thermal2.mtx",           // 24
     "test_matrices/tmt_sym.mtx",            // 25
     "test_matrices/offshore.mtx",            // 26
     "test_matrices/tmt_unsym.mtx"           // 27

    };
    


    for(int matrix=0; matrix<1; matrix++){
    int num_vecs = 10;

    magma_z_sparse_matrix hA, hA2, dA;

    magma_z_csr_mtx( &hA, filename[matrix] );

    hA2.storage_type = Magma_SELLC;
    hA2.blocksize = 8;
    hA2.alignment = 4;
    magma_z_mconvert( hA, &hA2, Magma_CSR, hA2.storage_type );

    // copy matrix to GPU
    magma_z_mtransfer( hA2, &dA, Magma_CPU, Magma_DEV);

    int n=hA2.num_cols;

    magmaDoubleComplex one = MAGMA_Z_MAKE(1.0, 0.0);
    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);

    magma_z_vector x1_h, x2_h, x1_d, x2_d;

    magma_z_vinit( &x1_h, Magma_CPU, n*num_vecs, one );

    for(int j=0; j<n; j++){
        for(int i=0; i<num_vecs; i++){
        x1_h.val[i+j*num_vecs] = MAGMA_Z_MAKE(double(i+1), 0.0);
        }
    }
  //  for(int i=0; i<num_vecs; i++)
    //    x1_h.val[i*n] = MAGMA_Z_MAKE(double(i+1), 0.0);

    magma_z_vtransfer( x1_h, &x1_d, Magma_CPU, Magma_DEV );


    //magma_z_vinit( &x1_d, Magma_DEV, n*num_vecs, one );

    magma_z_vinit( &x2_d, Magma_DEV, n*num_vecs, zero );



    magma_zprint_gpu( n, num_vecs, x1_d.val, n );

        real_Double_t  gpu_time;
        real_Double_t FLOPS = 2.0*num_vecs*n/1e9;
        gpu_time = magma_wtime();
    //magma_zlobpcg_shift( n, num_vecs, 2, x1_d.val );

    magma_z_spmv( one, dA, x1_d, zero, x2_d );

    //magma_zorthomgs(n, num_vecs, x1_d.val );
        gpu_time = magma_wtime() - gpu_time;
        printf( "blocksize: %d   GFLOP/s:  %.2e\n",num_vecs, FLOPS/gpu_time );//GFLOPS

    magma_zprint_gpu( n, num_vecs, x2_d.val, n );



    magma_z_vfree(&x1_d);
    magma_z_vfree(&x2_d);

    magma_z_vinit( &x1_d, Magma_DEV, n, one );

    magma_z_vinit( &x2_d, Magma_DEV, n, zero );

    magma_z_spmv( one, dA, x1_d, zero, x2_d );

        magma_z_vvisu( x2_d, 0,n);

    magma_z_vfree(&x1_h);
    magma_z_vfree(&x1_d);
    magma_z_vfree(&x2_d);

    magma_z_mfree(&hA);
    magma_z_mfree(&hA2);
    // free GPU memory
    magma_z_mfree(&dA);
    }



    TESTING_FINALIZE();
    return 0;
}
