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

// includes CUDA
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <cusparse_v2.h>
#include <cuda_profiler_api.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "../include/magmasparse.h"
#include "magma_lapack.h"
#include "testings.h"

#define PRECISION_z


/* ////////////////////////////////////////////////////////////////////////////
   -- running magma_zcg magma_zcg_merge 
*/
int main( int argc, char** argv)
{
    TESTING_INIT();
magma_setdevice(2);

    const char *filename[] =
    {
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

    };
    
    double start, end;

   // for(int matrix=16; matrix<20; matrix++){
    for(int matrix=2; matrix<3; matrix++){
   // for(int matrix=4; matrix<11; matrix++){
    int num_vecs = 10;

    magmaDoubleComplex one = MAGMA_Z_MAKE(1.0, 0.0);
    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);



    magma_z_sparse_matrix hA, hA2, hA3, dA, hAD, hADD, dAD, dADD, M, hM, hL, hU, dL, dU, hF, hG, hI, hJ, D, R, dD, dR;



    magma_z_csr_mtx( &hA, filename[matrix] );

    magma_z_vector x, x1, x2, b;
        magma_z_vinit( &b, Magma_DEV, hA.num_cols, one );
        magma_z_vinit( &x, Magma_DEV, hA.num_cols, zero );
        magma_z_vinit( &x1, Magma_DEV, hA.num_cols, zero );
        magma_z_vinit( &x2, Magma_DEV, hA.num_cols, zero );


    
    magma_zcsrsplit( 256, hA, &D, &R );

        magma_z_mtransfer( hA, &dA, Magma_CPU, Magma_DEV );
        magma_z_mtransfer( D, &dD, Magma_CPU, Magma_DEV );
        magma_z_mtransfer( R, &dR, Magma_CPU, Magma_DEV );


    real_Double_t res;

    magma_zresidual( dA,  b, x, &res );

    printf("res before: %.2e\n", res);

    for(int k=0; k<100; k++)
        magma_zbajac_csr( dD, dR, b, &x );

    magma_zresidual( dA,  b, x, &res );

    printf("res after: %.2e\n", res);

}


    TESTING_FINALIZE();
    return 0;
}
