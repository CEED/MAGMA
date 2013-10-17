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
#include "mkl_spblas.h"

#define PRECISION_z
#if defined(PRECISION_z)
#define MKL_ADDR(a) (MKL_Complex16*)(a)
#elif defined(PRECISION_c)
#define MKL_ADDR(a) (MKL_Complex8*)(a)
#else
#define MKL_ADDR(a) (a)
#endif


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing matrix vector product
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

    int id = -1, matrix = 0, i, ione = 1, *pntre, num_rblocks, num_cblocks;
    double start, end, work[1];
    for( int i = 1; i < argc; ++i ) {
        if ( strcmp("--id", argv[i]) == 0 ) {
            id = atoi( argv[++i] );
        }
    }
    if (id > -1) printf( "\n    Usage: ./testing_z_mv --id %d\n\n",id );
    printf( "=============================\n");
    printf( "matrix number   CSR-SpMV [s]\n");
    printf( "-----------------------------\n");
    for(matrix=27; matrix<28; matrix++)
    {
        int runs=100;

        magma_z_sparse_matrix hA, hB, hC, dA, dB;
        magma_z_vector hx, hy, dx, dy, dz;

        magmaDoubleComplex one  = MAGMA_Z_MAKE(1.0, 0.0);
        magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
  
        // init matrix on CPU
        if (id > -1) matrix = id;
        magma_z_csr_mtx( &hA, filename[matrix] );
        //printf( "\nmatrix read: %d-by-%d with %d nonzeros\n\n",hA.num_rows,hA.num_cols,hA.nnz );

        // copy matrix to GPU
        magma_z_mtransfer( hA, &dA, Magma_CPU, Magma_DEV);
        // init vectors on GPU    
        magma_z_vinit( &dx, Magma_DEV, hA.num_cols, one );
        magma_z_vinit( &dy, Magma_DEV, hA.num_cols, zero );
        magma_z_vinit( &dz, Magma_DEV, hA.num_cols, zero );

        // experiment setup
        int nIter = 1000;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // run SpMV kernel
        cudaDeviceSynchronize();
        cudaEventRecord(start, 0);

        for ( i=0; i<nIter; i++ ){
            magma_zbicgmerge_spmv1( dA.num_rows, dz.val, dz.val, dA.val, dA.row, dA.col, dz.val,  dx.val,  dy.val,  dz.val );
            //magma_z_spmv( one, dA, dx, zero, dy );
        }

        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float msecTotal = 0.0;
        cudaEventElapsedTime(&msecTotal, start, stop);

        // Compute and print the performance
        double msecPerMatrixMul = msecTotal / nIter;
        double flopsPerMatrixMul = 2.0 * dA.nnz;
        double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
        printf(
            "Performance= %.2f GFlop/s, Time= %.3f msec, SPMV FLOPS= %.0f Ops\n",
            gigaFlops,
            msecPerMatrixMul,
            flopsPerMatrixMul);



        //printf( "%d     %d  %d\n",matrix, dA.num_rows, dA.nnz );

        // free CPU memory
        magma_z_mfree(&hA);
        // free GPU memory
        magma_z_mfree(&dA);
        magma_z_vfree(&dx);
        magma_z_vfree(&dy);
        magma_z_vfree(&dz);

        if (id > -1) break;
    }
    printf( "=============================\n");

    TESTING_FINALIZE();
    return 0;
}
