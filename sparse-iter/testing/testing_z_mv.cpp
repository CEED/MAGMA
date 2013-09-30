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

     "test_matrices/jj_StocF-1465.mtx",      // 16
     "test_matrices/jj_Geo_1438.mtx",
     "test_matrices/jj_Emilia_923.mtx",
     "test_matrices/jj_ML_Geer.mtx",
     "test_matrices/jj_Flan_1565.mtx",
     "test_matrices/jj_Hook_1498.mtx",
     "test_matrices/jj_Long_Coup_dt0.mtx",
     "test_matrices/jj_Long_Coup_dt6.mtx",
     "test_matrices/jj_Cube_Coup_dt6.mtx",
     "test_matrices/jj_Cube_Coup_dt0.mtx",
     "test_matrices/jj_CoupCons3D.mtx",
     "test_matrices/jj_ML_Laplace.mtx",      // 27
    };

    int id = -1, matrix = 0, i, ione = 1, *pntre, num_rblocks, num_cblocks;
    double start, end, work[1];
    for( int i = 1; i < argc; ++i ) {
        if ( strcmp("--id", argv[i]) == 0 ) {
            id = atoi( argv[++i] );
        }
    }
    if (id > -1) printf( "\n    Usage: ./testing_z_mv --id %d\n\n",id );

    for(matrix=16; matrix<27; matrix++)
    {
        magma_z_sparse_matrix hA, hB, hC, dA, dB;
        magma_z_vector hx, hy, dx, dy;

        magmaDoubleComplex one  = MAGMA_Z_MAKE(1.0, 0.0);
        magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
  
        // init matrix on CPU
        if (id > -1) matrix = id;
        magma_z_csr_mtx( &hA, filename[matrix] );
        printf( "\nmatrix read: %d-by-%d with %d nonzeros\n\n",hA.num_rows,hA.num_cols,hA.nnz );
        // conver to ELLPACKT and copy to GPU
        magma_z_mconvert(  hA, &hB, Magma_CSR, Magma_ELLPACKT);
        magma_z_mtransfer( hB, &dB, Magma_CPU, Magma_DEV);
        magma_z_mfree(&hB);

        // init CPU vectors
        hC.blocksize = min(hA.num_rows, 8);
        num_rblocks = (hA.num_rows+hC.blocksize-1)/hC.blocksize;
        num_cblocks = (hA.num_cols+hC.blocksize-1)/hC.blocksize;
        magma_z_vinit( &hx, Magma_CPU, num_cblocks*hC.blocksize, one );
        magma_z_vinit( &hy, Magma_CPU, num_rblocks*hC.blocksize, zero );
        printf( " num_rblocks=%d, block_size=%d\n",num_rblocks,hC.blocksize );

        // calling MKL with CSR
        pntre = (int*)malloc( (hA.num_rows+1)*sizeof(int) );
        pntre[0] = 0;
        for (i=0; i<hA.num_rows; i++ ) pntre[i] = hA.row[i+1];
        start = magma_wtime(); 
        for (i=0; i<10; i++ )
        mkl_zcsrmv( "N", &hA.num_rows, &hA.num_cols, 
                    MKL_ADDR(&one), "GFNC", MKL_ADDR(hA.val), hA.col, hA.row, pntre, 
                                            MKL_ADDR(hx.val), 
                    MKL_ADDR(&zero),        MKL_ADDR(hy.val) );
        end = magma_wtime();
        printf( "\n > MKL  : %.2e seconds (CSR).\n",(end-start)/10 );
        free(pntre);
        //magma_zprint( 10,1, hy.val,10 );

        // calling MKL with BSR
        if( hA.num_rows / hC.blocksize < 10000) {
            // conver to BSR for MKL
            magma_z_mconvert( hA, &hC, Magma_CSR, Magma_BCSR);
            pntre = (int*)malloc( (num_rblocks+1)*sizeof(int) );
            pntre[0] = 0;
            for (i=0; i<num_rblocks; i++ ) pntre[i] = hC.row[i+1];
            start = magma_wtime(); 
            for (i=0; i<10; i++ )
            mkl_zbsrmv( "N", &num_rblocks, &num_cblocks, &hC.blocksize, 
                        MKL_ADDR(&one), "GFNC", MKL_ADDR(hC.val), hC.col, hC.row, pntre, 
                                                MKL_ADDR(hx.val), 
                        MKL_ADDR(&zero),        MKL_ADDR(hy.val) );
            end = magma_wtime();
            printf( " > MKL  : %.2e seconds (BSR, %dx%d,%d).\n",(end-start)/10,num_rblocks,num_cblocks,hC.blocksize );
            free(pntre);
            magma_z_mfree(&hC);
            //magma_zprint( 10,1, hy.val,10 );
        } else {
            printf( " > MKL  : ?? seconds (BSR, %dx%d,%d), too big.\n",num_rblocks,num_cblocks,hC.blocksize );
        }
        // init GPU vectors
        magma_z_vinit( &dx, Magma_DEV, hA.num_cols, one );
        magma_z_vinit( &dy, Magma_DEV, hA.num_cols, zero );
        //magma_z_vvisu( dx, 0,10);

        // copy matrix to GPU
        magma_z_mtransfer( hA, &dA, Magma_CPU, Magma_DEV);

        // SpMV on GPU (CSR)
        magma_device_sync(); start = magma_wtime(); 
        for (i=0; i<10; i++)
            magma_z_spmv( one, dA, dx, zero, dy);
        magma_device_sync(); end = magma_wtime(); 
        printf( " > MAGMA: %.2e seconds (CSR).\n",(end-start)/10 );
        //magma_z_vvisu( dy, 0,10);
        // SpMV on GPU (ELLPACKT)
        magma_device_sync(); start = magma_wtime(); 
        for (i=0; i<10; i++)
        magma_z_spmv( one, dB, dx, zero, dy);
        magma_device_sync(); end = magma_wtime(); 
        printf( " > MAGMA: %.2e seconds (ELLPACKT).\n",(end-start)/10 );
        //magma_z_vvisu( dy, 0,10);
        magma_zgetvector( hA.num_rows, dy.val,1, hx.val,1 );
        for( i=0; i<hA.num_rows; i++ ) hx.val[i] = MAGMA_Z_SUB( hx.val[i], hy.val[i] );
        printf( " >> error = %.2e\n\n",lapackf77_zlange( "F", &hA.num_rows, &ione, hx.val, &hA.num_rows, work ) );

        // free CPU memory
        magma_z_mfree(&hA);
        magma_z_vfree(&hx);
        magma_z_vfree(&hy);
        // free GPU memory
        magma_z_mfree(&dA);
        magma_z_mfree(&dB);
        magma_z_vfree(&dx);
        magma_z_vfree(&dy);

        if (id > -1) break;
    }

    TESTING_FINALIZE();
    return 0;
}
