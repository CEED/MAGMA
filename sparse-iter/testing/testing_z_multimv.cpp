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
#include <cusparse_v2.h>
#include <cuda_profiler_api.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "../include/magmasparse.h"
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
 /*    "/home/hanzt/sparse_matrices/mtx/Trefethen_20.mtx",
     "/home/hanzt/sparse_matrices/mtx/Trefethen_20000.mtx",*/
     "/home/hanzt/sparse_matrices/mtx/audikw_1.mtx",
     "/home/hanzt/sparse_matrices/mtx/bone010.mtx",
     "/home/hanzt/sparse_matrices/mtx/boneS10.mtx",
     "/home/hanzt/sparse_matrices/mtx/bmw3_2.mtx",
     "/home/hanzt/sparse_matrices/mtx/bmwcra_1.mtx",
     "/home/hanzt/sparse_matrices/mtx/cage10.mtx",
     "/home/hanzt/sparse_matrices/mtx/cant.mtx",
     "/home/hanzt/sparse_matrices/mtx/crankseg_2.mtx",
//     "/home/hanzt/sparse_matrices/mtx/jj_Cube_Coup_dt0.mtx",
     "/home/hanzt/sparse_matrices/mtx/dielFilterV2real.mtx",    //10
     "/home/hanzt/sparse_matrices/mtx/F1.mtx",   
     "/home/hanzt/sparse_matrices/mtx/Fault_639.mtx",
     "/home/hanzt/sparse_matrices/mtx/Hook_1498.mtx",
     "/home/hanzt/sparse_matrices/mtx/inline_1.mtx",
     "/home/hanzt/sparse_matrices/mtx/ldoor.mtx",
     "/home/hanzt/sparse_matrices/mtx/m_t1.mtx",
//     "/home/hanzt/sparse_matrices/mtx/jj_ML_Geer.mtx",
     "/home/hanzt/sparse_matrices/mtx/pwtk.mtx",
     "/home/hanzt/sparse_matrices/mtx/shipsec1.mtx",
     "/home/hanzt/sparse_matrices/mtx/Trefethen_20000.mtx",

   //  "/home/hanzt/sparse_matrices/mtx/Serena_b.mtx",

    // "/home/hanzt/sparse_matrices/mtx/circuit5M.mtx",

   //  "/home/hanzt/sparse_matrices/mtx/FullChip.mtx",

//     "/home/hanzt/sparse_matrices/mtx/af_shell10.mtx",
//     "/home/hanzt/sparse_matrices/mtx/kim2.mtx",

     "/home/hanzt/sparse_matrices/mtx/StocF-1465.mtx",
     "/home/hanzt/sparse_matrices/mtx/kkt_power.mtx",
     "/home/hanzt/sparse_matrices/mtx/bmwcra_1.mtx",
     "/home/hanzt/sparse_matrices/mtx/thermal2.mtx",

     "/home/hanzt/sparse_matrices/mtx/pre2.mtx",

     "/home/hanzt/sparse_matrices/mtx/xenon2.mtx",
     "/home/hanzt/sparse_matrices/mtx/stomach.mtx",



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

    magma_int_t id = -1, matrix = 0, i, ione = 1, *pntre, 
                                    num_rblocks, num_cblocks;
    double start, end, work[1];
    for( int i = 1; i < argc; ++i ) {
        if ( strcmp("--id", argv[i]) == 0 ) {
            id = atoi( argv[++i] );
        }
    }
    if (id > -1) printf( "\n    Usage: ./testing_z_mv --id %d\n\n",id );



    for(matrix=12; matrix<40; matrix++)
    {
        magma_z_sparse_matrix hA, hB, hC, dA, dB, hD, dD, hE, dE;
        magma_z_vector hx, hy, dx, dy, dx2, dy2;


        printf( "# ");

        magmaDoubleComplex one  = MAGMA_Z_MAKE(1.0, 0.0);
        magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
  
        // init matrix on CPU
        if (id > -1) matrix = id;
        magma_z_csr_mtx( &hA, filename[matrix] );
        printf( "#vectors & MAGMA-SELLCM  & CUSPARSE\n");
        printf( "#  sec - GFLOPS & sec - GFLOPS\n");

        // convert to SELLC-mod and copy to GPU
        hD.blocksize = 8;
        hD.alignment = 4;
        magma_z_mconvert(  hA, &hD, Magma_CSR, Magma_SELLC);
        magma_z_mtransfer( hD, &dD, Magma_CPU, Magma_DEV);

        // copy matrix to GPU
        magma_z_mtransfer( hA, &dA, Magma_CPU, Magma_DEV);


        for(int k=2; k<131; k+=2){
        //printf("number of vectors:%d\n",k);
        printf( "%d  &",k );
        magma_int_t num_vecs=k; 

        double FLOPS = (2.*hA.nnz*num_vecs)/1e9;
        //printf("FLOPS: %d  %d %ld  ", hA.nnz,num_vecs, FLOPS);

        //printf( "\nmatrix read: %d-by-%d with %d nonzeros\n\n",hA.num_rows,hA.num_cols,hA.nnz );

        // init CPU vectors
        hC.blocksize = min(hA.num_rows, 8);
        num_rblocks = (hA.num_rows+hC.blocksize-1)/hC.blocksize;
        num_cblocks = (hA.num_cols+hC.blocksize-1)/hC.blocksize;
        magma_z_vinit( &hx, Magma_CPU, num_cblocks*hC.blocksize, one );
        magma_z_vinit( &hy, Magma_CPU, num_rblocks*hC.blocksize, zero );

        // init GPU vectors
        magma_z_vinit( &dx, Magma_DEV, hA.num_cols*num_vecs, one );
        magma_z_vinit( &dy, Magma_DEV, hA.num_cols*num_vecs, zero );
        magma_z_vinit( &dx2, Magma_DEV, hA.num_cols*num_vecs, one );
        magma_z_vinit( &dy2, Magma_DEV, hA.num_cols*num_vecs, zero );




       // SpMV on GPU (SELLC-mod)

        magma_device_sync(); start = magma_wtime(); 
        //cudaProfilerStart();
        for (i=0; i<10; i++)
            magma_z_spmv( one, dD, dx2, zero, dy2);
        //cudaProfilerStop();
        magma_device_sync(); end = magma_wtime(); 
        //printf( " > MAGMA: %.2e seconds (SELLC-mod).\n",(end-start)/100 );
        printf( "  %.2e  &",(end-start)/10 );
        printf( "  %.2e  &",FLOPS/((end-start)/10.0) );//GFLOPS

        //printf( "  %.2e  &",(end-start)/10 );
       // magma_z_vvisu( dy2, 0,20);
       // magma_z_vvisu( dy2, (num_vecs)*hA.num_rows-20, 20);


        //magma_zprint_gpu( hA.num_rows, num_vecs, dy2.val, hA.num_rows );




      // SpMV on GPU (CUSPARSE - CSR)
        // CUSPARSE context //
        #ifdef PRECISION_d
        cusparseHandle_t cusparseHandle = 0;
        cusparseStatus_t cusparseStatus;
        cusparseStatus = cusparseCreate(&cusparseHandle);


        cusparseMatDescr_t descr = 0;
        cusparseStatus = cusparseCreateMatDescr(&descr);


        cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);
        double alpha = 1.0;
        double beta = 0.0;



        magma_device_sync(); start = magma_wtime(); 
        cudaProfilerStart();
        for (i=0; i<10; i++)
        cusparseZcsrmm2(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE, 
                            CUSPARSE_OPERATION_NON_TRANSPOSE, 
                            hA.num_rows, num_vecs, hA.num_cols, hA.nnz, 
                            &alpha, descr, dA.val, dA.row, dA.col, dx2.val,
                            hA.num_cols, &beta, dy2.val, hA.num_cols);
        cudaProfilerStop();
        magma_device_sync(); end = magma_wtime(); 
        //printf( " > CUSPARSE: %.2e seconds (CSR).\n",(end-start)/100 );
        printf( "  %.2e  &",(end-start)/10 );
        printf( "  %.2e  &",FLOPS/((end-start)/10.0) );//GFLOPS
        //magma_z_vvisu( dy2, 0, 20);
        //magma_z_vvisu( dy2, (num_vecs)*hA.num_rows-20, 20);

        cusparseDestroy( cusparseHandle );



        #endif

        magma_z_vfree(&hx);
        magma_z_vfree(&hy);
        // free GPU memory
        magma_z_vfree(&dx);
        magma_z_vfree(&dy);
        magma_z_vfree(&dx2);
        magma_z_vfree(&dy2);
        printf("\n");
        }
        printf("\n");
        printf("\n");
/*
        magma_zgetvector( hA.num_rows, dy2.val,1, hx.val,1 );
        magma_zgetvector( hA.num_rows, dy.val,1, hy.val,1 );
        for(int i=0; i<hA.num_rows; i++)
            if(abs(MAGMA_Z_REAL(hx.val[i])-MAGMA_Z_REAL(hy.val[i]))>0.001 ){
                printf("error in index %d : %f vs %f difference. %e\n", i, MAGMA_Z_REAL(hx.val[i]), MAGMA_Z_REAL(hy.val[i]), MAGMA_Z_REAL(hx.val[i])-MAGMA_Z_REAL(hy.val[i]));
            }
        //for( i=0; i<hA.num_rows; i++ ) hx.val[i] = MAGMA_Z_SUB( hx.val[i], hy.val[i] );
        //printf( " >> error = %.2e\n\n",lapackf77_zlange( "F", &hA.num_rows, &ione, hx.val, &hA.num_rows, work ) );
*/
        // free CPU memory
        magma_z_mfree(&dD);
        magma_z_mfree(&dA);
        magma_z_mfree(&hA);



        if (id > -1) break;
    }

    TESTING_FINALIZE();
    return 0;
}
