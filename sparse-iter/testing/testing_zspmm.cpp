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

#ifdef MAGMA_WITH_MKL
    #include "mkl_spblas.h"
    
    #define PRECISION_z
    #if defined(PRECISION_z)
    #define MKL_ADDR(a) (MKL_Complex16*)(a)
    #elif defined(PRECISION_c)
    #define MKL_ADDR(a) (MKL_Complex8*)(a)
    #else
    #define MKL_ADDR(a) (a)
    #endif
#endif

// includes, project
#include "flops.h"
#include "magma.h"
#include "magmasparse.h"
#include "magma_lapack.h"
#include "testings.h"


/* ////////////////////////////////////////////////////////////////////////////
   -- testing sparse matrix vector product
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    magma_z_sparse_matrix hA, hA_SELLP, hA_ELL, dA, dA_SELLP, dA_ELL;
    hA_SELLP.blocksize = 8;
    hA_SELLP.alignment = 8;
    double start, end;
    magma_int_t *pntre;

    magmaDoubleComplex one  = MAGMA_Z_MAKE(1.0, 0.0);
    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
    
    magma_int_t i, j;
    for( i = 1; i < argc; ++i ) {
        if ( strcmp("--blocksize", argv[i]) == 0 ) {
            hA_SELLP.blocksize = atoi( argv[++i] );
        }else if ( strcmp("--alignment", argv[i]) == 0 ) {
            hA_SELLP.alignment = atoi( argv[++i] );
        }else
            break;
    }
    printf( "\n#    usage: ./run_zspmv"
        " [ --blocksize %d --alignment %d (for SELLP) ]"
        " matrices \n\n", (int) hA_SELLP.blocksize, (int) hA_SELLP.alignment );

    while(  i < argc ){

        if( strcmp("LAPLACE2D", argv[i]) == 0 && i+1 < argc ){   // Laplace test
            i++;
            magma_int_t laplace_size = atoi( argv[i] );
            magma_zm_5stencil(  laplace_size, &hA );
        } else {                        // file-matrix test
            magma_z_csr_mtx( &hA,  argv[i]  ); 
        }

        printf( "# matrix info: %d-by-%d with %d nonzeros\n",
                            (int) hA.num_rows,(int) hA.num_cols,(int) hA.nnz );

        real_Double_t FLOPS = 2.0*hA.nnz/1e9;

        magma_z_vector hx, hy, dx, dy;

        // m - number of rows for the sparse matrix
        // n - number of vectors to be multiplied in the SpMM product
        magma_int_t m, n;

        m = hA.num_rows;
        n = 48;

        // init CPU vectors
        magma_z_vinit( &hx, Magma_CPU, m*n, one );
        magma_z_vinit( &hy, Magma_CPU, m*n, zero );

        // init DEV vectors
        magma_z_vinit( &dx, Magma_DEV, m*n, one );
        magma_z_vinit( &dy, Magma_DEV, m*n, zero );


        // calling MKL with CSR
        pntre = (magma_int_t*)malloc( (m+1)*sizeof(magma_int_t) );
        pntre[0] = 0;
        for (j=0; j<m; j++ ) pntre[j] = hA.row[j+1];

        #ifdef MAGMA_WITH_MKL
            MKL_INT num_rows = hA.num_rows;
            MKL_INT num_cols = hA.num_cols;
            MKL_INT nnz = hA.nnz;
            MKL_INT num_vecs = n;

            MKL_INT *col;
            TESTING_MALLOC_CPU( col, MKL_INT, nnz );
            for( magma_int_t t=0; t < hA.nnz; ++t ) {
                col[ t ] = hA.col[ t ];
            }
            MKL_INT *row;
            TESTING_MALLOC_CPU( row, MKL_INT, num_rows );
            for( magma_int_t t=0; t < hA.num_rows; ++t ) {
                row[ t ] = hA.col[ t ];
            }

            // === Call MKL with consecutive SpMVs, using mkl_zcsrmv ===
            // warmp up
            mkl_zcsrmv( "N", &num_rows, &num_cols, 
                        MKL_ADDR(&one), "GFNC", MKL_ADDR(hA.val), col, row, pntre, 
                                                MKL_ADDR(hx.val), 
                        MKL_ADDR(&zero),        MKL_ADDR(hy.val) );
    
            start = magma_wtime(); 
            for (j=0; j<10; j++ )
                mkl_zcsrmv( "N", &num_rows, &num_cols, 
                        MKL_ADDR(&one), "GFNC", MKL_ADDR(hA.val), col, row, pntre, 
                                                MKL_ADDR(hx.val), 
                        MKL_ADDR(&zero),        MKL_ADDR(hy.val) );
            end = magma_wtime();
            printf( "\n > MKL SpMVs : %.2e seconds %.2e GFLOP/s    (CSR).\n",
                                            (end-start)/10, FLOPS*10/(end-start) );
    
            // === Call MKL with blocked SpMVs, using mkl_zcsrmm ===
            char transa = 'n';
            MKL_INT ldb = n, ldc=n;
            char matdescra[6] = {'g', 'l', 'n', 'c', 'x', 'x'};
    
            // warm up
            mkl_zcsrmm( &transa, &num_rows, &num_vecs, &num_cols, MKL_ADDR(&one), matdescra, 
                      MKL_ADDR(hA.val), col, row, pntre,
                      MKL_ADDR(hx.val), &ldb,
                      MKL_ADDR(&zero),        
                      MKL_ADDR(hy.val), &ldc );
    
            start = magma_wtime();
            for (j=0; j<10; j++ )
                mkl_zcsrmm( &transa, &num_rows, &num_vecs, &num_cols, MKL_ADDR(&one), matdescra, 
                          MKL_ADDR(hA.val), col, row, pntre,
                          MKL_ADDR(hx.val), &ldb,
                          MKL_ADDR(&zero),        
                          MKL_ADDR(hy.val), &ldc );
            end = magma_wtime();
            printf( "\n > MKL SpMM  : %.2e seconds %.2e GFLOP/s    (CSR).\n",
                    (end-start)/10, FLOPS*10.*n/(end-start) );

            TESTING_FREE_CPU( row );
            TESTING_FREE_CPU( col );
            free(pntre);
        #endif // MAGMA_WITH_MKL

        // copy matrix to GPU
        magma_z_mtransfer( hA, &dA, Magma_CPU, Magma_DEV);
        // SpMV on GPU (CSR)
        magma_device_sync(); start = magma_wtime(); 
        for (j=0; j<10; j++)
            magma_z_spmv( one, dA, dx, zero, dy);
        magma_device_sync(); end = magma_wtime(); 
        printf( " > MAGMA: %.2e seconds %.2e GFLOP/s    (standard CSR).\n",
                                        (end-start)/10, FLOPS*10.*n/(end-start) );
        magma_z_mfree(&dA);

        // convert to ELL and copy to GPU
        magma_z_mconvert(  hA, &hA_ELL, Magma_CSR, Magma_ELL);
        magma_z_mtransfer( hA_ELL, &dA_ELL, Magma_CPU, Magma_DEV);
        magma_z_mfree(&hA_ELL);
        // SpMV on GPU (ELL)
        magma_device_sync(); start = magma_wtime(); 
        for (j=0; j<10; j++)
            magma_z_spmv( one, dA_ELL, dx, zero, dy);
        magma_device_sync(); end = magma_wtime(); 
        printf( " > MAGMA: %.2e seconds %.2e GFLOP/s    (standard ELL).\n",
                                        (end-start)/10, FLOPS*10.*n/(end-start) );
        magma_z_mfree(&dA_ELL);


        // convert to SELLP and copy to GPU
        magma_z_mconvert(  hA, &hA_SELLP, Magma_CSR, Magma_SELLP);
        magma_z_mtransfer( hA_SELLP, &dA_SELLP, Magma_CPU, Magma_DEV);
        magma_z_mfree(&hA_SELLP);
        // SpMV on GPU (SELLP)
        magma_device_sync(); start = magma_wtime(); 
        for (j=0; j<10; j++)
            magma_z_spmv( one, dA_SELLP, dx, zero, dy);
        magma_device_sync(); end = magma_wtime(); 
        printf( " > MAGMA: %.2e seconds %.2e GFLOP/s    (SELLP).\n",
                                        (end-start)/10, FLOPS*10.*n/(end-start) );
        magma_z_mfree(&dA_SELLP);



        // SpMV on GPU (CUSPARSE - CSR)
        // CUSPARSE context //
        #ifdef PRECISION_d
        magma_device_sync(); start = magma_wtime(); 
        cusparseHandle_t cusparseHandle = 0;
        cusparseStatus_t cusparseStatus;
        cusparseStatus = cusparseCreate(&cusparseHandle);

        cusparseMatDescr_t descr = 0;
        cusparseStatus = cusparseCreateMatDescr(&descr);

        cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);
        double alpha = 1.0;
        double beta = 0.0;

        // copy matrix to GPU
        magma_d_mtransfer( hA, &dA, Magma_CPU, Magma_DEV);

        for (j=0; j<10; j++)                    
        cusparseZcsrmm(cusparseHandle,
            CUSPARSE_OPERATION_NON_TRANSPOSE, 
                    dA.num_rows,   n, dA.num_cols, dA.nnz, 
                    &alpha, descr, dA.val, dA.row, dA.col,
                    dx.val, dA.num_cols, &beta, dy.val, dA.num_cols);
        magma_device_sync(); end = magma_wtime(); 
        printf( " > CUSPARSE: %.2e seconds %.2e GFLOP/s    (CSR).\n",
                                        (end-start)/10, FLOPS*10*n/(end-start) );

        cusparseDestroyMatDescr( descr );
        cusparseDestroy( cusparseHandle );

        magma_d_mfree(&dA);

        #endif

        printf("\n\n");


        // free CPU memory
        magma_z_mfree(&hA);
        magma_z_vfree(&hx);
        magma_z_vfree(&hy);
        // free GPU memory
        magma_z_vfree(&dx);
        magma_z_vfree(&dy);

        i++;

    }

    TESTING_FINALIZE();
    return 0;
}
