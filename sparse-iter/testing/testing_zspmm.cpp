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
#include "magma_lapack.h"
#include "testings.h"
#include "common_magmasparse.h"


/* ////////////////////////////////////////////////////////////////////////////
   -- testing sparse matrix vector product
*/
int main(  int argc, char** argv )
{
    magma_int_t info = 0;
    TESTING_INIT();
    magma_queue_t queue=NULL;
    magma_queue_create( &queue );
    
    magma_z_matrix hA={Magma_CSR}, hA_SELLP={Magma_CSR}, hA_ELL={Magma_CSR}, 
    dA={Magma_CSR}, dA_SELLP={Magma_CSR}, dA_ELL={Magma_CSR};
    
    magma_z_matrix hx={Magma_CSR}, hy={Magma_CSR}, dx={Magma_CSR}, 
    dy={Magma_CSR}, hrefvec={Magma_CSR}, hcheck={Magma_CSR};
        
    hA_SELLP.blocksize = 8;
    hA_SELLP.alignment = 8;
    real_Double_t start, end, res;
    #ifdef MAGMA_WITH_MKL
        magma_int_t *pntre=NULL;
    #endif
    cusparseHandle_t cusparseHandle = NULL;
    cusparseMatDescr_t descr = NULL;

    magmaDoubleComplex c_one  = MAGMA_Z_MAKE(1.0, 0.0);
    magmaDoubleComplex c_zero = MAGMA_Z_MAKE(0.0, 0.0);
    
    magma_int_t i, j;
    for( i = 1; i < argc; ++i ) {
        if ( strcmp("--blocksize", argv[i]) == 0 ) {
            hA_SELLP.blocksize = atoi( argv[++i] );
        } else if ( strcmp("--alignment", argv[i]) == 0 ) {
            hA_SELLP.alignment = atoi( argv[++i] );
        } else
            break;
    }
    printf("\n#    usage: ./run_zspmm"
           " [ --blocksize %d --alignment %d (for SELLP) ]"
           " matrices \n\n", (int) hA_SELLP.blocksize, (int) hA_SELLP.alignment );

    while( i < argc ) {
        if ( strcmp("LAPLACE2D", argv[i]) == 0 && i+1 < argc ) {   // Laplace test
            i++;
            magma_int_t laplace_size = atoi( argv[i] );
            CHECK( magma_zm_5stencil(  laplace_size, &hA, queue ));
        } else {                        // file-matrix test
            CHECK( magma_z_csr_mtx( &hA,  argv[i], queue ));
        }

        printf("%% matrix info: %d-by-%d with %d nonzeros\n",
                            (int) hA.num_rows,(int) hA.num_cols,(int) hA.nnz );

        real_Double_t FLOPS = 2.0*hA.nnz/1e9;



        // m - number of rows for the sparse matrix
        // n - number of vectors to be multiplied in the SpMM product
        magma_int_t m, n;

        m = hA.num_rows;
        n = 48;

        // init CPU vectors
        CHECK( magma_zvinit( &hx, Magma_CPU, m, n, c_one, queue ));
        CHECK( magma_zvinit( &hy, Magma_CPU, m, n, c_zero, queue ));

        // init DEV vectors
        CHECK( magma_zvinit( &dx, Magma_DEV, m, n, c_one, queue ));
        CHECK( magma_zvinit( &dy, Magma_DEV, m, n, c_zero, queue ));


        // calling MKL with CSR
        #ifdef MAGMA_WITH_MKL
            CHECK( magma_imalloc_cpu( &pntre, m + 1 ) );
            pntre[0] = 0;
            for (j=0; j < m; j++ ) {
                pntre[j] = hA.row[j+1];
            }

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
                        MKL_ADDR(&c_one), "GFNC", MKL_ADDR(hA.val), col, row, pntre,
                                                MKL_ADDR(hx.val),
                        MKL_ADDR(&c_zero),        MKL_ADDR(hy.val) );
    
            start = magma_wtime();
            for (j=0; j<10; j++ )
                mkl_zcsrmv( "N", &num_rows, &num_cols,
                        MKL_ADDR(&c_one), "GFNC", MKL_ADDR(hA.val), col, row, pntre,
                                                MKL_ADDR(hx.val),
                        MKL_ADDR(&c_zero),        MKL_ADDR(hy.val) );
            end = magma_wtime();
            printf( "\n > MKL SpMVs : %.2e seconds %.2e GFLOP/s    (CSR).\n",
                                            (end-start)/10, FLOPS*10/(end-start) );
    
            // === Call MKL with blocked SpMVs, using mkl_zcsrmm ===
            char transa = 'n';
            MKL_INT ldb = n, ldc=n;
            char matdescra[6] = {'g', 'l', 'n', 'c', 'x', 'x'};
    
            // warm up
            mkl_zcsrmm( &transa, &num_rows, &num_vecs, &num_cols, MKL_ADDR(&c_one), matdescra,
                      MKL_ADDR(hA.val), col, row, pntre,
                      MKL_ADDR(hx.val), &ldb,
                      MKL_ADDR(&c_zero),
                      MKL_ADDR(hy.val), &ldc );
    
            start = magma_wtime();
            for (j=0; j<10; j++ )
                mkl_zcsrmm( &transa, &num_rows, &num_vecs, &num_cols, MKL_ADDR(&c_one), matdescra,
                          MKL_ADDR(hA.val), col, row, pntre,
                          MKL_ADDR(hx.val), &ldb,
                          MKL_ADDR(&c_zero),
                          MKL_ADDR(hy.val), &ldc );
            end = magma_wtime();
            printf( "\n > MKL SpMM  : %.2e seconds %.2e GFLOP/s    (CSR).\n",
                    (end-start)/10, FLOPS*10.*n/(end-start) );

            TESTING_FREE_CPU( row );
            TESTING_FREE_CPU( col );
            row = NULL;
            col = NULL;

        #endif // MAGMA_WITH_MKL

        // copy matrix to GPU
        CHECK( magma_zmtransfer( hA, &dA, Magma_CPU, Magma_DEV, queue ));
        // SpMV on GPU (CSR)
        start = magma_sync_wtime( queue );
        for (j=0; j<10; j++)
            CHECK( magma_z_spmv( c_one, dA, dx, c_zero, dy, queue ));
        end = magma_sync_wtime( queue );
        printf( " > MAGMA: %.2e seconds %.2e GFLOP/s    (standard CSR).\n",
                                        (end-start)/10, FLOPS*10.*n/(end-start) );

        CHECK( magma_zmtransfer( dy, &hrefvec , Magma_DEV, Magma_CPU, queue ));
        magma_zmfree(&dA, queue );


        // convert to SELLP and copy to GPU
        CHECK( magma_zmconvert(  hA, &hA_SELLP, Magma_CSR, Magma_SELLP, queue ));
        CHECK( magma_zmtransfer( hA_SELLP, &dA_SELLP, Magma_CPU, Magma_DEV, queue ));
        magma_zmfree(&hA_SELLP, queue );
        magma_zmfree( &dy, queue );
        CHECK( magma_zvinit( &dy, Magma_DEV, dx.num_rows, dx.num_cols, c_zero, queue ));
        // SpMV on GPU (SELLP)
        start = magma_sync_wtime( queue );
        for (j=0; j<10; j++)
            CHECK( magma_z_spmv( c_one, dA_SELLP, dx, c_zero, dy, queue ));
        end = magma_sync_wtime( queue );
        printf( " > MAGMA: %.2e seconds %.2e GFLOP/s    (SELLP).\n",
                                        (end-start)/10, FLOPS*10.*n/(end-start) );

        CHECK( magma_zmtransfer( dy, &hcheck , Magma_DEV, Magma_CPU, queue ));
        res = 0.0;
        for(magma_int_t k=0; k<hA.num_rows; k++ )
            res=res + MAGMA_Z_REAL(hcheck.val[k]) - MAGMA_Z_REAL(hrefvec.val[k]);
        printf("%% |x-y|_F = %8.2e\n", res);
        if ( res < .000001 )
            printf("%% tester spmm SELL-P:  ok\n");
        else
            printf("%% tester spmm SELL-P:  failed\n");
        magma_zmfree( &hcheck, queue );
        magma_zmfree(&dA_SELLP, queue );



        // SpMV on GPU (CUSPARSE - CSR)
        // CUSPARSE context //
        magma_zmfree( &dy, queue );
        CHECK( magma_zvinit( &dy, Magma_DEV, dx.num_rows, dx.num_cols, c_zero, queue ));
        //#ifdef PRECISION_d
        start = magma_sync_wtime( queue );
        CHECK_CUSPARSE( cusparseCreate( &cusparseHandle ));
        CHECK_CUSPARSE( cusparseSetStream( cusparseHandle, queue ));
        CHECK_CUSPARSE( cusparseCreateMatDescr( &descr ));
        CHECK_CUSPARSE( cusparseSetMatType( descr, CUSPARSE_MATRIX_TYPE_GENERAL ));
        CHECK_CUSPARSE( cusparseSetMatIndexBase( descr, CUSPARSE_INDEX_BASE_ZERO ));
        magmaDoubleComplex alpha = c_one;
        magmaDoubleComplex beta = c_zero;

        // copy matrix to GPU
        CHECK( magma_zmtransfer( hA, &dA, Magma_CPU, Magma_DEV, queue) );

        for (j=0; j<10; j++)
        cusparseZcsrmm(cusparseHandle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
                    dA.num_rows,   n, dA.num_cols, dA.nnz,
                    &alpha, descr, dA.dval, dA.drow, dA.dcol,
                    dx.dval, dA.num_cols, &beta, dy.dval, dA.num_cols);
        end = magma_sync_wtime( queue );
        printf( " > CUSPARSE: %.2e seconds %.2e GFLOP/s    (CSR).\n",
                                        (end-start)/10, FLOPS*10*n/(end-start) );

        CHECK( magma_zmtransfer( dy, &hcheck , Magma_DEV, Magma_CPU, queue ));
        res = 0.0;
        for(magma_int_t k=0; k<hA.num_rows; k++ )
            res=res + MAGMA_Z_REAL(hcheck.val[k]) - MAGMA_Z_REAL(hrefvec.val[k]);
        printf("%% |x-y|_F = %8.2e\n", res);
        if ( res < .000001 )
            printf("%% tester spmm cuSPARSE:  ok\n");
        else
            printf("%% tester spmm cuSPARSE:  failed\n");
        magma_zmfree( &hcheck, queue );

        cusparseDestroyMatDescr( descr ); 
        cusparseDestroy( cusparseHandle );
        descr = NULL;
        cusparseHandle = NULL;
        //#endif

        printf("\n\n");


        // free CPU memory
        magma_zmfree(&hA, queue );
        magma_zmfree(&hx, queue );
        magma_zmfree(&hy, queue );
        magma_zmfree(&hrefvec, queue );
        // free GPU memory
        magma_zmfree(&dx, queue );
        magma_zmfree(&dy, queue );
        magma_zmfree(&dA, queue);

        i++;
    }

cleanup:
    #ifdef MAGMA_WITH_MKL
        magma_free_cpu(pntre);
    #endif
    cusparseDestroyMatDescr( descr ); 
    cusparseDestroy( cusparseHandle );
    magma_zmfree(&hA, queue );
    magma_zmfree(&dA, queue );
    magma_zmfree(&hA_ELL, queue );
    magma_zmfree(&dA_ELL, queue );
    magma_zmfree(&hA_SELLP, queue );
    magma_zmfree(&dA_SELLP, queue );
    
    magma_queue_destroy( queue );
    TESTING_FINALIZE();
    return info;
}
