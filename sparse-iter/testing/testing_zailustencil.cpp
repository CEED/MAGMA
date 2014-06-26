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

    const char *filename[] =
    {
            "/home/hanzt/sparse_matrices/mtx/Trefethen_20.mtx", //        n:19 nnz:147 nnz/n:7 max_nnz_row:9                            4
            "/home/hanzt/sparse_matrices/mtx/Trefethen_20.mtx", //        n:19 nnz:147 nnz/n:7 max_nnz_row:9                            4
            "/home/hanzt/sparse_matrices/mtx/Trefethen_200.mtx", //        n:19 nnz:147 nnz/n:7 max_nnz_row:9                            4
            "/home/hanzt/sparse_matrices/mtx/Trefethen_2000.mtx", //        n:19 nnz:147 nnz/n:7 max_nnz_row:9                            4
            "/home/hanzt/sparse_matrices/mtx/Trefethen_20000.mtx", //        n:19 nnz:147 nnz/n:7 max_nnz_row:9                            4


            "/home/hanzt/sparse_matrices/mtx/parabolic_fem.mtx", //       n:525825 nnz:3674625 nnz/n:6 max_nnz_row:7
            "/home/hanzt/sparse_matrices/mtx/ecology2.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
            "/home/hanzt/sparse_matrices/mtx/apache2.mtx", //             n:715176 nnz:4817870 nnz/n:6 max_nnz_row:8 
            "/home/hanzt/sparse_matrices/mtx/G3_circuit.mtx", //          n:1585478 nnz:7660826 nnz/n:4 max_nnz_row:6 
            "/home/hanzt/sparse_matrices/mtx/thermal2.mtx", //            n:1228045 nnz:8580313 nnz/n:6 max_nnz_row:11 
            "/home/hanzt/sparse_matrices/mtx/af_shell3.mtx", //           n:504855 nnz:17562051 nnz/n:34 max_nnz_row:40   
            //"/home/hanzt/sparse_matrices/mtx/offshore.mtx", //            n:259789 nnz:4242673 nnz/n:16 max_nnz_row:31 
  


            "/home/hanzt/sparse_matrices/mtx/Trefethen_20.mtx", //        n:19 nnz:147 nnz/n:7 max_nnz_row:9                            4
            "/home/hanzt/sparse_matrices/mtx/Trefethen_200.mtx", //        n:19 nnz:147 nnz/n:7 max_nnz_row:9                            4
            "/home/hanzt/sparse_matrices/mtx/Trefethen_2000.mtx", //        n:19 nnz:147 nnz/n:7 max_nnz_row:9                            4
            "/home/hanzt/sparse_matrices/mtx/Trefethen_20000.mtx", //        n:19 nnz:147 nnz/n:7 max_nnz_row:9                            4

            "/home/hanzt/sparse_matrices/mtx/thermal2.mtx", //            n:1228045 nnz:8580313 nnz/n:6 max_nnz_row:11 
            "/home/hanzt/sparse_matrices/mtx/ecology2.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
            "/home/hanzt/sparse_matrices/mtx/G3_circuit.mtx", //          n:1585478 nnz:7660826 nnz/n:4 max_nnz_row:6 
            "/home/hanzt/sparse_matrices/mtx/parabolic_fem.mtx", //       n:525825 nnz:3674625 nnz/n:6 max_nnz_row:7
            "/home/hanzt/sparse_matrices/mtx/apache2.mtx", //             n:715176 nnz:4817870 nnz/n:6 max_nnz_row:8 
            "/home/hanzt/sparse_matrices/mtx/offshore.mtx", //            n:259789 nnz:4242673 nnz/n:16 max_nnz_row:31 
            "/home/hanzt/sparse_matrices/mtx/af_shell3.mtx", //           n:504855 nnz:17562051 nnz/n:34 max_nnz_row:40     



    };
    
    real_Double_t start, end;
//magma_int_t n =8;
 //    for(int matrix=1; matrix<101; matrix=matrix+100){
 //    for(int matrix=1; matrix<11; matrix++){
   for(int matrix=0; matrix<10; matrix++){
  //  for(int matrix=4; matrix<10; matrix++){
    int num_vecs = 10;


    magma_z_sparse_matrix hA, hAL, hALCOO, hAU,  hAUT, hAUCOO, hAcusparse, hA3U, hAtmp, dA, hAD, hADD, dAD, dADD, hLU, dAL, dAU, dL, dU, hL, hU, hAt, hUT, hUTCOO;


// matrix from UFMC
    
   // printf("#  ");

/*
    // generate matrix of desired structure and size dense
    n=16*matrix;   // size is n*n
    magma_int_t nn = n*n;
    magma_int_t offdiags = n-1;
    magma_index_t *diag_offset;
    magmaDoubleComplex *diag_vals;
    magma_zmalloc_cpu( &diag_vals, offdiags+1 );
    magma_index_malloc_cpu( &diag_offset, offdiags+1 );

    for(int i=0; i<n; i++)
        diag_offset[i] = i;

    for(int i=0; i<n; i++)
        diag_vals[i] = MAGMA_Z_MAKE( -1.0, 0.0 );
    diag_vals[0] = MAGMA_Z_MAKE( 22*(double) n, 0.0 );
    magma_zmgenerator( nn, offdiags, diag_offset, diag_vals, &hA );
    /*

    magma_z_sparse_matrix hAB;
    hAB.blocksize = 16;
    magma_z_mconvert(hA, &hAB, Magma_CSR, Magma_BCSR);
    magma_z_mfree(&hA);
    magma_z_mconvert( hAB, &hA, Magma_BCSR, Magma_CSR);




    // generate matrix of desired structure and size 2d 5-point stencil
    int n=64;   // size is n*n
    magma_int_t nn = n*n;
    magma_int_t offdiags = 2;
    magma_index_t *diag_offset;
    magmaDoubleComplex *diag_vals;
    magma_zmalloc_cpu( &diag_vals, offdiags+1 );
    magma_index_malloc_cpu( &diag_offset, offdiags+1 );
    diag_offset[0] = 0;
    diag_offset[1] = 1;
    diag_offset[2] = n;

    diag_vals[0] = MAGMA_Z_MAKE( 4.0, 0.0 );
    diag_vals[1] = MAGMA_Z_MAKE( -1.0, 0.0 );
    diag_vals[2] = MAGMA_Z_MAKE( -1.0, 0.0 );
    magma_zmgenerator( nn, offdiags, diag_offset, diag_vals, &hA );
*/

/*
    // generate matrix of desired structure and size 3d 5-point stencil
    int n=4;   // size is n*n*n
    magma_int_t nn = n*n*n;
    magma_int_t offdiags = 3;
    magma_index_t *diag_offset;
    magmaDoubleComplex *diag_vals;
    magma_zmalloc_cpu( &diag_vals, offdiags+1 );
    magma_index_malloc_cpu( &diag_offset, offdiags+1 );
    diag_offset[0] = 0;
    diag_offset[1] = 1;
    diag_offset[2] = n;
    diag_offset[3] = n*n;

    diag_vals[0] = MAGMA_Z_MAKE( 6.0, 0.0 );
    diag_vals[1] = MAGMA_Z_MAKE( -1.0, 0.0 );
    diag_vals[2] = MAGMA_Z_MAKE( -1.0, 0.0 );
    diag_vals[3] = MAGMA_Z_MAKE( -1.0, 0.0 );
    magma_zmgenerator( nn, offdiags, diag_offset, diag_vals, &hA );


*/
//printf("generate matrix:");

    // generate matrix of desired structure and size 3d 27-point stencil
    int n=64;   // size is n*n*n
    magma_int_t nn = n*n*n;
    magma_int_t offdiags = 13;
    magma_index_t *diag_offset;
    magmaDoubleComplex *diag_vals;
    magma_zmalloc_cpu( &diag_vals, offdiags+1 );
    magma_index_malloc_cpu( &diag_offset, offdiags+1 );
    diag_offset[0] = 0;
    diag_offset[1] = 1;
    diag_offset[2] = n-1;
    diag_offset[3] = n;
    diag_offset[4] = n+1;
    diag_offset[5] = n*n-n-1;
    diag_offset[6] = n*n-n;
    diag_offset[7] = n*n-n+1;
    diag_offset[8] = n*n-1;
    diag_offset[9] = n*n;
    diag_offset[10] = n*n+1;
    diag_offset[11] = n*n+n-1;
    diag_offset[12] = n*n+n;
    diag_offset[13] = n*n+n+1;

    diag_vals[0] = MAGMA_Z_MAKE( 26.0, 0.0 );
    diag_vals[1] = MAGMA_Z_MAKE( -1.0, 0.0 );
    diag_vals[2] = MAGMA_Z_MAKE( -1.0, 0.0 );
    diag_vals[3] = MAGMA_Z_MAKE( -1.0, 0.0 );
    diag_vals[4] = MAGMA_Z_MAKE( -1.0, 0.0 );
    diag_vals[5] = MAGMA_Z_MAKE( -1.0, 0.0 );
    diag_vals[6] = MAGMA_Z_MAKE( -1.0, 0.0 );
    diag_vals[7] = MAGMA_Z_MAKE( -1.0, 0.0 );
    diag_vals[8] = MAGMA_Z_MAKE( -1.0, 0.0 );
    diag_vals[9] = MAGMA_Z_MAKE( -1.0, 0.0 );
    diag_vals[10] = MAGMA_Z_MAKE( -1.0, 0.0 );
    diag_vals[11] = MAGMA_Z_MAKE( -1.0, 0.0 );
    diag_vals[12] = MAGMA_Z_MAKE( -1.0, 0.0 );
    diag_vals[13] = MAGMA_Z_MAKE( -1.0, 0.0 );
    magma_zmgenerator( nn, offdiags, diag_offset, diag_vals, &hA );

//printf("done.\n");

/*
    magma_z_sparse_matrix hAB;
    hAB.blocksize = 2;
    magma_z_mconvert(hA, &hAB, Magma_CSR, Magma_BCSR);
    magma_z_mfree(&hA);
    magma_z_mconvert( hAB, &hA, Magma_BCSR, Magma_CSR);

    */
  // printf("# n:%d nnz:%d\n", hA.num_rows, hA.nnz);

/*

    // generate matrix of desired structure and size 2d 9-point stencil
    n=85;   // size is n*n
    magma_int_t nn = n*n;
    magma_int_t offdiags = 5;
    magma_index_t *diag_offset;
    magmaDoubleComplex *diag_vals;
    magma_zmalloc_cpu( &diag_vals, offdiags+1 );
    magma_index_malloc_cpu( &diag_offset, offdiags+1 );
    diag_offset[0] = 0;
    diag_offset[1] = 1;
    diag_offset[2] = 2;
    diag_offset[3] = n-1;
    diag_offset[4] = n;
    diag_offset[5] = n+1;
   // diag_offset[6] = 2*n-1;
    diag_vals[0] = MAGMA_Z_MAKE( 20.0, 0.0 );
    diag_vals[1] = MAGMA_Z_MAKE( -4.0, 0.0 );
    diag_vals[2] = MAGMA_Z_MAKE( -1.0, 0.0 );
    diag_vals[3] = MAGMA_Z_MAKE( -1.0, 0.0 );
    diag_vals[4] = MAGMA_Z_MAKE( -4.0, 0.0 );
    diag_vals[5] = MAGMA_Z_MAKE( -1.0, 0.0 );
 //   diag_vals[6] = MAGMA_Z_MAKE( 0.0, 0.0 );
    magma_zmgenerator( nn, offdiags, diag_offset, diag_vals, &hA );
    
    printf("# n:%d nnz:%d\n", hA.num_rows, hA.nnz);
*/
/*
    // generate matrix of desired structure and size 2d 27-point stencil
    n=16*matrix;   // size is n*n
    magma_int_t nn = n*n;
    magma_int_t offdiags = 11;
    magma_index_t *diag_offset;
    magmaDoubleComplex *diag_vals;
    magma_zmalloc_cpu( &diag_vals, offdiags+1 );
    magma_index_malloc_cpu( &diag_offset, offdiags+1 );
    diag_offset[0] = 0;
    diag_offset[1] = 1;
    diag_offset[2] = 2;
    diag_offset[3] = 3;
    diag_offset[4] = n-3;
    diag_offset[5] = n-2;
    diag_offset[6] = n-1;
    diag_offset[7] = n;
    diag_offset[8] = n+1;
    diag_offset[9] = n+2;
    diag_offset[10] = n+3;


   // diag_offset[6] = 2*n-1;
    diag_vals[0] = MAGMA_Z_MAKE( 80.0, 0.0 );
    diag_vals[1] = MAGMA_Z_MAKE( -6.0, 0.0 );
    diag_vals[2] = MAGMA_Z_MAKE( -4.0, 0.0 );
    diag_vals[3] = MAGMA_Z_MAKE( -2.0, 0.0 );
    diag_vals[4] = MAGMA_Z_MAKE( -1.0, 0.0 );
    diag_vals[5] = MAGMA_Z_MAKE( -2.0, 0.0 );
    diag_vals[6] = MAGMA_Z_MAKE( -4.0, 0.0 );
    diag_vals[7] = MAGMA_Z_MAKE( -6.0, 0.0 );
    diag_vals[8] = MAGMA_Z_MAKE( -4.0, 0.0 );
    diag_vals[9] = MAGMA_Z_MAKE( -2.0, 0.0 );
    diag_vals[10] = MAGMA_Z_MAKE( -1.0, 0.0 );
 //   diag_vals[6] = MAGMA_Z_MAKE( 0.0, 0.0 );
    magma_zmgenerator( nn, offdiags, diag_offset, diag_vals, &hA );

    magma_z_sparse_matrix hAB;
    hAB.blocksize = 4;
    magma_z_mconvert(hA, &hAB, Magma_CSR, Magma_BCSR);
    magma_z_mfree(&hA);
    magma_z_mconvert( hAB, &hA, Magma_BCSR, Magma_CSR);


    printf("# n:%d nnz:%d\n", hA.num_rows, hA.nnz);

*/
    //magma_z_csr_mtx( &hA, filename[matrix] ); int n;

    magma_zmscale( &hA, Magma_UNITDIAG );
//magma_z_mvisu(hA);

//printf("# n:%d nnz:%d\n", hA.num_rows, hA.nnz);

    //    printf("# runtime 1000 runs - blocksize 256 shift 1 \n");

   //     printf("#########################################################################\n");   

        //printf("# iters  sec(MAGMA)  sec(CUSPARSE)  avg res         min res       max res\n");   
     //   printf("# blocksize  ||A-LU||_F  runtime \n");   

for(int ilu=16; ilu>15; ilu=ilu/2 ){



    //magma_zilustruct( &hA, ilu);

    magma_z_mconvert( hA, &hAtmp, Magma_CSR, Magma_CSR );

    magma_z_sparse_matrix hACSRCOO, hAinitguess, dAinitguess;
    magma_z_mconvert( hA, &hACSRCOO, Magma_CSR, Magma_CSRCOO );
    magma_zmreorder( hACSRCOO, n, 8, &hAinitguess );

    magma_z_mtransfer( hAinitguess, &dAinitguess, Magma_CPU, Magma_DEV );

    real_Double_t FLOPS = 2.0*hA.nnz/1e9;

    magma_z_mtransfer( hA, &dA, Magma_CPU, Magma_DEV );
    
    // need only lower triangular
    hAL.diagorder_type == Magma_UNITY;
    magma_z_mconvert( hA, &hAL, Magma_CSR, Magma_CSRL );
    magma_z_mconvert( hAL, &hALCOO, Magma_CSR, Magma_CSRCOO );

    // need only upper triangular
    magma_z_mconvert( hA, &hAU, Magma_CSR, Magma_CSRU );
    magma_z_cucsrtranspose(  hAU, &hAUT );
    magma_z_mconvert( hAUT, &hAUCOO, Magma_CSR, Magma_CSRCOO );


// reordering for higher cache efficiency
magma_z_sparse_matrix hALCOOr, hAUCOOr;

//magma_zmreorder( hALCOO, n, 2, &hALCOOr );
//magma_zmreorder( hAUCOO, n, 2, &hAUCOOr );
// reordering

    //magma_z_mtransfer( hALCOO, &dAL, Magma_CPU, Magma_DEV );
    //magma_z_mtransfer( hAUCOO, &dAU, Magma_CPU, Magma_DEV );

            real_Double_t t_cusparse, t_chow;

            magmaDoubleComplex alpha = MAGMA_Z_MAKE(1.0, 0.0);
            magmaDoubleComplex beta = MAGMA_Z_MAKE(0.0, 0.0);


            // CUSPARSE context //
            cusparseHandle_t cusparseHandle;
            cusparseStatus_t cusparseStatus;
            cusparseStatus = cusparseCreate(&cusparseHandle);
             if(cusparseStatus != 0)    printf("error in Handle.\n");
            cusparseMatDescr_t descrA;
            cusparseStatus = cusparseCreateMatDescr(&descrA);
             if(cusparseStatus != 0)    printf("error in MatrDescr.\n");
            cusparseStatus =
            cusparseSetMatType(descrA,CUSPARSE_MATRIX_TYPE_GENERAL);
             if(cusparseStatus != 0)    printf("error in MatrType.\n");
            cusparseStatus =
            cusparseSetMatDiagType (descrA, CUSPARSE_DIAG_TYPE_NON_UNIT);
             if(cusparseStatus != 0)    printf("error in DiagType.\n");
            cusparseStatus =
            cusparseSetMatIndexBase(descrA,CUSPARSE_INDEX_BASE_ZERO);
             if(cusparseStatus != 0)    printf("error in IndexBase.\n");
            cusparseSolveAnalysisInfo_t info;
            cusparseStatus =
            cusparseCreateSolveAnalysisInfo(&info);
             if(cusparseStatus != 0)    printf("error in info.\n");

            magma_device_sync(); start = magma_wtime(); 
            cusparseStatus =
            cusparseZcsrsv_analysis( cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                        dA.num_rows, dA.nnz, descrA,
                            dA.val, dA.row, dA.col, info); 
             if(cusparseStatus != 0)    printf("error in analysis.\n");
            cusparseStatus =
            cusparseZcsrilu0( cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                              dA.num_rows, descrA, 
                             (magmaDoubleComplex*) dA.val, (const int *) dA.row, (const int *) dA.col, info);
             if(cusparseStatus != 0)    printf("error in ILU.\n");

            magma_device_sync(); end = magma_wtime(); 
            t_cusparse = end-start;

            cusparseDestroySolveAnalysisInfo( info );
             if(cusparseStatus != 0)    printf("error in info-free.\n");

            // end CUSPARSE context //

            magma_z_mtransfer( dA, &hAcusparse, Magma_DEV, Magma_CPU );



    for(int iters=0; iters<31; iters++){



    real_Double_t resavg = 0.0;
    real_Double_t nonlinresavg = 0.0;
    int numavg = 1;

    //multiple runs
    for(int z=0; z<numavg; z++){

    real_Double_t res = 0.0;
    real_Double_t nonlinres = 0.0;

    magma_z_mtransfer( hALCOO, &dL, Magma_CPU, Magma_DEV );
    magma_z_mtransfer( hALCOO, &dU, Magma_CPU, Magma_DEV );



    // the homomorphism
    magma_index_t *p, *p_h;
    magma_index_malloc( &p, hALCOO.nnz );
    magma_index_malloc_cpu( &p_h, hALCOO.nnz );
    for(int i=0; i<hALCOO.nnz; i++)
        p_h[i] = i;

    int limit=hALCOO.nnz;

    for(int i=0; i< limit; i++){
        int idx1 = rand()%limit;
        int idx2 = rand()%limit;
        int tmp = p_h[idx1];
        p_h[idx1] = p_h[idx2];
        p_h[idx2] = tmp;
    }

//printf("good\n");

    cublasSetVector( hALCOO.nnz , 
                sizeof( magma_index_t ), p_h, 1, p, 1 );


   // magma_zmhom( hALCOO, 2, p );


    magma_zmhom_fd( hALCOO, n, 2, p );

    magma_device_sync(); start = magma_wtime(); 




    for(int i=0; i<iters; i++){
        //magma_zailu_csr_s( dAL, dAU, dL, dU );
        magma_zailu_csr_a( dAinitguess, dL, dU );
        //magma_zailu_csr_s_debug( p, dAL, dAU, dL, dU );
        //printf("\n\n\n");
        //magma_z_mvisu(dU);
    }

    magma_device_sync(); end = magma_wtime();
    t_chow = end-start;

    magma_free( p );

    magma_z_mtransfer( dL, &hL, Magma_DEV, Magma_CPU );
    magma_z_mtransfer( dU, &hU, Magma_DEV, Magma_CPU );

    magma_z_mfree(&dL);
    magma_z_mfree(&dU);

    magma_z_LUmergein( hL, hU, &hAtmp);



    magma_zfrobenius( hAtmp, hAcusparse, &res );

    magma_z_cucsrtranspose(  hU, &hUT );

    magma_znonlinres(   hA, hL, hUT, &hLU, &nonlinres ); 
    /*
    if( res != res ){
        nonlinresavg += nonlinresavg/(i+1);
        resavg += resavg/(i+1);
    }
    else{*/
        nonlinresavg += nonlinres;
        resavg += res;
    //}

//    printf(" %d    %.2e   ",1* iters, t_chow);
//    printf(" %.2e    %.4e    %.4e   \n", t_cusparse, res, nonlinres);

//printf(" %d  %d  %d  %.2e  %.2e\n", iters, hA.num_rows, hA.nnz, t_chow*1.0, t_cusparse*1000.0);


    magma_z_mfree(&hL);
    magma_z_mfree(&hU);
    magma_z_mfree(&hUT);


    }//multiple runs

    nonlinresavg = nonlinresavg/numavg;
    resavg = resavg/numavg;

    printf(" %d    %.2e   ",1* iters, t_chow);
    printf(" %.2e    %.4e    %.4e   \n", t_cusparse, resavg, nonlinresavg);

    }// iters

    magma_z_mfree(&hAtmp);
    magma_z_mfree(&hAL);
    magma_z_mfree(&hALCOO);
    magma_z_mfree(&hAU);
    magma_z_mfree(&hAUT);
    magma_z_mfree(&hAUCOO);
    magma_z_mfree(&hAcusparse);
    magma_z_mfree(&dA);

    magma_z_mfree(&hALCOOr);
    magma_z_mfree(&hAUCOOr);

}

}


    TESTING_FINALIZE();
    return 0;
}
