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


int main( int argc, char** argv)
{
    TESTING_INIT();

   
    
    real_Double_t start, end;
    real_Double_t t_cusparse, t_chow;

    magma_z_sparse_matrix hA, hAL, hALCOO, hAU,  hAUt, hAUCOO, hAcusparse, hAtmp, dA, hLU, dL, dU, hL, hU, hUT;


        //################################################################//
        //                      matrix generator                          //
        //################################################################//

    // generate matrix of desired structure and size (3d 27-point stencil)
   /* int n=4;   // size is n*n*n
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

    // now set some entries to zero (boundary...)
    for(int  i=0; i<n*n; i++ ){
    for(int  j=0; j<n; j++ ){
        magma_index_t row = i*n+j;
        magma_index_t l_bound = i*n;
        magma_index_t u_bound = (i+1)*n;
        for(int  k=hA.row[row]; k<hA.row[row+1]; k++){

            if((hA.col[k] == row-1 ||
                hA.col[k] == row-n-1 ||
                hA.col[k] == row+n-1 ||
                hA.col[k] == row-n*n+n-1 ||
                hA.col[k] == row+n*n-n-1 ||
                hA.col[k] == row-n*n-1 ||
                hA.col[k] == row+n*n-1 ||
                hA.col[k] == row-n*n-n-1 ||
                hA.col[k] == row+n*n+n-1 ) && (row+1)%n == 1 )
                    
                    hA.val[k] = MAGMA_Z_MAKE( 0.0, 0.0 );

            if((hA.col[k] == row+1 ||
                hA.col[k] == row-n+1 ||
                hA.col[k] == row+n+1 ||
                hA.col[k] == row-n*n+n+1 ||
                hA.col[k] == row+n*n-n+1 ||
                hA.col[k] == row-n*n+1 ||
                hA.col[k] == row+n*n+1 ||
                hA.col[k] == row-n*n-n+1 ||
                hA.col[k] == row+n*n+n+1 ) && (row)%n ==n-1 )
                    
                    hA.val[k] = MAGMA_Z_MAKE( 0.0, 0.0 );
        }
        
    }
    }*/
    int n=64;
    magma_zm_27stencil(  n, &hA );

        //################################################################//
        //                  end matrix generator                          //
        //################################################################//

    // scale to unit diagonal
    magma_zmscale( &hA, Magma_UNITDIAG );

    real_Double_t FLOPS = 2.0*hA.nnz/1e9;


        //################################################################//
        //                  cuSPARSE reference ILU                        //
        //################################################################//

        magmaDoubleComplex alpha = MAGMA_Z_MAKE(1.0, 0.0);
        magmaDoubleComplex beta = MAGMA_Z_MAKE(0.0, 0.0);

        magma_z_mtransfer( hA, &dA, Magma_CPU, Magma_DEV );

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
        cusparseZcsrsv_analysis( cusparseHandle, 
                                 CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                 dA.num_rows, dA.nnz, descrA,
                                 dA.val, dA.row, dA.col, info); 
         if(cusparseStatus != 0)    printf("error in analysis.\n");

        cusparseStatus =
        cusparseZcsrilu0( cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                          dA.num_rows, descrA, 
                         (magmaDoubleComplex*) dA.val, (const int *) dA.row, 
                         (const int *) dA.col, info);
         if(cusparseStatus != 0)    printf("error in ILU.\n");

        magma_device_sync(); end = magma_wtime(); 
        t_cusparse = end-start;

        cusparseDestroySolveAnalysisInfo( info );
         if(cusparseStatus != 0)    printf("error in info-free.\n");

        // end CUSPARSE context //

        magma_z_mtransfer( dA, &hAcusparse, Magma_DEV, Magma_CPU );

        //################################################################//
        //                  end cuSPARSE reference ILU                    //
        //################################################################//

    // reorder the matrix determining the update processing order
    magma_z_sparse_matrix hAcopy, hACSRCOO, hAinitguess, dAinitguess;



    magma_z_mtransfer( hA, &hAcopy, Magma_CPU, Magma_CPU );

    // ---------------- iteration matrices ------------------- //
    // possibility to increase fill-in in ILU-(m)

    //ILU-m levels
    for( int levels = 0; levels < 8; levels++){ //ILU-m levels
    magma_zsymbilu( &hAcopy, levels, &hAL, &hAUt ); 
    printf("\n#================================================================================#\n");
    printf("# ILU-(m)  #nnz  iters  blocksize  t_chow   t_cusparse     error        ILUres \n");
    // add a unit diagonal to L for the algorithm
    magma_zmLdiagadd( &hAL ); 

    // transpose U for the algorithm
    magma_z_cucsrtranspose(  hAUt, &hAU );
    magma_z_mfree( &hAUt );
    // scale to unit diagonal
    //magma_zmscale( &hAU, Magma_UNITDIAG );


/*
    // need only lower triangular
    magma_z_mfree(&hAL);
    hAL.diagorder_type == Magma_UNITY;
    magma_z_mconvert( hA, &hAL, Magma_CSR, Magma_CSRL );
*/

    // ---------------- initial guess ------------------- //
    magma_z_mconvert( hAcopy, &hACSRCOO, Magma_CSR, Magma_CSRCOO );
    int blocksize = 4;
    magma_zmreorder( hACSRCOO, n, blocksize, blocksize, blocksize, &hAinitguess );
    magma_z_mtransfer( hACSRCOO, &dAinitguess, Magma_CPU, Magma_DEV );
    magma_z_mfree(&hACSRCOO);


        //################################################################//
        //                        iterative ILU                           //
        //################################################################//
    // number of AILU sweeps
    for(int iters=0; iters<31; iters++){

    // take average results for residuals
    real_Double_t resavg = 0.0;
    real_Double_t iluresavg = 0.0;
    int nnz, numavg = 1;
    //multiple runs
    for(int z=0; z<numavg; z++){

        real_Double_t res = 0.0;
        real_Double_t ilures = 0.0;

        // transfer the factor L and U
        magma_z_mtransfer( hAL, &dL, Magma_CPU, Magma_DEV );
        magma_z_mtransfer( hAU, &dU, Magma_CPU, Magma_DEV );

        // iterative ILU embedded in timing
        magma_device_sync(); start = magma_wtime(); 
        for(int i=0; i<iters; i++){
cudaProfilerStart();
            magma_zailu_csr_a( dAinitguess, dL, dU );
cudaProfilerStop();
        }
        magma_device_sync(); end = magma_wtime();
        t_chow = end-start;

        // check the residuals
        magma_z_mtransfer( dL, &hL, Magma_DEV, Magma_CPU );
        magma_z_mtransfer( dU, &hU, Magma_DEV, Magma_CPU );
        magma_z_cucsrtranspose(  hU, &hUT );

        magma_z_mfree(&dL);
        magma_z_mfree(&dU);

        magma_zmlumerge( hL, hUT, &hAtmp);
        // frobenius norm of error
        magma_zfrobenius( hAcusparse, hAtmp, &res );

        // ilu residual
        magma_zilures(   hA, hL, hUT, &hLU, &ilures ); 

        iluresavg += ilures;
        resavg += res;
        nnz = hAtmp.nnz;

        magma_z_mfree(&hL);
        magma_z_mfree(&hU);
        magma_z_mfree(&hUT);
        magma_z_mfree(&hAtmp);




    }//multiple runs

    iluresavg = iluresavg/numavg;
    resavg = resavg/numavg;

    printf(" %d      %d      %d       %d        %.2e   ",
                              levels, nnz, 1* iters, blocksize, t_chow);
    printf(" %.2e    %.4e    %.4e   \n", t_cusparse, resavg, iluresavg);

    }// iters
    printf("#================================================================================#\n");
    }// levels

    // free all memory
    magma_z_mfree(&hAL);
    magma_z_mfree(&hAU);
    magma_z_mfree(&hAcusparse);
    magma_z_mfree(&dA);
    magma_z_mfree(&dAinitguess);
    magma_z_mfree(&hA);
    magma_z_mfree(&hAcopy);

    //}// multiple matrices

    TESTING_FINALIZE();
    return 0;
}
