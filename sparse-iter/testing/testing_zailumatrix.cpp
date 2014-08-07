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
        //                      read matrix from file                     //
        //################################################################//
    const char *filename[] =
    {
//"/mnt/sparse_matrices/mtx/s3dkq4m2.mtx",

  //          "/mnt/sparse_matrices/mtx/Trefethen_20.mtx", //        n:19 nnz:147 nnz/n:7 max_nnz_row:9                            4
           // "/mnt/sparse_matrices/mtx/Trefethen_200.mtx", //        n:19 nnz:147 nnz/n:7 max_nnz_row:9                            4
           // "/mnt/sparse_matrices/mtx/Trefethen_2000.mtx", //        n:19 nnz:147 nnz/n:7 max_nnz_row:9                            4
           // "/mnt/sparse_matrices/mtx/Trefethen_20000.mtx", //        n:19 nnz:147 nnz/n:7 max_nnz_row:9                            4
            "/mnt/sparse_matrices/mtx/af_shell3.mtx", //           n:504855 nnz:17562051 nnz/n:34 max_nnz_row:40     
            "/mnt/sparse_matrices/mtx/apache2.mtx", //             n:715176 nnz:4817870 nnz/n:6 max_nnz_row:8 
            "/mnt/sparse_matrices/mtx/ecology2.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
            "/mnt/sparse_matrices/mtx/G3_circuit.mtx", //          n:1585478 nnz:7660826 nnz/n:4 max_nnz_row:6 
            "/mnt/sparse_matrices/mtx/offshore.mtx", //            n:259789 nnz:4242673 nnz/n:16 max_nnz_row:31 
            "/mnt/sparse_matrices/mtx/parabolic_fem.mtx", //       n:525825 nnz:3674625 nnz/n:6 max_nnz_row:7
            "/mnt/sparse_matrices/mtx/thermal2.mtx", //            n:1228045 nnz:8580313 nnz/n:6 max_nnz_row:11 
    };
    for(int matrix=0; matrix<1; matrix=matrix+1){

    printf("\n\n\n\n\n");
    magma_z_csr_mtx( &hA, filename[matrix] );

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
    for( int levels = 0; levels < 1; levels++){ //ILU-m levels
    magma_zsymbilu( &hAcopy, levels, &hAL, &hAUt ); 
    printf("\n#=========================================================================================#\n");
    printf("# ILU-(m)  #n       #nnz       iters  blocksize  t_chow   t_cusparse     error        ILUres \n");
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
    int blocksize = 1;
  //  magma_zmreorder( hACSRCOO, n, blocksize, blocksize, blocksize, &hAinitguess );
    magma_z_mtransfer( hACSRCOO, &dAinitguess, Magma_CPU, Magma_DEV );
    magma_z_mfree(&hACSRCOO);


        //################################################################//
        //                        iterative ILU                           //
        //################################################################//
    // number of AILU sweeps
    for(int iters=1; iters<2; iters++){

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
            magma_zailu_csr_a( dAinitguess, dL, dU );
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

    printf(" %d      %d      %d      %d       %d        %.2e   ",
                              levels, hA.num_rows, nnz, 1* iters, blocksize, t_chow);
    printf(" %.2e    %.4e    %.4e   \n", t_cusparse, resavg, iluresavg);

    }// iters
    printf("\n#=========================================================================================#\n");
    }// levels

    // free all memory
    magma_z_mfree(&hAL);
    magma_z_mfree(&hAU);
    magma_z_mfree(&hAcusparse);
    magma_z_mfree(&dA);
    magma_z_mfree(&dAinitguess);
    magma_z_mfree(&hA);
    magma_z_mfree(&hAcopy);

    }// multiple matrices

    TESTING_FINALIZE();
    return 0;
}
