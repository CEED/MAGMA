/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Hartwig Anzt 

       @precisions normal z -> s d c
*/
// includes CUDA
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <cusparse_v2.h>
#include <cuda_profiler_api.h>

// project includes
#include "common_magma.h"
#include "../include/magmasparse.h"

#include <assert.h>


#define PRECISION_z






/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

    Purpose
    =======

    Prepares the ICC preconditioner via the asynchronous ICC iteration.

    Arguments
    =========

    magma_z_sparse_matrix A                   input matrix A
    magma_z_preconditioner *precond           preconditioner parameters

    ========================================================================  */

magma_int_t
magma_zaiccsetup( magma_z_sparse_matrix A, magma_z_preconditioner *precond ){
/*
        // copy matrix into preconditioner parameter
            magma_z_mtransfer(A, &(precond->M), Magma_DEV, Magma_DEV);



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

            cusparseStatus =
            cusparseCreateSolveAnalysisInfo( &(precond->cuinfo) );
             if(cusparseStatus != 0)    printf("error in info.\n");

            // end CUSPARSE context //

            cusparseStatus =
            cusparseZcsrsv_analysis( cusparseHandle, 
                        CUSPARSE_OPERATION_NON_TRANSPOSE, 
                        precond->M.num_rows, precond->M.nnz, descrA,
                        precond->M.val, precond->M.row, precond->M.col, 
                        precond->cuinfo); 
             if(cusparseStatus != 0)    printf("error in analysis.\n");

            cusparseStatus =
            cusparseZcsric0(  cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                              precond->M.num_rows, descrA, 
                              precond->M.val, 
                              precond->M.row, 
                              precond->M.col, 
                              precond->cuinfo);
             if(cusparseStatus != 0)    printf("error in ICC.\n");


   //magma_z_mtransfer( hA, &(precond->M), Magma_CPU, Magma_DEV );
    magma_z_mconvert( hA, &hL , Magma_CSR, Magma_CSRL );
    magma_z_mconvert( hA, &hU , Magma_CSR, Magma_CSRU );
    magma_z_mtransfer( hL, &(precond->L), Magma_CPU, Magma_DEV );
    magma_z_mtransfer( hU, &(precond->U), Magma_CPU, Magma_DEV );

    magma_z_mfree(&hA);
    magma_z_mfree(&hL);
    magma_z_mfree(&hU);


    magma_z_mfree(&hA);
    magma_z_mtransfer( precond->M, &hA, Magma_DEV, Magma_CPU );

*/
    magma_z_sparse_matrix hA, dA, hAD, hADD, dAD, dADD, hL, hU;

    magma_z_mtransfer( A, &hA, Magma_DEV, Magma_CPU );

    hAD.storage_type = Magma_CSRCSCL;
    magma_z_mconvert( hA, &hAD, Magma_CSR, hAD.storage_type );
    magma_z_mtransfer( hAD, &dAD, Magma_CPU, Magma_DEV );

   #ifdef PRECISION_d
    // scale initial guess
    for(magma_int_t z=0; z<hA.num_rows; z++){
        real_Double_t s = 0.0;
        for(magma_int_t f=hA.row[z]; f<hA.row[z+1]; f++)
            s+= MAGMA_Z_REAL(hA.val[f])*MAGMA_Z_REAL(hA.val[f]);
        s = 1.0/sqrt(   s  );
        for(magma_int_t f=hA.row[z]; f<hA.row[z+1]; f++)
            hA.val[f] = hA.val[f] * s ;
           
    }
   #endif
    magma_z_mfree(&hAD);
    magma_z_mconvert( hA, &hAD, Magma_CSR, hAD.storage_type );
    magma_z_mtransfer( hAD, &dADD, Magma_CPU, Magma_DEV );
    magma_z_mfree(&hAD);
    magma_z_mfree(&hA);


    for(int i=0; i<20; i++){
        magma_zaic_csr_c( dAD, dADD );
    }
    magma_z_mfree(&dAD);

    magma_z_mtransfer( dADD, &hADD, Magma_DEV, Magma_CPU );

    magma_z_mfree(&dADD);
    magma_z_mconvert( hADD, &hA, Magma_CSRCSCL, Magma_CSR );
    magma_z_mfree(&hADD);    
 
    magma_z_mconvert( hA, &hL , Magma_CSR, Magma_CSRL );
    magma_z_mconvert( hA, &hU , Magma_CSR, Magma_CSRU );
    magma_z_mtransfer( hL, &(precond->L), Magma_CPU, Magma_DEV );
    magma_z_mtransfer( hU, &(precond->U), Magma_CPU, Magma_DEV );

    magma_z_mfree(&hA);

    magma_z_mfree(&hL);
    magma_z_mfree(&hU);


  //  cusparseDestroyMatDescr( descrA );
    //cusparseDestroy( cusparseHandle );
    return MAGMA_SUCCESS;

}




