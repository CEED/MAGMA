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
#include "magmasparse.h"

#include <assert.h>


#define PRECISION_z


/**
    Purpose
    -------

    Prepares the ILU preconditioner via the iterative ILU iteration.

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                input matrix A

    @param[in][out]
    precond     magma_z_preconditioner*
                preconditioner parameters
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgepr
    ********************************************************************/

magma_int_t
magma_ziterilusetup( 
    magma_z_matrix A, 
    magma_z_matrix b,                                 
    magma_z_preconditioner *precond,
    magma_queue_t queue ){

    magma_z_matrix hAh, hA, hL, hU, hAcopy, hAL, hAU, hAUt, hUT, hAtmp,
                        hACSRCOO, dAinitguess, dL, dU, DL, RL, DU, RU;

    // copy original matrix as CSRCOO to device
    magma_zmtransfer(A, &hAh, A.memory_location, Magma_CPU, queue );
    magma_zmconvert( hAh, &hA, hAh.storage_type, Magma_CSR , queue );
    magma_zmfree(&hAh, queue );

    magma_zmtransfer( hA, &hAcopy, Magma_CPU, Magma_CPU , queue );

    // in case using fill-in
    magma_zsymbilu( &hAcopy, precond->levels, &hAL, &hAUt,  queue ); 
    // add a unit diagonal to L for the algorithm
    magma_zmLdiagadd( &hAL , queue ); 
    // transpose U for the algorithm
    magma_z_cucsrtranspose(  hAUt, &hAU , queue );
    magma_zmfree( &hAUt , queue );

    // ---------------- initial guess ------------------- //
    magma_zmconvert( hAcopy, &hACSRCOO, Magma_CSR, Magma_CSRCOO , queue );
    magma_zmtransfer( hACSRCOO, &dAinitguess, Magma_CPU, Magma_DEV , queue );
    magma_zmfree(&hACSRCOO, queue );
    magma_zmfree(&hAcopy, queue );

    // transfer the factor L and U
    magma_zmtransfer( hAL, &dL, Magma_CPU, Magma_DEV , queue );
    magma_zmtransfer( hAU, &dU, Magma_CPU, Magma_DEV , queue );
    magma_zmfree(&hAL, queue );
    magma_zmfree(&hAU, queue );

    for(int i=0; i<precond->sweeps; i++){
        magma_ziterilu_csr( dAinitguess, dL, dU , queue );
    }

    magma_zmtransfer( dL, &hL, Magma_DEV, Magma_CPU , queue );
    magma_zmtransfer( dU, &hU, Magma_DEV, Magma_CPU , queue );
    magma_z_cucsrtranspose(  hU, &hUT , queue );

    magma_zmfree(&dL, queue );
    magma_zmfree(&dU, queue );
    magma_zmfree(&hU, queue );
    magma_zmlumerge( hL, hUT, &hAtmp, queue );

    magma_zmfree(&hL, queue );
    magma_zmfree(&hUT, queue );

    magma_zmtransfer( hAtmp, &precond->M, Magma_CPU, Magma_DEV , queue );

    hAL.diagorder_type = Magma_UNITY;
    magma_zmconvert(hAtmp, &hAL, Magma_CSR, Magma_CSRL, queue );
    hAL.storage_type = Magma_CSR;
    magma_zmconvert(hAtmp, &hAU, Magma_CSR, Magma_CSRU, queue );
    hAU.storage_type = Magma_CSR;

    magma_zmfree(&hAtmp, queue );

    magma_zcsrsplit( 256, hAL, &DL, &RL , queue );
    magma_zcsrsplit( 256, hAU, &DU, &RU , queue );

    magma_zmtransfer( DL, &precond->LD, Magma_CPU, Magma_DEV , queue );
    magma_zmtransfer( DU, &precond->UD, Magma_CPU, Magma_DEV , queue );

    // for cusparse uncomment this
    magma_zmtransfer( hAL, &precond->L, Magma_CPU, Magma_DEV , queue );
    magma_zmtransfer( hAU, &precond->U, Magma_CPU, Magma_DEV , queue );
    
/*

    //-- for ba-solve uncomment this

    if( RL.nnz != 0 )
        magma_zmtransfer( RL, &precond->L, Magma_CPU, Magma_DEV , queue );
    else{ 
        precond->L.nnz = 0;
        precond->L.val = NULL;
        precond->L.col = NULL;
        precond->L.row = NULL;
        precond->L.blockinfo = NULL;
    }

    if( RU.nnz != 0 )
        magma_zmtransfer( RU, &precond->U, Magma_CPU, Magma_DEV , queue );
    else{ 
        precond->U.nnz = 0;
        precond->L.val = NULL;
        precond->L.col = NULL;
        precond->L.row = NULL;
        precond->L.blockinfo = NULL;
    }

    //-- for ba-solve uncomment this
*/

        // extract the diagonal of L into precond->d 
    magma_zjacobisetup_diagscal( precond->L, &precond->d, queue );
    magma_zvinit( &precond->work1, Magma_DEV, hA.num_rows, 1, MAGMA_Z_ZERO, queue );
    
    // extract the diagonal of U into precond->d2  
    magma_zjacobisetup_diagscal( precond->U, &precond->d2, queue );
    magma_zvinit( &precond->work2, Magma_DEV, hA.num_rows, 1, MAGMA_Z_ZERO, queue );

    magma_zmfree(&hAL, queue );
    magma_zmfree(&hAU, queue );
    magma_zmfree(&DL, queue );
    magma_zmfree(&RL, queue );
    magma_zmfree(&DU, queue );
    magma_zmfree(&RU, queue );

    // CUSPARSE context //
    cusparseHandle_t cusparseHandle;
    cusparseStatus_t cusparseStatus;

    cusparseStatus = cusparseCreate(&cusparseHandle);
     if(cusparseStatus != 0)    printf("error in Handle.\n" );

    cusparseMatDescr_t descrL;
    cusparseStatus = cusparseCreateMatDescr(&descrL);
     if(cusparseStatus != 0)    printf("error in MatrDescr.\n" );

    cusparseStatus =
    cusparseSetMatType(descrL,CUSPARSE_MATRIX_TYPE_TRIANGULAR);
     if(cusparseStatus != 0)    printf("error in MatrType.\n" );

    cusparseStatus =
    cusparseSetMatDiagType (descrL, CUSPARSE_DIAG_TYPE_UNIT);
     if(cusparseStatus != 0)    printf("error in DiagType.\n" );

    cusparseStatus =
    cusparseSetMatIndexBase(descrL,CUSPARSE_INDEX_BASE_ZERO);
     if(cusparseStatus != 0)    printf("error in IndexBase.\n" );

    cusparseStatus =
    cusparseSetMatFillMode(descrL,CUSPARSE_FILL_MODE_LOWER);
     if(cusparseStatus != 0)    printf("error in fillmode.\n" );


    cusparseStatus = cusparseCreateSolveAnalysisInfo(&precond->cuinfoL ); 
     if(cusparseStatus != 0)    printf("error in info.\n" );

    cusparseStatus =
    cusparseZcsrsv_analysis(cusparseHandle, 
        CUSPARSE_OPERATION_NON_TRANSPOSE, precond->L.num_rows, 
        precond->L.nnz, descrL, 
        precond->L.val, precond->L.row, precond->L.col, precond->cuinfoL);
     if(cusparseStatus != 0)    printf("error in analysis.\n" );

    cusparseDestroyMatDescr( descrL  );

    cusparseMatDescr_t descrU;
    cusparseStatus = cusparseCreateMatDescr(&descrU );
     if(cusparseStatus != 0)    printf("error in MatrDescr.\n" );

    cusparseStatus =
    cusparseSetMatType(descrU,CUSPARSE_MATRIX_TYPE_TRIANGULAR );
     if(cusparseStatus != 0)    printf("error in MatrType.\n" );

    cusparseStatus =
    cusparseSetMatDiagType (descrU, CUSPARSE_DIAG_TYPE_NON_UNIT );
     if(cusparseStatus != 0)    printf("error in DiagType.\n" );

    cusparseStatus =
    cusparseSetMatIndexBase(descrU,CUSPARSE_INDEX_BASE_ZERO );
     if(cusparseStatus != 0)    printf("error in IndexBase.\n" );

    cusparseStatus =
    cusparseSetMatFillMode(descrU,CUSPARSE_FILL_MODE_UPPER);
     if(cusparseStatus != 0)    printf("error in fillmode.\n" );

    cusparseStatus = cusparseCreateSolveAnalysisInfo(&precond->cuinfoU ); 
     if(cusparseStatus != 0)    printf("error in info.\n" );

    cusparseStatus =
    cusparseZcsrsv_analysis(cusparseHandle, 
        CUSPARSE_OPERATION_NON_TRANSPOSE, precond->U.num_rows, 
        precond->U.nnz, descrU, 
        precond->U.val, precond->U.row, precond->U.col, precond->cuinfoU  );
     if(cusparseStatus != 0)    printf("error in analysis.\n" );

    cusparseDestroyMatDescr( descrU  );
    cusparseDestroy( cusparseHandle  );

    return MAGMA_SUCCESS;

}






/**
    Purpose
    -------

    Prepares the IC preconditioner via the iterative IC iteration.

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                input matrix A

    @param[in][out]
    precond     magma_z_preconditioner*
                preconditioner parameters
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zhepr
    ********************************************************************/

magma_int_t
magma_zitericsetup( 
    magma_z_matrix A, 
    magma_z_matrix b, 
    magma_z_preconditioner *precond,
    magma_queue_t queue ){


    magma_z_matrix hAh, hA, hAtmp, hAL, hAUt, hALt, hM, hACSRCOO, dAinitguess, dL;



    // copy original matrix as CSRCOO to device
    magma_zmtransfer(A, &hAh, A.memory_location, Magma_CPU, queue );
    magma_zmconvert( hAh, &hA, hAh.storage_type, Magma_CSR , queue );
    magma_zmfree(&hAh, queue );

    // in case using fill-in
    magma_zsymbilu( &hA, precond->levels, &hAL, &hAUt , queue ); 

    // need only lower triangular
    magma_zmfree(&hAUt, queue );
    magma_zmfree(&hAL, queue );
    magma_zmconvert( hA, &hAtmp, Magma_CSR, Magma_CSRL , queue );
    magma_zmfree(&hA, queue );

    // ---------------- initial guess ------------------- //
    magma_zmconvert( hAtmp, &hACSRCOO, Magma_CSR, Magma_CSRCOO , queue );
    //int blocksize = 1;
    //magma_zmreorder( hACSRCOO, n, blocksize, blocksize, blocksize, &hAinitguess , queue );
    magma_zmtransfer( hACSRCOO, &dAinitguess, Magma_CPU, Magma_DEV , queue );
    magma_zmfree(&hACSRCOO, queue );
    magma_zmtransfer( hAtmp, &dL, Magma_CPU, Magma_DEV , queue );
    magma_zmfree(&hAtmp, queue );

    for(int i=0; i<precond->sweeps; i++){
        magma_ziteric_csr( dAinitguess, dL , queue );
    }
    magma_zmtransfer( dL, &hAL, Magma_DEV, Magma_CPU , queue );
    magma_zmfree(&dL, queue );
    magma_zmfree(&dAinitguess, queue );


    // for CUSPARSE
    magma_zmtransfer( hAL, &precond->M, Magma_CPU, Magma_DEV , queue );

    // Jacobi setup
    magma_zjacobisetup_matrix( precond->M, &precond->L, &precond->d , queue );    

    // for Jacobi, we also need U
    magma_z_cucsrtranspose(   hAL, &hALt , queue );
    magma_z_matrix d_h;
    magma_zjacobisetup_matrix( hALt, &hM, &d_h , queue );

    magma_zmtransfer( hM, &precond->U, Magma_CPU, Magma_DEV , queue );

    magma_zmfree(&hM, queue );

    magma_zmfree(&d_h, queue );


        // copy the matrix to precond->L and (transposed) to precond->U
    magma_zmtransfer(precond->M, &(precond->L), Magma_DEV, Magma_DEV, queue );
    magma_zmtranspose( precond->L, &(precond->U), queue );

    // extract the diagonal of L into precond->d 
    magma_zjacobisetup_diagscal( precond->L, &precond->d, queue );
    magma_zvinit( &precond->work1, Magma_DEV, hAL.num_rows, 1, MAGMA_Z_ZERO, queue );

    // extract the diagonal of U into precond->d2
    magma_zjacobisetup_diagscal( precond->U, &precond->d2, queue );
    magma_zvinit( &precond->work2, Magma_DEV, hAL.num_rows, 1, MAGMA_Z_ZERO, queue );


    magma_zmfree(&hAL, queue );
    magma_zmfree(&hALt, queue );


    // CUSPARSE context //
    cusparseHandle_t cusparseHandle;
    cusparseStatus_t cusparseStatus;
    cusparseStatus = cusparseCreate(&cusparseHandle );
     if(cusparseStatus != 0)    printf("error in Handle.\n" );

    cusparseMatDescr_t descrL;
    cusparseStatus = cusparseCreateMatDescr(&descrL );
     if(cusparseStatus != 0)    printf("error in MatrDescr.\n" );

    cusparseStatus =
    cusparseSetMatType(descrL,CUSPARSE_MATRIX_TYPE_TRIANGULAR );
     if(cusparseStatus != 0)    printf("error in MatrType.\n" );

    cusparseStatus =
    cusparseSetMatDiagType (descrL, CUSPARSE_DIAG_TYPE_NON_UNIT );
     if(cusparseStatus != 0)    printf("error in DiagType.\n" );

    cusparseStatus =
    cusparseSetMatIndexBase(descrL,CUSPARSE_INDEX_BASE_ZERO );
     if(cusparseStatus != 0)    printf("error in IndexBase.\n" );

    cusparseStatus =
    cusparseSetMatFillMode(descrL,CUSPARSE_FILL_MODE_LOWER );
     if(cusparseStatus != 0)    printf("error in fillmode.\n" );


    cusparseStatus = cusparseCreateSolveAnalysisInfo(&precond->cuinfoL ); 
     if(cusparseStatus != 0)    printf("error in info.\n" );

    cusparseStatus =
    cusparseZcsrsv_analysis(cusparseHandle, 
        CUSPARSE_OPERATION_NON_TRANSPOSE, precond->M.num_rows, 
        precond->M.nnz, descrL, 
        precond->M.val, precond->M.row, precond->M.col, precond->cuinfoL  );
     if(cusparseStatus != 0)    printf("error in analysis L.\n" );

    cusparseDestroyMatDescr( descrL );

    cusparseMatDescr_t descrU;
    cusparseStatus = cusparseCreateMatDescr(&descrU );
     if(cusparseStatus != 0)    printf("error in MatrDescr.\n" );

    cusparseStatus =
    cusparseSetMatType(descrU,CUSPARSE_MATRIX_TYPE_TRIANGULAR );
     if(cusparseStatus != 0)    printf("error in MatrType.\n" );

    cusparseStatus =
    cusparseSetMatDiagType (descrU, CUSPARSE_DIAG_TYPE_NON_UNIT );
     if(cusparseStatus != 0)    printf("error in DiagType.\n" );

    cusparseStatus =
    cusparseSetMatIndexBase(descrU,CUSPARSE_INDEX_BASE_ZERO);
     if(cusparseStatus != 0)    printf("error in IndexBase.\n" );

    cusparseStatus =
    cusparseSetMatFillMode(descrU,CUSPARSE_FILL_MODE_LOWER );
     if(cusparseStatus != 0)    printf("error in fillmode.\n" );

    cusparseStatus = cusparseCreateSolveAnalysisInfo(&precond->cuinfoU ); 
     if(cusparseStatus != 0)    printf("error in info.\n" );

    cusparseStatus =
    cusparseZcsrsv_analysis(cusparseHandle, 
        CUSPARSE_OPERATION_TRANSPOSE, precond->M.num_rows, 
        precond->M.nnz, descrU, 
        precond->M.val, precond->M.row, precond->M.col, precond->cuinfoU  );
     if(cusparseStatus != 0)    printf("error in analysis U.\n" );

    cusparseDestroyMatDescr( descrU  );
    cusparseDestroy( cusparseHandle  );

    return MAGMA_SUCCESS;

}


/**
    Purpose
    -------

    Updates an existing preconditioner via additional iterative IC sweeps for 
    previous factorization initial guess (PFIG).
    See  Anzt et al., Parallel Computing, 2015.

    Arguments
    ---------

    @param[in]
    precond         magma_z_preconditioner*
                    preconditioner parameters

    @param[in]
    magma_int_t     number of updates
    
    @param[in]
    queue       magma_queue_t
                Queue to execute in.
                
    @ingroup magmasparse_zhepr
    ********************************************************************/

magma_int_t
magma_zitericupdate( 
    magma_z_matrix A, 
    magma_z_preconditioner *precond, 
    magma_int_t updates,
    magma_queue_t queue ){

    magma_z_matrix hALt;


    // copy original matrix as CSRCOO to device

    for(int i=0; i<updates; i++){
        magma_ziteric_csr( A, precond->M , queue );
    }
    //magma_zmtransfer( precond->M, &precond->M, Magma_DEV, Magma_DEV , queue );
    magma_zmfree(&precond->L, queue );
    magma_zmfree(&precond->U, queue );
    magma_zmfree( &precond->d , queue );


    // Jacobi setup
    magma_zjacobisetup_matrix( precond->M, &precond->L, &precond->d , queue );    

    // for Jacobi, we also need U
    magma_z_cucsrtranspose(   precond->M, &hALt , queue );
    magma_z_matrix d_h;
    magma_zjacobisetup_matrix( hALt, &precond->U, &d_h , queue );


    magma_zmfree(&d_h, queue );


    magma_zmfree(&hALt, queue );

/*    // no need, as the structure of L remains

    magma_z_matrix hAL, hALt, hM, dL;

    // CUSPARSE context //
    cusparseHandle_t cusparseHandle;
    cusparseStatus_t cusparseStatus;
    cusparseStatus = cusparseCreate(&cusparseHandle, queue );
     if(cusparseStatus != 0)    printf("error in Handle.\n" );

    cusparseMatDescr_t descrL;
    cusparseStatus = cusparseCreateMatDescr(&descrL, queue );
     if(cusparseStatus != 0)    printf("error in MatrDescr.\n" );

    cusparseStatus =
    cusparseSetMatType(descrL,CUSPARSE_MATRIX_TYPE_TRIANGULAR, queue );
     if(cusparseStatus != 0)    printf("error in MatrType.\n" );

    cusparseStatus =
    cusparseSetMatDiagType (descrL, CUSPARSE_DIAG_TYPE_NON_UNIT, queue );
     if(cusparseStatus != 0)    printf("error in DiagType.\n" );

    cusparseStatus =
    cusparseSetMatIndexBase(descrL,CUSPARSE_INDEX_BASE_ZERO, queue );
     if(cusparseStatus != 0)    printf("error in IndexBase.\n" );

    cusparseStatus =
    cusparseSetMatFillMode(descrL,CUSPARSE_FILL_MODE_LOWER, queue );
     if(cusparseStatus != 0)    printf("error in fillmode.\n" );


    cusparseStatus = cusparseCreateSolveAnalysisInfo(&precond->cuinfoL, queue ); 
     if(cusparseStatus != 0)    printf("error in info.\n" );

    cusparseStatus =
    cusparseZcsrsv_analysis(cusparseHandle, 
        CUSPARSE_OPERATION_NON_TRANSPOSE, precond->M.num_rows, 
        precond->M.nnz, descrL, 
        precond->M.val, precond->M.row, precond->M.col, precond->cuinfoL , queue );
     if(cusparseStatus != 0)    printf("error in analysis L.\n" );

    cusparseDestroyMatDescr( descrL , queue );

    cusparseMatDescr_t descrU;
    cusparseStatus = cusparseCreateMatDescr(&descrU, queue );
     if(cusparseStatus != 0)    printf("error in MatrDescr.\n" );

    cusparseStatus =
    cusparseSetMatType(descrU,CUSPARSE_MATRIX_TYPE_TRIANGULAR, queue );
     if(cusparseStatus != 0)    printf("error in MatrType.\n" );

    cusparseStatus =
    cusparseSetMatDiagType (descrU, CUSPARSE_DIAG_TYPE_NON_UNIT, queue );
     if(cusparseStatus != 0)    printf("error in DiagType.\n" );

    cusparseStatus =
    cusparseSetMatIndexBase(descrU,CUSPARSE_INDEX_BASE_ZERO, queue );
     if(cusparseStatus != 0)    printf("error in IndexBase.\n" );

    cusparseStatus =
    cusparseSetMatFillMode(descrU,CUSPARSE_FILL_MODE_LOWER, queue );
     if(cusparseStatus != 0)    printf("error in fillmode.\n" );

    cusparseStatus = cusparseCreateSolveAnalysisInfo(&precond->cuinfoU, queue ); 
     if(cusparseStatus != 0)    printf("error in info.\n" );

    cusparseStatus =
    cusparseZcsrsv_analysis(cusparseHandle, 
        CUSPARSE_OPERATION_TRANSPOSE, precond->M.num_rows, 
        precond->M.nnz, descrU, 
        precond->M.val, precond->M.row, precond->M.col, precond->cuinfoU , queue );
     if(cusparseStatus != 0)    printf("error in analysis U.\n" );

    cusparseDestroyMatDescr( descrU , queue );
    cusparseDestroy( cusparseHandle  );
*/
    //magma_zmfree(&precond->M, queue );
    // keep initial guess for possibility of PFIG
    //magma_zmtransfer( dAinitguess, &precond->M, Magma_DEV, Magma_DEV , queue );
    //magma_zmfree(&dAinitguess, queue );

    return MAGMA_SUCCESS;

}


/**
    Purpose
    -------

    Performs the left triangular solves using the IC preconditioner via Jacobi.

    Arguments
    ---------

    @param[in]
    b           magma_z_matrix
                RHS

    @param[out]
    x           magma_z_matrix*
                vector to precondition

    @param[in]
    precond     magma_z_preconditioner*
                preconditioner parameters
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgepr
    ********************************************************************/

magma_int_t
magma_zapplyiteric_l( 
    magma_z_matrix b, 
    magma_z_matrix *x, 
    magma_z_preconditioner *precond,
    magma_queue_t queue ){

    magma_int_t dofs = precond->L.num_rows;
    magma_z_solver_par jacobiiter_par;
    jacobiiter_par.maxiter = precond->maxiter;

    // compute c = D^{-1}b and copy c as initial guess to x
    magma_zjacobisetup_vector_gpu( dofs, b, precond->d, 
                                                precond->work1, x, queue ); 

    // copy c as initial guess to x
    //magma_zcopy( dofs, c.val, 1 , x->val, 1 , queue );                      

    // Jacobi iterator
    magma_zjacobiiter_precond( precond->L, x, &jacobiiter_par, precond , queue ); 

    return MAGMA_SUCCESS;

}


/**
    Purpose
    -------

    Performs the right triangular solves using the IC preconditioner via Jacobi.

    Arguments
    ---------

    @param[in]
    b           magma_z_matrix
                RHS

    @param[out]
    x           magma_z_matrix*
                vector to precondition

    @param[in]
    precond     magma_z_preconditioner*
                preconditioner parameters
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgepr
    ********************************************************************/

magma_int_t
magma_zapplyiteric_r( 
    magma_z_matrix b, 
    magma_z_matrix *x, 
    magma_z_preconditioner *precond,
    magma_queue_t queue ){

    magma_int_t dofs = precond->U.num_rows;
    magma_z_solver_par jacobiiter_par;
    jacobiiter_par.maxiter = precond->maxiter;

    // compute c = D^{-1}b and copy c as initial guess to x
    magma_zjacobisetup_vector_gpu( dofs, b, precond->d, 
                                                precond->work1, x, queue ); 

    // copy c as initial guess to x
    //magma_zcopy( dofs, c.val, 1 , x->val, 1 , queue );                      

    // Jacobi iterator
    magma_zjacobiiter_precond( precond->U, x, &jacobiiter_par, precond , queue ); 

    return MAGMA_SUCCESS;

}
