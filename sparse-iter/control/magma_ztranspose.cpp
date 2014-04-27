/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> s d c
       @author Hartwig Anzt

*/
#include "magma_lapack.h"
#include "common_magma.h"
#include "../include/magmasparse.h"

#include <assert.h>

// includes CUDA
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <cusparse_v2.h>
#include <cuda_profiler_api.h>

/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

    Purpose
    =======

    Helper function to transpose CSR matrix.


    Arguments
    =========

    magma_z_sparse_matrix A             input matrix (CSR)
    magma_z_sparse_matrix *B            output matrix (CSR)



    ========================================================================  */


magma_int_t 
magma_z_csrtranspose( magma_z_sparse_matrix A, magma_z_sparse_matrix *B ){

    if( A.storage_type != Magma_CSR ){
        magma_z_sparse_matrix ACSR, BCSR;
        magma_z_mconvert( A, &ACSR, A.storage_type, Magma_CSR );
        magma_z_cucsrtranspose( ACSR, &BCSR );
        magma_z_mconvert( BCSR, B, Magma_CSR, A.storage_type );
        
    }
    else{

        magma_int_t i, j, k, new_nnz=0, lrow, lcol;

        // fill in information for B
        B->storage_type = Magma_CSR;
        B->memory_location = A.memory_location;
        B->num_rows = A.num_rows;
        B->num_cols = A.num_cols;
        B->nnz = A.nnz;
        B->max_nnz_row = A.max_nnz_row;
        B->diameter = A.diameter;

        magma_zmalloc_cpu( &B->val, A.nnz );
        magma_indexmalloc_cpu( &B->row, A.num_rows+1 );
        magma_indexmalloc_cpu( &B->col, A.nnz );

        for( magma_int_t i=0; i<A.nnz; i++){
            B->val[i] = A.val[i];
            B->col[i] = A.col[i];
        }
        for( magma_int_t i=0; i<A.num_rows+1; i++){
            B->row[i] = A.row[i];
        }
        for( lrow = 0; lrow < A.num_rows; lrow++ ){
            B->row[lrow] = new_nnz;
            for( i=0; i<A.num_rows; i++ ){
                for( j=A.row[i]; j<A.row[i+1]; j++ ){
                    if( A.col[j] == lrow ){
                        B->val[ new_nnz ] = A.val[ j ];
                        B->col[ new_nnz ] = i;
                        new_nnz++; 
                    }
                }
            }
        }
        B->row[ B->num_rows ] = new_nnz;

        return MAGMA_SUCCESS;

    }
}


/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

    Purpose
    =======

    Helper function to transpose CSR matrix. 
    Using the CUSPARSE CSR2CSC function.


    Arguments
    =========

    magma_z_sparse_matrix A             input matrix (CSR)
    magma_z_sparse_matrix *B            output matrix (CSR)



    ========================================================================  */

magma_int_t 
magma_z_cucsrtranspose( magma_z_sparse_matrix A, magma_z_sparse_matrix *B ){
// for symmetric matrices: convert to csc using cusparse


    if( A.storage_type != Magma_CSR ){
        magma_z_sparse_matrix ACSR, BCSR;
        magma_z_mconvert( A, &ACSR, A.storage_type, Magma_CSR );
        magma_z_cucsrtranspose( ACSR, &BCSR );
        magma_z_mconvert( BCSR, B, Magma_CSR, A.storage_type );
        
    }
    else{

        magma_z_sparse_matrix A_d, B_d;

        magma_z_mtransfer( A, &A_d, A.memory_location, Magma_DEV );

        magma_z_mtransfer( A, &B_d, A.memory_location, Magma_DEV );


        // CUSPARSE context //
        cusparseHandle_t handle;
        cusparseStatus_t cusparseStatus;
        cusparseStatus = cusparseCreate(&handle);
         if(cusparseStatus != 0)    printf("error in Handle.\n");


        cusparseMatDescr_t descrA;
        cusparseMatDescr_t descrB;
        cusparseStatus = cusparseCreateMatDescr(&descrA);
        cusparseStatus = cusparseCreateMatDescr(&descrB);
         if(cusparseStatus != 0)    printf("error in MatrDescr.\n");

        cusparseStatus =
        cusparseSetMatType(descrA,CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatType(descrB,CUSPARSE_MATRIX_TYPE_GENERAL);
         if(cusparseStatus != 0)    printf("error in MatrType.\n");

        cusparseStatus =
        cusparseSetMatIndexBase(descrA,CUSPARSE_INDEX_BASE_ZERO);
        cusparseSetMatIndexBase(descrB,CUSPARSE_INDEX_BASE_ZERO);
         if(cusparseStatus != 0)    printf("error in IndexBase.\n");

        cusparseStatus = 
        cusparseZcsr2csc( handle, A.num_rows, A.num_rows, A.nnz,
                         A_d.val, A_d.row, A_d.col, B_d.val, B_d.col, B_d.row,
                         CUSPARSE_ACTION_NUMERIC, 
                         CUSPARSE_INDEX_BASE_ZERO);
         if(cusparseStatus != 0)    
                printf("error in transpose: %d.\n", cusparseStatus);

        cusparseDestroyMatDescr( descrA );
        cusparseDestroyMatDescr( descrB );
        cusparseDestroy( handle );

        // end CUSPARSE context //

        magma_z_mtransfer( B_d, B, Magma_DEV, Magma_CPU );
        magma_z_mfree( &A_d );
        magma_z_mfree( &B_d );

        return MAGMA_SUCCESS;
    }
}




