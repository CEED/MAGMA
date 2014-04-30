/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Hartwig Anzt
*/

#include <fstream>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <iostream>
#include <ostream>
#include <assert.h>
#include <stdio.h>
#include <math.h>       /* fabs */
#include "../include/magmasparse_z.h"
#include "../../include/magma.h"
#include "../include/mmio.h"


// includes CUDA
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <cusparse_v2.h>
#include <cuda_profiler_api.h>

using namespace std;

#define PRECISION_z

/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

    Purpose
    =======

    Computes the Frobenius norm of the difference between the CSR matrices A 
    and B. They need to share the same sparsity pattern!


    Arguments
    =========

    magma_z_sparse_matrix A              sparse matrix in CSR
    magma_z_sparse_matrix B              sparse matrix in CSR    
    real_Double_t *res                   residual; 

    ========================================================================  */

magma_int_t 
magma_zfrobenius( magma_z_sparse_matrix A, magma_z_sparse_matrix B, 
                  real_Double_t *res ){

    real_Double_t tmp2;
    magma_int_t i,j;
    
    for(i=0; i<A.num_rows; i++){
        for(j=A.row[i]; j<A.row[i+1]; j++){

            tmp2 = (real_Double_t) fabs( MAGMA_Z_REAL(A.val[j] )
                                            - MAGMA_Z_REAL(B.val[j]) );

            (*res) = (*res) + tmp2* tmp2;
        }      
    }

    (*res) =  sqrt((*res));

    return MAGMA_SUCCESS; 
}



/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

    Purpose
    =======

    Computes the nonlinear residual A- LU and returns the difference as
    well es the Frobenius norm of the difference


    Arguments
    =========

    magma_z_sparse_matrix A              input sparse matrix in CSR
    magma_z_sparse_matrix L              input sparse matrix in CSR    
    magma_z_sparse_matrix U              input sparse matrix in CSR    
    magma_z_sparse_matrix U              output sparse matrix in A-LU in CSR    
    real_Double_t *res                   output residual; 

    ========================================================================  */

magma_int_t 
magma_znonlinres(   magma_z_sparse_matrix A, 
                    magma_z_sparse_matrix L,
                    magma_z_sparse_matrix U, 
                    magma_z_sparse_matrix *LU, 
                    real_Double_t *res ){

    real_Double_t tmp2;
    magma_int_t i,j,k,m;

    magma_z_sparse_matrix L_d, U_d, LU_d;

    magma_z_mtransfer( L, &L_d, Magma_CPU, Magma_DEV ); 
    magma_z_mtransfer( U, &U_d, Magma_CPU, Magma_DEV ); 

    magma_z_mtransfer( U, &LU_d, Magma_CPU, Magma_DEV ); 

    cudaFree( LU_d.val );
    cudaFree( LU_d.col );
    cudaFree( LU_d.row );


    // CUSPARSE context //
    cusparseHandle_t handle;
    cusparseStatus_t cusparseStatus;
    cusparseStatus = cusparseCreate(&handle);
     if(cusparseStatus != 0)    printf("error in Handle.\n");


    cusparseMatDescr_t descrL;
    cusparseMatDescr_t descrU;
    cusparseMatDescr_t descrLU;
    cusparseStatus = cusparseCreateMatDescr(&descrL);
    cusparseStatus = cusparseCreateMatDescr(&descrU);
    cusparseStatus = cusparseCreateMatDescr(&descrLU);
     if(cusparseStatus != 0)    printf("error in MatrDescr.\n");

    cusparseStatus =
    cusparseSetMatType(descrL,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatType(descrU,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatType(descrLU,CUSPARSE_MATRIX_TYPE_GENERAL);
     if(cusparseStatus != 0)    printf("error in MatrType.\n");

    cusparseStatus =
    cusparseSetMatIndexBase(descrL,CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatIndexBase(descrU,CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatIndexBase(descrLU,CUSPARSE_INDEX_BASE_ZERO);
     if(cusparseStatus != 0)    printf("error in IndexBase.\n");

    // end CUSPARSE context //

    // multiply L and U on the device


    magma_int_t baseC;
    // nnzTotalDevHostPtr points to host memory
    magma_index_t *nnzTotalDevHostPtr = (magma_index_t*) &LU_d.nnz;
    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
    cudaMalloc((void**)&LU_d.row, sizeof(magma_index_t)*(L_d.num_rows+1));
    cusparseXcsrgemmNnz(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                L_d.num_rows, L_d.num_rows, L_d.num_rows, 
                                descrL, L_d.nnz, L_d.row, L_d.col,
                                descrU, U_d.nnz, U_d.row, U_d.col,
                                descrLU, LU_d.row, nnzTotalDevHostPtr );
    if (NULL != nnzTotalDevHostPtr){
        LU_d.nnz = *nnzTotalDevHostPtr;
    }else{
        cudaMemcpy(&LU_d.nnz, LU_d.row+m, sizeof(magma_index_t), 
                                            cudaMemcpyDeviceToHost);
        cudaMemcpy(&baseC, LU_d.row, sizeof(magma_index_t), 
                                            cudaMemcpyDeviceToHost);
        LU_d.nnz -= baseC;
    }
    cudaMalloc((void**)&LU_d.col, sizeof(magma_index_t)*LU_d.nnz);
    cudaMalloc((void**)&LU_d.val, sizeof(magmaDoubleComplex)*LU_d.nnz);
    cusparseZcsrgemm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                CUSPARSE_OPERATION_NON_TRANSPOSE, 
                    L_d.num_rows, L_d.num_rows, L_d.num_rows,
                    descrL, L_d.nnz,
                    L_d.val, L_d.row, L_d.col,
                    descrU, U_d.nnz,
                    U_d.val, U_d.row, U_d.col,
                    descrLU,
                    LU_d.val, LU_d.row, LU_d.col);



    cusparseDestroyMatDescr( descrL );
    cusparseDestroyMatDescr( descrU );
    cusparseDestroyMatDescr( descrLU );
    cusparseDestroy( handle );



    magma_z_mtransfer(LU_d, LU, Magma_DEV, Magma_CPU);
    magma_z_mfree( &L_d );
    magma_z_mfree( &U_d );
    magma_z_mfree( &LU_d );

    // compute Frobenius norm of A-LU
    for(i=0; i<A.num_rows; i++){
        for(j=A.row[i]; j<A.row[i+1]; j++){
            magma_index_t lcol = A.col[j];
            for(k=LU->row[i]; k<LU->row[i+1]; k++){
                if( LU->col[k] == lcol )
                    LU->val[k] = MAGMA_Z_MAKE(
                        MAGMA_Z_REAL( LU->val[k] )- MAGMA_Z_REAL( A.val[j] )
                                                , 0.0 );
            }
        }
    }

    for(i=0; i<LU->num_rows; i++){
        for(j=LU->row[i]; j<LU->row[i+1]; j++){
            tmp2 = (real_Double_t) fabs( MAGMA_Z_REAL(LU->val[j]) );
            (*res) = (*res) + tmp2* tmp2;
        }
    }

    (*res) =  sqrt((*res));

    return MAGMA_SUCCESS; 
}

