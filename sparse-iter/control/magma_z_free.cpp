/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

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
#include "../include/magmasparse_z.h"
#include "../../include/magma.h"
#include "../include/mmio.h"



using namespace std;








/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======

    Free the memory of a magma_z_vector.


    Arguments
    =========

    magma_z_vector x                     vector to free    

    =====================================================================  */

magma_int_t 
magma_z_vfree( magma_z_vector *x ){

    if( x->memory_location == Magma_CPU ){
        free( x->val );
        x->num_rows = 0;
        x->nnz = 0;
        return MAGMA_SUCCESS;     
    }
    if( x->memory_location == Magma_DEV ){
        cublasStatus stat;
        cudaFree( x->val );
        x->num_rows = 0;
        x->nnz = 0;
        if( ( int )stat != 0 ) {
            printf("Memory Free Error.\n");  
            return MAGMA_ERR_INVALID_PTR;
            exit(0);
        }
        else
            return MAGMA_SUCCESS;     
    }
    else{
        printf("Memory Free Error.\n");  
        return MAGMA_ERR_INVALID_PTR;
        exit(0);
    }
}



magma_int_t 
magma_z_mfree( magma_z_sparse_matrix *A ){

    if( A->memory_location == Magma_CPU ){
        if( A->storage_type = Magma_ELLPACK ){
            free( A->val );
            free( A->col );
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0;        
            return MAGMA_SUCCESS;                 
        } 
        if( A->storage_type = Magma_ELLPACKT ){
            free( A->val );
            free( A->col );
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0;        
            return MAGMA_SUCCESS;                 
        } 
        if( A->storage_type = Magma_CSR ){
            free( A->val );
            free( A->col );
            free( A->row );
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0;        
            return MAGMA_SUCCESS;                 
        } 
        if( A->storage_type = Magma_DENSE ){
            free( A->val );
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0;        
            return MAGMA_SUCCESS;                 
        } 
    }

    if( A->memory_location == Magma_DEV ){
       cublasStatus stat;
       if( A->storage_type = Magma_ELLPACK ){
            cudaFree( A->val );
            if( ( int )stat != 0 ) {
                printf("Memory Free Error.\n");  
                return MAGMA_ERR_INVALID_PTR;
                exit(0);
            }
            else
                return MAGMA_SUCCESS; 
            cudaFree( A->col );
            if( ( int )stat != 0 ) {
                printf("Memory Free Error.\n");  
                return MAGMA_ERR_INVALID_PTR;
                exit(0);
            }
            else
                return MAGMA_SUCCESS;

            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0;        
            return MAGMA_SUCCESS;                 
        } 
        if( A->storage_type = Magma_ELLPACKT ){
            cudaFree( A->val );
            if( ( int )stat != 0 ) {
                printf("Memory Free Error.\n");  
                return MAGMA_ERR_INVALID_PTR;
                exit(0);
            }
            else
                return MAGMA_SUCCESS; 
            cudaFree( A->col );
            if( ( int )stat != 0 ) {
                printf("Memory Free Error.\n");  
                return MAGMA_ERR_INVALID_PTR;
                exit(0);
            }
            else
                return MAGMA_SUCCESS;

            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0;        
            return MAGMA_SUCCESS;                 
        } 
        if( A->storage_type = Magma_CSR ){
            cudaFree( A->val );
            if( ( int )stat != 0 ) {
                printf("Memory Free Error.\n");  
                return MAGMA_ERR_INVALID_PTR;
                exit(0);
            }
            else
                return MAGMA_SUCCESS; 
            cudaFree( A->row );
            if( ( int )stat != 0 ) {
                printf("Memory Free Error.\n");  
                return MAGMA_ERR_INVALID_PTR;
                exit(0);
            }
            else
                return MAGMA_SUCCESS;
            cudaFree( A->col );
            if( ( int )stat != 0 ) {
                printf("Memory Free Error.\n");  
                return MAGMA_ERR_INVALID_PTR;
                exit(0);
            }
            else
                return MAGMA_SUCCESS;

            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0;        
            return MAGMA_SUCCESS;                 
        } 
        if( A->storage_type = Magma_DENSE ){
            cudaFree( A->val );
            if( ( int )stat != 0 ) {
                printf("Memory Free Error.\n");  
                return MAGMA_ERR_INVALID_PTR;
                exit(0);
            }
            else
                return MAGMA_SUCCESS; 

            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0;        
            return MAGMA_SUCCESS;                 
        }   
    }

    else{
        printf("Memory Free Error.\n");  
        return MAGMA_ERR_INVALID_PTR;
        exit(0);
    }
}



   


