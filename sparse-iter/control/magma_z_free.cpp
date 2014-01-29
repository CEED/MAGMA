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
#include "../include/magmasparse_z.h"
#include "../../include/magma.h"
#include "../include/mmio.h"



using namespace std;








/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

    Purpose
    =======

    Free the memory of a magma_z_vector.


    Arguments
    =========

    magma_z_vector *x                     vector to free    

    ========================================================================  */

magma_int_t 
magma_z_vfree( magma_z_vector *x ){

    if( x->memory_location == Magma_CPU ){
        magma_free_cpu( x->val );
        x->num_rows = 0;
        x->nnz = 0;
        return MAGMA_SUCCESS;     
    }
    else if( x->memory_location == Magma_DEV ){
        magma_free( x->val );
        
        x->num_rows = 0;
        x->nnz = 0;

        return MAGMA_SUCCESS;     
    }
    else{
        printf("Memory Free Error.\n");  
        return MAGMA_ERR_INVALID_PTR;
    }
}


/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

    Purpose
    =======

    Free the memory of a magma_z_sparse_matrix.


    Arguments
    =========

    magma_z_sparse_matrix *A                     matrix to free    

    ========================================================================  */

magma_int_t 
magma_z_mfree( magma_z_sparse_matrix *A ){

    if( A->memory_location == Magma_CPU ){
        if( A->storage_type == Magma_ELLPACK ){
            free( A->val );
            free( A->col );
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0;        
            return MAGMA_SUCCESS;                 
        } 
        if( A->storage_type == Magma_ELLPACKT ){
            free( A->val );
            free( A->col );
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0;        
            return MAGMA_SUCCESS;                 
        } 
        if( A->storage_type == Magma_ELLPACKRT ){
            free( A->val );
            free( A->row );
            free( A->col );
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0;        
            return MAGMA_SUCCESS;                 
        } 
        if( A->storage_type == Magma_SELLC ){
            free( A->val );
            free( A->row );
            free( A->col );
            free( A->blockinfo );
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0;        
            return MAGMA_SUCCESS;                 
        } 
        if( A->storage_type == Magma_CSR ){
            free( A->val );
            free( A->col );
            free( A->row );
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0;        
            return MAGMA_SUCCESS;                 
        } 
        if( A->storage_type == Magma_BCSR ){
            free( A->val );
            free( A->col );
            free( A->row );
            free( A->blockinfo );
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; 
            A->blockinfo = 0;        
            return MAGMA_SUCCESS;                 
        } 
        if( A->storage_type == Magma_DENSE ){
            free( A->val );
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0;        
            return MAGMA_SUCCESS;                 
        } 
    }

    if( A->memory_location == Magma_DEV ){
       if( A->storage_type == Magma_ELLPACK ){
            if( cudaFree( A->val ) != cudaSuccess ) {
                printf("Memory Free Error.\n");  
                return MAGMA_ERR_INVALID_PTR;
                exit(0);
            }
            if( cudaFree( A->col ) != cudaSuccess ) {
                printf("Memory Free Error.\n");  
                return MAGMA_ERR_INVALID_PTR;
                exit(0);
            }

            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0;        
            return MAGMA_SUCCESS;                 
        } 
        if( A->storage_type == Magma_ELLPACKT ){
            if( cudaFree( A->val ) != cudaSuccess ) {
                printf("Memory Free Error.\n");  
                return MAGMA_ERR_INVALID_PTR;
                exit(0);
            }
            if( cudaFree( A->col ) != cudaSuccess ) {
                printf("Memory Free Error.\n");  
                return MAGMA_ERR_INVALID_PTR;
                exit(0);
            }

            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0;        
            return MAGMA_SUCCESS;                 
        } 
        if( A->storage_type == Magma_ELLPACKRT ){
            if( cudaFree( A->val ) != cudaSuccess ) {
                printf("Memory Free Error.\n");  
                return MAGMA_ERR_INVALID_PTR;
                exit(0);
            }
            if( cudaFree( A->row ) != cudaSuccess ) {
                printf("Memory Free Error.\n");  
                return MAGMA_ERR_INVALID_PTR;
                exit(0);
            }
            if( cudaFree( A->col ) != cudaSuccess ) {
                printf("Memory Free Error.\n");  
                return MAGMA_ERR_INVALID_PTR;
                exit(0);
            }

            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0;        
            return MAGMA_SUCCESS;                 
        } 
        if( A->storage_type == Magma_SELLC ){
            if( cudaFree( A->val ) != cudaSuccess ) {
                printf("Memory Free Error.\n");  
                return MAGMA_ERR_INVALID_PTR;
                exit(0);
            }
            if( cudaFree( A->row ) != cudaSuccess ) {
                printf("Memory Free Error.\n");  
                return MAGMA_ERR_INVALID_PTR;
                exit(0);
            }
            if( cudaFree( A->col ) != cudaSuccess ) {
                printf("Memory Free Error.\n");  
                return MAGMA_ERR_INVALID_PTR;
                exit(0);
            }
            if( cudaFree( A->blockinfo ) != cudaSuccess ) {
                printf("Memory Free Error.\n");  
                return MAGMA_ERR_INVALID_PTR;
                exit(0);
            }
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0;        
            return MAGMA_SUCCESS;                 
        } 
        if( A->storage_type == Magma_CSR ){
            if( cudaFree( A->val ) != cudaSuccess ) {
                printf("Memory Free Error.\n");  
                return MAGMA_ERR_INVALID_PTR;
                exit(0);
            }
            if( cudaFree( A->row ) != cudaSuccess ) {
                printf("Memory Free Error.\n");  
                return MAGMA_ERR_INVALID_PTR;
                exit(0);
            }
            if( cudaFree( A->col ) != cudaSuccess ) {
                printf("Memory Free Error.\n");  
                return MAGMA_ERR_INVALID_PTR;
                exit(0);
            }

            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0;        
            return MAGMA_SUCCESS;                 
        } 
        if( A->storage_type == Magma_BCSR ){
            if( cudaFree( A->val ) != cudaSuccess ) {
                printf("Memory Free Error.\n");  
                return MAGMA_ERR_INVALID_PTR;
                exit(0);
            }
            if( cudaFree( A->row ) != cudaSuccess ) {
                printf("Memory Free Error.\n");  
                return MAGMA_ERR_INVALID_PTR;
                exit(0);
            }
            if( cudaFree( A->col ) != cudaSuccess ) {
                printf("Memory Free Error.\n");  
                return MAGMA_ERR_INVALID_PTR;
                exit(0);
            }
            free( A->blockinfo );
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0;        
            return MAGMA_SUCCESS;                 
        } 
        if( A->storage_type == Magma_DENSE ){
            if( cudaFree( A->val ) != cudaSuccess ) {
                printf("Memory Free Error.\n");  
                return MAGMA_ERR_INVALID_PTR;
                exit(0);
            }

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
    return MAGMA_SUCCESS;                 
}



   


