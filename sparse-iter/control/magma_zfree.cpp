/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Hartwig Anzt
*/

// includes, project
#include "common_magma.h"
#include "magmasparse_z.h"
#include "magma.h"
#include "mmio.h"








/**
    Purpose
    -------

    Free the memory of a magma_z_matrix.


    Arguments
    ---------

    @param[in,out]
    A           magma_z_matrix*
                matrix to free    
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmfree(
    magma_z_matrix *A,
    magma_queue_t queue )
{
    if ( A->memory_location == Magma_CPU ) {
       if ( A->storage_type == Magma_ELL || A->storage_type == Magma_ELLPACKT ){
            magma_free_cpu( A->val );
            magma_free_cpu( A->col );
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0;                      
        } 
        if (A->storage_type == Magma_ELLD ) {
            magma_free_cpu( A->val );
            magma_free_cpu( A->col );
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0;                       
        } 
        if ( A->storage_type == Magma_ELLRT ) {
            magma_free_cpu( A->val );
            magma_free_cpu( A->row );
            magma_free_cpu( A->col );
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0;                        
        } 
        if ( A->storage_type == Magma_SELLP ) {
            magma_free_cpu( A->val );
            magma_free_cpu( A->row );
            magma_free_cpu( A->col );
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0;                      
        } 
        if ( A->storage_type == Magma_CSR || A->storage_type == Magma_CSC 
                                        || A->storage_type == Magma_CSRD
                                        || A->storage_type == Magma_CSRL
                                        || A->storage_type == Magma_CSRU ) {
            magma_free_cpu( A->val );
            magma_free_cpu( A->col );
            magma_free_cpu( A->row );
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0;                      
        } 
        if (  A->storage_type == Magma_CSRCOO ) {
            magma_free_cpu( A->val );
            magma_free_cpu( A->col );
            magma_free_cpu( A->row );
            magma_free_cpu( A->rowidx );
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0;                     
        } 
        if ( A->storage_type == Magma_BCSR ) {
            magma_free_cpu( A->val );
            magma_free_cpu( A->col );
            magma_free_cpu( A->row );
            magma_free_cpu( A->blockinfo );
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; 
            A->blockinfo = 0;                    
        } 
        if ( A->storage_type == Magma_DENSE ) {
            magma_free_cpu( A->val );
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0;                      
        } 
        A->val = NULL;
        A->col = NULL;
        A->row = NULL;
        A->rowidx = NULL;
        A->blockinfo = NULL;
        A->diag = NULL;
        A->dval = NULL;
        A->dcol = NULL;
        A->drow = NULL;
        A->drowidx = NULL;
        A->ddiag = NULL;
        return MAGMA_SUCCESS; 
    }

    if ( A->memory_location == Magma_DEV ) {
       if ( A->storage_type == Magma_ELL || A->storage_type == Magma_ELLPACKT ){
            if ( magma_free( A->dval ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");  
                return MAGMA_ERR_INVALID_PTR;
                
            }
            if ( magma_free( A->dcol ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");  
                return MAGMA_ERR_INVALID_PTR;
                
            }

            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0;                    
        } 
        if ( A->storage_type == Magma_ELLD ) {
            if ( magma_free( A->dval ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");  
                return MAGMA_ERR_INVALID_PTR;
                
            }
            if ( magma_free( A->dcol ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");  
                return MAGMA_ERR_INVALID_PTR;
                
            }

            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0;                   
        } 
        if ( A->storage_type == Magma_ELLRT ) {
            if ( magma_free( A->dval ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");  
                return MAGMA_ERR_INVALID_PTR;
                
            }
            if ( magma_free( A->drow ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");  
                return MAGMA_ERR_INVALID_PTR;
                
            }
            if ( magma_free( A->dcol ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");  
                return MAGMA_ERR_INVALID_PTR;
                
            }

            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0;                         
        } 
        if ( A->storage_type == Magma_SELLP ) {
            if ( magma_free( A->dval ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");  
                return MAGMA_ERR_INVALID_PTR;
                
            }
            if ( magma_free( A->drow ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");  
                return MAGMA_ERR_INVALID_PTR;
                
            }
            if ( magma_free( A->dcol ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");  
                return MAGMA_ERR_INVALID_PTR;
                
            }
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0;                  
        } 
        if ( A->storage_type == Magma_CSR || A->storage_type == Magma_CSC 
                                        || A->storage_type == Magma_CSRD
                                        || A->storage_type == Magma_CSRL
                                        || A->storage_type == Magma_CSRU ) {
            if ( magma_free( A->dval ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");  
                return MAGMA_ERR_INVALID_PTR;
                
            }
            if ( magma_free( A->drow ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");  
                return MAGMA_ERR_INVALID_PTR;
                
            }
            if ( magma_free( A->dcol ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");  
                return MAGMA_ERR_INVALID_PTR;
                
            }

            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0;                    
        } 
        if (  A->storage_type == Magma_CSRCOO ) {
            if ( magma_free( A->dval ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");  
                return MAGMA_ERR_INVALID_PTR;
                
            }
            if ( magma_free( A->drow ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");  
                return MAGMA_ERR_INVALID_PTR;
                
            }
            if ( magma_free( A->dcol ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");  
                return MAGMA_ERR_INVALID_PTR;
                
            }
            if ( magma_free( A->drowidx ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");  
                return MAGMA_ERR_INVALID_PTR;
                
            }

            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0;                      
        } 
        if ( A->storage_type == Magma_BCSR ) {
            if ( magma_free( A->dval ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");  
                return MAGMA_ERR_INVALID_PTR;
                
            }
            if ( magma_free( A->drow ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");  
                return MAGMA_ERR_INVALID_PTR;
                
            }
            if ( magma_free( A->dcol ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");  
                return MAGMA_ERR_INVALID_PTR;
                
            }
            free( A->blockinfo );
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0;                  
        } 
        if ( A->storage_type == Magma_DENSE ) {
            if ( magma_free( A->dval ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");  
                return MAGMA_ERR_INVALID_PTR;
                
            }

            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0;        
                
        }   
        A->val = NULL;
        A->col = NULL;
        A->row = NULL;
        A->rowidx = NULL;
        A->blockinfo = NULL;
        A->diag = NULL;
        A->dval = NULL;
        A->dcol = NULL;
        A->drow = NULL;
        A->drowidx = NULL;
        A->ddiag = NULL;
        return MAGMA_SUCCESS; 
    }

    else {
        printf("Memory Free Error.\n");  
        return MAGMA_ERR_INVALID_PTR;
    }
    return MAGMA_SUCCESS;
}



   


