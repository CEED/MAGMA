/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Hartwig Anzt
*/
#include "magmasparse_internal.h"

#define THRESHOLD 10e-99


/**
    Purpose
    -------

    Takes a matrix and extracts a slice for solving the system in parallel:
    
        B = A( i:i+n, : ) 
        
    The last slice might be smaller. For the non-local parts, it is the identity.
    comm contains 1es in the locations that are non-local but needed to 
    solve local system.


    Arguments
    ---------
    
    @param[in]
    num_slices  magma_int_t
                number of slices
    
    @param[in]
    slice       magma_int_t
                slice id (0.. num_slices-1)

    @param[in]
    A           magma_z_matrix
                sparse matrix in CSR

    @param[out]
    B           magma_z_matrix*
                sparse matrix in CSR
                
   @param[out]          
    comm_i      magma_int_t*
                communication plan
 
    @param[out]          
    comm_v      magmaDoubleComplex*
                communication plan

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmslice(
    magma_int_t num_slices,
    magma_int_t slice,
    magma_z_matrix A, 
    magma_z_matrix *B,
    magma_index_t *comm_i,
    magmaDoubleComplex *comm_v,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    if ( A.memory_location == Magma_CPU
            && A.storage_type == Magma_CSR ){
        CHECK( magma_zmconvert( A, B, Magma_CSR, Magma_CSR, queue ) );
        magma_free_cpu( B->col );
        magma_free_cpu( B->val );
        
        magma_int_t i,j,k, nnz;
        magma_index_t col;
        magma_int_t size = magma_ceildiv( A.num_rows, num_slices ); 
        magma_int_t start = slice*size;
        magma_int_t end = min( (slice+1)*size, A.num_rows );
        // correct size for last slice
        size = end-start;
        
        // count elements - identity for rest
        nnz = A.row[ end ] - A.row[ start ] + ( A.num_rows - size );
        CHECK( magma_index_malloc_cpu( &B->col, nnz ) );
        CHECK( magma_zmalloc_cpu( &B->val, nnz ) );
        
        // for the communication plan
        CHECK( magma_index_malloc_cpu( &comm_i, A.num_rows ) );
        CHECK( magma_zmalloc_cpu( &comm_v, A.num_rows ) );
        for( i=0; i<A.num_rows; i++ ) {
            comm_i[i] = 0;
            comm_v[i] = MAGMA_Z_ZERO;
        }
        
        k=0;
        // identity above slice
        for( i=0; i<start; i++ ) {
            B->row[i+1]   = B->row[i]+1;
            B->val[k] = MAGMA_Z_ONE;
            B->col[k] = i;
            k++;
        }
        
        // slice        
        for( i=start; i<end; i++ ) {
            B->row[i+1]   = B->row[i] + (A.row[i+1]-A.row[i]);
            for( j=A.row[i]; j<A.row[i+1]; j++ ){
                B->val[k] = A.val[j];
                col = A.col[j];
                B->col[k] = col;
                // communication plan
                comm_i[ col ] = 1;
                comm_v[ col ] = comm_v[ col ] 
                                + MAGMA_Z_MAKE( MAGMA_Z_ABS( A.val[j] ), 0.0 );
                k++;
            }
        }
        
        
        // identity below slice
        for( i=end; i<A.num_rows; i++ ) {
            B->row[i+1] = B->row[i]+1;
            B->val[k] = MAGMA_Z_ONE;
            B->col[k] = i;
            k++;
        }
        B->nnz = k;
        
    }
    else {
        printf("error: mslice only supported for CSR matrices on the CPU: %d %d.\n", 
                (int)A.memory_location, (int) A.storage_type );
        info = MAGMA_ERR_NOT_SUPPORTED;
    }
cleanup:
    return info;
}




