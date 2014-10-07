/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Hartwig Anzt

*/

#include "common_magma.h"
#include "magmasparse.h"

#if (GPUSHMEM < 200)
   #define BLOCK_SIZE1 256
   #define BLOCK_SIZE2 1
#else
   #define BLOCK_SIZE1 256
   #define BLOCK_SIZE2 1
#endif


// copy nonzeros into new structure
__global__ void 
magma_zmcsrgpu_kernel1( int num_rows,  
                 magmaDoubleComplex *A_val, 
                 magma_index_t *A_rowptr, 
                 magma_index_t *A_colind,
                 magmaDoubleComplex *B_val, 
                 magma_index_t *B_rowptr, 
                 magma_index_t *B_colind ){

    int row = blockIdx.x*blockDim.x+threadIdx.x;
    int j;

    if(row<num_rows){
        magmaDoubleComplex zero = MAGMA_Z_ZERO;
        int start = A_rowptr[ row ];
        int new_location = start;
        int end = A_rowptr[ row+1 ];
        for( j=start; j<end; j++ ){
            if( A_val[j] != zero ){
       //         B_val[new_location] = A_val[j];
       //         B_colind[new_location] = A_colind[j];
                new_location++;
            } 
        }
        // this is not a correctr rowpointer! this is nn_z in this row!
        B_rowptr[ row ] = new_location-start;
    }
}


// generate a valid rowpointer
__global__ void 
magma_zmcsrgpu_kernel2( int num_rows,  
                 magma_index_t *B_rowptr,
                 magma_index_t *A_rowptr ){

    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int j, nnz = 0;

    if( idx == 0 ){
    A_rowptr[ 0 ] = nnz;
        for( j=0; j<num_rows; j++ ){
            nnz+=B_rowptr[ j ];
            A_rowptr[ j+1 ] = nnz;
        }
    }
}



// copy new structure into original matrix
__global__ void 
magma_zmcsrgpu_kernel3( int num_rows,  
                 magmaDoubleComplex *B_val, 
                 magma_index_t *B_rowptr, 
                 magma_index_t *B_colind,
                 magma_index_t *B2_rowptr, 
                 magmaDoubleComplex *A_val, 
                 magma_index_t *A_rowptr, 
                 magma_index_t *A_colind
                                            ){

    int row = blockIdx.x*blockDim.x+threadIdx.x;
    int j, new_location;
    
    if(row<num_rows){
    new_location = A_rowptr[ row ];
        int start = B2_rowptr[ row ];
        int end = B2_rowptr[ row+1 ];
        magmaDoubleComplex zero = MAGMA_Z_ZERO;
        for( j=start; j<end; j++ ){
            if( A_val[j] != zero ){
                B_val[new_location] = A_val[j];
                B_colind[new_location] = A_colind[j];
                new_location++;
            } 
               // A_val[ j ] = B_val[ j ];
               // A_colind[ j ] = B_colind[ j ];
        }
    }
}


/**
    Purpose
    -------

    Removes zeros in a CSR matrix. This is a GPU implementation of the 
    CSR compressor.

    Arguments
    ---------

    @param
    A           magma_z_sparse_matrix*
                input/output matrix 

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmcsrcompressor_gpu( magma_z_sparse_matrix *A ){

    if( A->memory_location == Magma_DEV && A->storage_type == Magma_CSR ){

        magma_int_t stat;
        magma_z_sparse_matrix B, B2;

        stat = magma_index_malloc( &B.row, A->num_rows + 1 );
        if( stat != 0 )
        {printf("Memory Allocation Error for B\n"); exit(0); }

        stat = magma_index_malloc( &B2.row, A->num_rows + 1 );
        if( stat != 0 )
        {printf("Memory Allocation Error for B2\n"); exit(0); }
        magma_index_copyvector( (A->num_rows+1), A->row, 1, B2.row, 1 );

        dim3 grid1( (A->num_rows+BLOCK_SIZE1-1)/BLOCK_SIZE1, 1, 1);  

        // copying the nonzeros into B and write in B.row how many there are
        magma_zmcsrgpu_kernel1<<< grid1, BLOCK_SIZE1, 0, magma_stream >>>
                ( A->num_rows, A->val, A->row, A->col, B.val, B.row, B.col );

        // correct the row pointer
        dim3 grid2( 1, 1, 1);  
        magma_zmcsrgpu_kernel2<<< grid2, BLOCK_SIZE2, 0, magma_stream >>>
                ( A->num_rows, B.row, A->row );
        // access the true number of nonzeros
        magma_index_t *cputmp;
        magma_index_malloc_cpu( &cputmp, 1 );
        magma_index_getvector( 1, A->row+(A->num_rows-1), 1, cputmp, 1 );
        A->nnz = (magma_int_t) cputmp[0];

        // reallocate with right size
        stat = magma_zmalloc( &B.val, A->nnz );
        if( stat != 0 )
            {printf("Memory Allocation Error for A\n"); exit(0); }
        stat = magma_index_malloc( &B.col, A->nnz );
        if( stat != 0 )
            {printf("Memory Allocation Error for A\n"); exit(0); }


        // copy correct values back
        magma_zmcsrgpu_kernel3<<< grid1, BLOCK_SIZE1, 0, magma_stream >>>
                ( A->num_rows, B.val, B.row, B.col, B2.row, A->val, A->row, A->col );

        magma_free( A->col );
        magma_free( A->val );
    A->col = B.col;
    A->val = B.val;

        magma_free( B.row );
        magma_free( B2.row );
       // magma_free( B.col );
       // magma_free( B.val );

        return MAGMA_SUCCESS; 
    }
    else{

        magma_z_sparse_matrix dA, CSRA;
        magma_storage_t A_storage = A->storage_type;
        magma_location_t A_location = A->memory_location;
        magma_z_mconvert( *A, &CSRA, A->storage_type, Magma_CSR );
        magma_z_mtransfer( *A, &dA, A->memory_location, Magma_DEV );

        magma_zmcsrcompressor_gpu( &dA );

        magma_z_mfree( &dA );
        magma_z_mfree( A );
        magma_z_mtransfer( dA, &CSRA, Magma_DEV, A_location );
        magma_z_mconvert( CSRA, A, Magma_CSR, A_storage );
        magma_z_mfree( &dA );
        magma_z_mfree( &CSRA );    

        return MAGMA_SUCCESS; 
    }
}


