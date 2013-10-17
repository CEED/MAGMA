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

    Copies a matrix from memory location src to memory location dst.


    Arguments
    =========

    magma_z_sparse_matrix A              sparse matrix A    
    magma_z_sparse_matrix *B             copy of A      
    magma_location_t src                 original location A
    magma_location_t dst                 location of the copy of A
   

    =====================================================================  */

magma_int_t 
magma_z_mtransfer( magma_z_sparse_matrix A, 
                   magma_z_sparse_matrix *B, 
                   magma_location_t src, 
                   magma_location_t dst){
    cublasStatus stat;

    // first case: copy matrix from host to device
    if( src == Magma_CPU && dst == Magma_DEV ){
        //CSR-type
        if( A.storage_type == Magma_CSR ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->memory_location = Magma_DEV;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            stat = cublasAlloc( A.nnz, sizeof( magmaDoubleComplex ), ( void** )&B->val );
            if( ( int )stat != 0 ) {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = cublasAlloc(  A.num_rows+1 , sizeof( magma_int_t ), ( void** )&B->row );
            if( ( int )stat != 0 ) {printf("Memory Allocation Error transferring matrix\n"); exit(0); } 
            stat = cublasAlloc( A.nnz, sizeof( magma_int_t ), ( void** )&B->col );
            if( ( int )stat != 0 ) {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            // data transfer
            cublasSetVector( A.nnz , sizeof( magmaDoubleComplex ), A.val, 1, B->val, 1 );
            cublasSetVector( A.num_rows+1 , sizeof( magma_int_t )  , A.row, 1, B->row, 1 );
            cublasSetVector( A.nnz , sizeof( magma_int_t )  , A.col, 1, B->col, 1 ); 
        } 
        //ELLPACK-type
        if( A.storage_type == Magma_ELLPACK ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->memory_location = Magma_DEV;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            stat = cublasAlloc( A.num_rows * A.max_nnz_row, sizeof( magmaDoubleComplex ), ( void** )&B->val );
            if( ( int )stat != 0 ) {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = cublasAlloc( A.num_rows * A.max_nnz_row, sizeof( magma_int_t ), ( void** )&B->col );
            if( ( int )stat != 0 ) {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            // data transfer
            cublasSetVector( A.num_rows * A.max_nnz_row , sizeof( magmaDoubleComplex ), A.val, 1, B->val, 1 );
            cublasSetVector( A.num_rows * A.max_nnz_row , sizeof( magma_int_t )  , A.col, 1, B->col, 1 ); 
        } 
        //ELLPACKT-type
        if( A.storage_type == Magma_ELLPACKT ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->memory_location = Magma_DEV;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            stat = cublasAlloc( A.num_rows * A.max_nnz_row, sizeof( magmaDoubleComplex ), ( void** )&B->val );
            if( ( int )stat != 0 ) {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = cublasAlloc( A.num_rows * A.max_nnz_row, sizeof( magma_int_t ), ( void** )&B->col );
            if( ( int )stat != 0 ) {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            // data transfer
            cublasSetVector( A.num_rows * A.max_nnz_row , sizeof( magmaDoubleComplex ), A.val, 1, B->val, 1 );
            cublasSetVector( A.num_rows * A.max_nnz_row , sizeof( magma_int_t )  , A.col, 1, B->col, 1 ); 
        } 
        //BCSR-type
        if( A.storage_type == Magma_BCSR ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->memory_location = Magma_DEV;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            B->blocksize = A.blocksize;
            B->numblocks = A.numblocks;
            magma_int_t size_b = A.blocksize;
            magma_int_t c_blocks = ceil( (float)A.num_cols / (float)size_b );     // max number of blocks per row
            magma_int_t r_blocks = ceil( (float)A.num_rows / (float)size_b );     // max number of blocks per column
            // memory allocation
            stat = cublasAlloc( size_b*size_b*A.numblocks, sizeof( magmaDoubleComplex ), ( void** )&B->val );
            if( ( int )stat != 0 ) {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = cublasAlloc(  r_blocks+1 , sizeof( magma_int_t ), ( void** )&B->row );
            if( ( int )stat != 0 ) {printf("Memory Allocation Error transferring matrix\n"); exit(0); } 
            stat = cublasAlloc( A.numblocks, sizeof( magma_int_t ), ( void** )&B->col );
            if( ( int )stat != 0 ) {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            magma_imalloc_cpu( &B->blockinfo, r_blocks * c_blocks );
            // data transfer
            cublasSetVector( size_b*size_b*A.numblocks , sizeof( magmaDoubleComplex ), A.val, 1, B->val, 1 );
            cublasSetVector( r_blocks+1 , sizeof( magma_int_t )  , A.row, 1, B->row, 1 );
            cublasSetVector( A.numblocks , sizeof( magma_int_t )  , A.col, 1, B->col, 1 ); 
            for( magma_int_t i=0; i<r_blocks * c_blocks; i++ ){
                B->blockinfo[i] = A.blockinfo[i];
            }
        }
        //DENSE-type
        if( A.storage_type == Magma_DENSE ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->memory_location = Magma_DEV;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            stat = cublasAlloc( A.num_rows*A.num_cols, sizeof( magmaDoubleComplex ), ( void** )&B->val );
            if( ( int )stat != 0 ) {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            // data transfer
            cublasSetVector( A.num_rows*A.num_cols , sizeof( magmaDoubleComplex ), A.val, 1, B->val, 1 );
        }
    }

    // second case: copy matrix from host to host
    if( src == Magma_CPU && dst == Magma_CPU ){
        //CSR-type
        if( A.storage_type == Magma_CSR ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->memory_location = Magma_CPU;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            magma_zmalloc_cpu( &B->val, A.nnz );
            magma_imalloc_cpu( &B->row, A.num_rows+1 );
            magma_imalloc_cpu( &B->col, A.nnz );
            // data transfer
            for( magma_int_t i=0; i<A.nnz; i++ ){
                B->val[i] = A.val[i];
                B->col[i] = A.col[i];
            }
            for( magma_int_t i=0; i<A.num_rows+1; i++ ){
                B->row[i] = A.row[i];
            }
        } 
        //ELLPACK-type
        if( A.storage_type == Magma_ELLPACK ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->memory_location = Magma_CPU;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            magma_zmalloc_cpu( &B->val, A.num_rows*A.max_nnz_row );
            magma_imalloc_cpu( &B->col, A.num_rows*A.max_nnz_row );
            // data transfer
            for( magma_int_t i=0; i<A.num_rows*A.max_nnz_row; i++ ){
                B->val[i] = A.val[i];
                B->col[i] = A.col[i];
            }
        }
        //ELLPACKT-type
        if( A.storage_type == Magma_ELLPACKT ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->memory_location = Magma_CPU;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            magma_zmalloc_cpu( &B->val, A.num_rows*A.max_nnz_row );
            magma_imalloc_cpu( &B->col, A.num_rows*A.max_nnz_row );
            // data transfer
            for( magma_int_t i=0; i<A.num_rows*A.max_nnz_row; i++ ){
                B->val[i] = A.val[i];
                B->col[i] = A.col[i];
            }
        } 
        //DENSE-type
        if( A.storage_type == Magma_DENSE ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->memory_location = Magma_CPU;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            magma_zmalloc_cpu( &B->val, A.num_rows*A.num_cols );
            // data transfer
            for( magma_int_t i=0; i<A.num_rows*A.num_cols; i++ ){
                B->val[i] = A.val[i];
            }
        }
    }

    // third case: copy matrix from device to host
    if( src == Magma_DEV && dst == Magma_CPU ){
        //CSR-type
        if( A.storage_type == Magma_CSR ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->memory_location = Magma_CPU;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            magma_zmalloc_cpu( &B->val, A.nnz );
            magma_imalloc_cpu( &B->row, A.num_rows+1 );
            magma_imalloc_cpu( &B->col, A.nnz );
            // data transfer
            cublasGetVector( A.nnz, sizeof( magmaDoubleComplex ), A.val, 1, B->val, 1 );
            cublasGetVector( A.num_rows+1, sizeof( magma_int_t ), A.row, 1, B->row, 1 );            
            cublasGetVector( A.nnz, sizeof( magma_int_t ), A.col, 1, B->col, 1 );
        } 
        //ELLPACK-type
        if( A.storage_type == Magma_ELLPACK ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->memory_location = Magma_CPU;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            magma_zmalloc_cpu( &B->val, A.num_rows*A.max_nnz_row );
            magma_imalloc_cpu( &B->col, A.num_rows*A.max_nnz_row );
            // data transfer
            cublasGetVector( A.num_rows*A.max_nnz_row, sizeof( magmaDoubleComplex ), A.val, 1, B->val, 1 );
            cublasGetVector( A.num_rows*A.max_nnz_row, sizeof( magma_int_t ), A.col, 1, B->col, 1 );
        } 
        //ELLPACKT-type
        if( A.storage_type == Magma_ELLPACKT ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->memory_location = Magma_CPU;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            magma_zmalloc_cpu( &B->val, A.num_rows*A.max_nnz_row );
            magma_imalloc_cpu( &B->col, A.num_rows*A.max_nnz_row );
            // data transfer
            cublasGetVector( A.num_rows*A.max_nnz_row, sizeof( magmaDoubleComplex ), A.val, 1, B->val, 1 );
            cublasGetVector( A.num_rows*A.max_nnz_row, sizeof( magma_int_t ), A.col, 1, B->col, 1 );
        } 
        //BCSR-type
        if( A.storage_type == Magma_BCSR ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->memory_location = Magma_CPU;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            B->blocksize = A.blocksize;
            B->numblocks = A.numblocks;
            magma_int_t size_b = A.blocksize;
            magma_int_t c_blocks = ceil( (float)A.num_cols / (float)size_b );     // max number of blocks per row
            magma_int_t r_blocks = ceil( (float)A.num_rows / (float)size_b );     // max number of blocks per column
            // memory allocation
            magma_zmalloc_cpu( &B->val, A.numblocks * A.blocksize * A.blocksize );
            magma_imalloc_cpu( &B->row, r_blocks+1 );
            magma_imalloc_cpu( &B->col, A.numblocks );
            magma_imalloc_cpu( &B->blockinfo, r_blocks * c_blocks );
            // data transfer
            cublasGetVector( A.numblocks * A.blocksize * A.blocksize, sizeof( magmaDoubleComplex ), A.val, 1, B->val, 1 );
            cublasGetVector( r_blocks+1, sizeof( magma_int_t ), A.row, 1, B->row, 1 );            
            cublasGetVector(  A.numblocks, sizeof( magma_int_t ), A.col, 1, B->col, 1 );
            for( magma_int_t i=0; i<r_blocks * c_blocks; i++ ){
                B->blockinfo[i] = A.blockinfo[i];
            }
        } 
        //DENSE-type
        if( A.storage_type == Magma_DENSE ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->memory_location = Magma_CPU;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            magma_zmalloc_cpu( &B->val, A.num_rows*A.num_cols );
            // data transfer
            cublasGetVector( A.num_rows*A.num_cols, sizeof( magmaDoubleComplex ), A.val, 1, B->val, 1 );
        }
    }

    // fourth case: copy matrix from device to device
    if( src == Magma_DEV && dst == Magma_DEV ){
        //CSR-type
        if( A.storage_type == Magma_CSR ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->memory_location = Magma_DEV;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            stat = cublasAlloc( A.nnz, sizeof( magmaDoubleComplex ), ( void** )&B->val );
            if( ( int )stat != 0 ) {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = cublasAlloc(  A.num_rows+1 , sizeof( magma_int_t ), ( void** )&B->row );
            if( ( int )stat != 0 ) {printf("Memory Allocation Error transferring matrix\n"); exit(0); } 
            stat = cublasAlloc( A.nnz, sizeof( magma_int_t ), ( void** )&B->col );
            if( ( int )stat != 0 ) {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            // data transfer
            cudaMemcpy( B->val, A.val, A.nnz*sizeof( magmaDoubleComplex ), cudaMemcpyDeviceToDevice );
            cudaMemcpy( B->row, A.row, (A.num_rows+1)*sizeof( magma_int_t ), cudaMemcpyDeviceToDevice );
            cudaMemcpy( B->col, A.col, A.nnz*sizeof( magma_int_t ), cudaMemcpyDeviceToDevice );
        } 
        //ELLPACK-type
        if( A.storage_type == Magma_ELLPACK ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->memory_location = Magma_DEV;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            stat = cublasAlloc( A.num_rows * A.max_nnz_row, sizeof( magmaDoubleComplex ), ( void** )&B->val );
            if( ( int )stat != 0 ) {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = cublasAlloc( A.num_rows * A.max_nnz_row, sizeof( magma_int_t ), ( void** )&B->col );
            if( ( int )stat != 0 ) {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            // data transfer
            cudaMemcpy( B->val, A.val, A.num_rows*A.max_nnz_row*sizeof( magmaDoubleComplex ), cudaMemcpyDeviceToDevice );
            cudaMemcpy( B->col, A.col, A.num_rows*A.max_nnz_row*sizeof( magma_int_t ), cudaMemcpyDeviceToDevice );
        } 
        //ELLPACKT-type
        if( A.storage_type == Magma_ELLPACKT ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->memory_location = Magma_DEV;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            stat = cublasAlloc( A.num_rows * A.max_nnz_row, sizeof( magmaDoubleComplex ), ( void** )&B->val );
            if( ( int )stat != 0 ) {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = cublasAlloc( A.num_rows * A.max_nnz_row, sizeof( magma_int_t ), ( void** )&B->col );
            if( ( int )stat != 0 ) {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            // data transfer
            cudaMemcpy( B->val, A.val, A.num_rows*A.max_nnz_row*sizeof( magmaDoubleComplex ), cudaMemcpyDeviceToDevice );
            cudaMemcpy( B->col, A.col, A.num_rows*A.max_nnz_row*sizeof( magma_int_t ), cudaMemcpyDeviceToDevice );
        } 
        //BCSR-type
        if( A.storage_type == Magma_BCSR ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->memory_location = Magma_DEV;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            B->blocksize = A.blocksize;
            B->numblocks = A.numblocks;
            magma_int_t size_b = A.blocksize;
            magma_int_t c_blocks = ceil( (float)A.num_cols / (float)size_b );     // max number of blocks per row
            magma_int_t r_blocks = ceil( (float)A.num_rows / (float)size_b );     // max number of blocks per column
            // memory allocation
            stat = cublasAlloc( size_b*size_b*A.numblocks, sizeof( magmaDoubleComplex ), ( void** )&B->val );
            if( ( int )stat != 0 ) {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = cublasAlloc(  r_blocks+1 , sizeof( magma_int_t ), ( void** )&B->row );
            if( ( int )stat != 0 ) {printf("Memory Allocation Error transferring matrix\n"); exit(0); } 
            stat = cublasAlloc( A.numblocks, sizeof( magma_int_t ), ( void** )&B->col );
            if( ( int )stat != 0 ) {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            magma_imalloc_cpu( &B->blockinfo, r_blocks * c_blocks );
            // data transfer
            cudaMemcpy( B->val, A.val, size_b*size_b*A.numblocks*sizeof( magmaDoubleComplex ), cudaMemcpyDeviceToDevice );
            cudaMemcpy( B->row, A.row, (r_blocks+1)*sizeof( magma_int_t ), cudaMemcpyDeviceToDevice );
            cudaMemcpy( B->col, A.col, A.numblocks*sizeof( magma_int_t ), cudaMemcpyDeviceToDevice );
            for( magma_int_t i=0; i<r_blocks * c_blocks; i++ ){
                B->blockinfo[i] = A.blockinfo[i];
            }
        }
        //DENSE-type
        if( A.storage_type == Magma_DENSE ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->memory_location = Magma_DEV;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            stat = cublasAlloc( A.num_rows*A.num_cols, sizeof( magmaDoubleComplex ), ( void** )&B->val );
            if( ( int )stat != 0 ) {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            // data transfer
            cudaMemcpy( B->val, A.val, A.num_rows*A.num_cols*sizeof( magmaDoubleComplex ), cudaMemcpyDeviceToDevice );
        }
    }


    return MAGMA_SUCCESS;
}

























/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======

    Copies a vector from memory location src to memory location dst.


    Arguments
    =========

    magma_z_vector x              vector x    
    magma_z_vector *y             copy of x      
    magma_location_t src          original location x
    magma_location_t dst          location of the copy of x
   

    =====================================================================  */

magma_int_t 
magma_z_vtransfer( magma_z_vector x, 
                   magma_z_vector *y, 
                   magma_location_t src, 
                   magma_location_t dst){

    cublasStatus stat;

    // first case: copy matrix from host to device
    if( src == Magma_CPU && dst == Magma_DEV ){
        // fill in information for B
        y->memory_location = Magma_DEV;
        y->num_rows = x.num_rows;
        y->nnz = x.nnz;
        // memory allocation
        stat = cublasAlloc( x.num_rows, sizeof( magmaDoubleComplex ), ( void** )&y->val );
        if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring vector\n"); exit(0); }
        // data transfer
        cublasSetVector( x.num_rows , sizeof( magmaDoubleComplex ), x.val, 1, y->val, 1 );
    }
    // second case: copy matrix from host to host
    if( src == Magma_CPU && dst == Magma_CPU ){
        // fill in information for B
        y->memory_location = Magma_DEV;
        y->num_rows = x.num_rows;
        y->nnz = x.nnz;
        // memory allocation
        magma_zmalloc_cpu( &y->val, x.num_rows );
        // data transfer
        for( magma_int_t i=0; i<x.num_rows; i++ )
            y->val[i] = x.val[i];
    }
    // third case: copy matrix from device to host
    if( src == Magma_DEV && dst == Magma_CPU ){
        // fill in information for B
        y->memory_location = Magma_DEV;
        y->num_rows = x.num_rows;
        y->nnz = x.nnz;
        // memory allocation
        magma_zmalloc_cpu( &y->val, x.num_rows );
        // data transfer
        cublasGetVector( x.num_rows, sizeof( magmaDoubleComplex ), x.val, 1, y->val, 1 );
    }
    // fourth case: copy matrix from device to device
    if( src == Magma_DEV && dst == Magma_DEV ){
        // fill in information for B
        y->memory_location = Magma_DEV;
        y->num_rows = x.num_rows;
        y->nnz = x.nnz;
        // memory allocation
        stat = cublasAlloc( x.num_rows, sizeof( magmaDoubleComplex ), ( void** )&y->val );
        if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring vector\n"); exit(0); }
        // data transfer
        cudaMemcpy( y->val, x.val, x.num_rows*sizeof( magmaDoubleComplex ), cudaMemcpyDeviceToDevice );
    }

    return MAGMA_SUCCESS;
}


