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
            // memory allocation
            stat = cublasAlloc( A.num_rows * A.max_nnz_row, sizeof( magmaDoubleComplex ), ( void** )&B->val );
            if( ( int )stat != 0 ) {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = cublasAlloc( A.num_rows * A.max_nnz_row, sizeof( magma_int_t ), ( void** )&B->col );
            if( ( int )stat != 0 ) {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            // data transfer
            cublasSetVector( A.num_rows * A.max_nnz_row , sizeof( magmaDoubleComplex ), A.val, 1, B->val, 1 );
            cublasSetVector( A.num_rows * A.max_nnz_row , sizeof( magma_int_t )  , A.col, 1, B->col, 1 ); 
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
            // memory allocation
            B->val = new magmaDoubleComplex[A.nnz];
            B->row = new magma_int_t[A.num_rows+1];
            B->col = new magma_int_t[A.nnz];
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
            // memory allocation
            B->val = new magmaDoubleComplex[A.num_rows*A.max_nnz_row];
            B->col = new magma_int_t[A.num_rows*A.max_nnz_row];
            // data transfer
            for( magma_int_t i=0; i<A.num_rows*A.max_nnz_row; i++ ){
                B->val[i] = A.val[i];
                B->col[i] = A.col[i];
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
            // memory allocation
            B->val = new magmaDoubleComplex[A.nnz];
            B->row = new magma_int_t[A.num_rows+1];
            B->col = new magma_int_t[A.nnz];
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
            // memory allocation
            B->val = new magmaDoubleComplex[A.num_rows*A.max_nnz_row];
            B->col = new magma_int_t[A.num_rows*A.max_nnz_row];
            // data transfer
            cublasGetVector( A.num_rows*A.max_nnz_row, sizeof( magmaDoubleComplex ), A.val, 1, B->val, 1 );
            cublasGetVector( A.num_rows*A.max_nnz_row, sizeof( magma_int_t ), A.col, 1, B->col, 1 );
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
            // memory allocation
            stat = cublasAlloc( A.num_rows * A.max_nnz_row, sizeof( magmaDoubleComplex ), ( void** )&B->val );
            if( ( int )stat != 0 ) {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = cublasAlloc( A.num_rows * A.max_nnz_row, sizeof( magma_int_t ), ( void** )&B->col );
            if( ( int )stat != 0 ) {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            // data transfer
            cudaMemcpy( B->val, A.val, A.num_rows*A.max_nnz_row*sizeof( magmaDoubleComplex ), cudaMemcpyDeviceToDevice );
            cudaMemcpy( B->col, A.col, A.num_rows*A.max_nnz_row*sizeof( magma_int_t ), cudaMemcpyDeviceToDevice );
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
        if( ( int )stat != 0 ) {printf("Memory Allocation Error transferring vector\n"); exit(0); }
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
        y->val = new magmaDoubleComplex[x.num_rows]; 
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
        y->val = new magmaDoubleComplex[x.num_rows]; 
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
        if( ( int )stat != 0 ) {printf("Memory Allocation Error transferring vector\n"); exit(0); }
        // data transfer
        cudaMemcpy( y->val, x.val, x.num_rows*sizeof( magmaDoubleComplex ), cudaMemcpyDeviceToDevice );
    }

    return MAGMA_SUCCESS;
}


