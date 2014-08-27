/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions mixed zc -> ds
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
#include "magmasparse_z.h"
#include "magmasparse_zc.h"
#include "magma.h"
#include "mmio.h"
#include "common_magma.h"



using namespace std;


/**
    Purpose
    -------

    convertes magma_c_vector from C to Z

    Arguments
    ---------

    @param
    x           magma_c_vector
                input vector descriptor

    @param
    y           magma_z_vector*
                output vector descriptor

    @ingroup magmasparse_caux
    ********************************************************************/

magma_int_t
magma_vector_clag2z( magma_c_vector x, magma_z_vector *y )
{
    magma_int_t info;
    if( x.memory_location == Magma_DEV){
        y->memory_location = x.memory_location;
        y->num_rows = x.num_rows;
        y->nnz = x.nnz;
        magma_zmalloc( &y->val, x.num_rows );
        magmablas_clag2z( x.num_rows, 1, x.val, x.num_rows, 
                                    y->val, x.num_rows, &info );
        return MAGMA_SUCCESS;
    }
    else if( x.memory_location == Magma_CPU ){
        y->memory_location = x.memory_location;
        y->num_rows = x.num_rows;
        y->nnz = x.nnz;
        magma_zmalloc_cpu( &y->val, x.num_rows );

        magma_int_t one= 1;
        magma_int_t info;
        lapackf77_clag2z( &x.num_rows, &one, 
                       x.val, &x.num_rows, 
                       y->val, &x.num_rows, &info);
        return MAGMA_SUCCESS;

    }
    else
        return MAGMA_ERR_NOT_SUPPORTED;
}



/**
    Purpose
    -------

    convertes magma_c_sparse_matrix from C to Z

    Arguments
    ---------

    @param
    A           magma_c_sparse_matrix
                input matrix descriptor

    @param
    B           magma_z_sparse_matrix*
                output matrix descriptor

    @ingroup magmasparse_caux
    ********************************************************************/

magma_int_t
magma_sparse_matrix_clag2z( magma_c_sparse_matrix A, magma_z_sparse_matrix *B )
{
    magma_int_t info;
    if( A.memory_location == Magma_DEV){
        B->storage_type = A.storage_type;
        B->memory_location = A.memory_location;
        B->num_rows = A.num_rows;
        B->num_cols = A.num_cols;
        B->nnz = A.nnz;
        B->max_nnz_row = A.max_nnz_row;
        if( A.storage_type == Magma_CSR ){
            magma_zmalloc( &B->val, A.nnz );
            magmablas_clag2z_sparse( A.nnz, 1, A.val, A.nnz, 
                                            B->val, A.nnz, &info );
            B->row = A.row;
            B->col = A.col;
            return MAGMA_SUCCESS;
        }
        if( A.storage_type == Magma_ELLPACK ){
            magma_zmalloc( &B->val, A.num_rows*A.max_nnz_row );
            magmablas_clag2z_sparse( A.num_rows*A.max_nnz_row, 1, A.val, 
            A.num_rows*A.max_nnz_row, B->val, A.num_rows*A.max_nnz_row, &info );
            B->col = A.col;
            return MAGMA_SUCCESS;
        }
        if( A.storage_type == Magma_ELL ){
            magma_zmalloc( &B->val, A.num_rows*A.max_nnz_row );
            magmablas_clag2z_sparse( A.num_rows*A.max_nnz_row, 1, A.val, 
            A.num_rows*A.max_nnz_row, B->val, A.num_rows*A.max_nnz_row, &info );
            B->col = A.col;
            return MAGMA_SUCCESS;
        }
        if( A.storage_type == Magma_DENSE ){
            magma_zmalloc( &B->val, A.num_rows*A.num_cols );
            magmablas_clag2z_sparse( A.num_rows, A.num_cols, A.val, A.num_rows, 
                    B->val, A.num_rows, &info );
            return MAGMA_SUCCESS;
        }
        else
            return MAGMA_ERR_NOT_SUPPORTED;
    }
    else
        return MAGMA_ERR_NOT_SUPPORTED;

}

