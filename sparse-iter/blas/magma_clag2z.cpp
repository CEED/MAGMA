/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

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
#include "../include/magmasparse_z.h"
#include "../../include/magma.h"
#include "../include/mmio.h"
#include "common_magma.h"



using namespace std;












/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======

    convertes magma_c_vector from C to Z

    Arguments
    =========

    magma_c_vector x        input vector descriptor
    magma_z_vector *y       output vector descriptor

    =====================================================================  */

magma_int_t
magma_vector_clag2z( magma_c_vector x, magma_z_vector *y )
{
    magma_int_t *info,*iter;
    if( x.memory_location == Magma_DEV){
        y->memory_location = x.memory_location;
        y->num_rows = x.num_rows;
        y->nnz = x.nnz;
        magma_zmalloc( &y->val, x.num_rows );
        magmablas_clag2z( x.num_rows, 1, x.val, 1, y->val, 1, info );
        return MAGMA_SUCCESS;
    }
    else
        return MAGMA_ERR_NOT_SUPPORTED;
}



/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======

    convertes magma_c_sparse_matrix from C to Z

    Arguments
    =========

    magma_c_sparse_matrix A         input matrix descriptor
    magma_z_sparse_matrix *B        output matrix descriptor

    =====================================================================  */

magma_int_t
magma_sparse_matrix_clag2z( magma_c_sparse_matrix A, magma_z_sparse_matrix *B )
{
    magma_int_t *info, *iter;
    if( A.memory_location == Magma_DEV){
        B->storage_type = A.storage_type;
        B->memory_location = A.memory_location;
        B->num_rows = A.num_rows;
        B->num_cols = A.num_cols;
        B->nnz = A.nnz;
        B->max_nnz_row = A.max_nnz_row;
        if( A.storage_type == Magma_CSR ){
            magma_zmalloc( &B->val, A.nnz );
            magmablas_clag2z( A.nnz, 1, A.val, 1, B->val, 1, info );
            B->row = A.row;
            B->col = A.col;
            return MAGMA_SUCCESS;
        }
        if( A.storage_type == Magma_ELLPACK ){
            magma_zmalloc( &B->val, A.num_rows*A.max_nnz_row );
            magmablas_clag2z( A.num_rows*A.max_nnz_row, 1, A.val, 1, B->val, 1, info );
            B->col = A.col;
            return MAGMA_SUCCESS;
        }
        if( A.storage_type == Magma_ELLPACKT ){
            magma_zmalloc( &B->val, A.num_rows*A.max_nnz_row );
            magmablas_clag2z( A.num_rows*A.max_nnz_row, 1, A.val, 1, B->val, 1, info );
            B->col = A.col;
            return MAGMA_SUCCESS;
        }
        if( A.storage_type == Magma_DENSE ){
            magma_zmalloc( &B->val, A.num_rows*A.num_cols );
            magmablas_clag2z( A.num_rows, A.num_cols, A.val, A.num_rows, B->val, A.num_rows, info );
            return MAGMA_SUCCESS;
        }
        else
            return MAGMA_ERR_NOT_SUPPORTED;
    }
    else
        return MAGMA_ERR_NOT_SUPPORTED;

}































