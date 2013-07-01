/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> c d s
       @author Hartwig Anzt

*/
#include "common_magma.h"
#include "../include/magmasparse_types.h"
#include "../include/magmasparse.h"




/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======

    For a given input matrix A and vectors x, y and scalars alpha, beta
    the wrapper determines the suitable SpMV computing
              y = alpha * A * x + beta * y.  
    Arguments
    =========

    magma_z_sparse_matrix A       sparse matrix A    
    magma_z_vector x              input vector x  
    magma_z_vector y              input vector y      
    magmaDoubleComplex alpha      scalar alpha
    magmaDoublComplex beta        scalar beta

    =====================================================================  */

magma_int_t
magma_z_SpMV(     magmaDoubleComplex alpha, magma_z_sparse_matrix A, 
                magma_z_vector x, magmaDoubleComplex beta, magma_z_vector y ){

    if( A.memory_location != x.memory_location || x.memory_location != y.memory_location ){
        printf(" error: linear algebra objects are not located in same memory!\n");
        // return some MAGMA_ERROR
    }

    // DEV case
    if( A.memory_location == x.memory_location == y.memory_location == Magma_DEV){
         if( A.storage_type == Magma_CSR )
             magma_zgecsrmv( MagmaNoTransStr, A.num_rows, A.num_cols, alpha, A.val, A.row, A.col, x.val, beta, y.val );
         if( A.storage_type == Magma_ELLPACK )
             magma_zgeellmv( MagmaNoTransStr, A.num_rows, A.num_cols, A.max_nnz_row, alpha, A.val, A.col, x.val, beta, y.val );
         if( A.storage_type == Magma_ELLPACKT )
             magma_zgeelltmv( MagmaNoTransStr, A.num_rows, A.num_cols, A.max_nnz_row, alpha, A.val, A.col, x.val, beta, y.val );

    }
    // CPU case missing!
     
 



    return MAGMA_SUCCESS;
}
