/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> s d c
       @author Hartwig Anzt

*/
#include "magma_lapack.h"
#include "common_magma.h"
#include "../include/magmasparse.h"

#include <assert.h>

// includes CUDA
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <cusparse_v2.h>
#include <cuda_profiler_api.h>

#define RTOLERANCE     lapackf77_dlamch( "E" )
#define ATOLERANCE     lapackf77_dlamch( "E" )

#define min(a, b) ((a) < (b) ? (a) : (b))

/** -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    -------

    Takes an strictly lower triangular matrix L and an upper triangular matrix U
    and merges them into a matrix A containing the upper and lower triangular 
    parts.

    Arguments
    ---------

    @param
    L           magma_z_sparse_matrix
                input strictly lower triangular matrix L

    @param
    U           magma_z_sparse_matrix
                input upper triangular matrix U
    
    @param
    A           magma_z_sparse_matrix*
                output matrix 

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmlumerge(    magma_z_sparse_matrix L, 
                    magma_z_sparse_matrix U,
                    magma_z_sparse_matrix *A){

    if( L.memory_location == Magma_CPU && U.memory_location == Magma_CPU ){
        
        magma_z_mtransfer( L, A, Magma_CPU, Magma_CPU );
        magma_free_cpu( A->col );
        magma_free_cpu( A->val );
        // make sure it is strictly lower triangular
        magma_int_t z = 0;
        for(magma_int_t i=0; i<A->num_rows; i++){
            for(magma_int_t j=L.row[i]; j<L.row[i+1]; j++){
                if( L.col[j] < i ){// make sure it is strictly lower triangular
                    z++;
                }
            }
            for(magma_int_t j=U.row[i]; j<U.row[i+1]; j++){
                z++;
            }
        }
        A->nnz = z;
        // fill A with the new structure;
        magma_index_malloc_cpu( &A->col, A->nnz );
        magma_zmalloc_cpu( &A->val, A->nnz );
        z = 0;
        for(magma_int_t i=0; i<A->num_rows; i++){
            A->row[i] = z;
            for(magma_int_t j=L.row[i]; j<L.row[i+1]; j++){
                if( L.col[j] < i ){// make sure it is strictly lower triangular
                    A->col[z] = L.col[j];
                    A->val[z] = L.val[j];
                    z++;
                }
            }
            for(magma_int_t j=U.row[i]; j<U.row[i+1]; j++){
                A->col[z] = U.col[j];
                A->val[z] = U.val[j];
                z++;
            }
        }
        A->row[A->num_rows] = z;
        A->nnz = z;
        return MAGMA_SUCCESS; 
    }
    else{

        printf("error: matrix not on CPU.\n"); 

        return MAGMA_SUCCESS; 
    }
}





