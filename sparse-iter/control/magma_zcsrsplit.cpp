/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Hartwig Anzt

*/
#include "magma_lapack.h"
#include "common_magma.h"
#include "magmasparse.h"

#include <assert.h>

// includes CUDA
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <cusparse_v2.h>
#include <cuda_profiler_api.h>


/**
    Purpose
    -------

    Splits a CSR matrix into two matrices, one containing the diagonal blocks
    with the diagonal element stored first, one containing the rest of the 
    original matrix.

    Arguments
    ---------

    @param[in]
    bsize       magma_int_t
                size of the diagonal blocks

    @param[in]
    A           magma_z_matrix
                CSR input matrix

    @param[out]
    D           magma_z_matrix*
                CSR matrix containing diagonal blocks

    @param[out]
    R           magma_z_matrix*
                CSR matrix containing rest
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_saux
    ********************************************************************/

extern "C" magma_int_t
magma_zcsrsplit(
    magma_int_t bsize,
    magma_z_matrix A,
    magma_z_matrix *D,
    magma_z_matrix *R,
    magma_queue_t queue )
{
    if (  A.memory_location == Magma_CPU &&
            (   A.storage_type == Magma_CSR ||
                A.storage_type == Magma_CSRCOO ) ) {

        magma_int_t i, k, j, nnz_diag, nnz_offd;
        
        magma_int_t stat_cpu = 0;
        D->val = NULL;
        D->col = NULL;
        D->row = NULL;
        D->rowidx = NULL;
        D->blockinfo = NULL;
        D->diag = NULL;
        D->dval = NULL;
        D->dcol = NULL;
        D->drow = NULL;
        D->drowidx = NULL;
        D->ddiag = NULL;
        R->val = NULL;
        R->col = NULL;
        R->row = NULL;
        R->rowidx = NULL;
        R->blockinfo = NULL;
        R->diag = NULL;
        R->dval = NULL;
        R->dcol = NULL;
        R->drow = NULL;
        R->drowidx = NULL;
        R->ddiag = NULL;

        nnz_diag = nnz_offd = 0;
        // Count the new number of nonzeroes in the two matrices
        for( i=0; i<A.num_rows; i+=bsize ){
            for( k=i; k<min(A.num_rows,i+bsize); k++ ){
                int check = 0;
                for( j=A.row[k]; j<A.row[k+1]; j++ ){
                    if ( A.col[j] < i )
                        nnz_offd++;
                    else if ( A.col[j] < i+bsize ){
                        if( A.col[j] == k ){
                            check = 1;
                        }
                        nnz_diag++;
                    }
                    else
                        nnz_offd++;
                }
                if( check == 0 ){
                    printf("error: matrix contains zero on diagonal at (%d,%d).\n", i, i);
                    magma_zmfree( &A, queue );
                    exit(-1);
                }
            }
        }

        // Allocate memory for the new matrices
        D->storage_type = Magma_CSRD;
        D->memory_location = A.memory_location;
        D->num_rows = A.num_rows;
        D->num_cols = A.num_cols;
        D->nnz = nnz_diag;

        R->storage_type = Magma_CSR;
        R->memory_location = A.memory_location;
        R->num_rows = A.num_rows;
        R->num_cols = A.num_cols;
        R->nnz = nnz_offd;

        stat_cpu += magma_zmalloc_cpu( &D->val, nnz_diag );
        stat_cpu += magma_index_malloc_cpu( &D->row, A.num_rows+1 );
        stat_cpu += magma_index_malloc_cpu( &D->col, nnz_diag );

        stat_cpu += magma_zmalloc_cpu( &R->val, nnz_offd );
        stat_cpu += magma_index_malloc_cpu( &R->row, A.num_rows+1 );
        stat_cpu += magma_index_malloc_cpu( &R->col, nnz_offd );
        
        if( stat_cpu != 0 ){
            magma_zmfree( D, queue );
            magma_zmfree( R, queue );
            return MAGMA_ERR_HOST_ALLOC;
        }
        

        // Fill up the new sparse matrices  
        D->row[0] = 0;
        R->row[0] = 0;

        nnz_offd = nnz_diag = 0;
        for( i=0; i<A.num_rows; i+=bsize) {
            for( k=i; k<min(A.num_rows,i+bsize); k++ ) {
                D->row[k+1] = D->row[k];
                R->row[k+1] = R->row[k];
     
                for( j=A.row[k]; j<A.row[k+1]; j++ ) {
                    if ( A.col[j] < i ) {
                        R->val[nnz_offd] = A.val[j];
                        R->col[nnz_offd] = A.col[j];
                        R->row[k+1]++;  
                        nnz_offd++;
                    }
                    else if ( A.col[j] < i+bsize ) {
                        // larger than diagonal remain as before
                        if ( A.col[j]>k ) {
                            D->val[nnz_diag] = A.val[ j ];
                            D->col[nnz_diag] = A.col[ j ];
                            D->row[k+1]++;
                        }
                        // diagonal is written first
                        else if ( A.col[j]==k ) {
                            D->val[D->row[k]] = A.val[ j ];
                            D->col[D->row[k]] = A.col[ j ];
                            D->row[k+1]++;
                        }
                        // smaller than diagonal are shifted one to the right 
                        // to have room for the diagonal
                        else {
                            D->val[nnz_diag+1] = A.val[ j ];
                            D->col[nnz_diag+1] = A.col[ j ];
                            D->row[k+1]++;
                        }
                        nnz_diag++;
                    }
                    else {
                        R->val[nnz_offd] = A.val[j];
                        R->col[nnz_offd] = A.col[j];
                        R->row[k+1]++;  
                        nnz_offd++;
                    }
                }
            }
        }
        return MAGMA_SUCCESS; 
    }
    else {
        magma_z_matrix Ah, ACSR, DCSR, RCSR, Dh, Rh;
        magma_zmtransfer( A, &Ah, A.memory_location, Magma_CPU, queue );
        magma_zmconvert( Ah, &ACSR, A.storage_type, Magma_CSR, queue );

        magma_zcsrsplit( bsize, ACSR, &DCSR, &RCSR, queue );

        magma_zmconvert( DCSR, &Dh, Magma_CSR, A.storage_type, queue );
        magma_zmconvert( RCSR, &Rh, Magma_CSR, A.storage_type, queue );

        magma_zmtransfer( Dh, D, Magma_CPU, A.memory_location, queue );
        magma_zmtransfer( Rh, R, Magma_CPU, A.memory_location, queue );

        magma_zmfree( &Ah, queue );
        magma_zmfree( &ACSR, queue );
        magma_zmfree( &Dh, queue );
        magma_zmfree( &DCSR, queue );
        magma_zmfree( &Rh, queue );
        magma_zmfree( &RCSR, queue );

        return MAGMA_SUCCESS; 
    }
}



