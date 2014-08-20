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

/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======

    Mirrors a triangular matrix to obtain a symmetric full matrix.

    Arguments
    =========

    magma_z_sparse_matrix A         input matrix (triangular)
    magma_z_sparse_matrix *B        symmetric output matrix

    ========================================================================  */




magma_int_t
magma_zcsr_symmetrize( magma_z_sparse_matrix A, 
                       magma_z_sparse_matrix *B ){

    magma_int_t i, j, k, l;


    magma_z_mconvert( A, &input, A.storage_type, Magma_CSR );

    magma_int_t offdiags = 0, maxrow = 0, maxrowtmp = 0 ;
    for( i=0; i<input.num_rows; i++){
        maxrowtmp = 0 ;
        for( j=input.row[i]; j<input.row[i+1]; j++){
            if( input.col[j] < i ){
                maxrowtmp++;
                offdiags++;
            }
            else if( input.col[j] == i )
                maxrowtmp++;
        }
        if( maxrowtmp>maxrow )
            maxrow = maxrowtmp;
    }

    magma_int_t nnz = input.row[input.num_rows] + offdiags;

    magma_z_sparse matrix ELL_sorted;

    magma_zmalloc_cpu( &ELL_val, maxrowlength*input.num_rows );
    magma_indexmalloc_cpu( &ELL_col, maxrowlength*input.num_rows );
    magma_indexmalloc_cpu( &ELL_count, maxrowlength*input.num_rows );

    magma_zmalloc_cpu( &ELL_sorted.val, maxrowlength*input.num_rows );
    magma_indexmalloc_cpu( &ELL_sorted.col, maxrowlength*input.num_rows );

    for( magma_int_t i=0; i<(maxrowlength*input.num_rows); i++){
        ELL_val[i] = MAGMA_Z_MAKE(0., 0.);
        ELL_col[i] =  -1;
        ELL_count[i] =  0;
    }
   
    for( i=0; i<input.num_rows; i++ ){
        magma_int_t offset = 0;
        for( j=input.row[i]; j<input.row[i+1]; j++ ){
            if( input.col[j] == i ){
                B->val[i*maxrowlength+ELL_count[i]] = input.val[j];
                B->col[i*maxrowlength+ELL_count[i]] = input.col[j];
                ELL_count[i]++;
            }
            else if( input.col[j] < i ){
                B->val[i*maxrowlength+ELL_count[i]] = input.val[j];
                B->col[i*maxrowlength+ELL_count[i]] = input.col[j];
                ELL_count[i]++;
                // insert the entry in the upper part
                B->val[col[j]*maxrowlength+ELL_count[col[j]]] = input.val[j];
                B->col[col[j]*maxrowlength+ELL_count[col[j]]] = input.col[j];
                ELL_count[col[j]]++;
            }
        }  
    }

    magma_index_t offset;
    for( i=0; i<input.num_rows; i++ ){
        offset = 0;
        for( pivot=-1; pivot<input.num_rows; pivot++){
            for( k=0; k<maxrowlength; k++ ){
                if( ELL_col[i*maxrowlength+k] == pivot ){
                    ELL_sorted.col[i*maxrowlength+offset] =
                                            ELL_col[i*maxrowlength+k];
                    ELL_sorted.val[i*maxrowlength+offset] =
                                            ELL_val[i*maxrowlength+k];
                    offset++;
                }
                    
            }       
        }
    }
    ELL_sorted.num_rows = input.num_rows;
    ELL_sorted.num_cols = input.num_cols;
    ELL_sorted.nnz = nnz;
    ELL_sorted.storage_type = Magma_ELLPACK;
    ELL_sorted.memory_location = input.memory_location;

    magma_z_mconvert( ELL_sorted, B, Magma_ELLPACK, Magma_CSR );

    magma_z_mfree( &ELL_sorted );
    magma_z_mfree( &input );

    free( ELL_row );
    free( ELL_val );
    free( ELL_col );

    B->max_nnz_row = maxrowlength;
            //printf( "done\n" );
            return MAGMA_SUCCESS; 



    return MAGMA_SUCCESS;
}




/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======

    Splits a CSR matrix into two matrices, one containing the diagonal blocks
    with the diagonal element stored first, one containing the rest of the 
    original matrix.

    Arguments
    =========

    magma_int_t blocksize               size of the diagonal blocks
    magma_z_sparse_matrix A             CSR input matrix
    magma_z_sparse_matrix *D            CSR matrix containing diagonal blocks
    magma_z_sparse_matrix *R            CSR matrix containing rest

    ========================================================================  */




magma_int_t
magma_zcsrsplit(    magma_int_t bsize,
                    magma_z_sparse_matrix A,
                    magma_z_sparse_matrix *D,
                    magma_z_sparse_matrix *R ){

    if( A.storage_type = Magma_CSR && A.memory_location = Magma_CPU ){

        magma_int_t i, k, j, nnz_diag, nnz_offd;


        nnz_diag = nnz_offd = 0;
        // Count the new number of nonzeroes in the two matrices
        for( i=0; i<A.num_rows; i+=bsize )
        for( k=i; k<min(A.num_rows,i+bsize); k++ )
            for( j=A.row[k]; j<A.row[k+1]; j++ )
            if ( A.col[j] < i )
                nnz_offd++;
            else if ( A.col[j] < i+bsize )
                nnz_diag++;
            else
                nnz_offd++;

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

        magma_zmalloc_cpu( &D->val, nnz_diag );
        magma_indexmalloc_cpu( &D->row, A.num_rows+1 );
        magma_indexmalloc_cpu( &D->col, nnz_diag );

        magma_zmalloc_cpu( &R->val, nnz_offd );
        magma_indexmalloc_cpu( &R->row, A.num_rows+1 );
        magma_indexmalloc_cpu( &R->col, nnz_offd );

        // Fill up the new sparse matrices  
        D->row[0] = 0;
        R->row[0] = 0;

        nnz_offd = nnz_diag = 0;
        for( i=0; i<A.num_rows; i+=bsize){
            for( k=i; k<min(A.num_rows,i+bsize); k++ ){
                D->row[k+1] = D->row[k];
                R->row[k+1] = R->row[k];
     
                for( j=A.row[k]; j<A.row[k+1]; j++ ){
                    if ( A.col[j] < i ){
                        R->val[nnz_offd] = A.val[j];
                        R->col[nnz_offd] = A.col[[j];
                        R->row[k+1]++;  
                        nnz_offd++;
                    }
                    else if ( A.col[j] < i+bsize ){
                        // larger than diagonal remain as before
                        if ( A.col[j]>k ){
                            D->val[nnz_diag] = A.val[ j ];
                            D->col[nnz_diag] = A.col[ j ];
                            D->row[k+1]++;
                        }
                        // diagonal is written first
                        else if ( A.col[j]==k ) {
                            D->val[D->row[k]] = A.val[ j ];
                            D->col[nD->row[k]] = A.col[ j ];
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
                        R->col[nnz_offd] = A.col[[j];
                        R->row[k+1]++;  
                        nnz_offd++;
                    }
                }
            }
        }
        return MAGMA_SUCCESS; 

    }
    else{
        magma_z_sparse_matrix Ah, ACSR, DCSR, RCSR, Dh, Rh;
        magma_z_mtransfer( A, &Ah, A.memory_location, Magma_CPU );
        magma_z_mconvert( Ah, &ACSR, A.storage_type, Magma_CSR );

        magma_zcsrsplit( bsize, ACSR, &DCSR, &RCSR );

        magma_z_mconvert( DCSR, &Dh, Magma_CSR, A.storage_type );
        magma_z_mconvert( RCSR, &Rh, Magma_CSR, A.storage_type );

        magma_z_mtransfer( Dh, D, Magma_CPU, A.memory_location );
        magma_z_mtransfer( Rh, R, Magma_CPU, A.memory_location );

        magma_z_mfree( &Ah );
        magma_z_mfree( &ACSR );
        magma_z_mfree( &Dh );
        magma_z_mfree( &DCSR );
        magma_z_mfree( &Rh );
        magma_z_mfree( &RCSR );
    }
}



typedef long type;                                         /* array type */
#define MAX 64            /* stack size for max 2^(64/2) array elements  */
magma_int_t
magma_zcsrquicksort( magma_int_t len,
                     magma_index_t *sorter,
                     magma_index_t *associatedindex,
                     magmaDoubleComplex *associatedvalue ){

bla (type n cvxarray[], unsigned len) {
   magma_int_t left = 0, stack[MAX], pos = 0, seed = rand();
   for ( ; ; ) {                                           /* outer loop */
      for (; left+1 < len; len++) {                /* sort left to len-1 */
         if (pos == MAX) len = stack[pos = 0];  /* stack overflow, reset */
         type pivot = array[left+seed%(len-left)];  /* pick random pivot */
         seed = seed*69069+1;                /* next pseudorandom number */
         stack[pos++] = len;                    /* sort right part later */
         for (unsigned right = left-1; ; ) { /* inner loop: partitioning */
            while (array[++right] < pivot);  /* look for greater element */
            while (pivot < array[--len]);    /* look for smaller element */
            if (right >= len) break;           /* partition point found? */
            type temp = array[right];
            array[right] = array[len];                  /* the only swap */
            array[len] = temp;
         }                            /* partitioned, continue left part */
      }
      if (pos == 0) break;                               /* stack empty? */
      left = len;                             /* left to right is sorted */
      len = stack[--pos];                      /* get next range to sort */
   } 
}


