/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

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
#include "common_magma.h"
#include <cuda.h>
//#include <cusparse_v2.h>

using namespace std;


/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

    Purpose
    =======

    Helper function to compress CSR containing zero-entries.


    Arguments
    =========

    magmaDoubleComplex ** val           input val pointer to compress
    magma_int_t ** row                  input row pointer to modify
    magma_int_t ** col                  input col pointer to compress
    magmaDoubleComplex ** valn          output val pointer
    magma_int_t ** rown                 output row pointer
    magma_int_t ** coln                 output col pointer
    magma_int_t *n                      number of rows in matrix



    ========================================================================  */


magma_int_t magma_z_csr_compressor( magmaDoubleComplex ** val, 
                                    magma_index_t ** row, 
                                    magma_index_t ** col, 
                                    magmaDoubleComplex ** valn, 
                                    magma_index_t ** rown, 
                                    magma_index_t ** coln, 
                                    magma_int_t *n,
                                    magma_int_t *alignedrows)
{
    magma_int_t i,j, nnz_new=0, (*row_nnz), nnz_this_row; 
    magma_indexmalloc_cpu( &(row_nnz), (*n) );
    magma_indexmalloc_cpu( rown, *n+1 );
    for( i=0; i<*n; i++ ){
        (*rown)[i] = nnz_new;
        nnz_this_row = 0;
        for( j=(*row)[i]; j<(*row)[i+1]; j++ ){
            if( MAGMA_Z_REAL((*val)[j]) != 0 ){
                nnz_new++;
                nnz_this_row++;
            }
        }
        row_nnz[i] = nnz_this_row;
    }
    (*rown)[*n] = nnz_new;

    magma_zmalloc_cpu( valn, nnz_new );
    magma_indexmalloc_cpu( coln, nnz_new );

    nnz_new = 0;
    for( i=0; i<*n; i++ ){
        for( j=(*row)[i]; j<(*row)[i+1]; j++ ){
            if( MAGMA_Z_REAL((*val)[j]) != 0 ){
                (*valn)[nnz_new]= (*val)[j];
                (*coln)[nnz_new]= (*col)[j];
                nnz_new++;
            }
        }
    }

    if( valn == NULL || coln == NULL || rown == NULL ){
        magma_free( valn );
        magma_free( coln );
        magma_free( rown );
        printf("error: memory allocation.\n");
        return MAGMA_ERR_HOST_ALLOC;
    }
    return MAGMA_SUCCESS;
}





/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

    Purpose
    =======

    Converter between different sparse storage formats.

    Arguments
    =========

    magma_z_sparse_matrix A              sparse matrix A    
    magma_z_sparse_matrix *B             copy of A in new format      
    magma_storage_t old_format           original storage format
    magma_storage_t new_format           new storage format

    ========================================================================  */

magma_int_t 
magma_z_mconvert( magma_z_sparse_matrix A, 
                  magma_z_sparse_matrix *B, 
                  magma_storage_t old_format, 
                  magma_storage_t new_format ){

    // check whether matrix on CPU
    if( A.memory_location == Magma_CPU ){

        // CSR to CSR
        if( old_format == Magma_CSR && new_format == Magma_CSR ){
            // fill in information for B
            B->storage_type = Magma_CSR;
            B->memory_location = A.memory_location;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;

            magma_zmalloc_cpu( &B->val, A.nnz );
            magma_indexmalloc_cpu( &B->row, A.num_rows+1 );
            magma_indexmalloc_cpu( &B->col, A.nnz );

            for( int i=0; i<A.nnz; i++){
                B->val[i] = A.val[i];
                B->col[i] = A.col[i];
            }
            for( int i=0; i<A.num_rows+1; i++){
                B->row[i] = A.row[i];
            }
            return MAGMA_SUCCESS; 
        }
        // CSR to ELLPACK    
        if( old_format == Magma_CSR && new_format == Magma_ELLPACK ){
            // fill in information for B
            B->storage_type = Magma_ELLPACK;
            B->memory_location = A.memory_location;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;

            // conversion
            magma_int_t i, j, *length, maxrowlength=0;
            magma_indexmalloc_cpu( &length, A.num_rows);

            for( i=0; i<A.num_rows; i++ ){
                length[i] = A.row[i+1]-A.row[i];
                if(length[i] > maxrowlength)
                     maxrowlength = length[i];
            }
            //printf( "Conversion to ELLPACK with %d elements per row: ",
                                                            // maxrowlength );
            //fflush(stdout);
             magma_zmalloc_cpu( &B->val, maxrowlength*A.num_rows );
             magma_indexmalloc_cpu( &B->col, maxrowlength*A.num_rows );
             for( magma_int_t i=0; i<(maxrowlength*A.num_rows); i++){
                  B->val[i] = MAGMA_Z_MAKE(0., 0.);
                  B->col[i] =  0;
             }
   
            for( i=0; i<A.num_rows; i++ ){
                 magma_int_t offset = 0;
                 for( j=A.row[i]; j<A.row[i+1]; j++ ){
                     B->val[i*maxrowlength+offset] = A.val[j];
                     B->col[i*maxrowlength+offset] = A.col[j];
                     offset++;
                 }
            }
            B->max_nnz_row = maxrowlength;
            //printf( "done\n" );
            return MAGMA_SUCCESS; 
        }

        // ELLPACK to CSR
        if( old_format == Magma_ELLPACK && new_format == Magma_CSR ){
            //printf( "Conversion to CSR: " );
            // fill in information for B
            B->storage_type = Magma_CSR;
            B->memory_location = A.memory_location;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;

            // conversion
            magma_int_t *row_tmp;
            magma_indexmalloc_cpu( &row_tmp, A.num_rows+1 );
            //fill the row-pointer
            for( magma_int_t i=0; i<A.num_rows+1; i++ )
                row_tmp[i] = i*A.max_nnz_row;
            //now use AA_ELL, IA_ELL, row_tmp as CSR with some zeros. 
            //The CSR compressor removes these
            magma_z_csr_compressor(&A.val, &row_tmp, &A.col, 
                       &B->val, &B->row, &B->col, &B->num_rows, &B->num_rows);  
            //printf( "done\n" );      
            return MAGMA_SUCCESS; 
        }        

        // CSR to ELL (former ELLPACKT, ELLPACK using column-major storage)
        if( old_format == Magma_CSR && new_format == Magma_ELL ){
            // fill in information for B
            B->storage_type = Magma_ELL;
            B->memory_location = A.memory_location;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;

            // conversion
            magma_int_t i, j, *length, maxrowlength=0;
            magma_indexmalloc_cpu( &length, A.num_rows);

            for( i=0; i<A.num_rows; i++ ){
                length[i] = A.row[i+1]-A.row[i];
                if(length[i] > maxrowlength)
                     maxrowlength = length[i];
            }
            magma_free_cpu( length );
            //printf( "Conversion to ELL with %d elements per row: ",
                                                           // maxrowlength );
            //fflush(stdout);
            magma_zmalloc_cpu( &B->val, maxrowlength*A.num_rows );
            magma_indexmalloc_cpu( &B->col, maxrowlength*A.num_rows );
            for( magma_int_t i=0; i<(maxrowlength*A.num_rows); i++){
                 B->val[i] = MAGMA_Z_MAKE(0., 0.);
                 B->col[i] =  0;
            }

            for( i=0; i<A.num_rows; i++ ){
                magma_int_t offset = 0;
                for( j=A.row[i]; j<A.row[i+1]; j++ ){
                     B->val[offset*A.num_rows+i] = A.val[j];
                     B->col[offset*A.num_rows+i] = A.col[j];
                     offset++;
                }
            }
            B->max_nnz_row = maxrowlength;
            //printf( "done\n" );
            return MAGMA_SUCCESS; 
        }

        // ELL (ELLPACKT) to CSR
        if( old_format == Magma_ELL && new_format == Magma_CSR ){
            //printf( "Conversion to CSR: " ); 
            //fflush(stdout);
            // fill in information for B
            B->storage_type = Magma_CSR;
            B->memory_location = A.memory_location;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;

            // conversion
            magma_int_t *row_tmp;
            magma_int_t *col_tmp;
            magmaDoubleComplex *val_tmp;
            magma_zmalloc_cpu( &val_tmp, A.num_rows*A.max_nnz_row );
            magma_indexmalloc_cpu( &row_tmp, A.num_rows+1 );
            magma_indexmalloc_cpu( &col_tmp, A.num_rows*A.max_nnz_row );
            //fill the row-pointer
            for( magma_int_t i=0; i<A.num_rows+1; i++ )
                row_tmp[i] = i*A.max_nnz_row;
            //transform RowMajor to ColMajor
            for( magma_int_t j=0;j<A.max_nnz_row;j++ ){
                for( magma_int_t i=0;i<A.num_rows;i++ ){
                    col_tmp[i*A.max_nnz_row+j] = A.col[j*A.num_rows+i];
                    val_tmp[i*A.max_nnz_row+j] = A.val[j*A.num_rows+i];
                }
            }    
            //now use AA_ELL, IA_ELL, row_tmp as CSR with some zeros. 
            //The CSR compressor removes these
            magma_z_csr_compressor(&val_tmp, &row_tmp, &col_tmp, 
                       &B->val, &B->row, &B->col, &B->num_rows, &B->num_rows); 

            //printf( "done\n" );      
            return MAGMA_SUCCESS; 
        }  
        // ELLPACK to CSR
        if( old_format == Magma_ELLPACK && new_format == Magma_CSR ){
            //printf( "Conversion to CSR: " );
            // fill in information for B
            B->storage_type = Magma_CSR;
            B->memory_location = A.memory_location;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;

            // conversion
            magma_int_t *row_tmp;
            magma_indexmalloc_cpu( &row_tmp, A.num_rows+1 );
            //fill the row-pointer
            for( magma_int_t i=0; i<A.num_rows+1; i++ )
                row_tmp[i] = i*A.max_nnz_row;
            //now use AA_ELL, IA_ELL, row_tmp as CSR with some zeros. 
            //The CSR compressor removes these
            magma_z_csr_compressor(&A.val, &row_tmp, &A.col, 
                       &B->val, &B->row, &B->col, &B->num_rows, &B->num_rows);  
            //printf( "done\n" );      
            return MAGMA_SUCCESS; 
        }        
        // CSR to ELLD (ELL with diagonal element first)
        if( old_format == Magma_CSR && new_format == Magma_ELLD ){
            // fill in information for B
            B->storage_type = Magma_ELLD;
            B->memory_location = A.memory_location;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;

            // conversion
            magma_int_t i, j, *length, maxrowlength=0;
            magma_indexmalloc_cpu( &length, A.num_rows);

            for( i=0; i<A.num_rows; i++ ){
                length[i] = A.row[i+1]-A.row[i];
                if(length[i] > maxrowlength)
                     maxrowlength = length[i];
            }
            magma_free_cpu( length );
            //printf( "Conversion to ELL with %d elements per row: ",
                                                           // maxrowlength );
            //fflush(stdout);
            magma_zmalloc_cpu( &B->val, maxrowlength*A.num_rows );
            magma_indexmalloc_cpu( &B->col, maxrowlength*A.num_rows );
            for( magma_int_t i=0; i<(maxrowlength*A.num_rows); i++){
                 B->val[i] = MAGMA_Z_MAKE(0., 0.);
                 B->col[i] =  0;
            }

            for( i=0; i<A.num_rows; i++ ){
                magma_int_t offset = 1;
                for( j=A.row[i]; j<A.row[i+1]; j++ ){
                    if( A.col[j] == i ){ // diagonal case
                        B->val[i] = A.val[j];
                        B->col[i] = A.col[j];
                    }else{
                        B->val[offset*A.num_rows+i] = A.val[j];
                        B->col[offset*A.num_rows+i] = A.col[j];
                        offset++;
                    }
                }
            }
            B->max_nnz_row = maxrowlength;
            //printf( "done\n" );
            return MAGMA_SUCCESS; 
        }
        // ELLD (ELL with diagonal element first) to CSR
        if( old_format == Magma_ELLD && new_format == Magma_CSR ){
            //printf( "Conversion to CSR: " ); 
            //fflush(stdout);
            // fill in information for B
            B->storage_type = Magma_CSR;
            B->memory_location = A.memory_location;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;

            // conversion
            magma_int_t *row_tmp;
            magma_int_t *col_tmp;
            magmaDoubleComplex *val_tmp;
            magma_zmalloc_cpu( &val_tmp, A.num_rows*A.max_nnz_row );
            magma_indexmalloc_cpu( &row_tmp, A.num_rows+1 );
            magma_indexmalloc_cpu( &col_tmp, A.num_rows*A.max_nnz_row );
            //fill the row-pointer
            for( magma_int_t i=0; i<A.num_rows+1; i++ )
                row_tmp[i] = i*A.max_nnz_row;
            //transform RowMajor to ColMajor 
            for( magma_int_t j=0;j<A.max_nnz_row;j++ ){
                for( magma_int_t i=0;i<A.num_rows;i++ ){
                    col_tmp[i*A.max_nnz_row+j] = A.col[j*A.num_rows+i];
                    val_tmp[i*A.max_nnz_row+j] = A.val[j*A.num_rows+i];
                }
            }    
            // sort the diagonal element into the right place
            magma_int_t *col_tmp2;
            magmaDoubleComplex *val_tmp2;
            magma_zmalloc_cpu( &val_tmp2, A.num_rows*A.max_nnz_row );
            magma_indexmalloc_cpu( &col_tmp2, A.num_rows*A.max_nnz_row );
            for( magma_int_t j=0;j<A.num_rows;j++ ){
                magmaDoubleComplex diagval = val_tmp[j*A.max_nnz_row];
                magma_index_t diagcol = col_tmp[j*A.max_nnz_row];
                magma_int_t smaller = 0;
                for( magma_int_t i=1;i<A.max_nnz_row;i++ ){
                    if( col_tmp[j*A.max_nnz_row+i] < diagcol )          
                        smaller++;
                }
                for( magma_int_t i=0;i<smaller;i++ ){                
                    col_tmp2[j*A.max_nnz_row+i] = col_tmp[j*A.max_nnz_row+i+1];
                    val_tmp2[j*A.max_nnz_row+i] = val_tmp[j*A.max_nnz_row+i+1];
                }
                col_tmp2[j*A.max_nnz_row+smaller] = col_tmp[j*A.max_nnz_row];
                val_tmp2[j*A.max_nnz_row+smaller] = val_tmp[j*A.max_nnz_row];
                for( magma_int_t i=smaller+1;i<A.max_nnz_row;i++ ){                
                    col_tmp2[j*A.max_nnz_row+i] = col_tmp[j*A.max_nnz_row+i];
                    val_tmp2[j*A.max_nnz_row+i] = val_tmp[j*A.max_nnz_row+i];
                }
            }   

            //now use AA_ELL, IA_ELL, row_tmp as CSR with some zeros. 
            //The CSR compressor removes these
            magma_z_csr_compressor(&val_tmp2, &row_tmp, &col_tmp2, 
                       &B->val, &B->row, &B->col, &B->num_rows, &B->num_rows); 

            //printf( "done\n" );      
            return MAGMA_SUCCESS; 
        }  
        // CSR to ELLDD (2x ELLD (first row, then col major)
        if( old_format == Magma_CSR && new_format == Magma_ELLDD ){         
            // fill in information for B
            B->storage_type = Magma_ELLDD;
            B->memory_location = A.memory_location;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;

            // conversion
            magma_int_t i, j;
            magma_z_sparse_matrix A_tmp;
            magma_z_mconvert( A, &A_tmp, Magma_CSR, Magma_ELLD );
            magma_int_t maxrowlength = A_tmp.max_nnz_row;
            B->max_nnz_row = maxrowlength;
            magma_int_t off = maxrowlength*A.num_rows;

            magma_zmalloc_cpu( &B->val, 2*off );
            magma_indexmalloc_cpu( &B->col, 2*off );

            magma_indexmalloc_cpu( &B->blockinfo, 2*off );
            
            for( i=0; i<off; i++){
                B->val[ i ] = A_tmp.val[ i ];
                B->col[ i ] = A_tmp.col[ i ];
            }
            for( i=0; i<off; i++){
                B->val[ i+off ] = A_tmp.val[ i ];
                B->col[ i+off ] = A_tmp.col[ i ];
            }
            for( j=0; j<maxrowlength; j++){
                for( i=0; i<A.num_rows; i++){
                    if( (i > B->col[j*A.num_rows+i]) 
                            && (MAGMA_Z_REAL(B->val[j*A.num_rows+i]) != 0.0 ) )
                        B->blockinfo[j*A.num_rows+i] = 1;
                    else
                        B->blockinfo[j*A.num_rows+i] = 0;                        
                }
            }
            for( j=0; j<maxrowlength; j++){
                for( i=0; i < A.num_rows; i++){
                    if( (i >= B->col[off+j*A.num_rows+i]) 
                        && (MAGMA_Z_REAL(B->val[off+j*A.num_rows+i]) != 0.0 ) )
                        B->blockinfo[off+j*A.num_rows+i] = 1;
                    else
                        B->blockinfo[off+j*A.num_rows+i] = 0;                        
                }
            }

            magma_z_mfree(&A_tmp);
            //printf( "done\n" );
            return MAGMA_SUCCESS; 
        }
        // ELLDD to CSR
        if( old_format == Magma_ELLDD && new_format == Magma_CSR ){         
            
            magma_int_t maxrowlength = A.max_nnz_row;
            magma_int_t off = A.max_nnz_row*A.num_rows;

            // conversion
            magma_int_t i, j;
            magma_z_sparse_matrix A_tmp;
            magma_zmalloc_cpu( &A_tmp.val, off );
            magma_indexmalloc_cpu( &A_tmp.col, off );
            A_tmp.num_rows = A.num_rows;
            A_tmp.num_cols = A.num_cols;
            A_tmp.nnz = A.nnz;
            A_tmp.max_nnz_row = A.max_nnz_row;
            A_tmp.storage_type = Magma_ELLD;
            A_tmp.memory_location = A.memory_location;

            for( i=0; i<off; i++){
                    A_tmp.col[ i ] = A.col[ i+off ];
                if( A.blockinfo[ i ] == 1 ){
                    A_tmp.val[ i ] = A.val[ i+off ];
                }
            }
            for( j=0; j<maxrowlength; j++){
                for( i=0; i<A.num_rows; i++){
                    if( A.blockinfo[ j*A.num_rows +i+off ] == 1 ){
                        magma_int_t row = A.col[ j*A.num_rows +i+off ];
                        magmaDoubleComplex val = A.val[ j*A.num_rows +i+off ];
                        for(int k=0; k<maxrowlength; k++){
                            if( A_tmp.col[row+k*A.num_rows] == i ){
                                A_tmp.val[ row+k*A.num_rows ] = val;
                            }
                        }
                    }
                }
            }


            magma_z_mconvert(A_tmp, B, Magma_ELLD, Magma_CSR );

            magma_z_mfree(&A_tmp);
            //printf( "done\n" );
            return MAGMA_SUCCESS; 
        }
        // CSR to ELLRT (also ELLPACKRT)
        if( old_format == Magma_CSR && new_format == Magma_ELLRT ){
            // fill in information for B
            B->storage_type = Magma_ELLRT;
            B->memory_location = A.memory_location;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;

            // conversion
            magma_int_t i, j, *length, maxrowlength=0;
            magma_indexmalloc_cpu( &length, A.num_rows);

            for( i=0; i<A.num_rows; i++ ){
                length[i] = A.row[i+1]-A.row[i];
                if(length[i] > maxrowlength)
                     maxrowlength = length[i];
            }
            magma_free_cpu( length );
            //printf( "Conversion to ELLRT with %d elements per row: ", 
            //                                                   maxrowlength );

            int num_threads = B->alignment * B->blocksize;
            int threads_per_row = B->alignment; 
            int rowlength = ( (int)
                    ((maxrowlength+threads_per_row-1)/threads_per_row) ) 
                                                            * threads_per_row;

            magma_zmalloc_cpu( &B->val, rowlength*A.num_rows );
            magma_indexmalloc_cpu( &B->col, rowlength*A.num_rows );
            magma_indexmalloc_cpu( &B->row, A.num_rows );
            for( magma_int_t i=0; i<rowlength*A.num_rows; i++){
                 B->val[i] = MAGMA_Z_MAKE(0., 0.);
                 B->col[i] =  0;
            }

            for( i=0; i<A.num_rows; i++ ){
                 magma_int_t offset = 0;
                 for( j=A.row[i]; j<A.row[i+1]; j++ ){
                     B->val[i*rowlength+offset] = A.val[j];
                     B->col[i*rowlength+offset] = A.col[j];
                     offset++;
                 }
                B->row[i] = A.row[i+1] - A.row[i];
            }
            B->max_nnz_row = maxrowlength;         
            //printf( "done\n" );
            return MAGMA_SUCCESS; 
        }

        // ELLRT to CSR
        if( old_format == Magma_ELLRT && new_format == Magma_CSR ){
            //printf( "Conversion to CSR: " ); 
            //fflush(stdout);
            // fill in information for B
            B->storage_type = Magma_CSR;
            B->memory_location = A.memory_location;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;

            int num_threads = A.alignment * A.blocksize;
            int threads_per_row = A.alignment; 
            int rowlength = ( (int)
                    ((A.max_nnz_row+threads_per_row-1)/threads_per_row) ) 
                                                            * threads_per_row;
            // conversion
            magma_int_t *row_tmp;
            magma_indexmalloc_cpu( &row_tmp, A.num_rows+1 );
            //fill the row-pointer
            for( magma_int_t i=0; i<A.num_rows+1; i++ )
                row_tmp[i] = i*rowlength;
            //now use AA_ELL, IA_ELL, row_tmp as CSR with some zeros. 
            //The CSR compressor removes these
            magma_z_csr_compressor(&A.val, &row_tmp, &A.col, 
                   &B->val, &B->row, &B->col, &B->num_rows, &B->num_rows);  
            //printf( "done\n" );      
            return MAGMA_SUCCESS; 
        }   
        // CSR to SELLC / SELLP
        // SELLC is SELLP using alignment 1
        // see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
        // A UNIFIED SPARSE MATRIX DATA FORMAT 
        // FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
        // in SELLP we modify SELLC:
        // alignment is posible such that multiple threads can be used for SpMV
        // so the rowlength is padded (SELLP) to a multiple of the alignment
        if( old_format == Magma_CSR && 
                (new_format == Magma_SELLC || new_format == Magma_SELLP ) ){
            // fill in information for B
            B->storage_type = new_format;
            if(B->alignment > 1)
                B->storage_type = Magma_SELLP;
            B->memory_location = A.memory_location;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->diameter = A.diameter;
            B->max_nnz_row = 0;
            magma_int_t C = B->blocksize;
            magma_int_t slices = ( A.num_rows+C-1)/(C);
            B->numblocks = slices;
            magma_int_t alignedlength, alignment = B->alignment;
            // conversion
            magma_int_t i, j, k, *length, maxrowlength=0;
            magma_indexmalloc_cpu( &length, C);
            // B-row points to the start of each slice
            magma_indexmalloc_cpu( &B->row, slices+1 );

            B->row[0] = 0;
            for( i=0; i<slices; i++ ){
                maxrowlength = 0;
                for(j=0; j<C; j++){
                    if(i*C+j<A.num_rows){
                        length[j] = A.row[i*C+j+1]-A.row[i*C+j];
                    }
                    else
                        length[j]=0;
                    if(length[j] > maxrowlength){
                         maxrowlength = length[j];
                    }
                }
                alignedlength = ((maxrowlength+alignment-1)/alignment) 
                                                                * alignment;
                B->row[i+1] = B->row[i] + alignedlength * C;
                if( alignedlength > B->max_nnz_row )
                    B->max_nnz_row = alignedlength;
            }
            B->nnz = B->row[slices];
            //printf( "Conversion to SELLC with %d slices of size %d and"
            //       " %d nonzeros.\n", slices, C, B->nnz );

            //fflush(stdout);
            magma_zmalloc_cpu( &B->val, B->row[slices] );
            magma_indexmalloc_cpu( &B->col, B->row[slices] );
            // zero everything
            for( i=0; i<B->row[slices]; i++ ){
              B->val[ i ] = MAGMA_Z_MAKE(0., 0.);
              B->col[ i ] =  0;
            }
            // fill in values
            for( i=0; i<slices; i++ ){
                for(j=0; j<C; j++){
                    magma_int_t line = i*C+j;
                    magma_int_t offset = 0;
                    if( line < A.num_rows){
                         for( k=A.row[line]; k<A.row[line+1]; k++ ){
                             B->val[ B->row[i] + j +offset*C ] = A.val[k];
                             B->col[ B->row[i] + j +offset*C ] = A.col[k];
                             offset++;
                         }
                    }
                }
            }
            return MAGMA_SUCCESS; 
        }
        // SELLC/SELLP to CSR    
        if( (old_format == Magma_SELLC || old_format == Magma_SELLP )
                                            && new_format == Magma_CSR ){
            // printf( "Conversion to CSR: " );
            // fill in information for B
            B->storage_type = Magma_CSR;
            B->memory_location = A.memory_location;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            magma_int_t C = A.blocksize;
            magma_int_t slices = A.numblocks;
            B->blocksize = A.blocksize;
            B->numblocks = A.numblocks;
            // conversion
            magma_int_t *row_tmp;
            magma_int_t *col_tmp;
            magmaDoubleComplex *val_tmp;
            magma_zmalloc_cpu( &val_tmp, A.max_nnz_row*(A.num_rows+C) );
            magma_indexmalloc_cpu( &row_tmp, A.num_rows+C );
            magma_indexmalloc_cpu( &col_tmp, A.max_nnz_row*(A.num_rows+C) );

            // zero everything
            for(magma_int_t i=0; i<A.max_nnz_row*(A.num_rows+C); i++ ){
              val_tmp[ i ] = MAGMA_Z_MAKE(0., 0.);
              col_tmp[ i ] =  0;
            }

            //fill the row-pointer
            for( magma_int_t i=0; i<A.num_rows+1; i++ ){
                row_tmp[i] = A.max_nnz_row*i;
                
            }

            //transform RowMajor to ColMajor
            for( magma_int_t k=0; k<slices; k++){
                magma_int_t blockinfo = (A.row[k+1]-A.row[k])/A.blocksize;
                for( magma_int_t j=0;j<C;j++ ){
                    for( magma_int_t i=0;i<blockinfo;i++ ){
                        col_tmp[ (k*C+j)*A.max_nnz_row+i ] = 
                                                A.col[A.row[k]+i*C+j];
                        val_tmp[ (k*C+j)*A.max_nnz_row+i ] = 
                                                A.val[A.row[k]+i*C+j];
                    }
                }    
            }

            //now use AA_ELL, IA_ELL, row_tmp as CSR with some zeros. 
            //The CSR compressor removes these

            magma_z_csr_compressor(&val_tmp, &row_tmp, &col_tmp, 
                       &B->val, &B->row, &B->col, &B->num_rows, &B->num_rows); 

            //printf( "done\n" );      
            return MAGMA_SUCCESS;  
        }
        // CSR to DENSE
        if( old_format == Magma_CSR && new_format == Magma_DENSE ){
            //printf( "Conversion to DENSE: " );
            // fill in information for B
            B->storage_type = Magma_DENSE;
            B->memory_location = A.memory_location;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;

            // conversion
            magma_zmalloc_cpu( &B->val, A.num_rows*A.num_cols );

             for( magma_int_t i=0; i<(A.num_rows)*(A.num_cols); i++){
                  B->val[i] = MAGMA_Z_MAKE(0., 0.);
             }

            for(magma_int_t i=0; i<A.num_rows; i++ ){
                 for(magma_int_t j=A.row[i]; j<A.row[i+1]; j++ )
                     B->val[i * (A.num_cols) + A.col[j] ] = A.val[ j ];
            }

            //printf( "done\n" );      
            return MAGMA_SUCCESS; 
        }
        // DENSE to CSR
        if( old_format == Magma_DENSE && new_format == Magma_CSR ){
            //printf( "Conversion to CSR: " );
            // fill in information for B
            B->storage_type = Magma_CSR;
            B->memory_location = A.memory_location;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;

            // conversion

            B->nnz=0;
            for( magma_int_t i=0; i<(A.num_rows)*(A.num_cols); i++ ){
                if( MAGMA_Z_REAL(A.val[i])!=0.0 )
                    (B->nnz)++;
            }
            magma_zmalloc_cpu( &B->val, B->nnz);
            magma_indexmalloc_cpu( &B->row, B->num_rows+1 );
            magma_indexmalloc_cpu( &B->col, B->nnz );

            magma_int_t i = 0;
            magma_int_t j = 0;
            magma_int_t k = 0;

            for(i=0; i<(A.num_rows)*(A.num_cols); i++)
            {
                if( i%(B->num_cols)==0 )
                {
                    (B->row)[k] = j;
                    k++;
                }
                if( MAGMA_Z_REAL(A.val[i])!=0 )
                {
                    (B->val)[j] = A.val[i];
                    (B->col)[j] = i%(B->num_cols);
                    j++;
                }

            }
            (B->row)[B->num_rows]=B->nnz;

            //printf( "done\n" );      
            return MAGMA_SUCCESS; 
        }
        // CSR to BCSR
        if( old_format == Magma_CSR && new_format == Magma_BCSR ){

            // fill in information for B
            B->storage_type = Magma_BCSR;
            B->memory_location = A.memory_location;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            //printf( "Conversion to BCSR(blocksize=%d): ",B->blocksize );

            magma_int_t i, j, k, l, numblocks;

            // conversion
            int size_b = B->blocksize;
            magma_int_t c_blocks = ceil( (float)A.num_cols / (float)size_b );     
                            // max number of blocks per row
            magma_int_t r_blocks = ceil( (float)A.num_rows / (float)size_b );     
                            // max number of blocks per column
            //printf("c_blocks: %d  r_blocks: %d  ", c_blocks, r_blocks);
         
            magma_indexmalloc_cpu( &B->blockinfo, c_blocks * r_blocks );
            if( B->blockinfo == NULL ){
                printf("error: memory allocation (B->blockinfo).\n");
                magma_free( B->blockinfo );
                return MAGMA_ERR_HOST_ALLOC;
            }
            for( i=0; i<c_blocks * r_blocks; i++ )
                B->blockinfo[i] = 0;
            #define  blockinfo(i,j)  blockinfo[(i)*c_blocks   + (j)]
            
            // fill in "1" in blockinfo if block is occupied
            for( i=0; i<A.num_rows; i++ ){
                for( j=A.row[i]; j<A.row[i+1]; j++ ){
                    k = floor(i / size_b);
                    l = floor(A.col[j] / size_b);
                    B->blockinfo(k,l) = 1;
                }
            } 

            // count blocks and fill rowpointer
            magma_indexmalloc_cpu( &B->row, r_blocks+1 );
            numblocks = 0;
            for( i=0; i<c_blocks * r_blocks; i++ ){
                if( i%c_blocks == 0)
                    B->row[i/c_blocks] = numblocks;
                if( B->blockinfo[i] != 0 ){
                    numblocks++;
                    B->blockinfo[i] = numblocks;
                }
            }
            B->row[r_blocks] = numblocks;
            //printf("number of blocks: %d  ", numblocks);
            B->numblocks = numblocks;

            magma_zmalloc_cpu( &B->val, numblocks * size_b * size_b );
            magma_indexmalloc_cpu( &B->col, numblocks  );
            if( B->val == NULL || B->col == NULL ){
                printf("error: memory allocation (B->val or B->col).\n");
                magma_free( B->blockinfo );
                if( B->val != NULL ) magma_free( B->val );
                if( B->col != NULL ) magma_free( B->col );
                return MAGMA_ERR_HOST_ALLOC;
            }

            for( i=0; i<numblocks * size_b * size_b; i++)
                B->val[i] = MAGMA_Z_MAKE(0.0, 0.0);

            // fill in col
            k = 0;
            for( i=0; i<c_blocks * r_blocks; i++ ){
                if( B->blockinfo[i] != 0 ){
                    B->col[k] = i%c_blocks;
                    k++;
                }
            }

            // fill in val
            for( i=0; i<A.num_rows; i++ ){
                for( j=A.row[i]; j<A.row[i+1]; j++ ){
                    k = floor(i / size_b);
                    l = floor(A.col[j] / size_b);
                // find correct block + take row into account + correct column
                    B->val[ (B->blockinfo(k,l)-1) * size_b * size_b + i%size_b 
                                        * size_b + A.col[j]%size_b ] = A.val[j];
                }
            } 

            // the values are now handled special: we want to transpose 
                                        //each block to be in MAGMA format
            magmaDoubleComplex *transpose;
            magma_zmalloc( &transpose, size_b*size_b );
            for( magma_int_t i=0; i<B->numblocks; i++ ){
                cudaMemcpy( transpose, B->val+i*size_b*size_b, 
        size_b*size_b*sizeof( magmaDoubleComplex ), cudaMemcpyHostToDevice );
                magmablas_ztranspose_inplace( size_b, transpose, size_b );
                cudaMemcpy( B->val+i*size_b*size_b, transpose, 
        size_b*size_b*sizeof( magmaDoubleComplex ), cudaMemcpyDeviceToHost );
            }
            /*
            printf("blockinfo for blocksize %d:\n", size_b);
            for( i=0; i<c_blocks; i++ ){
                for( j=0; j<c_blocks; j++ ){
                    printf("%d  ", B->blockinfo(i,j));
                }
                printf("\n");
            }
            printf("numblocks: %d\n", numblocks);
            printf("row:\n");
            for( i=0; i<r_blocks+1; i++ ){
                printf("%d  ", B->row[i]);
            }
            printf("\n");
            printf("col:\n");
            for( i=0; i<numblocks; i++ ){
                printf("%d  ", B->col[i]);
            }
            printf("\n");
            printf("val:\n");
            for( i=0; i<numblocks*size_b*size_b; i++ ){
                printf("%f\n", B->val[i]);
            }
            printf("\n");
            */


            //printf( "done\n" );      
            return MAGMA_SUCCESS; 

        }
        // BCSR to CSR
        if( old_format == Magma_BCSR && new_format == Magma_CSR ){
            printf( "Conversion to CSR: " );fflush(stdout);
            // fill in information for B
            B->storage_type = Magma_CSR;
            B->memory_location = A.memory_location;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;

            magma_int_t i, j, k, l, index;

            // conversion
            magma_int_t size_b = A.blocksize;
            magma_int_t c_blocks = ceil( (float)A.num_cols / (float)size_b );     
                // max number of blocks per row
            magma_int_t r_blocks = ceil( (float)A.num_rows / (float)size_b );     
                // max number of blocks per column
            //printf("c_blocks: %d  r_blocks: %d  ", c_blocks, r_blocks);
            //fflush(stdout);
            magmaDoubleComplex *val_tmp;      
            magma_zmalloc_cpu( &val_tmp, A.row[ r_blocks ] * size_b * size_b );
            magma_int_t *row_tmp;            
            magma_indexmalloc_cpu( &row_tmp, r_blocks*size_b+1 );   
                // larger than the final size due to overhead blocks
            magma_int_t *col_tmp;            
            magma_indexmalloc_cpu( &col_tmp, A.row[ r_blocks ] * size_b * size_b );
            if( col_tmp == NULL || val_tmp == NULL || row_tmp == NULL ){
                magma_free( B->val );
                magma_free( B->col );
                printf("error: memory allocation.\n");
                return MAGMA_ERR_HOST_ALLOC;
            }
            
            // fill row_tmp
            index = A.row[0];
            for( i = 0; i<r_blocks; i++ ){
                for( j=0; j<size_b; j++ ){            
                    row_tmp[ j + i * size_b] =  index;
                    index = index +  size_b * (A.row[i+1]-A.row[i]);
                }
            }
            if( r_blocks * size_b == A.num_rows ){
                // in this case the last entry of the row-pointer 
                        //has to be filled manually
                row_tmp[r_blocks*size_b] = A.row[r_blocks] * size_b * size_b;
            }

            // the val pointer has to be handled special: we need to transpose 
                        //each block back to row-major
            magmaDoubleComplex *transpose, *val_tmp2;
            magma_zmalloc( &transpose, size_b*size_b );
            magma_zmalloc_cpu( &val_tmp2, size_b*size_b*A.numblocks );
            for( magma_int_t i=0; i<A.numblocks; i++ ){
                cudaMemcpy( transpose, A.val+i*size_b*size_b, 
        size_b*size_b*sizeof( magmaDoubleComplex ), cudaMemcpyHostToDevice );
                magmablas_ztranspose_inplace( size_b, transpose, size_b );
                cudaMemcpy( val_tmp2+i*size_b*size_b, transpose, 
        size_b*size_b*sizeof( magmaDoubleComplex ), cudaMemcpyDeviceToHost );
            }

            // fill col and val
            index = 0;
            for( j=0; j<r_blocks; j++ ){
                for( i=A.row[j]; i<A.row[j+1]; i++){ // submatrix blocks
                    for( k =0; k<size_b; k++){ // row in submatrix
                        for( l =0; l<size_b; l++){ // col in submatrix
            // offset due to col in submatrix: l
            // offset due to submatrix block (in row): (i-A.row[j])*size_b
            // offset due to submatrix row: size_b*k*(A.row[j+1]-A.row[j])
            // offset due to submatrix block row: size_b*size_b*(A.row[j])
            col_tmp[ l + (i-A.row[j])*size_b + size_b*k*(A.row[j+1]-A.row[j]) 
                            + size_b*size_b*(A.row[j]) ] 
                   = A.col[i] * size_b + l;
            val_tmp[ l + (i-A.row[j])*size_b + size_b*k*(A.row[j+1]-A.row[j]) 
                            + size_b*size_b*(A.row[j]) ] 
                   = val_tmp2[index];
            index++;
                        }  
                    }
                }
            }
            /*
            printf("col_tmp:\n");
            for( i=0; i<A.row[ r_blocks ] * size_b * size_b; i++ )
                printf("%d  ", col_tmp[i]);
            printf("\n");
            printf("row_tmp:\n");
            for( i=0; i<r_blocks*size_b+1; i++ )
                printf("%d  ", row_tmp[i]);
            printf("\n");
            printf("val_tmp:\n");
            for( i=0; i<A.row[ r_blocks ] * size_b * size_b; i++ )
                printf("%2.0f  ", val_tmp[i]);
            printf("\n");
            */
            
            magma_z_csr_compressor(&val_tmp, &row_tmp, &col_tmp, 
                     &B->val, &B->row, &B->col, &B->num_rows, &B->num_rows); 

            B->nnz = B->row[B->num_rows];

            magma_free_cpu( val_tmp );
            magma_free_cpu( val_tmp2 );
            magma_free_cpu( row_tmp );
            magma_free_cpu( col_tmp );
        
            printf( "done.\n" );      
            return MAGMA_SUCCESS; 
        }
        else{
            printf("error: format not supported.\n");
            return MAGMA_ERR_NOT_SUPPORTED;
        }
    } // end CPU case
    else if( A.memory_location == Magma_DEV ){
            printf("error: conversion only supported on CPU.\n");
            return MAGMA_ERR_NOT_SUPPORTED;
/*
        // CSR to DENSE    
        if( old_format == Magma_CSR && new_format == Magma_DENSE ){
            // fill in information for B
            B->storage_type = Magma_DENSE;
            B->memory_location = A.memory_location;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;

            // CUSPARSE context //
            cusparseHandle_t cusparseHandle = 0;
            cusparseStatus_t cusparseStatus;
            cusparseStatus = cusparseCreate(&cusparseHandle);
            cusparseMatDescr_t descr = 0;
            cusparseStatus = cusparseCreateMatDescr(&descr);
            // end CUSPARSE context //

            magma_zmalloc( &B->val, A.num_rows*A.num_cols );

            // conversion using CUSPARSE
            cusparseZcsr2dense( cusparseHandle, A.num_rows, A.num_cols,
                                descr, A.val, A.row, A.col,
                                B->val, A.num_rows );
            return MAGMA_SUCCESS; 
        }
        // DENSE to CSR    
        if( old_format == Magma_DENSE && new_format == Magma_CSR ){
            // fill in information for B
            B->storage_type = Magma_CSR;
            B->memory_location = A.memory_location;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;

            // CUSPARSE context //
            cusparseHandle_t cusparseHandle = 0;
            cusparseStatus_t cusparseStatus;
            cusparseStatus = cusparseCreate(&cusparseHandle);
            cusparseMatDescr_t descr = 0;
            cusparseStatus = cusparseCreateMatDescr(&descr);
            // end CUSPARSE context //


            magma_int_t *nnz_per_row;
            magma_indexmalloc( &nnz_per_row, A.num_rows );
            //magma_zprint_gpu( A.num_rows, 1, nnz_per_row, A.num_rows )
            cusparseZnnz( cusparseHandle, CUSPARSE_DIRECTION_COLUMN,
                          A.num_rows, A.num_cols, 
                          descr,
                          A.val, A.num_rows, nnz_per_row, &B->nnz );

            magma_zmalloc( &B->val, B->nnz );
            magma_indexmalloc( &B->row, B->num_rows+1 );
            magma_indexmalloc( &B->col, B->nnz );

            // conversion using CUSPARSE
            cusparseZdense2csr( cusparseHandle, A.num_rows, A.num_cols,
                                descr,
                                A.val, A.num_rows, nnz_per_row,
                                B->val, B->row, B->col );

            magma_free( nnz_per_row );
            return MAGMA_SUCCESS; 
        }
        // CSR to BCSR
        if( old_format == Magma_CSR && new_format == Magma_BCSR ){
            //printf( "Conversion to BCSR: " );
            // fill in information for B
            B->storage_type = Magma_BCSR;
            B->memory_location = A.memory_location;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            int size_b = B->blocksize;

            // CUSPARSE context //
            cusparseHandle_t cusparseHandle = 0;
            cusparseStatus_t cusparseStatus;
            cusparseStatus = cusparseCreate(&cusparseHandle);
            cusparseMatDescr_t descr = 0;
            cusparseStatus = cusparseCreateMatDescr(&descr);
            // end CUSPARSE context //

            int base, nnzb;
            int mb = (A.num_rows + size_b-1)/size_b;
            // nnzTotalDevHostPtr points to host memory
            int *nnzTotalDevHostPtr = &nnzb;

            magma_indexmalloc( &B->row, mb+1 );
            cusparseXcsr2bsrNnz( cusparseHandle, CUSPARSE_DIRECTION_COLUMN,
                                 A.num_rows, A.num_cols, descr,
                                 A.row, A.col, size_b,
                                 descr, B->row, nnzTotalDevHostPtr );

            if (NULL != nnzTotalDevHostPtr){
                nnzb = *nnzTotalDevHostPtr;
            }else{
                cudaMemcpy(&nnzb, B->row+mb, sizeof(int), 
                                        cudaMemcpyDeviceToHost);
                cudaMemcpy(&base, B->row  , sizeof(int), 
                                        cudaMemcpyDeviceToHost);
                nnzb -= base;
            }
            B->numblocks = nnzb; // number of blocks

            magma_zmalloc( &B->val, nnzb*size_b*size_b );
            magma_indexmalloc( &B->col, nnzb );

            // conversion using CUSPARSE
            cusparseZcsr2bsr( cusparseHandle, CUSPARSE_DIRECTION_ROW,
                              A.num_rows, A.num_cols, descr,
                              A.val, A.row, A.col,
                              size_b, descr,
                              B->val, B->row, B->col);
            
            return MAGMA_SUCCESS; 
        }
        // BCSR to CSR
        if( old_format == Magma_BCSR && new_format == Magma_CSR ){
            //printf( "Conversion to CSR: " );
            // fill in information for B
            B->storage_type = Magma_CSR;
            B->memory_location = A.memory_location;
            B->diameter = A.diameter;

            magma_int_t size_b = A.blocksize;

            // CUSPARSE context //
            cusparseHandle_t cusparseHandle = 0;
            cusparseStatus_t cusparseStatus;
            cusparseStatus = cusparseCreate(&cusparseHandle);
            cusparseMatDescr_t descr = 0;
            cusparseStatus = cusparseCreateMatDescr(&descr);
            // end CUSPARSE context //

            int mb = (A.num_rows + size_b-1)/size_b;
            int nb = (A.num_cols + size_b-1)/size_b;
            int nnzb = A.numblocks; // number of blocks
            B->nnz  = nnzb * size_b * size_b; // number of elements
            B->num_rows = mb * size_b;
            B->num_cols = nb * size_b;

            magma_zmalloc( &B->val, B->nnz );
            magma_indexmalloc( &B->row, B->num_rows+1 );
            magma_indexmalloc( &B->col, B->nnz );

            // conversion using CUSPARSE
            cusparseZbsr2csr( cusparseHandle, CUSPARSE_DIRECTION_ROW,
                              mb, nb, descr, A.val, A.row, A.col, 
                              size_b, descr,
                              B->val, B->row, B->col );


            return MAGMA_SUCCESS; 
        }
        // CSR to CSC   
        if( old_format == Magma_CSR && new_format == Magma_CSC ){
            // fill in information for B
            B->storage_type = Magma_CSC;
            B->memory_location = A.memory_location;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;

            // CUSPARSE context //
            cusparseHandle_t cusparseHandle = 0;
            cusparseStatus_t cusparseStatus;
            cusparseStatus = cusparseCreate(&cusparseHandle);
            cusparseMatDescr_t descr = 0;
            cusparseStatus = cusparseCreateMatDescr(&descr);
            // end CUSPARSE context //

            magma_zmalloc( &B->val, B->nnz );
            magma_indexmalloc( &B->row, B->nnz );
            magma_indexmalloc( &B->col, B->num_cols+1 );

            // conversion using CUSPARSE
            cusparseZcsr2csc(cusparseHandle, A.num_rows, A.num_cols, A.nnz,
                             A.val, A.row, A.col, 
                             B->val, B->row, B->col, 
                             CUSPARSE_ACTION_NUMERIC, 
                             CUSPARSE_INDEX_BASE_ZERO);

            return MAGMA_SUCCESS; 
        }
        // CSC to CSR   
        if( old_format == Magma_CSC && new_format == Magma_CSR ){
            // fill in information for B
            B->storage_type = Magma_CSR;
            B->memory_location = A.memory_location;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;

            // CUSPARSE context //
            cusparseHandle_t cusparseHandle = 0;
            cusparseStatus_t cusparseStatus;
            cusparseStatus = cusparseCreate(&cusparseHandle);
            cusparseMatDescr_t descr = 0;
            cusparseStatus = cusparseCreateMatDescr(&descr);
            // end CUSPARSE context //

            magma_zmalloc( &B->val, B->nnz );
            magma_indexmalloc( &B->row, B->num_rows+1 );
            magma_indexmalloc( &B->col, B->nnz );

            // conversion using CUSPARSE
            cusparseZcsr2csc(cusparseHandle, A.num_rows, A.num_cols, A.nnz,
                             A.val, A.col, A.row, 
                             B->val, B->col, B->row, 
                             CUSPARSE_ACTION_NUMERIC, 
                             CUSPARSE_INDEX_BASE_ZERO);

            return MAGMA_SUCCESS; 
        }
        // CSR to COO
        if( old_format == Magma_CSR && new_format == Magma_COO ){
            // fill in information for B
            B->storage_type = Magma_COO;
            B->memory_location = A.memory_location;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;

            // CUSPARSE context //
            cusparseHandle_t cusparseHandle = 0;
            cusparseStatus_t cusparseStatus;
            cusparseStatus = cusparseCreate(&cusparseHandle);
            cusparseMatDescr_t descr = 0;
            cusparseStatus = cusparseCreateMatDescr(&descr);
            // end CUSPARSE context //

            magma_zmalloc( &B->val, B->nnz );
            magma_indexmalloc( &B->row, B->nnz );
            magma_indexmalloc( &B->col, B->nnz );

            cudaMemcpy( B->val, A.val, A.nnz*sizeof( magmaDoubleComplex ), 
                                                    cudaMemcpyDeviceToDevice );
            cudaMemcpy( B->col, A.col, A.nnz*sizeof( magma_int_t ), 
                                                    cudaMemcpyDeviceToDevice );

            // conversion using CUSPARSE
            cusparseXcsr2coo( cusparseHandle, A.row,
                              A.nnz, A.num_rows, B->row, 
                              CUSPARSE_INDEX_BASE_ZERO );

            return MAGMA_SUCCESS; 
        }
        // COO to CSR
        if( old_format == Magma_COO && new_format == Magma_CSR ){
            // fill in information for B
            B->storage_type = Magma_CSR;
            B->memory_location = A.memory_location;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;

            // CUSPARSE context //
            cusparseHandle_t cusparseHandle = 0;
            cusparseStatus_t cusparseStatus;
            cusparseStatus = cusparseCreate(&cusparseHandle);
            cusparseMatDescr_t descr = 0;
            cusparseStatus = cusparseCreateMatDescr(&descr);
            // end CUSPARSE context //

            magma_zmalloc( &B->val, B->nnz );
            magma_indexmalloc( &B->row, B->nnz );
            magma_indexmalloc( &B->col, B->nnz );

            cudaMemcpy( B->val, A.val, A.nnz*sizeof( magmaDoubleComplex ), 
                                                    cudaMemcpyDeviceToDevice );
            cudaMemcpy( B->col, A.col, A.nnz*sizeof( magma_int_t ), 
                                                    cudaMemcpyDeviceToDevice );

            // conversion using CUSPARSE
            cusparseXcoo2csr( cusparseHandle, A.row,
                              A.nnz, A.num_rows, B->row, 
                              CUSPARSE_INDEX_BASE_ZERO );            

            return MAGMA_SUCCESS; 
        }
        else{
             printf("error: format not supported.\n");
             return MAGMA_ERR_NOT_SUPPORTED;
        }
*/
    }
     
}



   


