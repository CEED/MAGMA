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

    Helper function to compress CSR containing zero-entries.


    Arguments
    =========

    magma_z_sparse_matrix A              sparse matrix A    
    magma_z_sparse_matrix *B             copy of A in new format      
    magma_storage_t old_format           original storage format
    magma_storage_t new_format           new storage format
    magma_mmajor_t old_major             old major type
    magma_mmajor_t new_major             new major type
   

    =====================================================================  */


magma_int_t csr_compressor(magmaDoubleComplex ** val, magma_int_t ** row, magma_int_t ** col, magmaDoubleComplex ** valn, magma_int_t ** rown, magma_int_t ** coln, magma_int_t *n)
{
    magma_int_t nnz_new=0; 
    for( magma_int_t i=0; i<(*row)[*n]; i++ )
        if( MAGMA_Z_REAL((*val)[i]) != 0 )
            nnz_new++;

    (*valn)=(magmaDoubleComplex*)malloc(nnz_new*sizeof(magmaDoubleComplex));
    (*coln)=(magma_int_t*)malloc(nnz_new*sizeof(magma_int_t));
    (*rown)=(magma_int_t*)malloc((*n+1)*sizeof(magma_int_t));
      
  
    magma_int_t nzcount=-1, nz_row=0;
    for(magma_int_t i=0; i<*n; i++)
    {
        nz_row=-1;
        for(magma_int_t nzj=(*row)[i]; nzj<(*row)[i+1]; nzj++)
        {
            if( MAGMA_Z_REAL((*val)[nzj]) != 0.0 )
            {
                  nzcount++;
                  nz_row++;
                  if(nz_row==0)
                      (*rown)[i]= nzcount;
          
            (*valn)[nzcount]= (*val)[nzj];
            (*coln)[nzcount]= (*col)[nzj];
            }
        }   
    }
    (*rown)[*n]=nnz_new;
    return MAGMA_SUCCESS;
}





/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======

    Converter between different sparse storage formats.


    Arguments
    =========

    magma_z_sparse_matrix A              sparse matrix A    
    magma_z_sparse_matrix *B             copy of A in new format      
    magma_storage_t old_format           original storage format
    magma_storage_t new_format           new storage format
    magma_mmajor_t old_major             old major type
    magma_mmajor_t new_major             new major type
   

    =====================================================================  */

magma_int_t 
magma_z_mconvert( magma_z_sparse_matrix A, 
                  magma_z_sparse_matrix *B, 
                  magma_storage_t old_format, 
                  magma_storage_t new_format,
                  magma_mmajor_t old_major,
                  magma_mmajor_t new_major ){

    // first case: CSR to ELLPACK
    if( old_format == Magma_CSR && new_format == Magma_ELLPACK ){

        // ELLPACK in RowMajor
        if( new_major == Magma_RowMajor ){
            // fill in information for B
            B->storage_type = Magma_ELLPACK;
            B->memory_location = A.memory_location;
            B->major_type = Magma_RowMajor;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;

            // conversion
            magma_int_t i, j, *length, maxrowlength=0;
            length = (magma_int_t*)malloc((A.num_rows)*sizeof(magma_int_t));

            #pragma omp parallel for
            for( i=0; i<A.num_rows; i++ ){
                length[i] = A.row[i+1]-A.row[i];
                if(length[i] > maxrowlength)
                     maxrowlength = length[i];
            }
            printf( "Conversion to ELLPACK RowMajor with %d elements per row: ", maxrowlength );

             B->val = (magmaDoubleComplex*)malloc((maxrowlength*A.num_rows)*sizeof(magmaDoubleComplex));
             B->col = (magma_int_t*)malloc((maxrowlength*A.num_rows)*sizeof(magma_int_t));
            #pragma omp parallel for
            for( i=0; i<A.num_rows; i++ ){
                magma_int_t offset = 0;
                for( j=A.row[i]; j<A.row[i+1]; j++ ){
                     B->val[offset*A.num_rows+i] = A.val[j];
                     B->col[offset*A.num_rows+i] = A.col[j];
                     offset++;
                }
            }
            B->max_nnz_row = maxrowlength;
            printf( "done\n" );
        }

        // ELLPACK in ColMajor    
        if( new_major == Magma_ColMajor ){
            // fill in information for B
            B->storage_type = Magma_ELLPACK;
            B->memory_location = A.memory_location;
            B->major_type = Magma_ColMajor;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;

            // conversion
            magma_int_t i, j, *length, maxrowlength=0;
            length = (magma_int_t*)malloc((A.num_rows)*sizeof(magma_int_t));

            #pragma omp parallel for
            for( i=0; i<A.num_rows; i++ ){
                length[i] = A.row[i+1]-A.row[i];
                if(length[i] > maxrowlength)
                     maxrowlength = length[i];
            }
            printf( "Conversion to ELLPACK RowMajor with %d elements per row: ", maxrowlength );

             B->val = (magmaDoubleComplex*)malloc((maxrowlength*A.num_rows)*sizeof(magmaDoubleComplex));
             B->col = (magma_int_t*)malloc((maxrowlength*A.num_rows)*sizeof(magma_int_t));
   
            #pragma omp parallel for
            for( i=0; i<A.num_rows; i++ ){
                 magma_int_t offset = 0;
                 for( j=A.row[i]; j<A.row[i+1]; j++ ){
                     B->val[i*maxrowlength+offset] = A.val[j];
                     B->col[i*maxrowlength+offset] = A.col[j];
                     offset++;
                 }
            }
            B->max_nnz_row = maxrowlength;
            printf( "done\n" );
        }
    }

    // second case: ELLPACK to CSR
    if( old_format == Magma_ELLPACK && new_format == Magma_CSR ){

        printf( "Conversion to CSR RowMajor: " );

        // ELLPACK in RowMajor
        if( old_major == Magma_RowMajor ){
            // fill in information for B
            B->storage_type = Magma_CSR;
            B->memory_location = A.memory_location;
            B->major_type = Magma_RowMajor;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;

            // conversion
            magma_int_t *row_tmp;
            magma_int_t *col_tmp;
            magmaDoubleComplex *val_tmp;
            col_tmp=(magma_int_t*)malloc((A.num_rows*A.max_nnz_row)*sizeof(magma_int_t));
            val_tmp=(magmaDoubleComplex*)malloc((A.num_rows*A.max_nnz_row)*sizeof(float));
            row_tmp=(magma_int_t*)malloc((A.num_rows+1)*sizeof(magma_int_t));
            //fill the row-pointer
            for( magma_int_t i=0; i<A.num_rows+1; i++ )
                row_tmp[i] = i*A.max_nnz_row;
            //transform RowMajor to ColMajor
            for( magma_int_t j=0;j<A.max_nnz_row;j++ ){
                for( magma_int_t i=0;i<A.num_rows;i++ ){
                    col_tmp[i*A.max_nnz_row+j] = A.col[j*A.num_rows+i];
                    val_tmp[i*A.max_nnz_row+j] = A.val[j*A.num_rows+i];
        //            printf(" inserted %f at %d\n",A.val[j*A.num_rows+i], A.col[j*A.num_rows+i]);
          //          printf(" inserted %f at %d\n",val_tmp[i*A.max_nnz_row+j], col_tmp[i*A.max_nnz_row+j]);
                }
            }    
            //now use AA_ELL, IA_ELL, row_tmp as CSR with some zeros. 
            //The CSR compressor removes these
            csr_compressor(&val_tmp, &row_tmp, &col_tmp, &B->val, &B->row, &B->col, &B->num_rows);  
        }  



        // ELLPACK in ColMajor
        if( old_major == Magma_ColMajor ){
            // fill in information for B
            B->storage_type = Magma_CSR;
            B->memory_location = A.memory_location;
            B->major_type = Magma_RowMajor;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;

            // conversion
            magma_int_t *row_tmp;
            magmaDoubleComplex *val_tmp;
            row_tmp=(magma_int_t*)malloc((A.num_rows+1)*sizeof(magma_int_t));
            //fill the row-pointer
            for( magma_int_t i=0; i<A.num_rows+1; i++ )
                row_tmp[i] = i*A.max_nnz_row;
            //now use AA_ELL, IA_ELL, row_tmp as CSR with some zeros. 
            //The CSR compressor removes these
            csr_compressor(&A.val, &row_tmp, &A.col, &B->val, &B->row, &B->col, &B->num_rows);  
        }  
      
            printf( "done\n" );      

    }

    return MAGMA_SUCCESS;  
}



   


