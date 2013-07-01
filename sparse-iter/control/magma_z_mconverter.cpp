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
#include "common_magma.h"



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

    =====================================================================  */

magma_int_t 
magma_z_mconvert( magma_z_sparse_matrix A, 
                  magma_z_sparse_matrix *B, 
                  magma_storage_t old_format, 
                  magma_storage_t new_format ){

    // check whether matrix on CPU
    if( A.memory_location == Magma_CPU ){

        // CSR to ELLPACK    
        if( old_format == Magma_CSR && new_format == Magma_ELLPACK ){
            // fill in information for B
            B->storage_type = Magma_ELLPACK;
            B->memory_location = A.memory_location;
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
            printf( "Conversion to ELLPACK with %d elements per row: ", maxrowlength );

             B->val = (magmaDoubleComplex*)malloc((maxrowlength*A.num_rows)*sizeof(magmaDoubleComplex));
             B->col = (magma_int_t*)malloc((maxrowlength*A.num_rows)*sizeof(magma_int_t));
             for( magma_int_t i=0; i<(maxrowlength*A.num_rows); i++){
                  B->val[i] = MAGMA_Z_MAKE(0., 0.);
                  B->col[i] =  0;
             }

            //memset(B->val, 0, (maxrowlength*A.num_rows)*sizeof(magmaDoubleComplex));  
            //memset(B->col, 0, (maxrowlength*A.num_rows)*sizeof(magma_int_t));  
   
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
            return MAGMA_SUCCESS; 
        }

        // ELLPACK to CSR
        if( old_format == Magma_ELLPACK && new_format == Magma_CSR ){
            printf( "Conversion to CSR: " );
            // fill in information for B
            B->storage_type = Magma_CSR;
            B->memory_location = A.memory_location;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;

            // conversion
            magma_int_t *row_tmp;
            magmaDoubleComplex *val_tmp;
            row_tmp = (magma_int_t*)malloc((A.num_rows+1)*sizeof(magma_int_t));
            //fill the row-pointer
            for( magma_int_t i=0; i<A.num_rows+1; i++ )
                row_tmp[i] = i*A.max_nnz_row;
            //now use AA_ELL, IA_ELL, row_tmp as CSR with some zeros. 
            //The CSR compressor removes these
            csr_compressor(&A.val, &row_tmp, &A.col, &B->val, &B->row, &B->col, &B->num_rows);  

            printf( "done\n" );      
            return MAGMA_SUCCESS; 
        }        

        // CSR to ELLPACKT
        if( old_format == Magma_CSR && new_format == Magma_ELLPACKT ){
            // fill in information for B
            B->storage_type = Magma_ELLPACKT;
            B->memory_location = A.memory_location;
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
            printf( "Conversion to ELLPACKT with %d elements per row: ", maxrowlength );

             B->val = (magmaDoubleComplex*)malloc((maxrowlength*A.num_rows)*sizeof(magmaDoubleComplex));
             B->col = (magma_int_t*)malloc((maxrowlength*A.num_rows)*sizeof(magma_int_t));
             for( magma_int_t i=0; i<(maxrowlength*A.num_rows); i++){
                  B->val[i] = MAGMA_Z_MAKE(0., 0.);
                  B->col[i] =  0;
             }

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
            return MAGMA_SUCCESS; 
        }

        // ELLPACKT to CSR
        if( old_format == Magma_ELLPACKT && new_format == Magma_CSR ){
            printf( "Conversion to CSR: " ); 
            // fill in information for B
            B->storage_type = Magma_CSR;
            B->memory_location = A.memory_location;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;

            // conversion
            magma_int_t *row_tmp;
            magma_int_t *col_tmp;
            magmaDoubleComplex *val_tmp;
            col_tmp=(magma_int_t*)malloc((A.num_rows*A.max_nnz_row)*sizeof(magma_int_t));
            val_tmp=(magmaDoubleComplex*)malloc((A.num_rows*A.max_nnz_row)*sizeof(magmaDoubleComplex));
            row_tmp=(magma_int_t*)malloc((A.num_rows+1)*sizeof(magma_int_t));
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
            csr_compressor(&val_tmp, &row_tmp, &col_tmp, &B->val, &B->row, &B->col, &B->num_rows); 

            printf( "done\n" );      
            return MAGMA_SUCCESS; 
        }  

        // CSR to DENSE
        if( old_format == Magma_CSR && new_format == Magma_DENSE ){
            printf( "Conversion to DENSE: " );
            // fill in information for B
            B->storage_type = Magma_DENSE;
            B->memory_location = A.memory_location;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;

            // conversion
            B->val = ( magmaDoubleComplex* )malloc((A.num_rows)*(A.num_cols)*sizeof( magmaDoubleComplex ));

             for( magma_int_t i=0; i<(A.num_rows)*(A.num_cols); i++){
                  B->val[i] = MAGMA_Z_MAKE(0., 0.);
             }

            for(magma_int_t i=0; i<A.num_rows; i++ ){
                 for(magma_int_t j=A.row[i]; j<A.row[i+1]; j++ )
                     B->val[i * (A.num_rows) + A.col[j] ] = A.val[ j ];
            }

            printf( "done\n" );      
            return MAGMA_SUCCESS; 
        }

        // DENSE to CSR
        if( old_format == Magma_DENSE && new_format == Magma_CSR ){
            printf( "Conversion to CSR: " );
            // fill in information for B
            B->storage_type = Magma_CSR;
            B->memory_location = A.memory_location;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;

            // conversion

            B->nnz=0;
            for( magma_int_t i=0; i<(A.num_rows)*(A.num_cols); i++ ){
                if( MAGMA_Z_REAL(A.val[i])!=0.0 )
                    (B->nnz)++;
            }

            B->val = ( magmaDoubleComplex* )malloc((B->nnz)*sizeof( magmaDoubleComplex ));
            B->col = ( magma_int_t* )malloc((B->nnz)*sizeof( magma_int_t ));
            B->row = ( magma_int_t* )malloc((B->num_rows+1)*sizeof( magma_int_t ));

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

            printf( "done\n" );      
            return MAGMA_SUCCESS; 
        }

        else{

            printf("error: formet not supported.\n");
            // return some MAGMA Error

        }
     

    } // end CPU case

    else{

        printf("error: matrix not on CPU.\n");
        // return some MAGMA Error

    }
     
}



   


