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

    Provides for a matrix A, a number of processors num_procs, 
    the processor id proc_id, and a number of matrix powers s, the 
    local matrix B to needed by this processor for the matrix power kernel.

    Arguments
    =========

    magma_sparse_matrix A                input matrix A
    magma_sparse_matrix *B                output matrix B
    magma_int_t num_procs                number of processors for matrix power kernel
    magma_int_t procs_id                 processor id
    magma_int_t s                        matrix powers

    =====================================================================  */



magma_int_t 
magma_z_mpksetup(  magma_z_sparse_matrix A, 
                   magma_z_sparse_matrix *B, 
                   magma_int_t num_procs, 
                   magma_int_t procs_id, 
                   magma_int_t s ){

    if( A.memory_location == Magma_CPU ){
        if( A.storage_type == Magma_CSR ){
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->storage_type = A.storage_type;
            B->memory_location = A.memory_location;
            magma_imalloc_cpu( &B->row, A.num_rows+1 );
                
            magma_int_t blocksize = ( A.num_rows+num_procs-1) / num_procs ;
            magma_int_t local_blocksize = min( blocksize, A.num_rows - procs_id * blocksize);
            //printf("local block size: %d\n", local_blocksize);

            magma_int_t *z, i, j, count, num_new_nnz;
            magma_imalloc_cpu( &z, A.num_rows );

            for( i=0; i<A.num_rows; i++ )
                    z[i] = 0;
            for( i=procs_id*(blocksize); i<(procs_id)*(blocksize)+ local_blocksize; i++ )
                    z[i] = 1;

            // determine the rows of A needed in local matrix B 
            for( count=0; count<s; count++ ){
                for( j=0; j<A.num_rows; j++ ){
                    if ( z[j] == 1 ){
                        for( i=A.row[j]; i<A.row[j+1]; i++ )
                            z[A.col[i]] = 1;
                    }
                }
            }
            
            // fill row pointer of B
            num_new_nnz = 0;
            for( i=0; i<A.num_rows; i++){
                B->row[i] = num_new_nnz;
                if( z[i] != 0 ){
                    num_new_nnz += A.row[i+1]-A.row[i];
                }            
            }
            B->row[B->num_rows] = num_new_nnz;
            B->nnz = num_new_nnz;
            magma_zmalloc_cpu( &B->val, num_new_nnz );
            magma_imalloc_cpu( &B->col, num_new_nnz );

            // fill val and col pointer of B
            num_new_nnz = 0;
            for( j=0; j<A.num_rows; j++){
                if( z[j] != 0 ){
                    for( i=A.row[j]; i<A.row[j+1]; i++ ){
                        B->col[num_new_nnz] = A.col[i];
                        B->val[num_new_nnz] = A.val[i];
                        num_new_nnz++;
                    }
                }
            }
        }
        else{
            magma_z_sparse_matrix C, D;
            magma_z_mconvert( A, &C, A.storage_type, Magma_CSR );
            magma_z_mpksetup(  C, &D, num_procs, procs_id, s );
            magma_z_mconvert( D, B, Magma_CSR, A.storage_type );
            magma_z_mfree(&C);
            magma_z_mfree(&D);
        }
    }
    else{
        magma_z_sparse_matrix C, D;
        magma_z_mtransfer( A, &C, A.memory_location, Magma_CPU );
        magma_z_mpksetup(  C, &D, num_procs, procs_id, s );
        magma_z_mtransfer( D, B, Magma_CPU, A.memory_location );
        magma_z_mfree(&C);
        magma_z_mfree(&D);
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

    Determines for a matrix A, a number of processors num_procs, 
    the processor id proc_id, and a number of matrix powers s, the number 
    of additional rows are needed by this processor for the matrix power kernel.

    Arguments
    =========

    magma_sparse_matrix A                input matrix A
    magma_int_t num_procs                number of processors for matrix power kernel
    magma_int_t procs_id                 processor id
    magma_int_t s                        matrix powers
    magma_int_t *num_add_rows            number of additional rows

    =====================================================================  */

magma_int_t 
magma_z_mpkinfo(   magma_z_sparse_matrix A, 
                   magma_int_t num_procs, 
                   magma_int_t procs_id, 
                   magma_int_t s,    
                   magma_int_t *num_add_rows ){

    if( A.memory_location == Magma_CPU ){
        if( A.storage_type == Magma_CSR ){
            magma_int_t blocksize = ( A.num_rows+num_procs-1) / num_procs ;
            magma_int_t local_blocksize = min( blocksize, A.num_rows - procs_id * blocksize);
            printf("local block size: %d\n", local_blocksize);

            magma_int_t *z, i, j, count;
            magma_imalloc_cpu( &z, A.num_rows );
            for( i=0; i<A.num_rows; i++ )
                    z[i] = 0;
            for( i=procs_id*(blocksize); i<(procs_id)*(blocksize)+ local_blocksize; i++ )
                    z[i] = 1;

            // determine the rows of A needed in local matrix B 
            for( count=0; count<s; count++ ){
                for( j=0; j<A.num_rows; j++ ){
                    if ( z[j] == 1 ){
                        for( i=A.row[j]; i<A.row[j+1]; i++ )
                            z[A.col[i]] = 1;
                    }
                }
            }

            (*num_add_rows) = 0;
            for( i=0; i<procs_id*blocksize; i++){
                if( z[i] != 0 )
                    (*num_add_rows)++;
            }
            for( i=(procs_id+1)*blocksize; i<A.num_rows; i++){
                if( z[i] != 0 )
                    (*num_add_rows)++;
            }
            printf("additional rows: %d\n", (*num_add_rows));
        

        /*
            // this part would allow to determine the additional rows needed
            magma_int_t *add_rows;
            magma_imalloc_cpu( &(add_rows), (*num_add_rows) );
            (*num_add_rows) = 0;
            for( i=0; i<procs_id*blocksize; i++){
                if( z[i] != 0 ){
                    (add_rows)[(*num_add_rows)] = i;
                    (*num_add_rows)++;
                }
            }
            for( i=(procs_id+1)*blocksize; i<A.num_rows; i++){
                if( z[i] != 0 ){
                    (add_rows)[(*num_add_rows)] = i;
                    (*num_add_rows)++;
                }
            }
            for( i=0; i<(*num_add_rows); i++){
                printf("%d  ", (add_rows)[i]);
            }
            printf("\n");
            magma_free_cpu(add_rows);
        */


        }
        else{
            magma_z_sparse_matrix C;
            magma_z_mconvert( A, &C, A.storage_type, Magma_CSR );
            magma_z_mpkinfo(  C, num_procs, procs_id, s, num_add_rows );
            magma_z_mfree(&C);
        }
    }
    else{
        magma_z_sparse_matrix C;
        magma_z_mtransfer( A, &C, A.memory_location, Magma_CPU );
        magma_z_mpkinfo(  C, num_procs, procs_id, s, num_add_rows );
        magma_z_mfree(&C);
    }

   return MAGMA_SUCCESS; 
}
