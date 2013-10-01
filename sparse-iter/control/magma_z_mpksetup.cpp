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

    Provides for a matrix A, a chunksize, an offset, 
    and a number of matrix powers s, the 
    local matrix B to needed by this processor for the matrix power kernel.

    Arguments
    =========

    magma_sparse_matrix A                input matrix A
    magma_sparse_matrix *B               output matrix B
    magma_int_t offset                   first row to compute
    magma_int_t chunksize                number of rows to compute
    magma_int_t s                        matrix powers

    =====================================================================  */



magma_int_t 
magma_z_mpksetup_one(  magma_z_sparse_matrix A, 
                       magma_z_sparse_matrix *B, 
                       magma_int_t offset, 
                       magma_int_t chunksize, 
                       magma_int_t s ){

    if( A.memory_location == Magma_CPU ){
        if( A.storage_type == Magma_CSR ){
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->storage_type = A.storage_type;
            B->memory_location = A.memory_location;
            magma_imalloc_cpu( &B->row, A.num_rows+1 );
                
            magma_int_t start = offset;
            magma_int_t end = offset + chunksize;
            //printf("local block size: %d offset: %d\n", chunksize, offset);

            magma_int_t *z1, *z2, i, j, count, num_new_nnz;
            magma_imalloc_cpu( &z1, A.num_rows );
            magma_imalloc_cpu( &z2, A.num_rows );
            for( i=0; i<A.num_rows; i++ )
                    z1[i] = 0;
            for( i=0; i<A.num_rows; i++ )
                    z2[i] = 0;
            for( i=start; i<end; i++ )
                    z1[i] = 1;

            // determine the rows of A needed in local matrix B 
            for( count=0; count<s-1; count++ ){
                for( j=0; j<A.num_rows; j++ ){
                    if ( z1[j] == 1 ){
                        for( i=A.row[j]; i<A.row[j+1]; i++ )
                            z2[A.col[i]] = 1;
                    }
                }
                for( i=0; i<A.num_rows; i++){
                    if( z1[i] == 1 || z2[i] ==1 )
                        z1[i] = 1; 
                }  
            }   
            // fill row pointer of B
            num_new_nnz = 0;
            for( i=0; i<A.num_rows; i++){
                B->row[i] = num_new_nnz;
                if( z1[i] != 0 ){
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
                if( z1[j] != 0 ){
                    for( i=A.row[j]; i<A.row[j+1]; i++ ){
                        B->col[num_new_nnz] = A.col[i];
                        B->val[num_new_nnz] = A.val[i];
                        num_new_nnz++;
                    }
                }
            }
            magma_free_cpu(z1);
            magma_free_cpu(z2);
        }
        else{
            magma_z_sparse_matrix C, D;
            magma_z_mconvert( A, &C, A.storage_type, Magma_CSR );
            magma_z_mpksetup_one(  C, &D, offset, chunksize, s );
            magma_z_mconvert( D, B, Magma_CSR, A.storage_type );
            magma_z_mfree(&C);
            magma_z_mfree(&D);
        }
    }
    else{
        magma_z_sparse_matrix C, D;
        magma_z_mtransfer( A, &C, A.memory_location, Magma_CPU );
        magma_z_mpksetup_one(  C, &D, offset, chunksize, s );
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

    Provides for a matrix A, a number of processors num_procs, 
    and a distribution *offset, *chunksize and s a set of matrices B
    each containing the matrix rows to compute the matrix power kernel.

    Arguments
    =========

    magma_sparse_matrix A                input matrix A
    magma_sparse_matrix B[MagmaMaxGPUs]  set of output matrices B
    magma_int_t num_procs                number of processors
    magma_int_t *offset                  array containing the offsets
    magma_int_t *chunksize               array containing the chunk sizes
    magma_int_t s                        matrix powers

    =====================================================================  */



magma_int_t 
magma_z_mpksetup(  magma_z_sparse_matrix A, 
                   magma_z_sparse_matrix B[MagmaMaxGPUs], 
                   magma_int_t num_procs,
                   magma_int_t *offset, 
                   magma_int_t *chunksize, 
                   magma_int_t s ){

    for(int procs=0; procs<num_procs; procs++){
            magma_z_mpksetup_one( A, &B[procs], offset[procs], chunksize[procs], s );        
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

    Provides for a matrix A, a chunksize, an offset, 
    and a number of matrix powers s, the number of additional rows 
    that are needed by this processor for the matrix power kernel.

    Arguments
    =========

    magma_sparse_matrix A                input matrix A
    magma_int_t offset                   first row to compute
    magma_int_t chunksize                number of rows to compute
    magma_int_t s                        matrix powers
    magma_int_t *num_add_rows            number of additional rows
    magma_int_t *add_rows                array for additional rows
    magma_int_t *num_add_vecs            number of additional vector entries
    magma_int_t *add_vecs                array for additional vector entries

    =====================================================================  */

magma_int_t 
magma_z_mpkinfo_one( magma_z_sparse_matrix A, 
                     magma_int_t offset, 
                     magma_int_t chunksize, 
                     magma_int_t s,    
                     magma_int_t *num_add_rows,
                     magma_int_t **add_rows,
                     magma_int_t *num_add_vecs,
                     magma_int_t **add_vecs ){

    if( A.memory_location == Magma_CPU ){
        if( A.storage_type == Magma_CSR ){
            magma_int_t start = offset;
            magma_int_t end = offset + chunksize;

            magma_int_t *z1, *z2, i, j, count;
            magma_imalloc_cpu( &z1, A.num_rows );
            magma_imalloc_cpu( &z2, A.num_rows );
            for( i=0; i<A.num_rows; i++ )
                    z1[i] = 0;
            for( i=0; i<A.num_rows; i++ )
                    z2[i] = 0;
            for( i=start; i<end; i++ )
                    z1[i] = 1;

            // determine the rows of A needed in local matrix B 
            for( count=0; count<s-1; count++ ){
                for( j=0; j<A.num_rows; j++ ){
                    if ( z1[j] == 1 ){
                        for( i=A.row[j]; i<A.row[j+1]; i++ )
                            z2[A.col[i]] = 1;
                    }
                }
                for( i=0; i<A.num_rows; i++){
                    if( z1[i] == 1 || z2[i] ==1 )
                        z1[i] = 1; 
                }  
            }   
            (*num_add_rows) = 0;
            for( i=0; i<start; i++){
                if( z1[i] != 0 )
                    (*num_add_rows)++;
            }
            for( i=end; i<A.num_rows; i++){
                if( z1[i] != 0 )
                    (*num_add_rows)++;
            }
            // this part determines the additional rows needed
            magma_imalloc_cpu( (add_rows), (*num_add_rows) );
            (*num_add_rows) = 0;
            for( i=0; i<start; i++){
                if( z1[i] != 0 ){
                    (*add_rows)[(*num_add_rows)] = i;
                    (*num_add_rows)++;
                }
            }
            for( i=end; i<A.num_rows; i++){
                if( z1[i] != 0 ){
                    (*add_rows)[(*num_add_rows)] = i;
                    (*num_add_rows)++;
                }
            }  


            // for the vector entries, we need one more loop
            for( j=0; j<A.num_rows; j++ ){
                if ( z1[j] == 1 ){
                    for( i=A.row[j]; i<A.row[j+1]; i++ )
                        z2[A.col[i]] = 1;
                }
            }
            for( i=0; i<A.num_rows; i++){
                if( z1[i] == 1 || z2[i] ==1 )
                    z1[i] = 1; 
            }  
            (*num_add_vecs) = 0;
            for( i=0; i<start; i++){
                if( z1[i] != 0 )
                    (*num_add_vecs)++;
            }
            for( i=end; i<A.num_rows; i++){
                if( z1[i] != 0 )
                    (*num_add_vecs)++;
            }
            // this part determines the additional vector entries needed
            magma_imalloc_cpu( (add_vecs), (*num_add_vecs) );
            (*num_add_vecs) = 0;
            for( i=0; i<start; i++){
                if( z1[i] != 0 ){
                    (*add_vecs)[(*num_add_vecs)] = i;
                    (*num_add_vecs)++;
                }
            }
            for( i=end; i<A.num_rows; i++){
                if( z1[i] != 0 ){
                    (*add_vecs)[(*num_add_vecs)] = i;
                    (*num_add_vecs)++;
                }
            }  


            magma_free_cpu(z1);
            magma_free_cpu(z2);


        }
        else{
            magma_z_sparse_matrix C;
            magma_z_mconvert( A, &C, A.storage_type, Magma_CSR );
            magma_z_mpkinfo_one(  C, offset, chunksize, s, num_add_rows, add_rows, num_add_vecs, add_vecs );
            magma_z_mfree(&C);
        }
    }
    else{
        magma_z_sparse_matrix C;
        magma_z_mtransfer( A, &C, A.memory_location, Magma_CPU );
        magma_z_mpkinfo_one(  C, offset, chunksize, s, num_add_rows, add_rows, num_add_vecs, add_vecs );
        magma_z_mfree(&C);
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

    Provides for a matrix A, a number of processors num_procs, 
    and a distribution *offset, *chunksize and s the number of 
    rows added to the different matrices.

    Arguments
    =========

    magma_sparse_matrix A                input matrix A
    magma_int_t num_procs                number of processors
    magma_int_t *offset                  array containing the offsets
    magma_int_t *chunksize               array containing the chunk sizes
    magma_int_t s                        matrix powers
    magma_int_t *num_add_rows            output array: number of additional rows
    magma_int_t *num_add_vecs            number of additional vector entries
    magma_int_t *add_vecs                array for additional vector entries

    =====================================================================  */



magma_int_t 
magma_z_mpkinfo(   magma_z_sparse_matrix A, 
                   magma_int_t num_procs,
                   magma_int_t *offset, 
                   magma_int_t *chunksize, 
                   magma_int_t s,
                   magma_int_t *num_add_rows,
                   magma_int_t **add_rows,
                   magma_int_t *num_add_vecs,
                   magma_int_t **add_vecs ){

    for(int procs=0; procs<num_procs; procs++){
        magma_z_mpkinfo_one( A, offset[procs], chunksize[procs], s, &num_add_rows[procs], &add_rows[procs], &num_add_vecs[procs], &add_vecs[procs] ); 
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

    Packs for a given vector x and a key add_rows a compressed version
    where the add_rows are in consecutive order.

    Arguments
    =========

    magma_int_t num_add_rows             number of elements to pack
    magma_int_t *add_rows                indices of elements to pack
    magmaDoubleComplex *x                uncompressed input vector
    magmaDoubleComplex *y                compressed output vector

    =====================================================================  */



magma_int_t 
magma_z_mpk_compress(    magma_int_t num_add_rows,
                         magma_int_t *add_rows,
                         magmaDoubleComplex *x,
                         magmaDoubleComplex *y ){
    #pragma unroll
    for(magma_int_t i=0; i<num_add_rows; i++){
        y[i] = x[ add_rows[i] ];
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

    Unpacks for a given vector x and a key add_rows the compressed version.

    Arguments
    =========

    magma_int_t num_add_rows             number of elements to pack
    magma_int_t *add_rows                indices of elements to pack
    magmaDoubleComplex *x                compressed input vector
    magmaDoubleComplex *y                uncompressed output vector

    =====================================================================  */



magma_int_t 
magma_z_mpk_uncompress(  magma_int_t num_add_rows,
                         magma_int_t *add_rows,
                         magmaDoubleComplex *x,
                         magmaDoubleComplex *y ){
    #pragma unroll
    for(magma_int_t i=0; i<num_add_rows; i++){
        y[add_rows[i]] = x[ i ];
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

    Unpacks for a given vector x and a key add_rows the compressed version.
    The selective version uncompresses only vectors within the area
    [ offset , offset+blocksize ]

    Arguments
    =========

    magma_int_t num_add_rows             number of elements to pack
    magma_int_t *add_rows                indices of elements to pack
    magma_int_t offset                   lower bound of vector chunk
    magma_int_t blocksize                number of locally computed elements
    magmaDoubleComplex *x                compressed input vector
    magmaDoubleComplex *y                uncompressed output vector

    =====================================================================  */



magma_int_t 
magma_z_mpk_uncompress_sel(    magma_int_t num_add_rows,
                         magma_int_t *add_rows,
                         magma_int_t offset,
                         magma_int_t blocksize,
                         magmaDoubleComplex *x,
                         magmaDoubleComplex *y ){
    #pragma unroll
    for(magma_int_t i=0; i<num_add_rows; i++){
 //       if( offset <= add_rows[i] )//&& add_rows[i] < offset+blocksize)
            y[add_rows[i]] = x[ i ];
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

    Packs the local matrix for higher SpMV performance.

    Arguments
    =========

    magma_int_t num_add_rows             number of elements to pack
    magma_int_t *add_rows                indices of elements to pack
    magmaDoubleComplex *x                uncompressed input vector
    magmaDoubleComplex *y                compressed output vector

    =====================================================================  */


magma_int_t
magma_z_mpk_mcompresso(      magma_z_sparse_matrix A,
                             magma_z_sparse_matrix *B,
                             magma_int_t offset,
                             magma_int_t blocksize,
                             magma_int_t num_add_rows,
                             magma_int_t *add_rows ){
    if( A.storage_type==Magma_CSR && A.memory_location==Magma_CPU ){
            int i=0,j,n=0,nnz=0;
            B->storage_type = Magma_CSR;
            B->memory_location = A.memory_location;
            B->num_rows = blocksize + num_add_rows;
            B->num_cols = A.num_cols;
            magma_imalloc_cpu( &B->row, B->num_rows+1 );
            for( j=0; j<blocksize+1; j++ ){
                (B->row)[n] = A.row[j+offset] - A.row[offset];
                i = (B->row)[j];
                n++;
            }
            for( j=0; j<num_add_rows; j++ ){
                i += A.row[add_rows[j]+1] - A.row[add_rows[j]];
                (B->row)[n] = i;
                n++;
            }
            B->nnz = i;
            magma_imalloc_cpu( &B->col, B->nnz );
            magma_zmalloc_cpu( &B->val, B->nnz );

            for( j=A.row[offset]; j<A.row[offset+blocksize]; j++ ){
                B->col[nnz] = A.col[j];
                B->val[nnz] = A.val[j];
                nnz++;
            }

            for( j=0; j<num_add_rows; j++ ){
                for( i=A.row[add_rows[j]]; i<A.row[add_rows[j]+1]; i++){
                    B->col[nnz] = A.col[i];
                    B->val[nnz] = A.val[i];
                    nnz++;
                }
            }
            //printf("\n");magma_z_mvisu(*B);printf("\n");
    }
    else{
        printf("not supported!\n");
    }

    return MAGMA_SUCCESS; 
}




