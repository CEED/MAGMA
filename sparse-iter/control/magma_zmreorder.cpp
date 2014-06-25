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

    Takes a matrix and a reordering scheme such that the output mat

    example:

        / a 0 0 b 0 \
        | 0 c 0 d 0 |
     A= | 0 e f g 0 |       b = 2
        | h 0 0 0 0 |
        \ i j 0 0 0 /

    will generate the projection:
    
    0 2 1 3 4 7 8 9 10 11
    
    according to
    
    a c b d e h f g i j    

    Arguments
    ---------

    @param
    A           magma_z_sparse_matrix
                input/output matrix 

    @param
    b           magma_int_t
                blocksize

    @param
    p           magma_index_t*
                homomorphism vector containing the indices


    @ingroup magmasparse_z
    ********************************************************************/

extern "C" magma_int_t
magma_zmreorder( magma_z_sparse_matrix A, magma_int_t n, magma_int_t b, magma_z_sparse_matrix *B ){

if( A.memory_location == Magma_CPU ){
        
        magma_int_t entry, i, j, k, l;
        magma_int_t rblock, r_blocks = (A.num_rows+b-1)/b;
        magma_int_t cblock, c_blocks = (A.num_cols+b-1)/b; 

        magma_index_t *p_h;
        magma_index_malloc_cpu( &p_h, A.num_rows );

//magma_z_mvisu(A);


        magma_int_t count=0;
        for( magma_int_t i=0; i<n; i+=b){
        for( magma_int_t j=0; j<n; j+=b){
        for( magma_int_t k=0; k<n; k+=b){

            for(magma_int_t b1=0; b1<b; b1++){
            for(magma_int_t b2=0; b2<b; b2++){
            for(magma_int_t b3=0; b3<b; b3++){
                magma_int_t row=(i+b1)*n*n+(j+b2)*n+(k+b3);
                magma_int_t bound = ( (row+1) < A.num_rows+1 ) ? 
                                            A.row[row+1] : A.nnz;

                p_h[row] = count;
                //printf("row: %d+(%d+%d)*%d+%d=%d\n",(i+b1)*n*n,j,b2,n,(k+b3), row);
                for( entry=A.row[row]; entry<bound; entry++){
                    p_h[entry] = count;
                    count++;
                }
            }
            }
            }


        }// row
        }// i
        }// p


        magma_int_t count=0, rowcount=0;
        for( magma_int_t i=0; i<n; i+=b){
        for( magma_int_t j=0; j<n; j+=b){
        for( magma_int_t k=0; k<n; k+=b){

            for(magma_int_t b1=0; b1<b; b1++){
            for(magma_int_t b2=0; b2<b; b2++){
            for(magma_int_t b3=0; b3<b; b3++){
                magma_int_t row=(i+b1)*n*n+(j+b2)*n+(k+b3);
                magma_int_t bound = ( (row+1) < A.num_rows+1 ) ? 
                                            A.row[row+1] : A.nnz;
                
                p_h[count] = row;
                B->row[rowcount] = count;
                //printf("row: %d+(%d+%d)*%d+%d=%d\n",(i+b1)*n*n,j,b2,n,(k+b3), row);
                for( entry=A.row[row]; entry<bound; entry++){
                    B->val[count] = A.val[entry];
                    B->col[count] = p_h[A.col[entry]];
                    B->rowidx[count] = p_h[A.rowidx[entry]];
                    count++;
                }
                rowcount++;
            }
            }
            }


        }// row
        }// i
        }// p
        B->row[rowcount] = count;


  //  for(i=0; i<A.nnz; i++)
  //  printf("%d \n", p_h[i]);
        //magma_setvector( A.nnz, sizeof(magma_index_t), p_h, 1, p, 1 );
        magma_free_cpu( p_h );
        return MAGMA_SUCCESS; 
    }
    else{

        magma_z_sparse_matrix hA, CSRA;
        magma_z_mtransfer( A, &hA, A.memory_location, Magma_CPU );
        magma_z_mconvert( hA, &CSRA, hA.storage_type, Magma_CSRCOO );

        magma_zmhom( CSRA, b, p );

        magma_z_mfree( &hA );
        magma_z_mfree( &CSRA );    

        return MAGMA_SUCCESS; 
    }
}





