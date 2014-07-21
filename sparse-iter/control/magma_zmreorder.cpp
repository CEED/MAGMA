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
    n           magma_int_t
                nodes in one dimension

    @param
    b           magma_int_t
                blocksize

    @param
    B           magma_z_sparse_matrix*
                new matrix filled with new indices


    @ingroup magmasparse_z
    ********************************************************************/

extern "C" magma_int_t
magma_zmreorder( magma_z_sparse_matrix A, magma_int_t n, magma_int_t a,
                 magma_int_t b, magma_int_t c, magma_z_sparse_matrix *B ){

if( A.memory_location == Magma_CPU && A.storage_type == Magma_CSRCOO ){
        
        magma_int_t entry, i, j, k, l;
        magma_z_mtransfer( A, B, Magma_CPU, Magma_CPU );

        magma_index_t *p;
        magma_index_malloc_cpu( &p, A.nnz );

        magma_int_t countn=0;
        for( magma_int_t i=0; i<n; i+=a){
        for( magma_int_t j=0; j<n; j+=b){
        for( magma_int_t k=0; k<n; k+=c){

            for(magma_int_t b1=0; b1<a; b1++){
            for(magma_int_t b2=0; b2<b; b2++){
            for(magma_int_t b3=0; b3<c; b3++){
                    entry = (i+b1)*n*n+(j+b2)*n+(k+b3);
                    p[countn] = entry;


         //           printf("p[%d]=%d\n",countn, entry);
                    countn++;

            }
            }
            }


        }// row
        }// i
        }// p
        countn = 0;
        for( i=0; i<A.num_rows; i++ ){

                for( j=A.row[p[i]]; j<A.row[p[i]+1]; j++ ){
       //         printf("j: %d\n", j);
                    B->val[countn] = A.val[j];
                    B->col[countn] = A.col[j];
                    B->rowidx[countn] = A.rowidx[j];
                    countn++;
                }
        }

        for( i=0; i<A.nnz; i++ ){
            //    printf("B[%d]:  %d  %d  %f\n",i, B->rowidx[i], B->col[i], B->val[i]);
        }



   /*             magma_int_t row=(i+b1)*n*n+(j+b2)*n+(k+b3);
                magma_int_t bound = ( (row+1) < A.num_rows+1 ) ? 
                                            A.row[row+1] : A.num_rows;
                printf("row: %d+(%d+%d)*%d+%d=%d\n",(i+b1)*n*n,j,b2,n,(k+b3), row);
                for( entry=A.row[row]; entry<bound; entry++){
                    p_h[count] = entry;
                    count++;
                }*/



        magma_free_cpu( p );



        
        return MAGMA_SUCCESS; 
    }
    else{

        magma_z_sparse_matrix hA, CSRA;
        magma_z_mtransfer( A, &hA, A.memory_location, Magma_CPU );
        magma_z_mconvert( hA, &CSRA, hA.storage_type, Magma_CSRCOO );

        magma_zmreorder( CSRA, n, a, b, c, B );

        magma_z_mfree( &hA );
        magma_z_mfree( &CSRA );    

        return MAGMA_SUCCESS; 
    }
}





