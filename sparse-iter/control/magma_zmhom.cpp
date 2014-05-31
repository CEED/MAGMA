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

    Takes a matrix and a blocksize b to generate a homomorphism that
    orders the matrix entries according to the subdomains of size b x b.
    Returns p on the device

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
    =========

    magma_z_sparse_matrix A          input/output matrix 
    magma_int_t b                    blocksize
    magma_index_t *p                 homomorphism vector containing the indices

    ========================================================================  */




extern "C" magma_int_t
magma_zmhom( magma_z_sparse_matrix A, magma_int_t b, magma_index_t *p ){

    if( A.memory_location == Magma_CPU && A.storage_type == Magma_CSRCOO ){
        
        magma_int_t i, j, k, l;
        magma_int_t rblock, r_blocks = (A.num_rows+b-1)/b;
        magma_int_t cblock, c_blocks = (A.num_cols+b-1)/b; 

        magma_index_t *p_h;
        magma_indexmalloc_cpu( &p_h, A.nnz );

        j=0;
        for( rblock=0; rblock<r_blocks; rblock++){
        for( cblock=0; cblock<c_blocks; cblock++){
            magma_int_t bound = A.nnz;
            bound = ( (rblock+1)*b < A.num_rows ) ? 
                                            A.row[(rblock+1)*b] : A.nnz;
            //printf("bound:%d\n", bound);
            //for( i=A.row[rblock*b]; i<bound; i++){
            for( i=A.row[rblock*b]; i<bound; i++){
                if( ( cblock*b <= A.col[i] && A.col[i] < (cblock+1)*b )
                        && 
                   ( rblock*b <= A.rowidx[i] && A.rowidx[i] < (rblock+1)*b ) ){

                    // insert this index at this point at the homomorphism
                        p_h[j] = i;
                        j++;
                 }
            }
        }// cblocks
        }// rblocks
        magma_setvector( A.nnz, sizeof(magma_index_t), p_h, 1, p, 1 );
/*
// debug
        magma_getvector( A.nnz, sizeof(magma_index_t), p, 1, p_h, 1 );

printf("A.nnz:%d\n", A.nnz);
for(int z=A.nnz-10; z<A.nnz; z++)
    printf("%d ", p_h[z]); 


        magma_index_t *t1, *t2;
        magma_indexmalloc_cpu( &t1, A.nnz );
        magma_indexmalloc_cpu( &t2, A.nnz );
for(int z=0; z<A.nnz; z++)
    t1[z] = 0;
for(int z=0; z<A.nnz; z++)
    t1[p_h[z]]++;

printf("\ndoubles:\n");
for(int z=0; z<A.nnz; z++){
    if( t1[z]>1 )
    printf(" double entry %d\n", z);
    }
    
printf("end\n");
// debug*/
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



