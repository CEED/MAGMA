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

    Takes a matrix obtained from a 3D FE discretization, and reorders it
    by restructuring the node numbering with 3D blocks of size a x b x c.
    the resulting matrix has some row-blocks reordered. Rows related 
    to spatially nearby nodes (within one of these blocks) are then also
    close in the matrix.


    Arguments
    ---------

    @param
    A           magma_z_sparse_matrix
                input/output matrix 

    @param
    n           magma_int_t
                nodes in one dimension

    @param
    a           magma_int_t
                blocksize x-discretization direction

    @param
    b           magma_int_t
                blocksize y-discretization direction

    @param
    c           magma_int_t
                blocksize z-discretization direction

    @param
    B           magma_z_sparse_matrix*
                new matrix filled with new indices


    @ingroup magmasparse_z
    ********************************************************************/

extern "C" magma_int_t
magma_zmreorder( magma_z_sparse_matrix A, magma_int_t n, magma_int_t a,
                 magma_int_t b, magma_int_t c, magma_z_sparse_matrix *B ){

if( A.memory_location == Magma_CPU && A.storage_type == Magma_CSRCOO ){
        
        magma_index_t entry;
        magma_z_mtransfer( A, B, Magma_CPU, Magma_CPU );

        magma_index_t *p;
        magma_index_malloc_cpu( &p, A.num_rows );

        magma_int_t countn=0;
        for( magma_index_t i=0; i<n; i+=a){
        for( magma_index_t j=0; j<n; j+=b){
        for( magma_index_t k=0; k<n; k+=c){

            for(magma_index_t b1=0; b1<a; b1++){
            for(magma_index_t b2=0; b2<b; b2++){
            for(magma_index_t b3=0; b3<c; b3++){
                    entry = (i+b1)*n*n+(j+b2)*n+(k+b3);
                    p[countn] = entry;
                    //printf("p[%d]=%d\n",countn, entry);
                    countn++;

            }
            }
            }


        }// i
        }// j
        }// k
        countn = 0;
        for( magma_index_t i=0; i<A.num_rows; i++ ){

                for( magma_index_t j=A.row[p[i]]; j<A.row[p[i]+1]; j++ ){
       //         printf("j: %d\n", j);
                    B->val[countn] = A.val[j];
                    B->col[countn] = A.col[j];
                    B->rowidx[countn] = A.rowidx[j];
                    countn++;
                }
        }

        magma_free_cpu( p );

/*
            // print the reordering
            magma_z_sparse_matrix Z;
            Z.storage_type = Magma_DENSE;
            Z.memory_location = A.memory_location;
            Z.num_rows = A.num_rows;
            Z.num_cols = A.num_cols;
            Z.nnz = A.nnz;

            // conversion
            magma_zmalloc_cpu( &Z.val, A.num_rows*A.num_cols );

            for( magma_int_t i=0; i<(A.num_rows)*(A.num_cols); i++){
                Z.val[i] = MAGMA_Z_MAKE(0., 0.);
            }

            for(magma_int_t i=0; i<A.nnz; i++ ){
                    Z.val[ B->rowidx[i] * (B->num_cols) + B->col[i] ] =  
                                            MAGMA_Z_MAKE( (double) i, 0.0 );
            }
            magma_z_mvisu( Z );

            magma_z_mfree( &Z );
            // end print the reordering
*/

        
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





