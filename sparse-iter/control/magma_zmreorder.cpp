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
magma_zmreorder3d( magma_z_sparse_matrix A, magma_int_t n, magma_int_t a,
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


        // randomize : generate a random reordering for p
        for( magma_index_t i=0; i<A.num_rows; i++){
                magma_index_t idx = rand()%A.num_rows;
                magma_index_t tmp = p[i];
                p[i] = p[idx];
                p[idx] = tmp;
        }   

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

        magma_zmreorder3d( CSRA, n, a, b, c, B );

        magma_z_mfree( &hA );
        magma_z_mfree( &CSRA );    

        return MAGMA_SUCCESS; 
    }
}


/** -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    -------

    Takes a matrix obtained from a 2D FE discretization, and reorders it
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
    B           magma_z_sparse_matrix*
                new matrix filled with new indices


    @ingroup magmasparse_z
    ********************************************************************/

extern "C" magma_int_t
magma_zmreorder2d( magma_z_sparse_matrix A, magma_int_t n, magma_int_t a,
                 magma_int_t b, magma_z_sparse_matrix *B ){

if( A.memory_location == Magma_CPU && A.storage_type == Magma_CSRCOO ){
        
        magma_index_t entry;
        magma_z_mtransfer( A, B, Magma_CPU, Magma_CPU );

        magma_index_t *p;
        magma_index_malloc_cpu( &p, A.num_rows );

        magma_int_t countn=0;
        for( magma_index_t i=0; i<n; i+=a){
        for( magma_index_t j=0; j<n; j+=b){

            for(magma_index_t b1=0; b1<a; b1++){
            for(magma_index_t b2=0; b2<b; b2++){
                    entry = (i+b1)*n+(j+b2);
                    p[countn] = entry;
                    //printf("p[%d]=%d\n",countn, entry);
                    countn++;

            }
            }


        }// i
        }// j

        // randomize : generate a random reordering for p
        for( magma_index_t i=0; i<A.num_rows; i++){
                magma_index_t idx = rand()%A.num_rows;
                magma_index_t tmp = p[i];
                p[i] = p[idx];
                p[idx] = tmp;
        }   

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

        magma_zmreorder2d( CSRA, n, a, b, B );

        magma_z_mfree( &hA );
        magma_z_mfree( &CSRA );    

        return MAGMA_SUCCESS; 
    }
}



























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
magma_zmreorder2( magma_z_sparse_matrix A, magma_int_t n, magma_int_t a,
                 magma_int_t b, magma_int_t c, magma_z_sparse_matrix *B ){

if( A.memory_location == Magma_CPU && A.storage_type == Magma_CSRCOO ){
        
        B->storage_type = A.storage_type;
        B->diagorder_type = A.diagorder_type;
        B->memory_location = Magma_CPU;
        B->num_rows = A.num_rows;
        B->num_cols = A.num_cols;
   
        magma_index_t i,j,k=0,entry;
        magma_index_malloc_cpu( &B->col, A.nnz*10 );
        magma_index_malloc_cpu( &B->row, A.num_rows+1 );
        magma_index_malloc_cpu( &B->rowidx, A.nnz*10 );
        magma_zmalloc_cpu( &B->val, A.nnz*10 );
        
    

        for( i=0; i<n; i++ ){
            B->row[i] = k;
        for( j=0; j<A.nnz; j++ ){
            if(     abs(i-A.col[j]) == 0  ||
                    abs(i-A.col[j]) == 1  ||
                    abs(i-A.col[j]) == n-1  ||
                    abs(i-A.col[j]) == n  ||
                    abs(i-A.col[j]) == n+1  ||
                    abs(i-A.col[j]) == n*n-n-1  ||
                    abs(i-A.col[j]) == n*n-n  ||
                    abs(i-A.col[j]) == n*n-n+1  ||
                    abs(i-A.col[j]) == n*n-1  ||
                    abs(i-A.col[j]) == n*n  ||
                    abs(i-A.col[j]) == n*n+1  ||
                    abs(i-A.col[j]) == n*n+n-1  ||
                    abs(i-A.col[j]) == n*n+n  ||
                    abs(i-A.col[j]) == n*n+n+1  ||
                    abs(i-A.rowidx[j]) == 0  ||
                    abs(i-A.rowidx[j]) == 1  ||
                    abs(i-A.rowidx[j]) == n-1  ||
                    abs(i-A.rowidx[j]) == n  ||
                    abs(i-A.rowidx[j]) == n+1  ||
                    abs(i-A.rowidx[j]) == n*n-n-1  ||
                    abs(i-A.rowidx[j]) == n*n-n  ||
                    abs(i-A.rowidx[j]) == n*n-n+1  ||
                    abs(i-A.rowidx[j]) == n*n-1  ||
                    abs(i-A.rowidx[j]) == n*n  ||
                    abs(i-A.rowidx[j]) == n*n+1  ||
                    abs(i-A.rowidx[j]) == n*n+n-1  ||
                    abs(i-A.rowidx[j]) == n*n+n  ||
                    abs(i-A.rowidx[j]) == n*n+n+1  ){

                B->col[k] = A.col[j];
                B->rowidx[k] = A.rowidx[j];
                B->val[k] = A.val[j];

                //printf("A.nnz:%d i:%d  k: %d<%d (%d %d) %f\n", A.nnz, i, k, A.nnz*3, B->rowidx[k], B->col[k], B->val[k]);
                k++;
            }

        }
        }
        B->nnz = k;
        B->row[n+1] = k;

    printf("B.nnz/A.nnz: %d / %d = %d\n", B->nnz, A.nnz,  B->nnz/A.nnz);

        return MAGMA_SUCCESS; 
    }
}





