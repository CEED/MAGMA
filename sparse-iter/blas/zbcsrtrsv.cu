/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s

*/

#include "common_magma.h"


#define  blockinfo(i,j)  blockinfo[(i)*c_blocks   + (j)]
#define A(i,j) A+((blockinfo(i,j)-1)*size_b*size_b)
#define x(i) x+(i*size_b)


/**
    Purpose
    -------
    
    For a Block-CSR ILU factorization, this routine performs the triangular 
    solves.
    
    Arguments
    ---------

    @param
    r_blocks    magma_int_t
                number of blocks

    @param
    size_b      magma_int_t
                blocksize in BCSR

    @param
    ipiv        magma_int_t*
                array containing pivots

    @param
    x           magmaDoubleComplex*
                input/output vector x


    @ingroup magmasparse_z
    ********************************************************************/

extern "C" magma_int_t
magma_zbcsrtrsv( magma_uplo_t uplo,
                 magma_int_t r_blocks,
                 magma_int_t c_blocks,
                 magma_int_t size_b, 
                 magmaDoubleComplex *A,
                 magma_index_t *blockinfo,   
                 magmaDoubleComplex *x ){

    // some useful variables
    magmaDoubleComplex one = MAGMA_Z_MAKE(1.0, 0.0);
    magmaDoubleComplex mone = MAGMA_Z_MAKE(-1.0, 0.0);
    magma_int_t j,k;

    if( uplo==MagmaLower ){ 
        // forward solve
        for( k=0; k<r_blocks; k++){
            // do the forward triangular solve for block M(k,k): L(k,k)y = b
            magma_ztrsv(MagmaLower, MagmaNoTrans, MagmaUnit, size_b, A(k,k), 
                                                             size_b, x(k), 1 );

             // update for all nonzero blocks below M(k,k) 
                    // the respective values of y
            for( j=k+1; j<c_blocks; j++ ){
                if( (blockinfo(j,k)!=0) ){
                    magmablas_zgemv( MagmaNoTrans, size_b, size_b, 
                                     mone, A(j,k), size_b,
                                     x(k), 1, one,  x(j), 1 );

                }
            }
        }
    }
    else if( uplo==MagmaUpper ){
        // backward solve
        for( k=r_blocks-1; k>=0; k--){
            // do the backward triangular solve for block M(k,k): U(k,k)x = y
            magma_ztrsv(MagmaUpper, MagmaNoTrans, MagmaNonUnit, size_b, A(k,k), 
                                                             size_b, x(k), 1 );

            // update for all nonzero blocks above M(k,k) 
                    // the respective values of y
            for( j=k-1; j>=0; j-- ){
                if( (blockinfo(j,k)!=0) ){
                    magmablas_zgemv( MagmaNoTrans, size_b, size_b, 
                                     mone, A(j,k), size_b,
                                     x(k), 1, one,  x(j), 1 );

                }
            }
        }
    }

    return MAGMA_SUCCESS;
}



