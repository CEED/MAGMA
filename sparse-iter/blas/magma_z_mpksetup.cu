/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
       @author Hartwig Anzt

*/

#include "common_magma.h"

#if (GPUSHMEM < 200)
   #define BLOCK_SIZE 128
#else
   #define BLOCK_SIZE 512
#endif


__global__ void 
compress_kernel(         int num_add_rows,
                         int *add_rows,
                         magmaDoubleComplex *x,
                         magmaDoubleComplex *y ){
    int row = blockDim.x * blockIdx.x + threadIdx.x ;
    if(row < num_add_rows ){
        y[ row ] = x [ add_rows[ row ] ];
    }
}

/**
    Purpose
    -------

    Packs a vector x using a key add_rows into the compressed version.


    Arguments
    ---------

    @param
    num_add_rowsmagma_int_t
                number of elements to unpack

    @param
    add_rows    magma_index_t*
                indices of elements to unpack

    @param
    x           magmaDoubleComplex*
                uncompressed input vector

    @param
    y           magmaDoubleComplex*
                compressed output vector


    @ingroup magmasparse_z
    ********************************************************************/



extern "C" int
magma_z_mpk_compress_gpu(   magma_int_t num_add_rows,
                         magma_index_t *add_rows,
                         magmaDoubleComplex *x,
                         magmaDoubleComplex *y ){

   dim3 grid( (num_add_rows+BLOCK_SIZE-1)/BLOCK_SIZE, 1, 1);

   compress_kernel<<< grid, BLOCK_SIZE, 0, magma_stream >>>
                  ( num_add_rows, add_rows, x, y );

    return MAGMA_SUCCESS; 
}




__global__ void 
uncompress_kernel(       int num_add_rows,
                         int *add_rows,
                         magmaDoubleComplex *x,
                         magmaDoubleComplex *y ){
    int row = blockDim.x * blockIdx.x + threadIdx.x ;
    if(row < num_add_rows ){
        y[ add_rows[ row ] ] = x [ row ];
    }
}

/**
    Purpose
    -------

    Unpacks a compressed vector x and a key add_rows on the GPU.


    Arguments
    ---------

    @param
    num_add_rowsmagma_int_t
                number of elements to unpack

    @param
    add_rows    magma_index_t*
                indices of elements to unpack

    @param
    x           magmaDoubleComplex*
                compressed input vector

    @param
    y           magmaDoubleComplex*
                uncompressed output vector


    @ingroup magmasparse_z
    ********************************************************************/



extern "C" int
magma_z_mpk_uncompress_gpu(   magma_int_t num_add_rows,
                         magma_index_t *add_rows,
                         magmaDoubleComplex *x,
                         magmaDoubleComplex *y ){

   dim3 grid( (num_add_rows+BLOCK_SIZE-1)/BLOCK_SIZE, 1, 1);

   uncompress_kernel<<< grid, BLOCK_SIZE, 0, magma_stream >>>
                  ( num_add_rows, add_rows, x, y );

    return MAGMA_SUCCESS; 
}


/**
    Purpose
    -------

    Distributes a compressed vector x after the SpMV using offset, 
                                                    blocksize and key.


    Arguments
    ---------

    @param
    offset      magma_int_t
                offset from 0

    @param
    blocksize   magma_int_t
                number of elements handled by GPU

    @param
    num_add_rowsmagma_int_t
                number of elements to pack

    @param
    add_rows    magma_index_t*
                indices of elements to pack

    @param
    x           magmaDoubleComplex*
                compressed input vector

    @param
    y           magmaDoubleComplex*
                uncompressed output vector


    @ingroup magmasparse_z
    ********************************************************************/



extern "C" int
magma_z_mpk_uncompspmv(  magma_int_t offset,
                         magma_int_t blocksize,
                         magma_int_t num_add_rows,
                         magma_index_t *add_rows,
                         magmaDoubleComplex *x,
                         magmaDoubleComplex *y ){

   dim3 grid( (num_add_rows+BLOCK_SIZE-1)/BLOCK_SIZE, 1, 1);

   magma_zcopyvector( blocksize, x, 1, y+offset, 1 );

   uncompress_kernel<<< grid, BLOCK_SIZE, 0, magma_stream >>>
                  ( num_add_rows, add_rows, x+blocksize, y );

    return MAGMA_SUCCESS; 
}


