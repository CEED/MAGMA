/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> c d s

*/

#include "common_magma.h"

#if (GPUSHMEM < 200)
   #define BLOCK_SIZE 128
#else
   #define BLOCK_SIZE 512
#endif



// every multiprocessor handles one BCSR-block
__global__ void 
zbcsrvalcpy_kernel( 
                  int size_b,
                  magma_int_t num_blocks,
                  magmaDoubleComplex **Aval, 
                  magmaDoubleComplex **Bval ){
    if(blockIdx.x*65535+blockIdx.y < num_blocks){
        // dA and dB iterate across row i
        magmaDoubleComplex *dA = Aval[ blockIdx.x*65535+blockIdx.y ];
        magmaDoubleComplex *dB = Bval[ blockIdx.x*65535+blockIdx.y ];
        int i = threadIdx.x;

        //dA += i;
        //dB += i;

        while( i<size_b*size_b ){
                dB[i] = dA[i]; //= MAGMA_Z_MAKE(5.0, 0.0);
                i+=BLOCK_SIZE;
        }
    }
}

// every multiprocessor handles one BCSR-block
__global__ void 
zbcsrvalzro_kernel( 
                  int size_b,
                  magma_int_t num_blocks,
                  magmaDoubleComplex **Bval ){
    if(blockIdx.x*65535+blockIdx.y < num_blocks){
        // dA and dB iterate across row i
        magmaDoubleComplex *dB = Bval[ blockIdx.x*65535+blockIdx.y ];
        int i = threadIdx.x;
        //dB += i;

        while( i<size_b*size_b ){
                dB[i] = MAGMA_Z_MAKE(0.0, 0.0);
                i+=BLOCK_SIZE;
        }
    }

}



/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======
    
    For a Block-CSR ILU factorization, this routine swaps rows in the vector *x
    according to the pivoting in *ipiv.
    
    Arguments
    =========

    magma_int_t r_blocks            number of blocks
    magma_int_t size_b              blocksize in BCSR
    magma_int_t *ipiv               array containing pivots
    magmaDoubleComplex *x           input/output vector x

    =====================================================================    */

extern "C" magma_int_t
magma_zbcsrvalcpy(  magma_int_t size_b, 
                    magma_int_t num_blocks, 
                    magma_int_t num_zero_blocks, 
                    magmaDoubleComplex **Aval, 
                    magmaDoubleComplex **Bval,
                    magmaDoubleComplex **Bval2 ){

 
        dim3 dimBlock( BLOCK_SIZE, 1, 1 );

        //dim3 dimGrid( num_blocks, 1, 1 );
        int dimgrid1 = 65535;
        int dimgrid2 = (num_blocks+65535-1)/65535;
        int dimgrid3 = (num_zero_blocks+65535-1)/65535;
        dim3 dimGrid( dimgrid2, dimgrid1, 1 );

        //printf( "num_blocks: %d  dimGrid:%d=%d x %d=%d  dimBlock%d \n", num_blocks, dimGrid.x, dimgrid1, dimGrid.y, dimgrid2, dimBlock.x);

        zbcsrvalcpy_kernel<<<dimGrid,dimBlock, 0, magma_stream >>>( size_b, num_blocks, Aval, Bval );

        dim3 dimGrid2( dimgrid3, dimgrid1, 1 );
        //printf( "num_blocks: %d  dimGrid:%d=%d x %d=%d  dimBlock%d \n", num_zero_blocks, dimGrid2.x, dimgrid1, dimGrid2.y, dimgrid3, dimBlock.x);

        zbcsrvalzro_kernel<<<dimGrid2,dimBlock, 0, magma_stream >>>( size_b, num_zero_blocks, Bval2 );

        return MAGMA_SUCCESS;

}



