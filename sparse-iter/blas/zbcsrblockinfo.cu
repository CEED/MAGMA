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


#define  blockinfo(i,j)  blockinfo[(i)*c_blocks   + (j)]
#define  val(i,j) val+((blockinfo(i,j)-1)*size_b*size_b)



// every thread initializes one entry
__global__ void 
zbcsrblockinfo5_kernel( 
                  magma_int_t num_blocks,
                  magmaDoubleComplex *address,
                  magmaDoubleComplex **AII ){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if( i < num_blocks ){
        *AII[ i ] = *address;
        if(i==0)
        printf("address: %d\n", address);
    }
}



/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======
    
    For a Block-CSR ILU factorization, this routine copies the filled blocks
    from the original matrix A and initializes the blocks that will later be 
    filled in the factorization process with zeros.
    
    Arguments
    =========


    magma_int_t size_b              blocksize in BCSR
    magma_int_t num_blocks          number of nonzero blocks
    magma_int_t num_zero_blocks     number of zero-blocks (will later be filled)
    magmaDoubleComplex **Aval       pointers to the nonzero blocks in A
    magmaDoubleComplex **Aval       pointers to the nonzero blocks in B
    magmaDoubleComplex **Aval       pointers to the zero blocks in B

    ======================================================================    */

extern "C" magma_int_t
magma_zbcsrblockinfo5(  magma_int_t lustep,
                        magma_int_t num_blocks, 
                        magma_int_t c_blocks, 
                        magma_int_t size_b,
                        magma_int_t *blockinfo,
                        magmaDoubleComplex *val,
                        magmaDoubleComplex **AII ){

 
        dim3 dimBlock( BLOCK_SIZE, 1, 1 );

        int dimgrid = (num_blocks+BLOCK_SIZE-1)/BLOCK_SIZE;
        dim3 dimGrid( dimgrid, 1, 1 );


        printf("dim grid: %d x %d", dimgrid, BLOCK_SIZE);
        magmaDoubleComplex **hAII;
        magma_malloc((void **)&hAII, num_blocks*sizeof(magmaDoubleComplex *));

        for(int i=0; i<num_blocks; i++){
           hAII[i] = val(lustep,lustep);
        }
        cublasSetVector( num_blocks, sizeof(magmaDoubleComplex *), 
                                                            hAII, 1, AII, 1 );
/*
    cublasSetVector( 1, sizeof(magmaDoubleComplex *), address, 1, daddress, 1 );
    zbcsrblockinfo5_kernel<<<dimGrid,dimBlock, 0, magma_stream >>>
                        ( num_blocks, daddress, AII );

*/
        return MAGMA_SUCCESS;

}



