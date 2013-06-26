/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> s d c

*/
#include "common_magma.h"

#define BLOCK_SIZE 64

/*********************************************************
 *
 * SWAP BLAS: permute to set of N elements
 *
 ********************************************************/
/*
 *  First version: line per line
 */
typedef struct {
    magmaDoubleComplex *A1;
    magmaDoubleComplex *A2;
    int n, lda1, lda2;
} magmagpu_zswap_params_t;

__global__ void magmagpu_zswap( magmagpu_zswap_params_t params )
{
    unsigned int x = threadIdx.x + __mul24(blockDim.x, blockIdx.x);
    unsigned int offset1 = __mul24( x, params.lda1);
    unsigned int offset2 = __mul24( x, params.lda2);
    if( x < params.n )
    {
        magmaDoubleComplex *A1  = params.A1 + offset1;
        magmaDoubleComplex *A2  = params.A2 + offset2;
        magmaDoubleComplex temp = *A1;
        *A1 = *A2;
        *A2 = temp;
    }
}

extern "C" void 
magmablas_zswap( magma_int_t n, magmaDoubleComplex *dA1T, magma_int_t lda1, 
                 magmaDoubleComplex *dA2T, magma_int_t lda2)
{
    int blocksize = 64;
    dim3 blocks( (n+blocksize-1) / blocksize, 1, 1);
    magmagpu_zswap_params_t params = { dA1T, dA2T, n, lda1, lda2 };
    magmagpu_zswap<<< blocks, blocksize, 0, magma_stream >>>( params );
}

