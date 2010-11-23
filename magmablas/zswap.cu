/*
    -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

       @precisions normal z -> s d c

*/

#include <stdio.h>
#include <cublas.h>

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
    cuDoubleComplex *A1;
    cuDoubleComplex *A2;
    int n, lda1, lda2;
} zswap_params_t;

__global__ void myzswap( zswap_params_t params )
{
    unsigned int x = threadIdx.x + __mul24(blockDim.x, blockIdx.x);
    unsigned int offset1 = __mul24( x, params.lda1);
    unsigned int offset2 = __mul24( x, params.lda2);
    if( x < params.n )
    {
        cuDoubleComplex *A1  = params.A1 + offset1;
        cuDoubleComplex *A2  = params.A2 + offset2;
        cuDoubleComplex temp = *A1;
        *A1 = *A2;
        *A2 = temp;
    }
}

extern "C" void 
magmablas_zswap( int n, cuDoubleComplex *dA1T, int lda1, 
                 cuDoubleComplex *dA2T, int lda2)
{
    int blocksize = 64;
    dim3 blocks = ( (params.n+blocksize-1) / blocksize, 1, 1);
    zswap_params_t params = { dA1T, dA2T, n, lda1, lda2 };
    myzswap<<< blocks, blocksize >>>( params );
}

