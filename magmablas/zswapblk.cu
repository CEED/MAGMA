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

/*********************************************************/
/*
*  Blocked version: swap several pair of line
 */
typedef struct {
    cuDoubleComplex *A1;
    cuDoubleComplex *A2;
    int n, lda1, lda2, npivots;
    short ipiv[BLOCK_SIZE];
} magmagpu_zswapblk_params_t;

__global__ void magmagpu_zswapblkrm( magmagpu_zswapblk_params_t params )
{
    unsigned int y = threadIdx.x + blockDim.x*blockIdx.x;
    if( y < params.n )
    {
        cuDoubleComplex *A1 = params.A1 + y - params.lda1;
        cuDoubleComplex *A2 = params.A2 + y;
      
        for( int i = 0; i < params.npivots; i++ )
        {
            A1 += params.lda1;
            if ( params.ipiv[i] == -1 )
                continue;
            cuDoubleComplex tmp1  = *A1;
            cuDoubleComplex *tmp2 = A2 + params.ipiv[i]*params.lda2;
            *A1   = *tmp2;
            *tmp2 = tmp1;
        }
    }
}

__global__ void magmagpu_zswapblkcm( magmagpu_zswapblk_params_t params )
{
    unsigned int y = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int offset1 = __mul24( y, params.lda1);
    unsigned int offset2 = __mul24( y, params.lda2);
    if( y < params.n )
    {
        cuDoubleComplex *A1 = params.A1 + offset1 - 1;
        cuDoubleComplex *A2 = params.A2 + offset2;
      
        for( int i = 0; i < params.npivots; i++ )
        {
            A1++;
            if ( params.ipiv[i] == -1 )
                continue;
            cuDoubleComplex tmp1  = *A1;
            cuDoubleComplex *tmp2 = A2 + params.ipiv[i];
            *A1   = *tmp2;
            *tmp2 = tmp1;
        }
    }
    __syncthreads();
}

extern "C" void 
magmablas_zswapblk( char storev, magma_int_t n, 
                    cuDoubleComplex *dA1T, magma_int_t lda1,
                    cuDoubleComplex *dA2T, magma_int_t lda2,
                    magma_int_t i1, magma_int_t i2,
                    magma_int_t *ipiv, magma_int_t inci, magma_int_t offset )
{
    int  blocksize = 64;
    dim3 blocks( (n+blocksize-1) / blocksize, 1, 1);
    int  k, im;
    
    /* Quick return */
    if ( n == 0 )
        return;
    
    if ( (storev == 'C') || (storev == 'c') ) {
        for( k=(i1-1); k<i2; k+=BLOCK_SIZE )
        {
            int sb = min(BLOCK_SIZE, i2-k);
            magmagpu_zswapblk_params_t params = { dA1T+k, dA2T, n, lda1, lda2, sb };
            for( int j = 0; j < sb; j++ )
            {
                im = ipiv[(k+j)*inci] - 1;
                if ( (k+j) == im)
                    params.ipiv[j] = -1;
                else
                    params.ipiv[j] = im - offset;
            }
            magmagpu_zswapblkcm<<< blocks, blocksize, 0, magma_stream >>>( params );
        }
    }else {
        for( k=(i1-1); k<i2; k+=BLOCK_SIZE )
        {
            int sb = min(BLOCK_SIZE, i2-k);
            magmagpu_zswapblk_params_t params = { dA1T+k*lda1, dA2T, n, lda1, lda2, sb };
            for( int j = 0; j < sb; j++ )
            {
                im = ipiv[(k+j)*inci] - 1;
                if ( (k+j) == im)
                    params.ipiv[j] = -1;
                else
                    params.ipiv[j] = im - offset;
            }
            magmagpu_zswapblkrm<<< blocks, blocksize, 0, magma_stream >>>( params );
        }
    }
}

