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

/*********************************************************/
/*
*  Blocked version: swap several pair of line
 */
typedef struct {
    cuDoubleComplex *A1;
    cuDoubleComplex *A2;
    int n, ldx1, ldx2, ldy1, ldy2, npivots;
    short ipiv[BLOCK_SIZE];
} zswapblk_params_t;

__global__ void myzswapblk( zswapblk_params_t params )
{
    unsigned int y = threadIdx.x + blockDim.x*blockIdx.x;
    /* unsigned int offset1 = __mul24( y, params.ldy1); */
    /* unsigned int offset2 = __mul24( y, params.ldy2); */
    if( y < params.n )
    {
        cuDoubleComplex *A1 = params.A1 + y - params.ldx1; /*offset1*/
        cuDoubleComplex *A2 = params.A2 + y;               /*offset2*/
      
        for( int i = 0; i < params.npivots; i++ )
        {
            A1 += params.ldx1;
            if ( params.ipiv[i] == -1 )
                continue;
            cuDoubleComplex tmp1  = *A1;
            cuDoubleComplex *tmp2 = A2 + params.ipiv[i]*params.ldx2;
            *A1   = *tmp2;
            *tmp2 = tmp1;
        }
    }
}

extern "C" void 
magmablas_zswapblk( int n, cuDoubleComplex *dA1T, int ldx1, int ldy1, 
                    cuDoubleComplex *dA2T, int ldx2, int ldy2,
                    int i1, int i2, int *ipiv, int inci, int offset )
{
    int  blocksize = 64;
    dim3 blocks( (n+blocksize-1) / blocksize, 1, 1);
    int  k, im;
    for( k=(i1-1); k<i2; k+=BLOCK_SIZE )
    {
        int sb = min(BLOCK_SIZE, i2-k);
        zswapblk_params_t params = { dA1T+k*ldx1, dA2T, n, ldx1, ldx2, ldy1, ldy2, sb };
        for( int j = 0; j < sb; j++ )
        {
            im = ipiv[(k+j)*inci] - 1;
            if ( (k+j) == im)
                params.ipiv[j] = -1;
            else
                params.ipiv[j] = im - offset;
        }
        myzswapblk<<< blocks, blocksize >>>( params );
    }
}

