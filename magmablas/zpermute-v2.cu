/*
    -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

       @precisions normal z -> s d c

*/

#include <stdio.h>

#define BLOCK_SIZE 64

typedef struct {
        double2 *A;
        int n, lda, j0;
        short ipiv[BLOCK_SIZE];
} zlaswp_params_t;

typedef struct {
        double2 *A;
        int n, lda, j0, npivots;
        short ipiv[BLOCK_SIZE];
} zlaswp_params_t2;


__global__ void myzlaswp2( zlaswp_params_t2 params )
{
        unsigned int tid = threadIdx.x + __mul24(blockDim.x, blockIdx.x);
        if( tid < params.n )
	{
                int lda = params.lda;
		double2 *A = params.A + tid + lda * params.j0;

		for( int i = 0; i < params.npivots; i++ )
		{
                 	int j = params.ipiv[i];
			double2 *p1 = A + i*lda;
			double2 *p2 = A + j*lda;
			double2 temp = *p1;
			*p1 = *p2;
			*p2 = temp;
		}
	}
}

extern "C" void zlaswp2( zlaswp_params_t &params );

extern "C" void zlaswp3( zlaswp_params_t2 &params )
{
 	int blocksize = 64;
	dim3 blocks = (params.n+blocksize-1) / blocksize;
	myzlaswp2<<< blocks, blocksize >>>( params );
}


extern "C" void 
magmablas_zpermute_long2( double2 *dAT, int lda, int *ipiv, int nb, int ind )
{
        int k;

        for( k = 0; k < nb-BLOCK_SIZE; k += BLOCK_SIZE )
        {
                //zlaswp_params_t params = { dAT, lda, lda, ind + k };
                zlaswp_params_t2 params = { dAT, lda, lda, ind + k, BLOCK_SIZE };
                for( int j = 0; j < BLOCK_SIZE; j++ )
                {
                        params.ipiv[j] = ipiv[ind + k + j] - k - 1;
                        ipiv[ind + k + j] += ind;
                }
                //zlaswp2( params );
	        zlaswp3( params );
        }

	int num_pivots = nb - k;

        zlaswp_params_t2 params = { dAT, lda, lda, ind + k, num_pivots};
        for( int j = 0; j < num_pivots; j++ )
        {
            params.ipiv[j] = ipiv[ind + k + j] - k - 1;
            ipiv[ind + k + j] += ind;
        }
        zlaswp3( params );
}

#undef BLOCK_SIZE
