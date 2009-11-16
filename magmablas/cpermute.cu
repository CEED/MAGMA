/*
    -- MAGMA (version 0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2009
*/

#define BLOCK_SIZE 64

typedef struct {
        float2 *A;
        int n, lda, j0;
        short ipiv[BLOCK_SIZE];
} claswp_params_t;

__global__ void mydlaswp_( claswp_params_t params )
{
        unsigned int tid = threadIdx.x + __mul24(blockDim.x, blockIdx.x);
        if( tid < params.n )
	{
                int lda = params.lda;
		float2 *A = params.A + tid + lda * params.j0;

		for( int i = 0; i < BLOCK_SIZE; i++ )
		{
                 	int j = params.ipiv[i];
			float2 *p1 = A + i*lda;
			float2 *p2 = A + j*lda;
			float2 temp = *p1;
			*p1 = *p2;
			*p2 = temp;
		}
	}
}

extern "C" void claswp2( claswp_params_t &params )
{
 	int blocksize = BLOCK_SIZE;
	dim3 blocks = (params.n+blocksize-1) / blocksize;
	mydlaswp_<<< blocks, blocksize >>>( params );
}


extern "C" void 
magmablas_cpermute_long( float2 *dAT, int lda, int *ipiv, int nb, int ind )
{
        // assert( (nb % BLOCK_SIZE) == 0 );
        for( int k = 0; k < nb; k += BLOCK_SIZE )
        {
                claswp_params_t params = { dAT, lda, lda, ind + k };
                for( int j = 0; j < BLOCK_SIZE; j++ )
                {
                        params.ipiv[j] = ipiv[ind + k + j] - k - 1;
                        ipiv[ind + k + j] += ind;
                }
                claswp2( params );
        }
}

#undef BLOCK_SIZE
