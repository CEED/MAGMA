/*
    -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010
*/

#define BLOCK_SIZE 64

typedef struct {
        float *A;
        int n, lda, j0;
        short ipiv[BLOCK_SIZE];
} slaswp_params_t;

__global__ void myslaswp_( slaswp_params_t params )
{
        unsigned int tid = threadIdx.x + __mul24(blockDim.x, blockIdx.x);
        if( tid < params.n )
	{
                int lda = params.lda;
		float *A = params.A + tid + lda * params.j0;

		for( int i = 0; i < BLOCK_SIZE; i++ )
		{
                 	int j = params.ipiv[i];
			float *p1 = A + i*lda;
			float *p2 = A + j*lda;
			float temp = *p1;
			*p1 = *p2;
			*p2 = temp;
		}
	}
}

extern "C" void slaswp2( slaswp_params_t &params )
{
 	int blocksize = 64;
	dim3 blocks = (params.n+blocksize-1) / blocksize;
	myslaswp_<<< blocks, blocksize >>>( params );
}


extern "C" void 
magmablas_spermute_long( float *dAT, int lda, int *ipiv, int nb, int ind )
{
        // assert( (nb % BLOCK_SIZE) == 0 );
        for( int k = 0; k < nb; k += BLOCK_SIZE )
        {
                slaswp_params_t params = { dAT, lda, lda, ind + k };
                for( int j = 0; j < BLOCK_SIZE; j++ )
                {
                        params.ipiv[j] = ipiv[ind + k + j] - k - 1;
                        ipiv[ind + k + j] += ind;
                }
                slaswp2( params );
        }
}

#undef BLOCK_SIZE
