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

typedef struct {
  cuDoubleComplex *A1;
  cuDoubleComplex *A2;
  int n, lda1, lda2;
} swap_params_t;

__global__ void myzswap( swap_params_t params )
{
  unsigned int x = threadIdx.x + blockDim.x*blockIdx.x;
  unsigned int offset1 = __mul24( x, params.lda1);
  unsigned int offset2 = __mul24( x, params.lda2);
  if( x < params.n )
    {
      cuDoubleComplex *A1 = params.A1 + offset1;
      cuDoubleComplex *A2 = params.A2 + offset2;
      cuDoubleComplex temp = *A1;
      *A1 = *A2;
      *A2 = temp;
    }
}

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

extern "C" void 
magmablas_zlaswp( int n, cuDoubleComplex *dAT, int lda, 
                  int i1, int i2, int *ipiv, int inci )
{
  int k;
  
  for( k=(i1-1); k<i2; k+=BLOCK_SIZE )
    {
      int sb = min(BLOCK_SIZE, i2-k);
      //zlaswp_params_t params = { dAT, lda, lda, ind + k };
      zlaswp_params_t2 params = { dAT+k*lda, n, lda, 0, sb };
      for( int j = 0; j < sb; j++ )
        {
          params.ipiv[j] = ipiv[(k+j)*inci] - k - 1;
        }
      zlaswp3( params );
    }
}

extern "C" void 
magmablas_zswap( int n, cuDoubleComplex *dA1T, int lda1, 
                 cuDoubleComplex *dA2T, int lda2)
{
    swap_params_t params = { dA1T, dA2T, n, lda1, lda2 };
    int  blocksize = 64;
    int  blocks = (params.n+blocksize-1) / blocksize;
    
    myzswap<<< blocks, blocksize >>>( params );
}

#undef BLOCK_SIZE
