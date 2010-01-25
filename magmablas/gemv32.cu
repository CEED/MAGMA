/*
    -- MAGMA (version 0.2) --
	Univ. of Tennessee, Knoxville
	Univ. of California, Berkeley
	Univ. of Colorado, Denver
	November 2009
*/

#include "cublas.h"
#include "magma.h"


__global__ void 
sgemvT32_kernel(int m, float alpha, float* A, int lda, float *x, float *y)
{
/*  -- MAGMA (version 0.2) --

    Purpose
    =======

    This routine computes y = alpha A^T x where A is single precision 
    array of dimension (32, M).
*/

    const int inx = threadIdx.x;
    const int iny = threadIdx.y;

    int ind  = iny + __mul24(blockIdx.x,32);
    ind = inx + __mul24(ind,lda);
    int ind2 = inx + __mul24(iny,32);

    A += ind;
    x += inx;

    float res = 0.f;

    __shared__ float buff[64];
    __shared__ float la[32][33];

    buff[ind2]  = x[0];

    #pragma unroll
    for(int j=0; j<16; j++)
      la[iny+__mul24(2,j)][inx] = A[j*__mul24(2,lda)];

    __syncthreads();

    // multiply with the sub-matrix
    #pragma unroll
    for(int j=0; j <16; j++)
      res += la[inx][j+iny*16]*buff[j+iny*16];

    ind = inx + __mul24(blockIdx.x,32);
    la[inx][iny]= res;

    __syncthreads();

    if (ind<m){
       res = la[inx][0] + la[inx][1];
       y[ind] = alpha*res;
    }
}

__global__ void 
dgemvT32_kernel(int m, double alpha, double* A, int lda, double *x, double *y)
{
/*  -- MAGMA (version 0.2) --

    Purpose
    =======

    This routine computes y = alpha A^T x where A is double precision
    array of dimension (32, M).
*/
 
    const int inx = threadIdx.x;
    const int iny = threadIdx.y;

    int ind  = iny + __mul24(blockIdx.x,32);
    ind = inx + __mul24(ind,lda);
    int ind2 = inx + __mul24(iny,32);

    A += ind;
    x += inx;

    double res = 0.f;

    __shared__ double buff[64];
    __shared__ double la[32][33];

    buff[ind2]  = x[0];
    #pragma unroll
    for(int j=0; j<16; j++)
      la[iny+__mul24(2, j)][inx] = A[j*__mul24(2,lda)];

    __syncthreads();

    #pragma unroll
    for(int j=0; j < 16; j++)
      res += la[inx][j+iny*16]*buff[j+iny*16];

    ind = inx + __mul24(blockIdx.x,32);
    la[inx][iny]= res;

    __syncthreads();

    if (ind<m){
      res = la[inx][0] + la[inx][1];
      y[ind] = alpha*res;
    }
}

__global__ void 
sgemv32_kernel(int n, float alpha, float* A, int lda, float *x, float *y)
{
/*  -- MAGMA (version 0.2) --

    Purpose
    =======

    This routine computes y = alpha A x where A is single precision
    array of dimension (N, 32).
*/

    int ind = blockIdx.x*32 + threadIdx.x;

    A += ind;
    x += threadIdx.x;

    float res = 0.f;

    __shared__ float buff[32];
    buff[threadIdx.x]  = x[0];

    __syncthreads();
    #pragma unroll
    for(int j=0; j < n; j++){
       res+=A[0]*buff[j];
       A+=lda;
    }

    if (ind<n)
      y[ind] = alpha*res;
}


__global__ void
dgemv32_kernel(int n, double alpha, double* A, int lda, double *x, double *y)
{
/*  -- MAGMA (version 0.2) --

    Purpose
    =======

    This routine computes y = alpha A x where A is double precision
    array of dimension (N, 32).
*/

    int ind = blockIdx.x*32 + threadIdx.x;

    A += ind;
    x += threadIdx.x;

    double res = 0.f;

    __shared__ double buff[32];
    buff[threadIdx.x]  = x[0];

    __syncthreads();
    #pragma unroll
    for(int j=0; j < n; j++){
       res+=A[0]*buff[j];
       A+=lda;
    }

    if (ind<n)
      y[ind] = alpha*res;
}


void magmablas_sgemv32(char tran, int n, float alpha, 
                       float *A, int lda, float *x, float *y)
{
/*  -- MAGMA (version 0.2) --

    Purpose
    =======

    This routine computes 
       y = alpha A^T x           for tran = 'T' / 't' or
       y = alpha A x 
    where A is single precision array of dimension (32, N) for 
    tran = 'T' / 't', or of dimension (N, 32) otherwise.
*/

    int blocks;
    if (n % 32 == 0)
      blocks = n/32;
    else
      blocks = n/32 + 1;
    dim3 grid(blocks, 1, 1);

    if (tran == 'T' || tran == 't'){
      dim3 threads(32, 2, 1);
      sgemvT32_kernel<<<grid, threads>>>(n, alpha, A, lda, x, y);
    }
    else 
    {
      dim3 threads(32, 1, 1);
      sgemv32_kernel<<<grid, threads>>>(n, alpha, A, lda, x, y);
    }
}


void magmablas_dgemv32(char tran, int n, double alpha, double *A, int lda,
		       double *x, double *y)
{
/*  -- MAGMA (version 0.2) --

    Purpose
    =======

    This routine computes
       y = alpha A^T x 	      	 for tran = 'T' / 't' or
      	y = alpha A x
    where A is double precision array of dimension (32, N) for
    tran = 'T' / 't', or of dimension (N, 32) otherwise.
*/

    int blocks;
    if (n % 32==0)
      blocks = n/32;
    else
      blocks = n/32 + 1;
    dim3 grid(blocks, 1, 1);

    if (tran == 'T' || tran == 't'){
      dim3 threads(32, 2, 1);
      dgemvT32_kernel<<<grid, threads>>>(n, alpha, A, lda, x, y);
    }
    else
    {
      dim3 threads(32, 1, 1);
      dgemv32_kernel<<<grid, threads>>>(n, alpha, A, lda, x, y);
    }
}
