/*
    -- MAGMA (version 0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2009
*/

#include "cublas.h"
#include "magma.h"
#include "constant.h"
#define magmablas_sgemv_fermi magmablas_sgemv
#define magmablas_sgemvt_fermi magmablas_sgemvt

#define num_threads 128
#define sgemv_bs 32
#define threadSize 128


__global__ void 
sgemv_kernel1_fermi(int n, int m, int n1, float* A, int lda, float *x, float *y)
{
  int ind = blockIdx.x*num_threads + threadIdx.x;

  A += ind;

  float res = 0.f;

  for(int i=0; i<n1; i += sgemv_bs ){

    #pragma unroll
    for(int j=0; j < sgemv_bs ; j++){
       res += A[0] * x[j];
       A   += lda;
    }
	x += sgemv_bs;
  }

  if (m>n1){

     for(int j=0; j<(m-n1); j++){
         res += A[0] * x[j];
         A   += lda;
     }
  }

  if (ind<n)
     y[ind] = res;

}

__global__ void 
sgemv_kernel2_fermi(int n, int m, int n1, float* A, int lda, float *x, float *y)
{
  int ind = blockIdx.x*num_threads + threadIdx.x;

  A += ind;
  x += threadIdx.x;

  float res = 0.f;

  __shared__ float buff[num_threads];
  for(int i=0; i<n1; i += num_threads ){
    __syncthreads();
    buff[threadIdx.x]  = x[i];

    __syncthreads();
    #pragma unroll
    for(int j=0; j < num_threads ; j++){
       res+=A[0]*buff[j];
       A+=lda;
    }
  }
  __syncthreads();

  if (m>n1){
     buff[threadIdx.x]  = x[n1];

     __syncthreads();
     for(int j=0; j<(m-n1); j++){
         res += A[0]*buff[j];
         A+=lda;
     }
  }

  if (ind<n)
     y[ind] = res;
}

extern "C" void
magmablas_sgemv_fermi(int n, int m, float *A, int lda, float *x, float *z)
{
/*  -- MAGMA (version 0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2009

    Purpose
    =======

    This routine computes z = A x on the GPU.

    N      - (input) INTEGER.
             On entry, N specifies the number of rows of the matrix A.

    M      - (input) INTEGER.
             On entry, M specifies the number of columns of the matrix A

    A      - (input) SINGLE PRECISION array of dimension ( LDA, m ) on the GPU.
   
    LDA    - (input) INTEGER.
             LDA specifies the leading dimension of A.

    X      - (input) SINGLE PRECISION array of dimension m.
     
    Z      - (output) SINGLE PRECISION array of	dimension m. 
             On exit Z = A X.

    ===================================================================== */

    int blocks;
    if (n % num_threads==0)
        blocks = n/num_threads;
    else
        blocks = n/num_threads + 1;

    dim3 grid(blocks, 1, 1);
    dim3 threads(num_threads, 1, 1);
    if(n<=8500) 
		sgemv_kernel1_fermi<<<grid, threads>>>(n, m, (m / sgemv_bs)*sgemv_bs, 
			                            A, lda, x, z);
    else
		sgemv_kernel2_fermi<<<grid, threads>>>(n, m, (m / num_threads)*num_threads, 
			                            A, lda, x, z);
}



__global__ void 
sgemvt_kernel1_fermi(int m, int n, float alpha, int n1, float* A, int lda,
              float *x, float *y)
{
	unsigned int tx = threadIdx.x;

	__shared__ float sdata[threadSize];
	
	volatile float *smem;

	float res;
	res = 0.0f;
     
	for(int i=0; i<n1; i+= threadSize)
	{
		res += A[tx + i + lda * blockIdx.y] * x[tx + i];
	}

	
	if(m > n1)
	{
		if( tx + n1 <  m )
		{
			res  += A[tx + n1 + lda *blockIdx.y] * x[tx + n1];
		}
		else 
		{
			res  += 0.0f;
		}
	}	

    sdata[tx] = res;
	__syncthreads();
    
    /*
	if(tx < 128) 
	{
		sdata[tx] += sdata[tx + 128];
	}
    __syncthreads();
	*/

	if(tx < 64) 
	{
		sdata[tx] += sdata[tx + 64];
	}
    __syncthreads();

	if(tx < 32)
	{
		smem = sdata;
		smem[tx] += smem[tx + 32];
		smem[tx] += smem[tx + 16];
		smem[tx] += smem[tx +  8];
		smem[tx] += smem[tx +  4];
		smem[tx] += smem[tx +  2];
		smem[tx] += smem[tx +  1];
	}

    if( tx == 0 ) 
	{
		y[blockIdx.y] = sdata[0]; 		

		if (blockIdx.y < n)
		{
			y[blockIdx.y] = y[blockIdx.y] * alpha;
		}
	}
}


__global__ void 
sgemvt_kernel2_fermi(int m, int n, float alpha,
               int n1, float* A, int lda, float *x, float *y)
{
  const int inx = threadIdx.x;
  const int iny = threadIdx.y;

  int ind  = iny + blockIdx.x * 16;
  ind = inx + ind * lda;
  int ind2 = inx + iny * 16;
  if (ind2>31)
     ind2-=32;

  A += ind;
  x += ind2;

  float res = 0.f;

  __shared__ float buff[32];
  __shared__ float la[16][17];

  for(int i=0; i<n1; i += 32 ){
     buff[ind2]  = x[i];
     #pragma unroll
     for(int j=0; j<4; j++)
        la[iny + j * 4][inx] = A[j* 4 * lda];

     __syncthreads();
     #pragma unroll
     for(int j=0; j < 4; j++)
       res += la[inx][iny*4+j]*buff[j+iny*4];

     A += 16;

     __syncthreads();
     //===========================================
     #pragma unroll
     for(int j=0; j<4; j++)
         la[iny+ j * 4][inx] = A[j* 4 * lda];

     __syncthreads();

     #pragma unroll
     for(int j=0; j < 4; j++)
        res += la[inx][iny*4+j]*buff[j+16+iny*4];
     A += 16;
  }

  __syncthreads(); // 1
  if (n>n1){
     if (ind2>=(n-n1))
	buff[ind2]=0.;
     else
        buff[ind2]  = x[n1];

     __syncthreads();
     #pragma unroll
     for(int j=0; j<4; j++)
         if (inx>=(n-n1))
            la[iny + j * 4][inx] =  0.f;
         else
            la[iny + j * 4][inx] = A[j* 4 * lda];

     __syncthreads();
     if (n-n1>4){
        #pragma unroll
        for(int j=0; j < 4; j++){
           ind =  j+iny*4;
           res += la[inx][ind]*buff[ind];
        }
	A += 16;
        __syncthreads();
	#pragma unroll
	for(int j=0; j<4; j++)
          if (inx+16>=(n-n1))
             la[iny+ j * 4][inx] = 0.f;
          else
             la[iny+ j * 4][inx] = A[j* 4* lda];

        __syncthreads();

        #pragma unroll
	for(int j=0; j < 4; j++){
           ind = j+4*iny;
           res += la[inx][ind]*buff[16+ind];
        }
     }
     else {
	#pragma unroll
        for(int j=0; j < 4; j++){
          ind = j+iny*4;
          res += la[inx][ind]*buff[ind];
        }
     }
  }

  __syncthreads();
  ind = inx + blockIdx.x * 16;
  la[inx][iny]= res;
  __syncthreads();
  if (ind<n && iny==0){
     res = la[inx][0] + la[inx][1] + la[inx][2] + la[inx][3];
     y[ind] = alpha*res;
  }
}

extern "C" void
magmablas_sgemvt1_fermi(int m, int n, float alpha, float *A, int lda,
                  float *x, float *z)
{
/*  -- MAGMA (version 0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2009

    Purpose
    =======

    This routine computes z = alpha A^t x on the GPU. 
    Recommended for large M and N.

    M      - (input) INTEGER.
             On entry, M specifies the number of rows of the matrix A.

    N      - (input) INTEGER.
             On entry, N specifies the number of columns of the matrix A

    A      - (input) SINGLE PRECISION array of dimension ( LDA, n ) on the GPU.

    LDA    - (input) INTEGER.
             LDA specifies the leading dimension of A.

    X      - (input) SINGLE PRECISION array of dimension m.

    Z      - (output) SINGLE PRECISION array of dimension n.
             On exit Z = alpha A^t X.

    ===================================================================== */

    dim3 grid    ( 1,  n,  1);
    dim3 threads ( threadSize,   1,  1);

    sgemvt_kernel1_fermi<<<grid, threads>>>( m, n, alpha, ( m / threadSize) * threadSize,
                                       A, lda, x, z);

									  
}

extern "C" void
magmablas_sgemvt2_fermi(int m, int n, float alpha, float *A, int lda,
                  float *x, float *z)
{
/*  -- MAGMA (version 0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2009

    Purpose
    =======

    This routine computes z = alpha A^t x on the GPU. Used in least squares 
    solver for N small (e.g. = BS, a block size of order 64, 128, etc).

    M      - (input) INTEGER.
             On entry, M specifies the number of rows of the matrix A.

    N      - (input) INTEGER.
             On entry, N specifies the number of columns of the matrix A

    A      - (input) SINGLE PRECISION array of dimension ( LDA, n ) on the GPU.

    LDA    - (input) INTEGER.
             LDA specifies the leading dimension of A.

    X      - (input) SINGLE PRECISION array of dimension m.

    Z      - (output) SINGLE PRECISION array of dimension n.
             On exit Z = alpha A^t X.

    ===================================================================== */

    int blocks;

    if (n % 16==0)
        blocks = n/16;
    else
        blocks = n/16 + 1;

    dim3 grid(blocks, 1, 1);
    dim3 threads(16, 4, 1);

    sgemvt_kernel2_fermi<<<grid, threads>>>(m, n, alpha, (m / 32)*32,
                                      A, lda, x, z);
}

extern "C" void
magmablas_sgemvt_fermi(int m, int n, float alpha, float *A, int lda, 
                 float *x, float *z)
{
/*  -- MAGMA (version 0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2009

    Purpose
    =======

    This routine computes z = alpha A^t x on the GPU.

    M      - (input) INTEGER.
             On entry, M specifies the number of rows of the matrix A.

    N      - (input) INTEGER.
             On entry, N specifies the number of columns of the matrix A

    A      - (input) SINGLE PRECISION array of dimension ( LDA, n ) on the GPU.

    LDA    - (input) INTEGER.
             LDA specifies the leading dimension of A.

    X      - (input) SINGLE PRECISION array of dimension m.

    Z      - (output) SINGLE PRECISION array of dimension n.
             On exit Z = alpha A^t X.

    ===================================================================== */

    if (n<=128)
      magmablas_sgemvt2_fermi(m, n, alpha, A, lda, x, z);
    else
      magmablas_sgemvt1_fermi(m, n, alpha, A, lda, x, z);
    

}

#undef num_threads
#undef sgemv_bs
#undef threadSize 
