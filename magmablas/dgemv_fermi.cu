/*
    -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010
*/

#include "cublas.h"
#include "magma.h"

#define num_threads 64
#define dgemv_bs 64
#define threadSize 128

__global__ void 
dgemv_kernel_fermi(int n, int m, int n1, double* A, int lda, double *x, double *y)
{
  int ind = blockIdx.x*num_threads + threadIdx.x;

  A += ind;
  x += threadIdx.x;

  double res = 0.f;

  __shared__ double buff[dgemv_bs];
  for(int i=0; i<n1; i += dgemv_bs ){
    __syncthreads();
    buff[threadIdx.x]  = x[i];

    __syncthreads();
    #pragma unroll
    for(int j=0; j < dgemv_bs ; j++){
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
magmablas_dgemv_fermi(int m, int n, double *A, int lda, double *x, double *z)
{
/*  -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

    Purpose
    =======

    This routine computes z = A x on the GPU.

    M      - (input) INTEGER.
             On entry, M specifies the number of rows of the matrix A.

    N      - (input) INTEGER.
             On entry, N specifies the number of columns of the matrix A

    A      - (input) DOUBLE PRECISION array of dimension (LDA, n) on the GPU.
   
    LDA    - (input) INTEGER.
             LDA specifies the leading dimension of A.

    X      - (input) DOUBLE PRECISION array of dimension n.
     
    Z      - (output) DOUBLE PRECISION array of	dimension m. 
             On exit Z = A X.

    ===================================================================== */

    int blocks;
    if (m % num_threads==0)
        blocks = m/num_threads;
    else
        blocks = m/num_threads + 1;

    dim3 grid(blocks, 1, 1);
    dim3 threads(num_threads, 1, 1);
 
    dgemv_kernel_fermi<<<grid, threads>>>(m, n, (n / dgemv_bs)*dgemv_bs, 
                                          A, lda, x, z);
}


__global__ void
dgemvt_kernel1_fermi(int m, int n, double alpha, int n1, double* A, int lda,
                     double *x, double *y)
{
	unsigned int tx = threadIdx.x;

	__shared__ double sdata[threadSize];

	volatile double *smem;

	double res;
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
dgemvt_kernel2_fermi(int m, int n, double alpha,
		     int n1, double* A, int lda, double *x, double *y)
{
  const int inx = threadIdx.x;
  const int iny = threadIdx.y;

  int ind  = iny + __mul24(blockIdx.x,16);
  ind = inx + __mul24(ind,lda);
  int ind2 = inx + __mul24(iny,16);
  if (ind2>31)
     ind2-=32;

  A += ind;
  x += ind2;
  if (ind2>31)
     ind2-=32;

  double res = 0.f;

  __shared__ double buff[32];
  __shared__ double la[16][17];

  for(int i=0; i<n1; i += 32 ){
     buff[ind2]  = x[i];
     #pragma unroll
     for(int j=0; j<4; j++)
        la[iny+__mul24(j,4)][inx] = A[j*__mul24(4,lda)];

     __syncthreads();
     #pragma unroll
     for(int j=0; j < 4; j++)
       res += la[inx][iny*4+j]*buff[j+iny*4];

     A += 16;
    __syncthreads();
     //===========================================
     #pragma unroll
     for(int j=0; j<4; j++)
         la[iny+__mul24(j,4)][inx] = A[j*__mul24(4,lda)];

     __syncthreads();

     #pragma unroll
     for(int j=0; j < 4; j++)
        res += la[inx][iny*4+j]*buff[j+16+iny*4];
     A += 16;
  }

  if (n>n1){
     if (ind2>=(n-n1))
        buff[ind2]=0.;
     else
        buff[ind2]  = x[n1];

     __syncthreads();
     #pragma unroll
     for(int j=0; j<4; j++)
         la[iny+__mul24(j,4)][inx] = A[j*__mul24(4,lda)];

     __syncthreads();
     if (n-n1>4){
        #pragma unroll
	for(int j=0; j < 4; j++)
           res += la[inx][iny*4+j]*buff[j+iny*4];

        A += 16;
        __syncthreads();
        #pragma unroll
          for(int j=0; j<4; j++)
            la[iny+__mul24(j,4)][inx] = A[j*__mul24(4,lda)];

        __syncthreads();

        #pragma unroll
        for(int j=0; j < 4; j++)
           res += la[inx][iny*4+j]*buff[j+16+iny*4];
     }
     else {
        #pragma unroll
        for(int j=0; j < 4; j++)
          res += la[inx][iny*4+j]*buff[j+iny*4];
     }
  }

  __syncthreads();
  ind = inx + __mul24(blockIdx.x,16);
  la[inx][iny]= res;
  __syncthreads();
  if (ind<n){
     res = la[inx][0] + la[inx][1] + la[inx][2] + la[inx][3];
     y[ind] = alpha*res;
  }
}

extern "C" void
magmablas_dgemvt1_fermi(int m, int n, double alpha, double *A, int lda,
                        double *x, double *z)
{
/*  -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

    Purpose
    =======

    This routine computes z = alpha A^t x on the GPU.
    Recommended for large M and N.

    M      - (input) INTEGER.
             On entry, M specifies the number of rows of the matrix A.

    N      - (input) INTEGER.
             On entry, N specifies the number of columns of the matrix A

    A      - (input) DOUBLE PRECISION array of dimension ( LDA, n ) on the GPU.

    LDA    - (input) INTEGER.
             LDA specifies the leading dimension of A.

    X      - (input) DOUBLE PRECISION array of dimension m.

    Z      - (output) DOUBLE PRECISION array of dimension n.
             On exit Z = alpha A^t X.

    ===================================================================== */


	dim3 grid    ( 1,  n,  1);
	dim3 threads ( threadSize,   1,  1);

	dgemvt_kernel1_fermi<<<grid, threads>>>( m, n, alpha, 
                                                (m/threadSize)*threadSize,
                                                A, lda, x, z);
}

extern "C" void
magmablas_dgemvt2_fermi(int m, int n, double alpha, double *A, int lda,
                        double *x, double *z)
{
/*  -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

    Purpose
    =======

    This routine computes z = alpha A^t x on the GPU. Used in least squares
    solver for N small (e.g. = BS, a block size of order 64, 128, etc).

    M      - (input) INTEGER.
             On entry, M specifies the number of rows of the matrix A.

    N      - (input) INTEGER.
             On entry, N specifies the number of columns of the matrix A

    A      - (input) DOUBLE PRECISION array of dimension ( LDA, n ) on the GPU.

    LDA    - (input) INTEGER.
             LDA specifies the leading dimension of A.

    X      - (input) DOUBLE PRECISION array of dimension m.

    Z      - (output) DOUBLE PRECISION array of dimension n.
             On exit Z = alpha A^t X.

    ===================================================================== */

    int blocks;

    if (n % 16==0)
        blocks = n/16;
    else
        blocks = n/16 + 1;

    dim3 grid(blocks, 1, 1);
    dim3 threads(16, 4, 1);

    dgemvt_kernel2_fermi<<<grid, threads>>>(m, n, alpha, (m / 32)*32,
                                            A, lda, x, z);
}

extern "C" void
magmablas_dgemvt_fermi(int m, int n, double alpha, double *A, int lda,
                       double *x, double *z)
{
/*  -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

    Purpose
    =======

    This routine computes z = alpha A^t x on the GPU.

    M      - (input) INTEGER.
             On entry, m specifies the number of rows of the matrix A.

    N      - (input) INTEGER.
             On entry, n specifies the number of columns of the matrix A

    A      - (input) SINGLE PRECISION array of dimension ( LDA, n ) on the GPU.

    LDA    - (input) INTEGER.
             LDA specifies the leading dimension of A.

    X      - (input) SINGLE PRECISION array of dimension m.

    Z      - (output) SINGLE PRECISION array of dimension n.
             On exit Z = alpha A^t X.

    ===================================================================== */

    if (n<=128)
      magmablas_dgemvt2_fermi(m, n, alpha, A, lda, x, z);
    else
      magmablas_dgemvt1_fermi(m, n, alpha, A, lda, x, z);
}

#undef num_threads
#undef dgemv_bs
