/*
    -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010
*/

#include "stdio.h"
#include "cublas.h"
#include "magma.h"
#include "constant.h"

#define num_threads 128
#define cgemv_bs 32
#define threadSize 128

#define MAGMA_Z_SET2REAL(v, t) v.x = (t); v.y = 0.0
#define magmablas_cgemv_fermi magmablas_cgemv 



inline __host__ __device__ float2 make_float2(float s)
{
	return make_float2(s, s);
}
inline __host__ __device__ float2 make_float2(int2 a)
{
	return make_float2(float(a.x), float(a.y));
}

// negate
inline __host__ __device__ float2 operator-(float2 &a)
{
	return make_float2(-a.x, -a.y);
}
// addition
inline __host__ __device__ float2 operator+(float2 a, float2 b)
{
	return make_float2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(float2 &a, float2 b)
{
	a.x += b.x; a.y += b.y;
}

// subtract
inline __host__ __device__ float2 operator-(float2 a, float2 b)
{
	return make_float2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(float2 &a, float2 b)
{
	a.x -= b.x; a.y -= b.y;
}

// multiply
inline __host__ __device__ float2 operator*(float2 a, float2 b)
{
    return make_float2(a.x * b.x - a.y * b.y, a.y * b.x + a.x * b.y);
}
inline __host__ __device__ float2 operator*(float2 a, float s)
{
	return make_float2(a.x * s, a.y * s);
}
inline __host__ __device__ float2 operator*(float s, float2 a)
{
	return make_float2(a.x * s, a.y * s);
}
inline __host__ __device__ void operator*=(float2 &a, float s)
{
	a.x *= s; a.y *= s;
}

inline __host__ __device__ float2 conjugate(float2 a)
{
   float2 b;
   b.x = a.x;
   b.y = 0.0f-a.y;
   return b;
}





__global__ void 
cgemvn_kernel1_fermi(int n, int m, int n1, float2 alpha, float2* A, int lda, float2 *x, float2 *y)
{
  int ind = blockIdx.x*num_threads + threadIdx.x;

  A += ind;

  float2 res;
  MAGMA_Z_SET2REAL(res, 0.0f);

  for(int i=0; i<n1; i += cgemv_bs ){

    #pragma unroll
    for(int j=0; j < cgemv_bs ; j++){
       res += A[0] * x[j];
       A   += lda;
    }
	x += cgemv_bs;
  }

  if (m>n1){

     for(int j=0; j<(m-n1); j++){
         res += A[0] * x[j];
         A   += lda;
     }
  }

  if (ind<n)
     y[ind] = alpha * res;

}

__global__ void 
cgemvn_kernel2_fermi(int n, int m, int n1, float2 alpha,  float2* A, int lda, float2 *x, float2 *y)
{
  int ind = blockIdx.x*num_threads + threadIdx.x;

  A += ind;
  x += threadIdx.x;

  float2 res;
  MAGMA_Z_SET2REAL(res, 0.0f);

  __shared__ float2 buff[num_threads];
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
     y[ind] = alpha * res;
}

extern "C" void
magmablas_cgemvn_fermi(int n, int m, float2 alpha, float2 *A, int lda, float2 *x, float2 *y)
{
/*  -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

    Purpose
    =======

    This routine computes Y = alpha A x on the GPU.

    N      - (input) INTEGER.
             On entry, N specifies the number of rows of the matrix A.

    M      - (input) INTEGER.
             On entry, M specifies the number of columns of the matrix A

    A      - (input) SINGLE PRECISION array of dimension ( LDA, m ) on the GPU.
   
    LDA    - (input) INTEGER.
             LDA specifies the leading dimension of A.

    X      - (input) SINGLE PRECISION array of dimension m.
     
    Y      - (output) SINGLE PRECISION array of	dimension m. 
             On exit Y = alpha A X.

    ===================================================================== */

    int blocks;
    if (n % num_threads==0)
        blocks = n/num_threads;
    else
        blocks = n/num_threads + 1;

    dim3 grid(blocks, 1, 1);
    dim3 threads(num_threads, 1, 1);
//    if(n<=8500) 
		cgemvn_kernel1_fermi<<<grid, threads>>>(n, m, (m / cgemv_bs)*cgemv_bs, 
			                           alpha, A, lda, x, y);
 /*   else 
		cgemvn_kernel2_fermi<<<grid, threads>>>(n, m, (m / num_threads)*num_threads, 
			                           alpha, A, lda, x, y);
  */
}



__global__ void 
cgemvt_kernel1_fermi(int m, int n, float2 alpha, int n1, float2* A, int lda,
              float2 *x, float2 *y)
{
	unsigned int tx = threadIdx.x;

	__shared__ float2 sdata[threadSize];
	

	float2 res;
    MAGMA_Z_SET2REAL(res, 0.0f);
	float2 zero;
    MAGMA_Z_SET2REAL(zero, 0.0f);
     
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
			res  += zero;
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
		sdata[tx] += sdata[tx + 32];
	}

    if(tx == 0)
	{
		for(int i=1;i<32;i++)
		{
			sdata[tx] += sdata[tx + i];
		}
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
cgemvt_kernel2_fermi(int m, int n, float2 alpha,
               int n1, float2* A, int lda, float2 *x, float2 *y)
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

  float2 res;
  MAGMA_Z_SET2REAL(res, 0.0f);
  float2 zero;
  MAGMA_Z_SET2REAL(zero, 0.0f);

  __shared__ float2 buff[32];
  __shared__ float2 la[16][17];

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
	buff[ind2]=zero;
     else
        buff[ind2]  = x[n1];

     __syncthreads();
     #pragma unroll
     for(int j=0; j<4; j++)
         if (inx>=(n-n1))
            la[iny + j * 4][inx] =  zero;
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
             la[iny+ j * 4][inx] = zero;
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
magmablas_cgemvt1_fermi(int m, int n, float2 alpha, float2 *A, int lda,
                  float2 *x, float2 *y)
{


    dim3 grid    ( 1,  n,  1);
    dim3 threads ( threadSize,   1,  1);

    cgemvt_kernel1_fermi<<<grid, threads>>>( m, n, alpha, ( m / threadSize) * threadSize,
                                       A, lda, x, y);

									  
}

extern "C" void
magmablas_cgemvt2_fermi(int m, int n, float2 alpha, float2 *A, int lda,
                  float2 *x, float2 *y)
{

    int blocks;

    if (n % 16==0)
        blocks = n/16;
    else
        blocks = n/16 + 1;

    dim3 grid(blocks, 1, 1);
    dim3 threads(16, 4, 1);

    cgemvt_kernel2_fermi<<<grid, threads>>>(m, n, alpha, (m / 32)*32,
                                      A, lda, x, y);
}

extern "C" void
magmablas_cgemvt_fermi(int m, int n, float2 alpha, float2 *A, int lda, 
                 float2 *x, float2 *y)
{
/*  -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

    Purpose
    =======

    This routine computes y = alpha *  A^t *  x on the GPU.

    M      - (input) INTEGER.
             On entry, M specifies the number of rows of the matrix A.

    N      - (input) INTEGER.
             On entry, N specifies the number of columns of the matrix A

    A      - (input) SINGLE PRECISION array of dimension ( LDA, n ) on the GPU.

    LDA    - (input) INTEGER.
             LDA specifies the leading dimension of A.

    X      - (input) SINGLE PRECISION array of dimension m.

    Y      - (output) SINGLE PRECISION array of dimension n.
             On exit Y = alpha A^t X.

    ===================================================================== */

    if (n<=128)
      magmablas_cgemvt2_fermi(m, n, alpha, A, lda, x, y);
    else
      magmablas_cgemvt1_fermi(m, n, alpha, A, lda, x, y);
    

}


extern "C" void
magmablas_cgemv_fermi(char flag, int m, int n, float2 alpha, float2 *A, int lda, float2 *x, float2 *y) 
{

	if (flag == 'N' || flag == 'n')
	{
		magmablas_cgemvn_fermi(m,  n, alpha, A, lda, x, y);
	}
	else if(flag == 'T' || flag == 't')
	{
		magmablas_cgemvt_fermi(m,  n, alpha, A, lda, x, y);
	}
	else 
	{
		printf("Not Available\n");
	}
}


#undef num_threads
#undef cgemv_bs
#undef threadSize 
