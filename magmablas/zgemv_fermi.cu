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
#define zgemv_bs 32
#define threadSize 128

#define MAGMA_Z_SET2REAL(v, t) v.x = (t); v.y = 0.0
#define magmablas_zgemv_fermi magmablas_zgemv


inline __host__ __device__ double2 make_double2(double s)
{
	return make_double2(s, s);
}
inline __host__ __device__ double2 make_double2(int2 a)
{
	return make_double2(double(a.x), double(a.y));
}

// negate
inline __host__ __device__ double2 operator-(double2 &a)
{
	return make_double2(-a.x, -a.y);
}
// addition
inline __host__ __device__ double2 operator+(double2 a, double2 b)
{
	return make_double2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(double2 &a, double2 b)
{
	a.x += b.x; a.y += b.y;
}

// subtract
inline __host__ __device__ double2 operator-(double2 a, double2 b)
{
	return make_double2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(double2 &a, double2 b)
{
	a.x -= b.x; a.y -= b.y;
}

// multiply
inline __host__ __device__ double2 operator*(double2 a, double2 b)
{
    return make_double2(a.x * b.x - a.y * b.y, a.y * b.x + a.x * b.y);
}
inline __host__ __device__ double2 operator*(double2 a, double s)
{
	return make_double2(a.x * s, a.y * s);
}
inline __host__ __device__ double2 operator*(double s, double2 a)
{
	return make_double2(a.x * s, a.y * s);
}
inline __host__ __device__ void operator*=(double2 &a, double s)
{
	a.x *= s; a.y *= s;
}

inline __host__ __device__ double2 conjugate(double2 a)
{
   double2 b;
   b.x = a.x;
   b.y = 0.0f-a.y;
   return b;
}





__global__ void 
zgemvn_kernel1_fermi(int n, int m, int n1, double2 alpha, double2* A, int lda, double2 *x, double2 *y)
{
  int ind = blockIdx.x*num_threads + threadIdx.x;

  A += ind;

  double2 res;
  MAGMA_Z_SET2REAL(res, 0.0f);

  for(int i=0; i<n1; i += zgemv_bs ){

    #pragma unroll
    for(int j=0; j < zgemv_bs ; j++){
       res += A[0] * x[j];
       A   += lda;
    }
	x += zgemv_bs;
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
zgemvn_kernel2_fermi(int n, int m, int n1, double2 alpha,  double2* A, int lda, double2 *x, double2 *y)
{
  int ind = blockIdx.x*num_threads + threadIdx.x;

  A += ind;
  x += threadIdx.x;

  double2 res;
  MAGMA_Z_SET2REAL(res, 0.0f);

  __shared__ double2 buff[num_threads];
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
magmablas_zgemvn_fermi(int n, int m, double2 alpha, double2 *A, int lda, double2 *x, double2 *y)
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
/*    if(n<=8500) 
		zgemvn_kernel1_fermi<<<grid, threads>>>(n, m, (m / zgemv_bs)*zgemv_bs, 
			                           alpha, A, lda, x, y);
    else */
		zgemvn_kernel2_fermi<<<grid, threads>>>(n, m, (m / num_threads)*num_threads, 
			                           alpha, A, lda, x, y);
}



__global__ void 
zgemvt_kernel1_fermi(int m, int n, double2 alpha, int n1, double2* A, int lda,
              double2 *x, double2 *y)
{
	unsigned int tx = threadIdx.x;

	__shared__ double2 sdata[threadSize];
	

	double2 res;
    MAGMA_Z_SET2REAL(res, 0.0f);
	double2 zero;
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
zgemvt_kernel2_fermi(int m, int n, double2 alpha,
               int n1, double2* A, int lda, double2 *x, double2 *y)
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

  double2 res;
  MAGMA_Z_SET2REAL(res, 0.0f);
  double2 zero;
  MAGMA_Z_SET2REAL(zero, 0.0f);

  __shared__ double2 buff[32];
  __shared__ double2 la[16][17];

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
magmablas_zgemvt1_fermi(int m, int n, double2 alpha, double2 *A, int lda,
                  double2 *x, double2 *y)
{


    dim3 grid    ( 1,  n,  1);
    dim3 threads ( threadSize,   1,  1);

    zgemvt_kernel1_fermi<<<grid, threads>>>( m, n, alpha, ( m / threadSize) * threadSize,
                                       A, lda, x, y);

									  
}

extern "C" void
magmablas_zgemvt2_fermi(int m, int n, double2 alpha, double2 *A, int lda,
                  double2 *x, double2 *y)
{

    int blocks;

    if (n % 16==0)
        blocks = n/16;
    else
        blocks = n/16 + 1;

    dim3 grid(blocks, 1, 1);
    dim3 threads(16, 4, 1);

    zgemvt_kernel2_fermi<<<grid, threads>>>(m, n, alpha, (m / 32)*32,
                                      A, lda, x, y);
}

extern "C" void
magmablas_zgemvt_fermi(int m, int n, double2 alpha, double2 *A, int lda, 
                 double2 *x, double2 *y)
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
      magmablas_zgemvt2_fermi(m, n, alpha, A, lda, x, y);
    else
      magmablas_zgemvt1_fermi(m, n, alpha, A, lda, x, y);
    

}


extern "C" void
magmablas_zgemv_fermi(char flag, int m, int n, double2 alpha, double2 *A, int lda, double2 *x, double2 *y) 
{

	if (flag == 'N' || flag == 'n')
	{
		magmablas_zgemvn_fermi(m,  n, alpha, A, lda, x, y);
	}
	else if(flag == 'T' || flag == 't')
	{
		magmablas_zgemvt_fermi(m,  n, alpha, A, lda, x, y);
	}
	else 
	{
		printf("Not Available\n");
	}
}


#undef num_threads
#undef zgemv_bs
#undef threadSize 
