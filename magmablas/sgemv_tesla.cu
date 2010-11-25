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
#define sgemv_bs 64
#define magmablas_sgemv_tesla magmablas_sgemv
#define magmablas_sgemvt_tesla magmablas_sgemvt

__global__ void 
sgemv_kernel(int n, int m, int n1, float* A, int lda, float *x, float *y)
{
  int ind = blockIdx.x*num_threads + threadIdx.x;

  A += ind;
  x += threadIdx.x;

  float res = 0.f;

  __shared__ float buff[sgemv_bs];
  for(int i=0; i<n1; i += sgemv_bs ){
    __syncthreads();
    buff[threadIdx.x]  = x[i];

    __syncthreads();
    #pragma unroll
    for(int j=0; j < sgemv_bs ; j++){
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

__global__ void
sgemv_kernel2(int n, int m, int n1, float* A, int lda, 
              float *x, int incx, float *y)
{
  int ind = blockIdx.x*num_threads + threadIdx.x;

  A += ind;
  x += threadIdx.x * incx;

  float res = 0.f;

  __shared__ float buff[sgemv_bs];
  for(int i=0; i<n1; i += sgemv_bs ){
    __syncthreads();
    buff[threadIdx.x]  = x[i*incx];

    __syncthreads();
    #pragma unroll
    for(int j=0; j < sgemv_bs ; j++){
       res+=A[0]*buff[j];
       A+=lda;
    }
  }
  __syncthreads();

  if (m>n1){
     buff[threadIdx.x]  = x[n1*incx];

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
magmablas_sgemv_tesla(int m, int n, float *A, int lda, float *x, float *z)
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
             On entry, N specifies the number of rows of the matrix A.

    N      - (input) INTEGER.
             On entry, M specifies the number of columns of the matrix A

    A      - (input) SINGLE PRECISION array of dimension ( LDA, n ) on the GPU.
   
    LDA    - (input) INTEGER.
             LDA specifies the leading dimension of A.

    X      - (input) SINGLE PRECISION array of dimension n.
     
    Z      - (output) SINGLE PRECISION array of	dimension m. 
             On exit Z = A X.

    ===================================================================== */

    int blocks;
    if (m % num_threads==0)
        blocks = m/num_threads;
    else
        blocks = m/num_threads + 1;

    dim3 grid(blocks, 1, 1);
    dim3 threads(num_threads, 1, 1);
 
    sgemv_kernel<<<grid, threads>>>(m, n, (n / sgemv_bs)*sgemv_bs, 
                                    A, lda, x, z);
}

extern "C" void
magmablas_sgemv2(int n, int m, float *A, int lda, float *x, int incx, float *z)
{
/*  -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

    Purpose
    =======

    This routine computes z = A x on the GPU.
    This version has INCX as an argument. 

    N      - (input) INTEGER.
             On entry, N specifies the number of rows of the matrix A.

    M      - (input) INTEGER.
             On entry, M specifies the number of columns of the matrix A

    A      - (input) SINGLE PRECISION array of dimension ( LDA, m ) on the GPU.

    LDA    - (input) INTEGER.
             LDA specifies the leading dimension of A.

    X      - (input) SINGLE PRECISION array of dimension n.

    INCX   - (input) Specifies the increment for the elements of X. 
             INCX must not be zero. Unchanged on exit.

    Z      - (output) SINGLE PRECISION array of dimension m.
             On exit Z = A X.

    ===================================================================== */

    int blocks;
    if (n % num_threads==0)
        blocks = n/num_threads;
    else
        blocks = n/num_threads + 1;

    dim3 grid(blocks, 1, 1);
    dim3 threads(num_threads, 1, 1);

    sgemv_kernel2<<<grid, threads>>>(n, m, (m / sgemv_bs)*sgemv_bs,
                                     A, lda, x, incx, z);
}

__global__ void 
sgemvt_kernel1(int n, int m, float alpha, int n1, float* A, int lda,
              float *x, float *y)
{
  const int inx = threadIdx.x;
  const int iny = threadIdx.y;

  int ind  = iny + __mul24(blockIdx.x,32);
  ind = inx + __mul24(ind,lda);
  int ind2 = inx + __mul24(iny,32);

  A += ind;
  x += ind2;

  float res = 0.f;

  __shared__ float buff[sgemv_bs];
  __shared__ float la[32][33];

  for(int i=0; i<n1; i += sgemv_bs ){
      buff[ind2]  = x[i];
      #pragma unroll
      for(int j=0; j<16; j++)
         la[iny+__mul24(j,2)][inx] = A[j*__mul24(2,lda)];

      __syncthreads();
      #pragma unroll
      for(int j=0; j < 16; j++)
        res += la[inx][iny*16+j]*buff[j+iny*16];

      A += 32;

      //===============================================
      #pragma unroll
      for(int j=0; j<16; j++)
         la[iny+__mul24(j,2)][inx] = A[j*__mul24(2,lda)];

      __syncthreads();

      #pragma unroll
      for(int j=0; j < 16; j++)
        res += la[inx][iny*16+j]*buff[j+32+iny*16];
      A += 32;
    }

    if (n>n1){
      if (ind2>=(n-n1))
         buff[ind2]=0.;
      else
         buff[ind2]  = x[n1];

      #pragma unroll
      for(int j=0; j<16; j++)
         la[iny+__mul24(j,2)][inx] = A[j*__mul24(2,lda)];

     __syncthreads();

     if (n-n1>16){
        #pragma unroll
        for(int j=0; j < 16; j++)
           res += la[inx][iny*16+j]*buff[j+iny*16];

        A += 32;
        #pragma unroll
        for(int j=0; j<16; j++)
          la[iny+__mul24(j,2)][inx] = A[j*__mul24(2,lda)];

        __syncthreads();

        #pragma unroll
        for(int j=0; j < 16; j++)
           res += la[inx][iny*16+j]*buff[j+32+iny*16];
     }
     else {
        #pragma unroll
        for(int j=0; j < 16; j++)
          res += la[inx][iny*16+j]*buff[j+iny*16];
     }
  }
  ind = inx + __mul24(blockIdx.x,32);

  la[inx][iny]= res;
  if (ind<m){
     res = la[inx][0] + la[inx][1];
     y[ind] = alpha*res;
  }
}

__global__ void 
sgemvt_kernel2(int n, int m, float alpha,
               int n1, float* A, int lda, float *x, float *y)
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

  float res = 0.f;

  __shared__ float buff[32];
  __shared__ float la[16][17];

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
            la[iny+__mul24(j,4)][inx] =  0.f;
         else
            la[iny+__mul24(j,4)][inx] = A[j*__mul24(4,lda)];

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
             la[iny+__mul24(j,4)][inx] = 0.f;
          else
             la[iny+__mul24(j,4)][inx] = A[j*__mul24(4,lda)];

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
  ind = inx + __mul24(blockIdx.x,16);
  la[inx][iny]= res;
  __syncthreads();
  if (ind<m && iny==0){
     res = la[inx][0] + la[inx][1] + la[inx][2] + la[inx][3];
     y[ind] = alpha*res;
  }
}

extern "C" void
magmablas_sgemvt1_tesla(int m, int n, float alpha, float *A, int lda,
                        float *x, float *z)
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

    A      - (input) SINGLE PRECISION array of dimension ( LDA, N ) on the GPU.

    LDA    - (input) INTEGER.
             LDA specifies the leading dimension of A.

    X      - (input) SINGLE PRECISION array of dimension m.

    Z      - (output) SINGLE PRECISION array of dimension n.
             On exit Z = alpha A^t X.

    ===================================================================== */
    int blocks;

    if (n % 32==0)
        blocks = n/32;
    else
        blocks = n/32 + 1;

    dim3 grid(blocks, 1, 1);
    dim3 threads(32, 2, 1);

    sgemvt_kernel1<<<grid, threads>>>(m, n, alpha, (m / sgemv_bs)*sgemv_bs,
                                      A, lda, x, z);
}

extern "C" void
magmablas_sgemvt2_tesla(int m, int n, float alpha, float *A, int lda,
                        float *x, float *z)
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

    sgemvt_kernel2<<<grid, threads>>>(m, n, alpha, (m / 32)*32,
                                      A, lda, x, z);
}

extern "C" void
magmablas_sgemvt_tesla(int m, int n, float alpha, float *A, int lda, 
                       float *x, float *z)
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
      magmablas_sgemvt2_tesla(m, n, alpha, A, lda, x, z);
    else
      magmablas_sgemvt1_tesla(m, n, alpha, A, lda, x, z);
}

#undef num_threads
#undef sgemv_bs
