/*
    -- MAGMA (version 0.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       June 2009
*/

#include "cublas.h"
#include "magma.h"

#define num_threads 64
#define sgemv_bs 64

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

extern "C" void
magmablas_sgemv(int n, int m, float *A, int lda, float *x, float *z)
{
/*  -- MAGMA (version 0.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       June 2009

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
 
    sgemv_kernel<<<grid, threads>>>(n, m, (m / sgemv_bs)*sgemv_bs, 
                                    A, lda, x, z);
}


__global__ void 
sgemvt_kernel(int n, int m, float alpha, int n1, float* A, int lda,
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
  if (ind<n){
     res = la[inx][0] + la[inx][1];
     y[ind] = alpha*res;
  }
}


extern "C" void
magmablas_sgemvt(int n, int m, float alpha, float *A, int lda, 
                 float *x, float *z)
{
/*  -- MAGMA (version 0.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       June 2009

    Purpose
    =======

    This routine computes z = alpha A^t x on the GPU.

    N      - (input) INTEGER.
             On entry, N specifies the number of rows of the matrix A.

    M      - (input) INTEGER.
             On entry, M specifies the number of columns of the matrix A

    A      - (input) SINGLE PRECISION array of dimension ( LDA, m ) on the GPU.

    LDA    - (input) INTEGER.
             LDA specifies the leading dimension of A.

    X      - (input) SINGLE PRECISION array of dimension n.

    Z      - (output) SINGLE PRECISION array of dimension n.
             On exit Z = alpha A^t X.

    ===================================================================== */

    int blocks;
    if (m % 32==0)
        blocks = m/32;
    else
        blocks = m/32 + 1;

    dim3 grid(blocks, 1, 1);
    dim3 threads(32, 2, 1);

    sgemvt_kernel<<<grid, threads>>>(n, m, alpha, (n / sgemv_bs)*sgemv_bs,
                                     A, lda, x, z);
}

#undef num_threads
#undef sgemv_bs
