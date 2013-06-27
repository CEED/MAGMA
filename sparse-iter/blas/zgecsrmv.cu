/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> c d s

*/
#include "common_magma.h"

#if (GPUSHMEM < 200)
   #define BLOCK_SIZE 128
#else
   #define BLOCK_SIZE 512
#endif



__global__ void 
zgecsrmv_kernel( int m, 
                 magmaDoubleComplex alpha, 
                 magmaDoubleComplex *d_val, 
                 int *d_rowptr, 
                 int *d_colind,
                 magmaDoubleComplex *d_x,
                 magmaDoubleComplex beta, 
                 magmaDoubleComplex *d_y)
{
  int index = blockIdx.x*blockDim.x+threadIdx.x;
  int j;
  magmaDoubleComplex tmp;

  if(index<m){
    tmp = 0;
    for( j=d_rowptr[index]; j<d_rowptr[index+1]; j++ ){
      tmp += d_val[j] * d_x[d_colind[j]];
    }
    d_y[index] = alpha * tmp + beta * d_y[index];
  }
}



/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======
    
    This routine computes y = alpha *  A^t *  x + beta * y on the GPU.
    
    Arguments
    =========

    magma_int_t m                   number of rows in A
    magma_int_t n                   number of columns in A 
    magmaDoubleComplex alpha        scalar multiplier
    magmaDoubleComplex *d_val       array containing values of A in CSR
    magma_int_t *d_rowptr           rowpointer of A in CSR
    magma_int_t *d_colind           columnindices of A in CSR
    magmaDoubleComplex *d_x         input vector x
    magmaDoubleComplex beta         scalar multiplier
    magmaDoubleComplex *d_y         input/output vector y

    =====================================================================    */

extern "C" magma_int_t
magma_zgecsrmv(char transA, 
               magma_int_t m, magma_int_t n, 
               magmaDoubleComplex alpha, 
               magmaDoubleComplex *d_val, 
               magma_int_t *d_rowptr, 
               magma_int_t *d_colind,
               magmaDoubleComplex *d_x,
               magmaDoubleComplex beta, 
               magmaDoubleComplex *d_y){


   dim3 grid(N/num_threads, 1, 1);
   dim3 threads(num_threads, 1, 1);

   zgecsrmv_kernel<<< grid, threads, 0, magma_stream >>>(d_A, d_I, d_J, N, d_X, d_Y);


   return MAGMA_SUCCESS;
}



