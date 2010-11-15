/*
    -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

       @precisions normal z -> s d c

*/

#include <cublas.h>
#include "magma.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- This is an auxiliary routine called from zgehrd.  The routine is called
      in 16 blocks, 32 thread per block and initializes to zero the 1st 
      32x32 block of A.
*/

__global__ void zset_to_zero(cuDoubleComplex *A, int lda){
    int ind = blockIdx.x*lda + threadIdx.x;
    
    A += ind;
    A[0] = MAGMA_Z_ZERO;
//   A[16*lda] = 0.;
}

__global__ void zset_nbxnb_to_zero(int nb, cuDoubleComplex *A, int lda){
   int ind = blockIdx.x*lda + threadIdx.x, i, j;

   A += ind;
   for(i=0; i<nb; i+=32){
     for(j=0; j<nb; j+=32)
         A[j] = MAGMA_Z_ZERO;
     A += 32*lda;
   }
}

void zzero_32x32_block(cuDoubleComplex *A, int lda)
{
  // zset_to_zero<<<16, 32>>>(A, lda);
  zset_to_zero<<<32, 32>>>(A, lda);
}

void zzero_nbxnb_block(int nb, cuDoubleComplex *A, int lda)
{
  zset_nbxnb_to_zero<<<32, 32>>>(nb, A, lda);
}



