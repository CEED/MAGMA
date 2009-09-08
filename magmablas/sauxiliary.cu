/*
    -- MAGMA (version 0.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       June 2009
*/

#include "cublas.h"
#include "magma.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- This is an auxiliary routine called from sgehrd.  The routine is called
      in 16 blocks, 32 thread per block and initializes to zero the 1st 
      32x32 block of A.
*/

__global__ void sset_to_zero(float *A, int lda){
   int ind = blockIdx.x*lda + threadIdx.x;

   A += ind;
   A[0] = 0.f;
//   A[16*lda] = 0.f;
}

__global__ void sset_nbxnb_to_zero(int nb, float *A, int lda){
   int ind = blockIdx.x*lda + threadIdx.x, i, j;

   A += ind;
   for(i=0; i<nb; i+=32){
     for(j=0; j<nb; j+=32)
        A[j] = 0.f;
     A += 32*lda;
   }
}

void szero_32x32_block(float *A, int lda)
{
  // sset_to_zero<<<16, 32>>>(A, lda);
  sset_to_zero<<<32, 32>>>(A, lda);
}

void szero_nbxnb_block(int nb, float *A, int lda)
{
  sset_nbxnb_to_zero<<<32, 32>>>(nb, A, lda);
}



