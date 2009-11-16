/*
    -- MAGMA (version 0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2009
*/

#include "cublas.h"
#include "magma.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- This is an auxiliary routine called from dgehrd.  The routine is called
      in 16 blocks, 32 thread per block and initializes to zero the 1st 
      32x32 block of A.
*/

__global__ void dset_to_zero(double *A, int lda){
   int ind = blockIdx.x*lda + threadIdx.x;

   A += ind;
   A[0] = 0.f;
//   A[16*lda] = 0.f;
}

__global__ void dset_nbxnb_to_zero(int nb, double *A, int lda){
   int ind = blockIdx.x*lda + threadIdx.x, i, j;

   A += ind;
   for(i=0; i<nb; i+=32){
     for(j=0; j<nb; j+=32)
        A[j] = 0.f;
     A += 32*lda;
   }
}

void dzero_32x32_block(double *A, int lda)
{
  // dset_to_zero<<<16, 32>>>(A, lda);
  dset_to_zero<<<32, 32>>>(A, lda);
}

void dzero_nbxnb_block(int nb, double *A, int lda)
{
  dset_nbxnb_to_zero<<<32, 32>>>(nb, A, lda);
}



