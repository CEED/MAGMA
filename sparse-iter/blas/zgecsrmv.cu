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

extern "C" magma_int_t
magma_zgecsrmv(char transA, 
               magma_int_t m, magma_int_t n, 
               magmaDoubleComplex alpha, 
               magmaDoubleComplex *d_val, 
               magma_int_t *d_rowptr, magma_int_t *d_colind,
               magmaDoubleComplex *d_x,
               magmaDoubleComplex beta, 
               magmaDoubleComplex *d_y)
{
/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======

    
    Arguments
    =========



    =====================================================================    */
/*
   dim3 grid(N/num_threads, 1, 1);
   dim3 threads(num_threads, 1, 1);

   smv_kernel<<<grid, threads>>>(d_A, d_I, d_J, N, d_X, d_Y);
*/

   return MAGMA_SUCCESS;
}


