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



//Michael Garland
__global__ void 
zgeelltmv_kernel( int num_rows, 
                 int num_cols,
                 int num_cols_per_row,
                 magmaDoubleComplex alpha, 
                 magmaDoubleComplex *d_val, 
                 int *d_colind,
                 magmaDoubleComplex *d_x,
                 magmaDoubleComplex beta, 
                 magmaDoubleComplex *d_y)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x ;
    if(row < num_rows ){
        magmaDoubleComplex dot = MAGMA_Z_MAKE(0.0, 0.0);
        for ( int n = 0; n < num_cols_per_row ; n ++){
            magma_int_t col = d_colind [ num_rows * n + row ];
            magmaDoubleComplex val = d_val [ num_rows * n + row ];
            if( val != 0)
                dot += val * d_x[col ];
        }
        d_y[ row ] = dot * alpha + beta * d_y [ row ];
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
    Input format is ELLPACKT.
    
    Arguments
    =========

    magma_int_t m                   number of rows in A
    magma_int_t n                   number of columns in A 
    magmaDoubleComplex alpha        scalar multiplier
    magmaDoubleComplex *d_val       array containing values of A in ELLPACK
    magma_int_t *d_colind           columnindices of A in ELLPACK
    magmaDoubleComplex *d_x         input vector x
    magmaDoubleComplex beta         scalar multiplier
    magmaDoubleComplex *d_y         input/output vector y

    =====================================================================    */

extern "C" magma_int_t
magma_zgeelltmv(const char *transA,
               magma_int_t m, magma_int_t n,
               magma_int_t nnz_per_row,
               magmaDoubleComplex alpha,
               magmaDoubleComplex *d_val,
               magma_int_t *d_colind,
               magmaDoubleComplex *d_x,
               magmaDoubleComplex beta,
               magmaDoubleComplex *d_y ){



   dim3 grid( (m+BLOCK_SIZE-1)/BLOCK_SIZE, 1, 1);

   zgeelltmv_kernel<<< grid, BLOCK_SIZE, 0 >>>
                  ( m, n, nnz_per_row, alpha, d_val, d_colind, d_x, beta, d_y );


   return MAGMA_SUCCESS;
}



