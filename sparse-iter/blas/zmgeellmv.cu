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
zmgeellmv_kernel( int num_rows, 
                  int num_cols,
                  int num_vecs,
                  int num_cols_per_row,
                  magmaDoubleComplex alpha, 
                  magmaDoubleComplex *d_val, 
                  int *d_colind,
                  magmaDoubleComplex *d_x,
                  magmaDoubleComplex beta, 
                  magmaDoubleComplex *d_y)
{
int row = blockDim.x * blockIdx.x + threadIdx.x ;

    extern __shared__ magmaDoubleComplex dot[];

    if(row < num_rows ){
        for( int i=0; i<num_vecs; i++)
                dot[ threadIdx.x + i*blockDim.x ] = MAGMA_Z_MAKE(0.0, 0.0);
        for ( int n = 0; n < num_cols_per_row ; n ++){
            int col = d_colind [ num_cols_per_row * row + n ];
            magmaDoubleComplex val = d_val [ num_cols_per_row * row + n ];
            if( val != 0){
                for( int i=0; i<num_vecs; i++)
                    dot[ threadIdx.x + i*blockDim.x ] += 
                                    val * d_x[col + i * num_cols ];
            }
        }
        for( int i=0; i<num_vecs; i++)
                d_y[ row + i*num_cols ] = dot[ threadIdx.x + i*blockDim.x ] 
                                * alpha + beta * d_y [ row + i * num_cols ];
    }
}





/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======
    
    This routine computes Y = alpha *  A *  X + beta * Y for X and Y sets of 
    num_vec vectors on the GPU. Input format is ELLPACK. 
    
    Arguments
    =========

    magma_int_t m                   number of rows in A
    magma_int_t n                   number of columns in A 
    mama_int_t num_vecs             number of vectors
    magmaDoubleComplex alpha        scalar multiplier
    magmaDoubleComplex *d_val       array containing values of A in ELLPACK
    magma_int_t *d_colind           columnindices of A in ELLPACK
    magmaDoubleComplex *d_x         input vector x
    magmaDoubleComplex beta         scalar multiplier
    magmaDoubleComplex *d_y         input/output vector y

    ======================================================================    */

extern "C" magma_int_t
magma_zmgeellmv(const char *transA,
               magma_int_t m, magma_int_t n,
               magma_int_t num_vecs,
               magma_int_t nnz_per_row,
               magmaDoubleComplex alpha,
               magmaDoubleComplex *d_val,
               magma_int_t *d_colind,
               magmaDoubleComplex *d_x,
               magmaDoubleComplex beta,
               magmaDoubleComplex *d_y ){



    dim3 grid( (m+BLOCK_SIZE-1)/BLOCK_SIZE, 1, 1);
    unsigned int MEM_SIZE =  num_vecs* BLOCK_SIZE 
                            * sizeof( magmaDoubleComplex ); // num_vecs vectors 
    zmgeellmv_kernel<<< grid, BLOCK_SIZE, MEM_SIZE >>>
        ( m, n, num_vecs, nnz_per_row, alpha, d_val, d_colind, d_x, beta, d_y );


   return MAGMA_SUCCESS;
}



