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
   #define BLOCK_SIZE 256
#else
   #define BLOCK_SIZE 256
#endif



__global__ void 
zgecsrmv_kernel( int num_rows, int num_cols, 
                 magmaDoubleComplex alpha, 
                 magmaDoubleComplex *d_val, 
                 int *d_rowptr, 
                 int *d_colind,
                 magmaDoubleComplex *d_x,
                 magmaDoubleComplex beta, 
                 magmaDoubleComplex *d_y){

    int row = blockIdx.x*blockDim.x+threadIdx.x;
    int j;

    if(row<num_rows){
        magmaDoubleComplex dot = MAGMA_Z_ZERO;
        int start = d_rowptr[ row ];
        int end = d_rowptr[ row+1 ];
        for( j=start; j<end; j++)
            dot += d_val[ j ] * d_x[ d_colind[j] ];
        d_y[ row ] =  dot *alpha + beta * d_y[ row ];
    }
}


__global__ void 
zgecsrmv_kernel_shift( int num_rows, int num_cols, 
                       magmaDoubleComplex alpha, 
                       magmaDoubleComplex lambda, 
                       magmaDoubleComplex *d_val, 
                       int *d_rowptr, 
                       int *d_colind,
                       magmaDoubleComplex *d_x,
                       magmaDoubleComplex beta, 
                       magmaDoubleComplex *d_y){

    int row = blockIdx.x*blockDim.x+threadIdx.x;
    int j;

    if(row<num_rows){
        magmaDoubleComplex dot = MAGMA_Z_ZERO;
        int start = d_rowptr[ row ];
        int end = d_rowptr[ row+1 ];
        for( j=start; j<end; j++)
            dot += d_val[ j ] * d_x[ d_colind[j] ];
        d_y[ row ] =  dot *alpha - lambda * d_x[ row ] + beta * d_y[ row ];
    }
}


/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======
    
    This routine computes y = alpha *  A *  x + beta * y on the GPU.
    
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
magma_zgecsrmv(     const char *transA,
                    magma_int_t m, magma_int_t n,
                    magmaDoubleComplex alpha,
                    magmaDoubleComplex *d_val,
                    magma_int_t *d_rowptr,
                    magma_int_t *d_colind,
                    magmaDoubleComplex *d_x,
                    magmaDoubleComplex beta,
                    magmaDoubleComplex *d_y ){

    dim3 grid( (m+BLOCK_SIZE-1)/BLOCK_SIZE, 1, 1);

    zgecsrmv_kernel<<< grid, BLOCK_SIZE, 0, magma_stream >>>(m, n, alpha,
                                                              d_val, d_rowptr, d_colind,
                                                              d_x, beta, d_y);

    return MAGMA_SUCCESS;
}



/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======
    
    This routine computes y = alpha * ( A -lambda I ) * x + beta * y on the GPU.
    
    Arguments
    =========

    magma_int_t m                   number of rows in A
    magma_int_t n                   number of columns in A 
    magmaDoubleComplex alpha        scalar multiplier
    magmaDoubleComplex alpha        scalar multiplier
    magmaDoubleComplex *d_val       array containing values of A in CSR
    magma_int_t *d_rowptr           rowpointer of A in CSR
    magma_int_t *d_colind           columnindices of A in CSR
    magmaDoubleComplex *d_x         input vector x
    magmaDoubleComplex beta         scalar multiplier
    magmaDoubleComplex *d_y         input/output vector y

    =====================================================================    */

extern "C" magma_int_t
magma_zgecsrmv_shift( const char *transA,
                      magma_int_t m, magma_int_t n,
                      magmaDoubleComplex alpha,
                      magmaDoubleComplex lambda,
                      magmaDoubleComplex *d_val,
                      magma_int_t *d_rowptr,
                      magma_int_t *d_colind,
                      magmaDoubleComplex *d_x,
                      magmaDoubleComplex beta,
                      magmaDoubleComplex *d_y ){

    dim3 grid( (m+BLOCK_SIZE-1)/BLOCK_SIZE, 1, 1);

    zgecsrmv_kernel_shift<<< grid, BLOCK_SIZE, 0, magma_stream >>>
                         (m, n, alpha, lambda, d_val, d_rowptr, d_colind, d_x, beta, d_y);

    return MAGMA_SUCCESS;
}



