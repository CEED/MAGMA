/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s

*/

#include "common_magma.h"

#if (GPUSHMEM < 200)
   #define BLOCK_SIZE 128
#else
   #define BLOCK_SIZE 512
#endif



//F. Vázquez, G. Ortega, J.J. Fernández, E.M. Garzón, Almeria University
__global__ void 
zgeellrtmv_kernel( int num_rows, 
                 int num_cols,
                 magmaDoubleComplex alpha, 
                 magmaDoubleComplex *d_val, 
                 int *d_colind,
                 int *d_rowlength,
                 magmaDoubleComplex *d_x,
                 magmaDoubleComplex beta, 
                 magmaDoubleComplex *d_y,
                 int T,
                 int alignment)
{
int idx = blockDim.x * blockIdx.x + threadIdx.x ; // global thread index
int idb = threadIdx.x ;  // local thread index
int idp = idb%T;  // number of threads assigned to one row
int i = idx/T;  // row index

extern __shared__ magmaDoubleComplex shared[];

    if(i < num_rows ){
        magmaDoubleComplex dot = MAGMA_Z_MAKE(0.0, 0.0);
        int max_ = (d_rowlength[i]+T-1)/T;  
            // number of elements each thread handles

        for ( int k = 0; k < max_ ; k++ ){

            // original code in paper (not working for me)
            //magmaDoubleComplex val = d_val[ k*(T*alignment)+(i*T)+idp ];  
            //int col = d_colind [ k*(T*alignment)+(i*T)+idp ];    

            // new code (working for me)        
            magmaDoubleComplex val = d_val[ k*(T)+(i*alignment)+idp ];
            int col = d_colind [ k*(T)+(i*alignment)+idp ];

            dot += val * d_x[ col ];
        }
        shared[idb]  = dot;
        if( idp < 16 ){
            shared[idb]+=shared[idb+16];
            if( idp < 8 ) shared[idb]+=shared[idb+8];
            if( idp < 4 ) shared[idb]+=shared[idb+4];
            if( idp < 2 ) shared[idb]+=shared[idb+2];
            if( idp == 0 ) {
                d_y[i] = (shared[idb]+shared[idb+1])*alpha + beta*d_y [i];
            }

        }
    }

}



/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

    Purpose
    =======
    
    This routine computes y = alpha *  A *  x + beta * y on the GPU.
    Input format is ELLPACKRT. The ideas are taken from 
    "Improving the performance of the sparse matrix
    vector product with GPUs", (CIT 2010), 
    and modified to provide correct values.

    
    Arguments
    =========
    const char *transA                  transpose info for matrix (not needed)
    magma_int_t m                       number of rows 
    magma_int_t n                       number of columns
    magma_int_t nnz_per_row             max number of nonzeros in a row
    magmaDoubleComplex alpha            scalar alpha
    magmaDoubleComplex *d_val           val array
    magma_int_t *d_colind               col indices  
    magma_int_t *d_rowlength            number of elements in each row
    magmaDoubleComplex *d_x             input vector x
    magmaDoubleComplex beta             scalar beta
    magmaDoubleComplex *d_y             output vector y
    magma_int_t num_threads             threads per block
    magma_int_t threads_per_row         threads assigned to each row

    =====================================================================    */

extern "C" magma_int_t
magma_zgeellrtmv(  magma_trans_t transA,
                   magma_int_t m, magma_int_t n,
                   magma_int_t nnz_per_row,
                   magmaDoubleComplex alpha,
                   magmaDoubleComplex *d_val,
                   magma_int_t *d_colind,
                   magma_int_t *d_rowlength,
                   magmaDoubleComplex *d_x,
                   magmaDoubleComplex beta,
                   magmaDoubleComplex *d_y,
                   magma_int_t num_threads,
                   magma_int_t threads_per_row ){


    int num_blocks = ( (threads_per_row*m+num_threads-1)
                                    /num_threads);


    int alignment = ((int)(nnz_per_row+threads_per_row-1)/threads_per_row)
                            *threads_per_row;


    dim3 grid( num_blocks, 1, 1);
    int Ms =  num_threads* sizeof( magmaDoubleComplex );
    //    printf("launch kernel: %d %d %d\n", grid.x, num_threads, Ms);
    zgeellrtmv_kernel<<< grid, num_threads, Ms, magma_stream >>>
             ( m, n, alpha, d_val, d_colind, d_rowlength, d_x, beta, d_y, 
                                                threads_per_row, alignment );


   return MAGMA_SUCCESS;
}


