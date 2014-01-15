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
        int max_ = (d_rowlength[i]+T-1)/T;  // number of elements each thread handles

        for ( int k = 0; k < max_ ; k++ ){
            //magmaDoubleComplex val = d_val[ k*(T*alignment)+(i*T)+idp ];
            //int col = d_colind [ k*(T*alignment)+(i*T)+idp ];
            magmaDoubleComplex val = d_val[ k*(T)+(i*alignment)+idp ];
            int col = d_colind [ k*(T)+(i*alignment)+idp ];
            dot += val * d_x[col ];
        }
        d_y[i] = (shared[idb])*alpha + beta*d_y [i];
        shared[idb]  = dot;
        d_y[i] = (shared[idb])*alpha + beta*d_y [i];

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
       November 2011

    Purpose
    =======
    
    This routine computes y = alpha *  A *  x + beta * y on the GPU.
    Input format is ELLPACKRT.
    
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
magma_zgeellrtmv(const char *transA,
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


    int alignment = ((int)(nnz_per_row+threads_per_row-1)/threads_per_row)*threads_per_row;


    dim3 grid( num_blocks, 1, 1);
    int Ms =  num_threads* sizeof( magmaDoubleComplex );
    zgeellrtmv_kernel<<< grid, num_threads, Ms, magma_stream >>>
             ( m, n, alpha, d_val, d_colind, d_rowlength, d_x, beta, d_y, threads_per_row, alignment );


   return MAGMA_SUCCESS;
}


