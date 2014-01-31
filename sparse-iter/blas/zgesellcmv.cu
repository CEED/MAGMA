/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s

*/
#include "cuda_runtime.h"
#include <stdio.h>
#include "common_magma.h"

#if (GPUSHMEM < 200)
   #define BLOCK_SIZE 128
#else
   #define BLOCK_SIZE 512
#endif


#define PRECISION_z


// SELLC SpMV kernel
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
__global__ void 
zgesellcmv_kernel(   int num_rows, 
                     int num_cols,
                     int blocksize,
                     magmaDoubleComplex alpha, 
                     magmaDoubleComplex *d_val, 
                     int *d_colind,
                     int *d_rowptr,
                    // int *d_blockinfo,
                     magmaDoubleComplex *d_x,
                     magmaDoubleComplex beta, 
                     magmaDoubleComplex *d_y)
{
    // threads assigned to rows
    int Idx = blockDim.x * blockIdx.x + threadIdx.x ;
    int offset = d_rowptr[ blockIdx.x ];
    int border = (d_rowptr[ blockIdx.x+1 ]-offset)/blocksize;
    if(Idx < num_rows ){
        magmaDoubleComplex dot = MAGMA_Z_MAKE(0.0, 0.0);
        for ( int n = 0; n < border; n++){ //d_blockinfo[ blockIdx.x ] ; n ++){
            int col = d_colind [offset+ blocksize * n + threadIdx.x ];
            magmaDoubleComplex val = d_val[offset+ blocksize * n + threadIdx.x];
            if( val != 0){
                  dot=dot+val*d_x[col];
            }
        }

        d_y[ Idx ] = dot * alpha + beta * d_y [ Idx ];
    }
}

// SELLC SpMV kernel modified assigning multiple threads to each row
__global__ void 
zgesellcmtmv_kernel( int num_rows, 
                     int num_cols,
                     int blocksize,
                     int T,
                     magmaDoubleComplex alpha, 
                     magmaDoubleComplex *d_val, 
                     int *d_colind,
                     int *d_rowptr,
                     magmaDoubleComplex *d_x,
                     magmaDoubleComplex beta, 
                     magmaDoubleComplex *d_y)
{
   // threads assigned to rows
    int idx = blockDim.x * blockIdx.x + threadIdx.x ;
    int offset = d_rowptr[ blockIdx.x ];

    int idb = threadIdx.x ;  // local thread index
    int idp = idb%T;  // number of thread in this row
    int i = idx/T;  // row index
    int ilocal = threadIdx.x/T;  // local row index

    extern __shared__ magmaDoubleComplex shared[];



    if(i < num_rows ){
        magmaDoubleComplex dot = MAGMA_Z_MAKE(0.0, 0.0);
        int max_ = ((d_rowptr[ blockIdx.x+1 ]-offset)/blocksize) / T;  
            // number of elements each thread handles
        for ( int k = 0; k < max_ ; k++ ){
    
            magmaDoubleComplex val = 
                        d_val[ offset + idp*blocksize*max_ + ilocal + blocksize * k ];
            int col = 
                        d_colind[ offset + idp*blocksize*max_ + ilocal + blocksize * k ];

            dot += val * d_x[ col ];
        }
        shared[idb]  = dot;
/*
        if( idp < 16 ){
            shared[idb]+=shared[idb+16];
            if( idp < 8 ) shared[idb]+=shared[idb+8];
            if( idp < 4 ) shared[idb]+=shared[idb+4];
            if( idp < 2 ) shared[idb]+=shared[idb+2];
            if( idp == 0 ) {
                d_y[i] = (shared[idb]+shared[idb+1])*alpha + beta*d_y [i];
            }

        }*/
        #if defined(PRECISION_z) || defined(PRECISION_c)
            if( idp < 16 ){
                shared[idb]+=shared[idb+16];
                if( idp < 8 ) shared[idb]+=shared[idb+8];
                if( idp < 4 ) shared[idb]+=shared[idb+4];
                if( idp < 2 ) shared[idb]+=shared[idb+2];
                if( idp == 0 ) {
                    d_y[i] = (shared[idb]+shared[idb+1])*alpha + beta*d_y [i];
                }

            }
        #endif
        #if defined(PRECISION_d)
            if( idp < 16 ){
                volatile double *temp2 = shared;
                temp2[ idb ] += temp2[ idb + 16 ];
                temp2[ idb ] += temp2[ idb + 8 ];
                temp2[ idb ] += temp2[ idb + 4 ];
                temp2[ idb ] += temp2[ idb + 2 ];
                if( idp == 0 ) {
                    d_y[i] = (temp2[idb]+temp2[idb+1])*alpha + beta*d_y [i];
                }
            }
        #endif
        #if defined(PRECISION_s)
            if( idp < 16 ){
                volatile double *temp2 = shared;
                temp2[ idb ] += temp2[ idb + 16 ];
                temp2[ idb ] += temp2[ idb + 8 ];
                temp2[ idb ] += temp2[ idb + 4 ];
                temp2[ idb ] += temp2[ idb + 2 ];
                if( idp == 0 ) {
                    d_y[i] = (temp2[idb]+temp2[idb+1])*alpha + beta*d_y [i];
                }
            }
        #endif
    }
}

/*
   printf("in kernel\n");
    // threads assigned to rows
    int idx = threadIdx.x ;
    int idy = threadIdx.y ;
    int i = blockDim.x * blockIdx.x + threadIdx.x ;
    int offset = d_rowptr[ blockIdx.x ];

   // extern __shared__ magmaDoubleComplex shared[];



    if(i < num_rows ){
    printf("thread %d %d\n", idx, idy);
        magmaDoubleComplex dot = MAGMA_Z_MAKE(0.0, 0.0);
        int max_ = ((d_rowptr[ blockIdx.x+1 ]-offset)/blocksize) / T;  
            // number of elements each thread handles
        for ( int k = 0; k < max_ ; k++ ){
    
            magmaDoubleComplex val = 
                        d_val[ offset + idy*blocksize*max_ + idx + blocksize * k ];
            int col = 
                        d_colind[ offset + idy*blocksize*max_ + idx  + blocksize * k ];

            dot += val * d_x[ col ];
        }
        shared[idx*32+idy]  = dot;
        if( idy < 16 ){
            shared[idy]+=shared[idy+16];
            if( idy < 8 ) shared[idy]+=shared[idy+8];
            if( idy < 4 ) shared[idy]+=shared[idy+4];
            if( idy < 2 ) shared[idy]+=shared[idy+2];
            if( idy == 0 ) {
                d_y[i] = (shared[idy]+shared[idy+1])*alpha + beta*d_y [i];
            }

        }
    }
    printf("end kernel\n");
*/

/*
    // threads assigned to rows
    int idx = blockDim.x * blockIdx.x + threadIdx.x ;
    int offset = d_rowptr[ blockIdx.x ];

    int idb = threadIdx.x ;  // local thread index
    int idp = idb%T;  // number of thread in this row
    int i = idx/T;  // row index
    int ilocal = threadIdx.x/T;  // local row index

    extern __shared__ magmaDoubleComplex shared[];



    if(i < num_rows ){
        magmaDoubleComplex dot = MAGMA_Z_MAKE(0.0, 0.0);
        int max_ = ((d_rowptr[ blockIdx.x+1 ]-offset)/blocksize) / T;  
            // number of elements each thread handles
        for ( int k = 0; k < max_ ; k++ ){
    
            magmaDoubleComplex val = 
                        d_val[ offset + idp*blocksize*max_ + ilocal + blocksize * k ];
            int col = 
                        d_colind[ offset + idp*blocksize*max_ + ilocal + blocksize * k ];

            dot += val * d_x[ col ];
        }
        shared[idb]  = dot;
        if( idp < 16 ){
            shared[idb]+=shared[idb+16];
            if( idp < 8 ) shared[idb]+=shared[idb+8];
            if( idp < 4 ) shared[idb]+=shared[idb+4];
            if( idp < 2 ) shared[idb]+=shared[idb+2];
            if( idp == 0 ) {
                d_y[i] = MAGMA_Z_MAKE((double) i, 0.0);//(shared[idb]+shared[idb+1])*alpha + beta*d_y [i];
            }

        }
    }

*/

/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

    Purpose
    =======
    
    This routine computes y = alpha *  A^t *  x + beta * y on the GPU.
    Input format is SELLC.
    
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

    ======================================================================    */

extern "C" magma_int_t
magma_zgesellcmv(   magma_trans_t transA,
                    magma_int_t m, magma_int_t n,
                    magma_int_t blocksize,
                    magma_int_t slices,
                    magma_int_t alignment,
                    magmaDoubleComplex alpha,
                    magmaDoubleComplex *d_val,
                    magma_int_t *d_colind,
                    magma_int_t *d_rowptr,
                    magma_int_t *d_blockinfo,
                    magmaDoubleComplex *d_x,
                    magmaDoubleComplex beta,
                    magmaDoubleComplex *d_y ){


    if( alignment == 1 ){
       // the kernel can only handle up to 65535 slices 
       // (~2M rows for blocksize 32)
       dim3 grid( slices, 1, 1);

       zgesellcmv_kernel<<< grid, blocksize, 0, magma_stream >>>
       ( m, n, blocksize, alpha,
            d_val, d_colind, d_rowptr, d_x, beta, d_y );

    }else{
        int num_threads = blocksize*alignment;


        int num_blocks = ( (m+num_threads-1)
                                        /num_threads);
        dim3 block_size;
        block_size.x = blocksize;
        block_size.y = alignment;


        dim3 grid( num_blocks, 1, 1);
        int Ms = num_threads * sizeof( magmaDoubleComplex );


       zgesellcmtmv_kernel<<< grid, num_threads, Ms, magma_stream >>>
       ( m, n, blocksize, alignment, alpha,
            d_val, d_colind, d_rowptr, d_x, beta, d_y );


    }

   return MAGMA_SUCCESS;
}

