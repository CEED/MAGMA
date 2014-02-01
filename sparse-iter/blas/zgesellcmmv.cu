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
   // T threads assigned to each row
    int idb = threadIdx.x ;  // local thread index
    int idp = idb%blocksize;  // local row index
    int bdx = blockIdx.y * 65535 + blockIdx.x; // global block index
    int row = bdx * blocksize + idp;  // row index
    int i = idb/blocksize;  // number of thread in row

    extern __shared__ magmaDoubleComplex shared[];

    if(row < num_rows ){
        magmaDoubleComplex dot = MAGMA_Z_MAKE(0.0, 0.0);
        int offset = d_rowptr[ bdx ];
        int max_ = ((d_rowptr[ bdx+1 ]-offset)/blocksize) / T;  

            // number of elements each thread handles
        for ( int k = 0; k < max_ ; k++ ){
            magmaDoubleComplex val = 
                        d_val[ offset + idb + blocksize*T*k ];
            int col = 
                    d_colind[ offset + idb + blocksize*T*k ];
            dot += val * d_x[ col ];

        }
        shared[idb]  = dot;

        __syncthreads();
        if( i < 16 ){
            shared[idb]+=shared[idb+blocksize*16];              __syncthreads();
            if( i < 8 ) shared[idb]+=shared[idb+blocksize*8];   __syncthreads();
            if( i < 4 ) shared[idb]+=shared[idb+blocksize*4];   __syncthreads();
            if( i < 2 ) shared[idb]+=shared[idb+blocksize*2];   __syncthreads();
            if( i == 0 ) {
                d_y[row] = 
                (shared[idb]+shared[idb+blocksize*1])*alpha + beta*d_y [row];
            }

        }
       // #endif
  /*      #if defined(PRECISION_d)
        if( i < 16 ){
            volatile double *temp2 = shared;
            if( i < 16 ) temp2[ idb ] += temp2[ idb+blocksize*16 ];
            if( i < 8 ) temp2[ idb ] += temp2[ idb+blocksize*8 ];
            if( i < 4 ) temp2[ idb ] += temp2[ idb+blocksize*4 ];
            if( i < 2 ) temp2[ idb ] += temp2[ idb+blocksize*2 ];
            if( i == 0 ) {
                d_y[row] = 
                (temp2[idb]+temp2[idb+blocksize*1])*alpha + beta*d_y [row];
            }
        }
        #endif
        #if defined(PRECISION_s)
        if( i < 16 ){
            volatile float *temp2 = shared;
            temp2[ idb ] += temp2[ idb+blocksize*16 ];
            temp2[ idb ] += temp2[ idb+blocksize*8 ];
            temp2[ idb ] += temp2[ idb+blocksize*4 ];
            temp2[ idb ] += temp2[ idb+blocksize*2 ];
            if( i == 0 ) {
                d_y[row] = 
                (temp2[idb]+temp2[idb+blocksize*1])*alpha + beta*d_y [row];
            }
        }
        #endif*/
    }
}



// SELLCM SpMV kernel
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
__global__ void 
zgesellcmtmv2d_kernel_8( int num_rows, 
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
   // T threads assigned to each row
    int idx = threadIdx.y ;     // thread in row
    int idy = threadIdx.x;      // local row
    int ldx = idx * blocksize + idy;
    int bdx = blockIdx.y * gridDim.x + blockIdx.x; // global block index
    int row = bdx * blocksize + idy;  // global row index

    extern __shared__ magmaDoubleComplex shared[];

    if(row < num_rows ){
        magmaDoubleComplex dot = MAGMA_Z_MAKE(0.0, 0.0);
        int offset = d_rowptr[ bdx ];
        int block = blocksize * T; // total number of threads

        int max_ = (d_rowptr[ bdx+1 ]-offset)/block;  
            // number of elements each thread handles
        for ( int k = 0; k < max_ ; k++ ){
            magmaDoubleComplex val = 
                        d_val[ offset + ldx + block*k ];
            int col = 
                    d_colind[ offset + ldx + block*k ];
            dot += val * d_x[ col ];
        }
        shared[ldx]  = dot;

        __syncthreads();
        if( idx < 4 ){
            shared[ldx]+=shared[ldx+blocksize*4];              
            __syncthreads();
            if( idx < 2 ) shared[ldx]+=shared[ldx+blocksize*2];   
            __syncthreads();
            if( idx == 0 ) {
                d_y[row] = 
                (shared[ldx]+shared[ldx+blocksize*1])*alpha + beta*d_y [row];
            }

        }

    }
}
// SELLCM SpMV kernel
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
__global__ void 
zgesellcmtmv2d_kernel_16( int num_rows, 
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
   // T threads assigned to each row
    int idx = threadIdx.y ;     // thread in row
    int idy = threadIdx.x;      // local row
    int ldx = idx * blocksize + idy;
    int bdx = blockIdx.y * gridDim.x + blockIdx.x; // global block index
    int row = bdx * blocksize + idy;  // global row index

    extern __shared__ magmaDoubleComplex shared[];

    if(row < num_rows ){
        magmaDoubleComplex dot = MAGMA_Z_MAKE(0.0, 0.0);
        int offset = d_rowptr[ bdx ];
        int block = blocksize * T; // total number of threads

        int max_ = (d_rowptr[ bdx+1 ]-offset)/block;  
            // number of elements each thread handles
        for ( int k = 0; k < max_ ; k++ ){
            magmaDoubleComplex val = 
                        d_val[ offset + ldx + block*k ];
            int col = 
                    d_colind[ offset + ldx + block*k ];
            dot += val * d_x[ col ];
        }
        shared[ldx]  = dot;

        __syncthreads();
        if( idx < 8 ){
            shared[ldx]+=shared[ldx+blocksize*8];              
            __syncthreads();
            if( idx < 4 ) shared[ldx]+=shared[ldx+blocksize*4];   
            __syncthreads();
            if( idx < 2 ) shared[ldx]+=shared[ldx+blocksize*2];   
            __syncthreads();
            if( idx == 0 ) {
                d_y[row] = 
                (shared[ldx]+shared[ldx+blocksize*1])*alpha + beta*d_y [row];
            }

        }

    }
}


// SELLCM SpMV kernel
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
__global__ void 
zgesellcmtmv2d_kernel_32( int num_rows, 
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
   // T threads assigned to each row
    int idx = threadIdx.y ;     // thread in row
    int idy = threadIdx.x;      // local row
    int ldx = idx * blocksize + idy;
    int bdx = blockIdx.y * gridDim.x + blockIdx.x; // global block index
    int row = bdx * blocksize + idy;  // global row index

    extern __shared__ magmaDoubleComplex shared[];

    if(row < num_rows ){
        magmaDoubleComplex dot = MAGMA_Z_MAKE(0.0, 0.0);
        int offset = d_rowptr[ bdx ];
        int block = blocksize * T; // total number of threads

        int max_ = (d_rowptr[ bdx+1 ]-offset)/block;  
            // number of elements each thread handles
        for ( int k = 0; k < max_ ; k++ ){
            magmaDoubleComplex val = 
                        d_val[ offset + ldx + block*k ];
            int col = 
                    d_colind[ offset + ldx + block*k ];
            dot += val * d_x[ col ];
        }
        shared[ldx]  = dot;

        __syncthreads();
        if( idx < 16 ){
            shared[ldx]+=shared[ldx+blocksize*16];              
            __syncthreads();
            if( idx < 8 ) shared[ldx]+=shared[ldx+blocksize*8];  
            __syncthreads();
            if( idx < 4 ) shared[ldx]+=shared[ldx+blocksize*4];   
            __syncthreads();
            if( idx < 2 ) shared[ldx]+=shared[ldx+blocksize*2];   
            __syncthreads();
            if( idx == 0 ) {
                d_y[row] = 
                (shared[ldx]+shared[ldx+blocksize*1])*alpha + beta*d_y [row];
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
    
    This routine computes y = alpha *  A^t *  x + beta * y on the GPU.
    Input format is SELLCM.
    
    Arguments
    =========

    magma_trans_t transA            transpose A?
    magma_int_t m                   number of rows in A
    magma_int_t n                   number of columns in A 
    magma_int_t blocksize           number of rows in one ELLPACKT-slice
    magma_int_t slices              number of slices in matrix
    magma_int_t alignment           number of threads assigned to one row
    magmaDoubleComplex alpha        scalar multiplier
    magmaDoubleComplex *d_val       array containing values of A in SELLCM
    magma_int_t *d_colind           columnindices of A in SELLCM
    magma_int_t *d_rowptr           rowpointer of SELLCM
    magmaDoubleComplex *d_x         input vector x
    magmaDoubleComplex beta         scalar multiplier
    magmaDoubleComplex *d_y         input/output vector y

    ======================================================================    */

extern "C" magma_int_t
magma_zgesellcmmv(  magma_trans_t transA,
                    magma_int_t m, magma_int_t n,
                    magma_int_t blocksize,
                    magma_int_t slices,
                    magma_int_t alignment,
                    magmaDoubleComplex alpha,
                    magmaDoubleComplex *d_val,
                    magma_int_t *d_colind,
                    magma_int_t *d_rowptr,
                    magmaDoubleComplex *d_x,
                    magmaDoubleComplex beta,
                    magmaDoubleComplex *d_y ){

    // using a 2D thread grid

    int num_threads = blocksize*alignment;
    if( num_threads > 512)
        printf("error: shared memory more than 512 threads requested.\n");


    dim3 block( blocksize, alignment, 1);

    int dimgrid1 = sqrt(slices);
    int dimgrid2 = (slices + dimgrid1 -1 ) / dimgrid1;

    dim3 grid( dimgrid1, dimgrid2, 1);
    int Ms = num_threads * sizeof( magmaDoubleComplex );
    //printf("launch kernel: %d x %d -> %d %d\n", 
                        //grid.x, grid.y, num_threads, Ms);
    if( alignment == 8)
        zgesellcmtmv2d_kernel_8<<< grid, block, Ms, magma_stream >>>
        ( m, n, blocksize, alignment, alpha,
            d_val, d_colind, d_rowptr, d_x, beta, d_y );

    else if( alignment == 16)
        zgesellcmtmv2d_kernel_16<<< grid, block, Ms, magma_stream >>>
        ( m, n, blocksize, alignment, alpha,
            d_val, d_colind, d_rowptr, d_x, beta, d_y );

    else if( alignment == 32)
        zgesellcmtmv2d_kernel_32<<< grid, block, Ms, magma_stream >>>
        ( m, n, blocksize, alignment, alpha,
            d_val, d_colind, d_rowptr, d_x, beta, d_y );

    else
        printf("error: currently only alignment 8, 16, 32 supported.\n");

   return MAGMA_SUCCESS;
}

