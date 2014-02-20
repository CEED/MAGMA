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

#define PRECISION_z






// SELLCM SpMV kernel 3D grid
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
__global__ void 
zmgesellcmtmv2d_kernel_8_3D( int num_rows, 
                     int num_cols,
                     int num_vecs,
                     int blocksize,
                     int T,
                     magmaDoubleComplex alpha, 
                     magmaDoubleComplex *d_val, 
                     magma_index_t *d_colind,
                     magma_index_t *d_rowptr,
                     magmaDoubleComplex *d_x,
                     magmaDoubleComplex beta, 
                     magmaDoubleComplex *d_y)
{
   // T threads assigned to each row
    int idx = threadIdx.y ;     // thread in row
    int idy = threadIdx.x;      // local row
    int idz = threadIdx.z;      // vector
    int ldx = idx * blocksize + idy;
    int ldz = idz * blocksize * T + idx * blocksize + idy;
    int bdx = blockIdx.y * gridDim.x + blockIdx.x; // global block index
    int row = bdx * blocksize + idy;  // global row index
    int vec = idz*num_rows;

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

            dot += val * d_x[ col+idz ];
        }
        shared[ldz]  = dot;

        __syncthreads();
        if( idx < 4 ){
            shared[ldz]+=shared[ldz+blocksize*4];               
            __syncthreads();
            if( idx < 2 ) shared[ldz]+=shared[ldz+blocksize*2];   
            __syncthreads();
            if( idx == 0 ) {
                d_y[row+idz] = 
                (shared[ldz]+shared[ldz+blocksize*1])*alpha 
                                            + beta*d_y [row+idz];
            }

        }

    }

}

// SELLCM SpMV kernel 3D grid
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
__global__ void 
zmgesellcmtmv2d_kernel_16_3D( int num_rows, 
                     int num_cols,
                     int num_vecs,
                     int blocksize,
                     int T,
                     magmaDoubleComplex alpha, 
                     magmaDoubleComplex *d_val, 
                     magma_index_t *d_colind,
                     magma_index_t *d_rowptr,
                     magmaDoubleComplex *d_x,
                     magmaDoubleComplex beta, 
                     magmaDoubleComplex *d_y)
{
   // T threads assigned to each row
    int idx = threadIdx.y ;     // thread in row
    int idy = threadIdx.x;      // local row
    int idz = threadIdx.z;      // vector
    int ldx = idx * blocksize + idy;
    int ldz = idz * blocksize * T + idx * blocksize + idy;
    int bdx = blockIdx.y * gridDim.x + blockIdx.x; // global block index
    int row = bdx * blocksize + idy;  // global row index
    int vec = idz*num_rows;

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

            dot += val * d_x[ col+idz ];
        }
        shared[ldz]  = dot;

        __syncthreads();
        if( idx < 8 ){
            shared[ldz]+=shared[ldz+blocksize*8];              
            __syncthreads();
            if( idx < 4 ) shared[ldz]+=shared[ldz+blocksize*4];   
            __syncthreads();
            if( idx < 2 ) shared[ldz]+=shared[ldz+blocksize*2];   
            __syncthreads();
            if( idx == 0 ) {
                d_y[row+idz] = 
                (shared[ldz]+shared[ldz+blocksize*1])*alpha 
                                            + beta*d_y [row+idz];
            }

        }

    }

}

// SELLCM SpMV kernel 3D grid
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
__global__ void 
zmgesellcmtmv2d_kernel_32_3D( int num_rows, 
                     int num_cols,
                     int num_vecs,
                     int blocksize,
                     int T,
                     magmaDoubleComplex alpha, 
                     magmaDoubleComplex *d_val, 
                     magma_index_t *d_colind,
                     magma_index_t *d_rowptr,
                     magmaDoubleComplex *d_x,
                     magmaDoubleComplex beta, 
                     magmaDoubleComplex *d_y)
{
   // T threads assigned to each row
    int idx = threadIdx.y ;     // thread in row
    int idy = threadIdx.x;      // local row
    int idz = threadIdx.z;      // vector
    int ldx = idx * blocksize + idy;
    int ldz = idz * blocksize * T + idx * blocksize + idy;
    int bdx = blockIdx.y * gridDim.x + blockIdx.x; // global block index
    int row = bdx * blocksize + idy;  // global row index
    int vec = idz*num_rows;

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

            dot += val * d_x[ col+idz ];
        }
        shared[ldz]  = dot;

        __syncthreads();
        if( idx < 16 ){
            shared[ldz]+=shared[ldz+blocksize*16];              
            __syncthreads();
            if( idx < 8 ) shared[ldz]+=shared[ldz+blocksize*8];  
            __syncthreads();
            if( idx < 4 ) shared[ldz]+=shared[ldz+blocksize*4];   
            __syncthreads();
            if( idx < 2 ) shared[ldz]+=shared[ldz+blocksize*2];   
            __syncthreads();
            if( idx == 0 ) {
                d_y[row+idz] = 
                (shared[ldz]+shared[ldz+blocksize*1])*alpha 
                                            + beta*d_y [row+idz];
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
    
    This routine computes Y = alpha *  A^t *  X + beta * Y on the GPU.
    Input format is SELLCM.
    
    Arguments
    =========

    magma_trans_t transA            transpose A?
    magma_int_t m                   number of rows in A
    magma_int_t n                   number of columns in A 
    magma_int_t num_vecs            number of columns in X and Y
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
magma_zmgesellcmmv( magma_trans_t transA,
                    magma_int_t m, magma_int_t n,
                    magma_int_t num_vecs,
                    magma_int_t blocksize,
                    magma_int_t slices,
                    magma_int_t alignment,
                    magmaDoubleComplex alpha,
                    magmaDoubleComplex *d_val,
                    magma_index_t *d_colind,
                    magma_index_t *d_rowptr,
                    magmaDoubleComplex *d_x,
                    magmaDoubleComplex beta,
                    magmaDoubleComplex *d_y ){

    // using a 2D thread grid

    int num_threads = num_vecs * blocksize*alignment;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch < 200 && num_threads > 256 )
        printf("error: too much shared memory requested.\n");
    else if (  num_threads > 1000 )
        printf("error: too much shared memory requested.\n");

    dim3 block( blocksize, alignment, num_vecs );

    int dimgrid1 = sqrt(slices);
    int dimgrid2 = (slices + dimgrid1 -1 ) / dimgrid1;

    dim3 grid( dimgrid1, dimgrid2, 1);
    int Ms =  num_threads * sizeof( magmaDoubleComplex );

    if( alignment == 8)
        zmgesellcmtmv2d_kernel_8_3D<<< grid, block, Ms, magma_stream >>>
        ( m, n, num_vecs, blocksize, alignment, alpha,
            d_val, d_colind, d_rowptr, d_x, beta, d_y );

    else if( alignment == 16)
        zmgesellcmtmv2d_kernel_16_3D<<< grid, block, Ms, magma_stream >>>
        ( m, n, num_vecs, blocksize, alignment, alpha,
            d_val, d_colind, d_rowptr, d_x, beta, d_y );

    else if( alignment == 32)
        zmgesellcmtmv2d_kernel_32_3D<<< grid, block, Ms, magma_stream >>>
        ( m, n, num_vecs, blocksize, alignment, alpha,
            d_val, d_colind, d_rowptr, d_x, beta, d_y );

    else{
        printf("error: alignment %d not supported.\n", alignment);
        exit(-1);
    }

   return MAGMA_SUCCESS;
}

