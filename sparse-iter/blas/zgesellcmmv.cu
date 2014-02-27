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

// #define TEXTURE


// SELLCM SpMV kernel
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
__global__ void 
zgesellcmtmv2d_kernel_4( int num_rows, 
                     int num_cols,
                     int blocksize,
                     int T,
                     magmaDoubleComplex alpha, 
                     magmaDoubleComplex *d_val, 
                     magma_index_t *d_colind,
                     magma_index_t *d_rowptr,
                     const magmaDoubleComplex* __restrict__ d_x,
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

        int kk, i1, i2;
        magmaDoubleComplex x1, x2, v1, v2;
        d_colind += offset + ldx ;
        d_val += offset + ldx;
        for ( kk = 0; kk < max_-1 ; kk+=2 ){
            i1 = d_colind[ block*kk];
            i2 = d_colind[ block*kk + block];

            x1 = d_x[ i1 ];   
            x2 = d_x[ i2 ]; 

            v1 = d_val[ block*kk ];
            v2 = d_val[ block*kk + block];

            dot += v1 * x1;
            dot += v2 * x2;
        }
  
        if (kk<max_){
           x1 = d_x[ d_colind[ block*kk] ];            
           v1 = d_val[ block*kk ];

            dot += v1 * x1;
        }

        shared[ldx]  = dot;

        __syncthreads();
        if( idx < 2 ){
            shared[ldx]+=shared[ldx+blocksize*2];              
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
zgesellcmtmv2d_kernel_8( int num_rows, 
                     int num_cols,
                     int blocksize,
                     int T,
                     magmaDoubleComplex alpha, 
                     magmaDoubleComplex *d_val, 
                     magma_index_t *d_colind,
                     magma_index_t *d_rowptr,
                     const magmaDoubleComplex* __restrict__ d_x,
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

        int kk, i1, i2;
        magmaDoubleComplex x1, x2, v1, v2;
        d_colind += offset + ldx ;
        d_val += offset + ldx;
        for ( kk = 0; kk < max_-1 ; kk+=2 ){
            i1 = d_colind[ block*kk];
            i2 = d_colind[ block*kk + block];

            x1 = d_x[ i1 ];   
            x2 = d_x[ i2 ]; 

            v1 = d_val[ block*kk ];
            v2 = d_val[ block*kk + block];

            dot += v1 * x1;
            dot += v2 * x2;
        }
  
        if (kk<max_){
           x1 = d_x[ d_colind[ block*kk] ];            
           v1 = d_val[ block*kk ];

            dot += v1 * x1;
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
                     magma_index_t *d_colind,
                     magma_index_t *d_rowptr,
                     const magmaDoubleComplex* __restrict__ d_x,
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
                     magma_index_t *d_colind,
                     magma_index_t *d_rowptr,
                     const magmaDoubleComplex* __restrict__ d_x,
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



/************************* same but using texture mem *************************/

#if defined(PRECISION_d) && defined(TEXTURE)

__inline__ __device__ double 
read_from_tex( cudaTextureObject_t texdx, const int& i){
  int2 temp = tex1Dfetch<int2>( texdx, i ); 
  return __hiloint2double(temp.y,temp.x);
}

// SELLCM SpMV kernel
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
__global__ void 
zgesellcmtmv2d_kernel_4_tex( int num_rows, 
                     int num_cols,
                     int blocksize,
                     int T,
                     magmaDoubleComplex alpha, 
                     magmaDoubleComplex *d_val, 
                     magma_index_t *d_colind,
                     magma_index_t *d_rowptr,
                     cudaTextureObject_t texdx,
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

        int kk, i1, i2;
        magmaDoubleComplex x1, x2, v1, v2;
        d_colind += offset + ldx ;
        d_val += offset + ldx;
        for ( kk = 0; kk < max_-1 ; kk+=2 ){
            i1 = d_colind[ block*kk];
            i2 = d_colind[ block*kk + block];

            x1 = read_from_tex( texdx, i1 );
            x2 = read_from_tex( texdx, i2 );

            v1 = d_val[ block*kk ];
            v2 = d_val[ block*kk + block];

            dot += v1 * x1;
            dot += v2 * x2;
        }
  
        if (kk<max_){
           x1 = read_from_tex( texdx, d_colind[ block*kk] );
           v1 = d_val[ block*kk ];

            dot += v1 * x1;
        }

        shared[ldx]  = dot;

        __syncthreads();
        if( idx < 2 ){
            shared[ldx]+=shared[ldx+blocksize*2];              
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
zgesellcmtmv2d_kernel_8_tex( int num_rows, 
                     int num_cols,
                     int blocksize,
                     int T,
                     magmaDoubleComplex alpha, 
                     magmaDoubleComplex *d_val, 
                     magma_index_t *d_colind,
                     magma_index_t *d_rowptr,
                     cudaTextureObject_t texdx,
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

        int kk, i1, i2;
        magmaDoubleComplex x1, x2, v1, v2;
        d_colind += offset + ldx ;
        d_val += offset + ldx;
        for ( kk = 0; kk < max_-1 ; kk+=2 ){
            i1 = d_colind[ block*kk];
            i2 = d_colind[ block*kk + block];

            x1 = read_from_tex( texdx, i1 );
            x2 = read_from_tex( texdx, i2 );

            v1 = d_val[ block*kk ];
            v2 = d_val[ block*kk + block];

            dot += v1 * x1;
            dot += v2 * x2;
        }
  
        if (kk<max_){
           x1 = read_from_tex( texdx, d_colind[ block*kk] );
           v1 = d_val[ block*kk ];

            dot += v1 * x1;
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
zgesellcmtmv2d_kernel_16_tex( int num_rows, 
                     int num_cols,
                     int blocksize,
                     int T,
                     magmaDoubleComplex alpha, 
                     magmaDoubleComplex *d_val, 
                     magma_index_t *d_colind,
                     magma_index_t *d_rowptr,
                     cudaTextureObject_t texdx,
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

            dot += val * read_from_tex( texdx, col );
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
zgesellcmtmv2d_kernel_32_tex( int num_rows, 
                     int num_cols,
                     int blocksize,
                     int T,
                     magmaDoubleComplex alpha, 
                     magmaDoubleComplex *d_val, 
                     magma_index_t *d_colind,
                     magma_index_t *d_rowptr,
                     cudaTextureObject_t texdx,
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

            dot += val * read_from_tex( texdx, col );
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

#endif

/*********************     end of texture versions   **************************/

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
                    magma_index_t *d_colind,
                    magma_index_t *d_rowptr,
                    magmaDoubleComplex *d_x,
                    magmaDoubleComplex beta,
                    magmaDoubleComplex *d_y ){

    // using a 2D thread grid

    int num_threads = blocksize*alignment;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch < 200 && num_threads > 256 )
        printf("error: too much shared memory requested.\n");

    dim3 block( blocksize, alignment, 1);

    int dimgrid1 = sqrt(slices);
    int dimgrid2 = (slices + dimgrid1 -1 ) / dimgrid1;

    dim3 grid( dimgrid1, dimgrid2, 1);
    int Ms = num_threads * sizeof( magmaDoubleComplex );

    #if defined(PRECISION_d) && defined(TEXTURE)

        // Create channel.
        cudaChannelFormatDesc channel_desc;
        channel_desc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindSigned);

        // Create resource descriptor.
        struct cudaResourceDesc resDescdx;
        memset(&resDescdx, 0, sizeof(resDescdx));
        resDescdx.resType = cudaResourceTypeLinear;
        resDescdx.res.linear.devPtr = (void*)d_x;
        resDescdx.res.linear.desc = channel_desc;
        resDescdx.res.linear.sizeInBytes = m*sizeof(double);

        // Specify texture object parameters.
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.filterMode     = cudaFilterModePoint;
        texDesc.readMode       = cudaReadModeElementType;

        // Create texture object.
        cudaTextureObject_t texdx = 0;
        cudaCreateTextureObject(&texdx, &resDescdx, &texDesc, NULL);

        cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

        if( alignment == 4)
            zgesellcmtmv2d_kernel_4_tex<<< grid, block, Ms, magma_stream >>>
            ( m, n, blocksize, alignment, alpha,
                d_val, d_colind, d_rowptr, texdx, beta, d_y );

        else if( alignment == 8)
            zgesellcmtmv2d_kernel_8_tex<<< grid, block, Ms, magma_stream >>>
            ( m, n, blocksize, alignment, alpha,
                d_val, d_colind, d_rowptr, texdx, beta, d_y );

        else if( alignment == 16)
            zgesellcmtmv2d_kernel_16_tex<<< grid, block, Ms, magma_stream >>>
            ( m, n, blocksize, alignment, alpha,
                d_val, d_colind, d_rowptr, texdx, beta, d_y );

        else if( alignment == 32)
            zgesellcmtmv2d_kernel_32_tex<<< grid, block, Ms, magma_stream >>>
            ( m, n, blocksize, alignment, alpha,
                d_val, d_colind, d_rowptr, texdx, beta, d_y );

        else{
            printf("error: alignment %d not supported.\n", alignment);
            exit(-1);
        }

        cudaDestroyTextureObject(texdx);

    #else 
        if( alignment == 4)
            zgesellcmtmv2d_kernel_4<<< grid, block, Ms, magma_stream >>>
            ( m, n, blocksize, alignment, alpha,
                d_val, d_colind, d_rowptr, d_x, beta, d_y );

        else if( alignment == 8)
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

        else{
            printf("error: alignment %d not supported.\n", alignment);
            exit(-1);
        }
    #endif

   return MAGMA_SUCCESS;
}

