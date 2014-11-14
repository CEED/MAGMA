/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       
       @author Azzam Haidar
       @author Tingxing Dong
*/
#include "common_magma.h"
#include "batched_kernel_param.h"

#define BLK_SIZE 256
// SWP_WIDTH is number of threads in a block
// 64 and 256 are better on Kepler; 
extern __shared__ magmaDoubleComplex shared_data[];


//=================================================================================================
static __device__ 
void zlaswp_rowparallel_devfunc(  
                              int n, int width, int height,
                              magmaDoubleComplex *dA, int lda, 
                              magmaDoubleComplex *dout, int ldo,
                              magma_int_t* pivinfo)
{

    //int height = k2- k1;
    //int height = blockDim.x;
    unsigned int tid = threadIdx.x;
    dA   += SWP_WIDTH * blockIdx.x * lda;
    dout += SWP_WIDTH * blockIdx.x * ldo;
    magmaDoubleComplex *sdata = shared_data;

    if(blockIdx.x == gridDim.x -1)
    {
       width = n - blockIdx.x * SWP_WIDTH;
    }

    if(tid < height)
    {
        int mynewroworig = pivinfo[tid]-1; //-1 to get the index in C
        int itsreplacement = pivinfo[mynewroworig] -1 ; //-1 to get the index in C
        #pragma unroll
        for(int i=0; i<width; i++)
        {
          sdata[ tid + i * height ]    = dA[ mynewroworig + i * lda ];
          dA[ mynewroworig + i * lda ] = dA[ itsreplacement + i * lda ];
        }
    }
    __syncthreads();

    if(tid < height)
    {
        // copy back the upper swapped portion of A to dout 
        #pragma unroll
        for(int i=0; i<width; i++)
        {
           dout[tid + i * ldo] = sdata[tid + i * height];
        }
    }
}

//=================================================================================================
// parallel swap the swaped dA(1:nb,i:n) is stored in dout 
//=================================================================================================
__global__ 
void zlaswp_rowparallel_kernel( 
                                int n, int width, int height,
                                magmaDoubleComplex *dinput, int ldi, 
                                magmaDoubleComplex *doutput, int ldo,
                                magma_int_t*  pivinfo)
{

    zlaswp_rowparallel_devfunc(n, width, height, dinput, ldi, doutput, ldo, pivinfo);

}
//=================================================================================================

__global__ 
void zlaswp_rowparallel_kernel_batched(
                                int n, int width, int height,
                                magmaDoubleComplex **input_array, int ldi, 
                                magmaDoubleComplex **output_array, int ldo,
                                magma_int_t** pivinfo_array)
{
    int batchid = blockIdx.z;
    zlaswp_rowparallel_devfunc(n, width, height, input_array[batchid], ldi, output_array[batchid], ldo, pivinfo_array[batchid]);
}


//=================================================================================================

//=================================================================================================
extern "C" void
magma_zlaswp_rowparallel_batched_q( magma_int_t n, 
                       magmaDoubleComplex** input_array, magma_int_t ldi,
                       magmaDoubleComplex** output_array, magma_int_t ldo,
                       magma_int_t k1, magma_int_t k2,
                       magma_int_t **pivinfo_array, 
                       magma_queue_t stream, magma_int_t batchCount )
{

    if(n == 0 ) return ;
    int height = k2-k1;
    if( height  > 1024) 
    {
       printf(" n=%d > 1024, not supported \n", n);

    }

    int blocks =  (n-1)/ SWP_WIDTH + 1;
    dim3  grid(blocks, 1, batchCount);

    if( n < SWP_WIDTH)
    {
        zlaswp_rowparallel_kernel_batched<<<grid, height, sizeof(magmaDoubleComplex) * height * n, stream >>>
                                           ( n, n, height, input_array, ldi, output_array, ldo, pivinfo_array ); 
    }
    else
    {
        zlaswp_rowparallel_kernel_batched<<< grid, height, sizeof(magmaDoubleComplex) * height * SWP_WIDTH , stream >>>
                                            (n, SWP_WIDTH, height, input_array, ldi, output_array, ldo, pivinfo_array ); 
 
    }
}

//=================================================================================================


extern "C" void
magma_zlaswp_rowparallel_batched( magma_int_t n, magmaDoubleComplex** input_array, magma_int_t ldi,
                   magmaDoubleComplex** output_array, magma_int_t ldo,
                   magma_int_t k1, magma_int_t k2,
                   magma_int_t **pivinfo_array, 
                   magma_int_t batchCount )
{
    magma_zlaswp_rowparallel_batched_q(n, input_array, ldi, output_array, ldo, k1, k2, pivinfo_array, magma_stream, batchCount);
}

//=================================================================================================




//=================================================================================================
extern "C" void
magma_zlaswp_rowparallel_q( magma_int_t n, 
                       magmaDoubleComplex* input, magma_int_t ldi,
                       magmaDoubleComplex* output, magma_int_t ldo,
                       magma_int_t k1, magma_int_t k2,
                       magma_int_t *pivinfo, 
                       magma_queue_t stream)
{
    if(n == 0 ) return ;
    int height = k2-k1;
    if( height  > MAX_NTHREADS) 
    {
       printf(" height=%d > %d, magma_zlaswp_rowparallel_q not supported \n", n,MAX_NTHREADS);

    }

    int blocks =  (n-1)/ SWP_WIDTH + 1;
    dim3  grid(blocks, 1, 1);

    if( n < SWP_WIDTH)
    {
        zlaswp_rowparallel_kernel<<<grid, height, sizeof(magmaDoubleComplex) * height * n, stream >>>
                                   ( n, n, height, input, ldi, output, ldo, pivinfo ); 
    }
    else
    {
        zlaswp_rowparallel_kernel<<< grid, height, sizeof(magmaDoubleComplex) * height * SWP_WIDTH , stream >>>
                                    (n, SWP_WIDTH, height, input, ldi, output, ldo, pivinfo ); 
    }
}


//=================================================================================================

extern "C" void
magma_zlaswp_rowparallel( magma_int_t n, magmaDoubleComplex* input, magma_int_t ldi,
                   magmaDoubleComplex* output, magma_int_t ldo,
                   magma_int_t k1, magma_int_t k2,
                   magma_int_t *pivinfo)
{
    magma_zlaswp_rowparallel_q(n, input, ldi, output, ldo, k1, k2, pivinfo, magma_stream);
}

//=================================================================================================





//=================================================================================================
//  serial swap that does swapping one row by one row
//=================================================================================================
__global__ void zlaswp_rowserial_kernel_batched( int n, magmaDoubleComplex **dA_array, int lda, int k1, int k2, magma_int_t** ipiv_array )
{
    magmaDoubleComplex* dA = dA_array[blockIdx.z];
    magma_int_t *d_ipiv = ipiv_array[blockIdx.z];
    
    unsigned int tid = threadIdx.x + blockDim.x*blockIdx.x;
    
    k1--;
    k2--;

    if( tid < n) {

        magmaDoubleComplex A1;

        for( int i1 = k1; i1 < k2; i1++ ) 
        {
            int i2 = d_ipiv[i1] - 1;  // Fortran index, switch i1 and i2
            if( i2 != i1)
            {
                A1 = dA[i1 + tid * lda];
                dA[i1 + tid * lda] = dA[i2 + tid * lda];
                dA[i2 + tid * lda] = A1;
            }
        }
    }
}

//=================================================================================================
//  serial swap that does swapping one row by one row, similar to LAPACK
//  K1, K2 are in Fortran indexing  
//=================================================================================================
extern "C" void
magma_zlaswp_rowserial_batched_q(magma_int_t n, magmaDoubleComplex** dA_array, magma_int_t lda,
                   magma_int_t k1, magma_int_t k2,
                   magma_int_t **ipiv_array, 
                   magma_queue_t stream, magma_int_t batchCount)
{

    if(n == 0 ) return ;

    int blocks =  (n-1)/ BLK_SIZE + 1;
    dim3  grid(blocks, 1, batchCount);

    zlaswp_rowserial_kernel_batched<<< grid, max(BLK_SIZE, n), 0, stream >>>(
        n, dA_array, lda, k1, k2, ipiv_array); 

}

extern "C" void
magma_zlaswp_rowserial_batched(magma_int_t n, magmaDoubleComplex** dA_array, magma_int_t lda,
                   magma_int_t k1, magma_int_t k2,
                   magma_int_t **ipiv_array, 
                   magma_int_t batchCount)
{
    magma_zlaswp_rowserial_batched_q(n, dA_array, lda, k1, k2, ipiv_array,  magma_stream, batchCount);
}




//=================================================================================================
//  serial swap that does swapping one column by one column
//=================================================================================================
__global__ void zlaswp_columnserial_kernel_batched( int n, magmaDoubleComplex **dA_array, int lda, int k1, int k2, magma_int_t** ipiv_array )
{
    magmaDoubleComplex* dA = dA_array[blockIdx.z];
    magma_int_t *d_ipiv = ipiv_array[blockIdx.z];
    
    unsigned int tid = threadIdx.x + blockDim.x*blockIdx.x;
    k1--;
    k2--;
    if( k1 < 0 || k2 < 0 ) return;


    if( tid < n) {
        magmaDoubleComplex A1;
        if(k1 <= k2)
        {
            for( int i1 = k1; i1 <= k2; i1++ ) 
            {
                int i2 = d_ipiv[i1] - 1;  // Fortran index, switch i1 and i2
                if( i2 != i1)
                {
                    A1 = dA[i1 * lda + tid];
                    dA[i1 * lda + tid] = dA[i2 * lda + tid];
                    dA[i2 * lda + tid] = A1;
                }
            }
        }else
        {
            for( int i1 = k1; i1 >= k2; i1-- ) 
            {
                int i2 = d_ipiv[i1] - 1;  // Fortran index, switch i1 and i2
                if( i2 != i1)
                {
                    A1 = dA[i1 * lda + tid];
                    dA[i1 * lda + tid] = dA[i2 * lda + tid];
                    dA[i2 * lda + tid] = A1;
                }
            }
        }
    }
}

//=================================================================================================
//  serial swap that does swapping one column by one column
//  K1, K2 are in Fortran indexing  
//=================================================================================================
extern "C" void
magma_zlaswp_columnserial_batched_q(magma_int_t n, magmaDoubleComplex** dA_array, magma_int_t lda,
                   magma_int_t k1, magma_int_t k2,
                   magma_int_t **ipiv_array, 
                   magma_queue_t stream, magma_int_t batchCount)
{

    if(n == 0 ) return ;

    int blocks =  (n-1)/ BLK_SIZE + 1;
    dim3  grid(blocks, 1, batchCount);

    zlaswp_columnserial_kernel_batched<<< grid, min(BLK_SIZE, n), 0, stream >>>(
        n, dA_array, lda, k1, k2, ipiv_array); 

}

extern "C" void
magma_zlaswp_columnserial_batched(magma_int_t n, magmaDoubleComplex** dA_array, magma_int_t lda,
                   magma_int_t k1, magma_int_t k2,
                   magma_int_t **ipiv_array, 
                   magma_int_t batchCount)
{
    magma_zlaswp_columnserial_batched_q(n, dA_array, lda, k1, k2, ipiv_array,  magma_stream, batchCount);
}

