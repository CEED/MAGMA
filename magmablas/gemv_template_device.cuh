/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Tingxing Dong
       @author Azzam Haidar

*/


#ifndef MAGMABLAS_GEMV_TEMPLATE_H
#define MAGMABLAS_GEMV_TEMPLATE_H

#include "gemm_template_device_defs.cuh"// use make_FloatingPoint

// op<trans>( x ) returns x or conj(x).
template< const int conjugate, typename T >
__host__ __device__ static inline
T op( T& x )
{
    if (conjugate == 1) {
        return conj(x);
    } else {
        return x;
    }
}


template<class T, const int BLK_X, const int BLK_Y, const int TILE_SIZE> 
static __device__ void
gemvn_template_device(
    int m, int n, T alpha,
    const T * __restrict__ A, int lda,
    const T * __restrict__ x, int incx, T beta,
    T       *y, int incy)
{

    if(m <=0 || n <= 0) return;

    int num_threads = blockDim.x * blockDim.y * blockDim.z;
    
    if(BLK_X * BLK_Y != num_threads) return;// need to launch exactly the same number of threads as template parameters indicate

    int thread_id = threadIdx.x + threadIdx.y * blockDim.x;

    // threads are all configurated locally
    int tx = thread_id % BLK_X;
    int ty = thread_id / BLK_X;

    int ind = blockIdx.x*TILE_SIZE + tx;

    __shared__ T sdata[BLK_X * BLK_Y];


    int st = blockIdx.x * TILE_SIZE;

    int ed = min(st+TILE_SIZE, magma_roundup(m,BLK_X));
    
    int iters = (ed-st)/BLK_X;

    for(int i=0; i<iters; i++)
    {   
        if(ind <m ) A += ind;

        T res = make_FloatingPoint(0.0, 0.0);
        
        if(ind < m )
        {
            for(int col=ty; col<n; col+=BLK_Y)
            {       
                res += A[col*lda] * x[col*incx];
            }
        }

        if(BLK_X >= num_threads) // indicated 1D threads configuration. Shared memory is not needed, reduction is done naturally
        {

            if(ty == 0 && ind < m)
            {
                y[ind*incy] = alpha*res + beta*y[ind*incy];
            }
        }
        else 
        {
            sdata[ty + tx * BLK_Y] = res;

            __syncthreads(); 

            if( BLK_Y > 16)
            { 
                magma_sum_reduce< BLK_Y >( ty, sdata + tx * BLK_Y);
            }
            else
            {
                if(ty == 0 && ind < m)
                {
                    for(int i=1; i< BLK_Y; i++)
                    {
                        sdata[tx * BLK_Y] += sdata[i + tx * BLK_Y]; 
                    }   
        
                }
            }

            if(ty == 0 && ind < m)
            {
                y[ind*incy] = alpha*sdata[tx * BLK_Y] + beta*y[ind*incy];
            }

            __syncthreads();

        }

 
        if( ind < m) A -= ind;

        ind += BLK_X;

    }
}





//////////////////////////////////////////////////////////////////////////////////////////


template<class T, const int BLK_X, const int BLK_Y, const int TILE_SIZE,  int CONJA> 
static __device__ void
gemvc_template_device(
    int m, int n, T alpha,
    const T * __restrict__ A, int lda,
    const T * __restrict__ x, int incx, T beta,
    T       *y, int incy)
{

    if(m <=0 || n <= 0) return;

    int num_threads = blockDim.x * blockDim.y * blockDim.z;
    
    if(BLK_X * BLK_Y != num_threads) return;// need to launch exactly the same number of threads as template parameters indicate

    int thread_id = threadIdx.x + threadIdx.y * blockDim.x;

    // threads are all configurated locally
    int tx = thread_id % BLK_X;
    int ty = thread_id / BLK_X;

    __shared__ T sdata[BLK_X * BLK_Y];

    T res;
    int mfull = (m / BLK_X) * BLK_X;
     
    int start = blockIdx.y * TILE_SIZE + ty;
    int iters;

    #define usefixedcondition 0

    #ifdef usefixedcondition
        /*fixed condition*/
        iters = TILE_SIZE / BLK_Y;
    #else
        /* flexible condition based on global n (has drops when size is roughly bigger than TILE_SIZE)*/
        //int iters = magma_ceildiv(min(n,TILE_SIZE), BLK_Y);

        /* flexible condition based on my local nloc=ed-st*/
        int st = blockIdx.y * TILE_SIZE;
        //int ed = magma_ceildiv( min(n, st + TILE_SIZE), BLK_Y ) * BLK_Y; 
        int ed = min(st+TILE_SIZE, magma_roundup(n,BLK_Y));
        iters = (ed-st)/BLK_Y;
    #endif


    if(tx < m) A += tx;
    
    for(int i = 0; i< iters; i++)// at 2Gflops/ overhead
    //for(int col=start; col < (blockIdx.y+1)*TILE_SIZE; col+= BLK_Y)// at least 3Gflop/s overhead
    {

        int col = start + i * BLK_Y;

        if( col < n) A += col*lda;

        res = make_FloatingPoint(0.0, 0.0);

        // partial sums
        for(int i=0; i < mfull; i += BLK_X) {
            res += op<CONJA>(A[i]) * x[(tx + i)*incx];
        }
        if ( tx + mfull < m ) {
            res += op<CONJA>(A[mfull]) * x[(tx + mfull)*incx];
        }

        sdata[tx + ty * BLK_X] = res;

        // tree reduction of partial sums,
        // from BLK_X sums to ... 128 to 64 to 32 ... to 1 sum in sdata[0]
        if( BLK_X > 16)
        { 
            magma_sum_reduce< BLK_X >( tx, sdata + ty * BLK_X);
        }
        else
        {
            __syncthreads();

            if(tx == 0 && col < n)
            {
                for(int i=1; i<m && i < BLK_X; i++)
                {
                    sdata[0 + ty * BLK_X] += sdata[i + ty * BLK_X];
                }
            }
            __syncthreads();

        }

        if ( tx == 0 && col < n) {
            y[col*incy] = alpha*sdata[0 + ty * BLK_X] + beta*y[col*incy];
        }

        __syncthreads();

        if( col < n)  A -= col * lda;

    }

}


#endif /* MAGMABLAS_GEMV_TEMPLATE_H  */
