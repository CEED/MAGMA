/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> s d c

*/
#include "common_magma.h"

// 512 is maximum number of threads for CUDA capability 1.x
#define BLOCK_SIZE 512

#define BLOCK_SIZEx  32
#define BLOCK_SIZEy  16

// ----------------------------------------
// Does sum reduction of array x, leaving total in x[0].
// Contents of x are destroyed in the process.
// With k threads, can reduce array up to 2*k in size.
// Assumes number of threads <= 1024 (which is max number of threads up to CUDA capability 3.0)
// Having n as template parameter allows compiler to evaluate some conditions at compile time.
template< int n >
__device__ void sum_reduce( /*int n,*/ int i, cuDoubleComplex* x )
{
    __syncthreads();
    if ( n > 1024 ) { if ( i < 1024 && i + 1024 < n ) { x[i] += x[i+1024]; }  __syncthreads(); }
    if ( n >  512 ) { if ( i <  512 && i +  512 < n ) { x[i] += x[i+ 512]; }  __syncthreads(); }
    if ( n >  256 ) { if ( i <  256 && i +  256 < n ) { x[i] += x[i+ 256]; }  __syncthreads(); }
    if ( n >  128 ) { if ( i <  128 && i +  128 < n ) { x[i] += x[i+ 128]; }  __syncthreads(); }
    if ( n >   64 ) { if ( i <   64 && i +   64 < n ) { x[i] += x[i+  64]; }  __syncthreads(); }
    if ( n >   32 ) { if ( i <   32 && i +   32 < n ) { x[i] += x[i+  32]; }  __syncthreads(); }
    // probably don't need __syncthreads for < 16 threads
    // because of implicit warp level synchronization.
    if ( n >   16 ) { if ( i <   16 && i +   16 < n ) { x[i] += x[i+  16]; }  __syncthreads(); }
    if ( n >    8 ) { if ( i <    8 && i +    8 < n ) { x[i] += x[i+   8]; }  __syncthreads(); }
    if ( n >    4 ) { if ( i <    4 && i +    4 < n ) { x[i] += x[i+   4]; }  __syncthreads(); }
    if ( n >    2 ) { if ( i <    2 && i +    2 < n ) { x[i] += x[i+   2]; }  __syncthreads(); }
    if ( n >    1 ) { if ( i <    1 && i +    1 < n ) { x[i] += x[i+   1]; }  __syncthreads(); }
}
// end sum_reduce

//==============================================================================

__global__
void magma_zlarfx_kernel( int m, cuDoubleComplex *v, cuDoubleComplex *tau,
                         cuDoubleComplex *c, int ldc, double *xnorm,
                         cuDoubleComplex *T, int it )
{
    if ( !MAGMA_Z_EQUAL(*tau, MAGMA_Z_ZERO) ) {
        const int i = threadIdx.x;
        cuDoubleComplex *dc = c + (blockIdx.x-it-1) * ldc;

        __shared__ cuDoubleComplex sum[ BLOCK_SIZE ];
        cuDoubleComplex lsum;

        /*  w := v' * C  */
        lsum = MAGMA_Z_ZERO;
        for( int j = i; j < m; j += BLOCK_SIZE ){
            if (j==0){
               lsum += MAGMA_Z_MUL( MAGMA_Z_ONE, dc[j] );
               v[j] = MAGMA_Z_ONE;
            }
            else
               lsum += MAGMA_Z_MUL( MAGMA_Z_CNJG( v[j] ), dc[j] );
        }
        sum[i] = lsum;
        sum_reduce< BLOCK_SIZE >( i, sum );

        /*  C := C - v * w  */
        __syncthreads();
        cuDoubleComplex z__1 = - MAGMA_Z_CNJG(*tau) * sum[0];
        if (blockIdx.x>it){
           for( int j = m-i-1; j>=0 ; j -= BLOCK_SIZE )
                 dc[j] += z__1 * v[j];
           __syncthreads();

           /* Adjust the rest of the column norms */
           if (i==0){
             double temp = MAGMA_Z_ABS( dc[0] ) / xnorm[blockIdx.x-it-1];
             temp = (temp + 1.) * (1. - temp);
             xnorm[blockIdx.x-it-1] = xnorm[blockIdx.x-it-1] * sqrt(temp); 
           }
        }
        else
        {
           if (blockIdx.x==it)
              *(T+it) = *tau;
           else
              *(T+blockIdx.x) = MAGMA_Z_CNJG(z__1);
        }
    }
}

//==============================================================================
__global__
void magma_zlarft2_kernel(cuDoubleComplex *T, int ldt, cuDoubleComplex *t)
{
   const int i = threadIdx.x;
   T += i;

   __shared__ cuDoubleComplex tlocal[ BLOCK_SIZE ];
   cuDoubleComplex res = MAGMA_Z_MAKE(0., 0.);

   tlocal[i] = t[i];
   __syncthreads();

   #pragma unroll
   for(int j=0; j<blockDim.x; j++)
      res +=  T[j*ldt]*tlocal[j];

   t[i] = res;
}

//==============================================================================

/*
    Apply a complex elementary reflector H to a complex M-by-N
    matrix C from the left. H is represented in the form
          H = I - tau * v * v'
    where tau is a complex scalar and v is a complex vector.
    If tau = 0, then H is taken to be the unit matrix.

    To apply H' (the conjugate transpose of H), supply conjg(tau) 
    instead tau.

    The norms of v(:, 1:n) are given as input in xnorm(1:n). On exit, the norms
    are adjusted to hold the norms of v(2:m,2:n). This is a difference with the 
    LAPACK's zlarf routine. 
 */
extern "C" void
magma_zlarfx_gpu(int m, int n, cuDoubleComplex *v, cuDoubleComplex *tau,
                cuDoubleComplex *c, int ldc, double *xnorm, 
                cuDoubleComplex *T, int i )
{
    int N = n + i + 1;

    dim3  blocks( N);
    dim3 threads( BLOCK_SIZE );

    magma_zlarfx_kernel<<< blocks, threads >>>( m, v, tau, c, ldc, xnorm, T+i*N, i);

    if (i > 0){
       dim3  blocks2( 1 );
       dim3 threads2( i );
       magma_zlarft2_kernel<<< blocks2, threads2 >>>( T, N, T+i*N);
    }
}

//==============================================================================
