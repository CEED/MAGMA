/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Azzam Haidar

*/
#include "common_magma.h"
#include "magma_templates.h"

// 512 is maximum number of threads for CUDA capability 1.x
#define BLOCK_SIZE 512

#define BLOCK_SIZEx  32
#define BLOCK_SIZEy  16


//==============================================================================
//==============================================================================

__global__
void magma_zlarf_kernel( int m, magmaDoubleComplex *v, magmaDoubleComplex *tau,
                         magmaDoubleComplex *c, int ldc )
{
    if ( !MAGMA_Z_EQUAL(*tau, MAGMA_Z_ZERO) ) {
        const int tx = threadIdx.x;
        magmaDoubleComplex *dc = c + blockIdx.x * ldc;

        __shared__ magmaDoubleComplex sum[ BLOCK_SIZE ];
        magmaDoubleComplex lsum;

        /* perform  w := v' * C  */
        if (tx==0)
            lsum = dc[0]; //since V[0] should be one
        else
            lsum = MAGMA_Z_ZERO;
        for( int j = tx+1; j < m; j += BLOCK_SIZE ){
            lsum += MAGMA_Z_MUL( MAGMA_Z_CNJG( v[j] ), dc[j] );
        }
        sum[tx] = lsum;
        magma_sum_reduce< BLOCK_SIZE >( tx, sum );

        /*  C := C - v * w  */
        __syncthreads();
        magmaDoubleComplex z__1 = - MAGMA_Z_CNJG(*tau) * sum[0];
        for( int j = m-tx-1; j>0 ; j -= BLOCK_SIZE )
             dc[j] += z__1 * v[j];

        if(tx==0) dc[0] += z__1;
    }
}

//==============================================================================
//==============================================================================

__global__
void magma_zlarf_smkernel( int m, int n, magmaDoubleComplex *v, magmaDoubleComplex *tau,
                           magmaDoubleComplex *c, int ldc )
{
    if ( ! MAGMA_Z_EQUAL(*tau, MAGMA_Z_ZERO) ) {
        const int i = threadIdx.x, col= threadIdx.y;

        for( int k = col; k < n; k += BLOCK_SIZEy ) {
            magmaDoubleComplex *dc = c + k * ldc;
    
            __shared__ magmaDoubleComplex sum[ BLOCK_SIZEx ][ BLOCK_SIZEy + 1];
            magmaDoubleComplex lsum;
    
            /*  w := v' * C  */
            lsum = MAGMA_Z_ZERO;
            for( int j = i; j < m; j += BLOCK_SIZEx ){
                if (j==0)
                   lsum += MAGMA_Z_MUL( MAGMA_Z_ONE, dc[j] );
                else
                   lsum += MAGMA_Z_MUL( MAGMA_Z_CNJG( v[j] ), dc[j] );
            }
            sum[i][col] = lsum;
            magma_sum_reduce_2d< BLOCK_SIZEx, BLOCK_SIZEy+1 >( i, col, sum );
    
            /*  C := C - v * w  */
            __syncthreads();
            magmaDoubleComplex z__1 = - MAGMA_Z_CNJG(*tau) * sum[0][col];
            for( int j = m-i-1; j>=0 ; j -= BLOCK_SIZEx ) {
                 if (j==0)
                    dc[j] += z__1;
                 else
                    dc[j] += z__1 * v[j];
            }
        }
    }
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

    This routine uses only one SM (block).
 */
extern "C" void
magma_zlarf_sm(int m, int n, magmaDoubleComplex *v, magmaDoubleComplex *tau,
               magmaDoubleComplex *c, int ldc)
{
    dim3  blocks( 1 );
    dim3 threads( BLOCK_SIZEx, BLOCK_SIZEy );

    magma_zlarf_smkernel<<< blocks, threads, 0, magma_stream >>>( m, n, v, tau, c, ldc );
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

 */

extern "C" magma_int_t
magma_zlarf_gpu(
    magma_int_t m,  magma_int_t n,
    magmaDoubleComplex *v, magmaDoubleComplex *tau,
    magmaDoubleComplex *c,  magma_int_t ldc)
{
    dim3 grid( n, 1, 1 );
    dim3 threads( BLOCK_SIZE );
    if ( n>0 ){
        magma_zlarf_kernel<<< grid, threads, 0, magma_stream >>>( m, v, tau, c, ldc);
    }

    // The computation can be done on 1 SM with the following routine.
    // magma_zlarf_sm(m, n, v, tau, c, ldc);

    return MAGMA_SUCCESS;
}

//==============================================================================
