/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
       @author Hartwig Anzt

*/
#include "common_magma.h"

#define BLOCK_SIZE 256

#define PRECISION_z


// initialize arrays with zero



// dot product for multiple vectors
__global__ void
magma_zmzdotc_one_kernel_1( 
    int Gs,
    int n, 
    magmaDoubleComplex * v0,
    magmaDoubleComplex * w0,
    magmaDoubleComplex * vtmp)
{
    extern __shared__ magmaDoubleComplex temp[]; 
    int Idx = threadIdx.x;   
    int i   = blockIdx.x * blockDim.x + Idx;

    // 1 vectors v(i)/w(i)
    
    temp[ Idx ]                 = ( i < n ) ?
                v0[ i ] * w0[ i ] : MAGMA_Z_ZERO;
    
    __syncthreads();
    if ( Idx < 128 ){
            temp[ Idx ] += temp[ Idx + 128 ];
    }
    __syncthreads();
    if ( Idx < 64 ){
            temp[ Idx ] += temp[ Idx + 64 ];
    }
    __syncthreads();
    #if defined(PRECISION_z) || defined(PRECISION_c)
        if( Idx < 32 ){
                temp[ Idx ] += temp[ Idx + 32 ];
                __syncthreads();
                temp[ Idx ] += temp[ Idx + 16 ];
                __syncthreads();
                temp[ Idx ] += temp[ Idx + 8 ];
                __syncthreads();
                temp[ Idx ] += temp[ Idx + 4 ];
                __syncthreads();
                temp[ Idx ] += temp[ Idx + 2 ];
                __syncthreads();
                temp[ Idx ] += temp[ Idx + 1 ];
                __syncthreads();
        }
    #endif
    #if defined(PRECISION_d)
        if( Idx < 32 ){
            volatile double *temp2 = temp;
                temp2[ Idx ] += temp2[ Idx + 32 ];
                temp2[ Idx ] += temp2[ Idx + 16 ];
                temp2[ Idx ] += temp2[ Idx + 8 ];
                temp2[ Idx ] += temp2[ Idx + 4 ];
                temp2[ Idx ] += temp2[ Idx + 2 ];
                temp2[ Idx ] += temp2[ Idx + 1 ];
        }
    #endif
    #if defined(PRECISION_s)
        if( Idx < 32 ){
            volatile float *temp2 = temp;
                temp2[ Idx ] += temp2[ Idx + 32 ];
                temp2[ Idx ] += temp2[ Idx + 16 ];
                temp2[ Idx ] += temp2[ Idx + 8 ];
                temp2[ Idx ] += temp2[ Idx + 4 ];
                temp2[ Idx ] += temp2[ Idx + 2 ];
                temp2[ Idx ] += temp2[ Idx + 1 ];
        }
    #endif
    if ( Idx == 0 ){
            vtmp[ blockIdx.x ] = temp[ 0 ];
    }
}



// block reduction for 1 vectors
__global__ void
magma_zmzdotc_one_kernel_2( 
    int Gs,
    int n, 
    magmaDoubleComplex * vtmp,
    magmaDoubleComplex * vtmp2 )
{
    extern __shared__ magmaDoubleComplex temp[];    
    int Idx = threadIdx.x;
    int blockSize = 128;
    int gridSize = blockSize  * 2 * gridDim.x; 

        int i = blockIdx.x * ( blockSize * 2 ) + Idx;   
        temp[Idx] = MAGMA_Z_ZERO;
        while (i < Gs ) {
            temp[ Idx  ] += vtmp[ i ]; 
            temp[ Idx  ] += 
                ( i + (blockSize) < Gs ) ? vtmp[ i + (blockSize) ] 
                                                : MAGMA_Z_ZERO;
            i += gridSize;
        }
    __syncthreads();
    if ( Idx < 64 ){
            temp[ Idx ] += temp[ Idx + 64 ];
    }
    __syncthreads();
    #if defined(PRECISION_z) || defined(PRECISION_c)
        if( Idx < 32 ){
                temp[ Idx ] += temp[ Idx + 32 ];
                __syncthreads();
                temp[ Idx ] += temp[ Idx + 16 ];
                __syncthreads();
                temp[ Idx ] += temp[ Idx + 8 ];
                __syncthreads();
                temp[ Idx ] += temp[ Idx + 4 ];
                __syncthreads();
                temp[ Idx ] += temp[ Idx + 2 ];
                __syncthreads();
                temp[ Idx ] += temp[ Idx + 1 ];
                __syncthreads();
        }
    #endif
    #if defined(PRECISION_d)
        if( Idx < 32 ){
            volatile double *temp2 = temp;
                temp2[ Idx ] += temp2[ Idx + 32 ];
                temp2[ Idx ] += temp2[ Idx + 16 ];
                temp2[ Idx ] += temp2[ Idx + 8 ];
                temp2[ Idx ] += temp2[ Idx + 4 ];
                temp2[ Idx ] += temp2[ Idx + 2 ];
                temp2[ Idx ] += temp2[ Idx + 1 ];
        }
    #endif
    #if defined(PRECISION_s)
        if( Idx < 32 ){
            volatile float *temp2 = temp;
                temp2[ Idx ] += temp2[ Idx + 32 ];
                temp2[ Idx ] += temp2[ Idx + 16 ];
                temp2[ Idx ] += temp2[ Idx + 8 ];
                temp2[ Idx ] += temp2[ Idx + 4 ];
                temp2[ Idx ] += temp2[ Idx + 2 ];
                temp2[ Idx ] += temp2[ Idx + 1 ];
        }
    #endif
    if ( Idx == 0 ){
            vtmp2[ blockIdx.x ] = temp[ 0 ];
    }
}

/**
    Purpose
    -------

    Computes the scalar product of a set of 1 vectors such that

    skp[0] = [ <v_0,w_0> ]

    Returns the vector skp.
    In case there are less dot products required, an easy workaround is
    given by doubling input.

    Arguments
    ---------

    @param[in]
    n           int
                length of v_i and w_i

    @param[in]                             
    v0          magmaDoubleComplex_ptr     
                input vector               

    @param[in]                                         
    w0          magmaDoubleComplex_ptr                 
                input vector                           

    @param[in]
    d1          magmaDoubleComplex_ptr 
                workspace

    @param[in]
    d2          magmaDoubleComplex_ptr 
                workspace

    @param[out]
    skp         magmaDoubleComplex_ptr 
                vector[4] of scalar products [<v_i, w_i>]
                This vector is located on the host

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zblas
    ********************************************************************/

extern "C" magma_int_t
magma_zmzdotc_one(
    int n,  
    magmaDoubleComplex_ptr v0, 
    magmaDoubleComplex_ptr w0,
    magmaDoubleComplex_ptr d1,
    magmaDoubleComplex_ptr d2,
    magmaDoubleComplex_ptr skp,
    magma_queue_t queue )
{
    // set queue for old dense routines
    magma_queue_t orig_queue;
    magmablasGetKernelStream( &orig_queue );

    int local_block_size=256;
    dim3 Bs( local_block_size );
    dim3 Gs( magma_ceildiv( n, local_block_size ) );
    dim3 Gs_next;
    int Ms = (local_block_size) * sizeof( magmaDoubleComplex ); // 1 skp 
    magmaDoubleComplex_ptr aux1 = d1, aux2 = d2;
    int b = 1;        


    magma_zmzdotc_one_kernel_1<<<Gs, Bs, Ms, queue>>>
            ( Gs.x, n, v0, w0, d1 );
   
    while( Gs.x > 1 ) {
        Gs_next.x = ( Gs.x+Bs.x-1 )/ Bs.x;
        if ( Gs_next.x == 1 ) Gs_next.x = 2;
        magma_zmzdotc_one_kernel_2<<< Gs_next.x/2, Bs.x/2, Ms/2, queue >>> 
                    ( Gs.x, n, aux1, aux2 );
        Gs_next.x = Gs_next.x /2;
        Gs.x = Gs_next.x;
        b = 1 - b;
        if ( b ) { aux1 = d1; aux2 = d2; }
        else   { aux2 = d1; aux1 = d2; }
    }
    
        // copy vectors to host
    magma_zgetvector( 1 , aux1, 1, skp, 1 );
    

   magmablasSetKernelStream( orig_queue );
   return MAGMA_SUCCESS;
}


