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
__global__ void
magma_zmzdotc_gpumemzero(  
    magmaDoubleComplex * d, 
    int n )
{
   int i = blockIdx.x * blockDim.x + threadIdx.x;

   if( i < n ){
    for( int j=0; j<4; j++)
      d[ i+j*n ] = MAGMA_Z_MAKE( 0.0, 0.0 );
    }
}


// dot product for multiple vectors
__global__ void
magma_zmzdotc_kernel_1( 
    int Gs,
    int n, 
    magmaDoubleComplex * v0,
    magmaDoubleComplex * w0,
    magmaDoubleComplex * v1,
    magmaDoubleComplex * w1,
    magmaDoubleComplex * v2,
    magmaDoubleComplex * w2,
    magmaDoubleComplex * v3,
    magmaDoubleComplex * w3,
    magmaDoubleComplex * vtmp)
{
    extern __shared__ magmaDoubleComplex temp[]; 
    int Idx = threadIdx.x;   
    int i   = blockIdx.x * blockDim.x + Idx;
    int j;

    // 4 vectors v(i)/w(i)
    
    temp[ Idx ]                 = ( i < n ) ?
                v0[ i ] * w0[ i ] : MAGMA_Z_ZERO;
                
    temp[ Idx + blockDim.x ]    = ( i < n ) ?
                v1[ i ] * w1[ i ] : MAGMA_Z_ZERO;
                
    temp[ Idx + 2*blockDim.x ]  = ( i < n ) ?
                v2[ i ] * w2[ i ] : MAGMA_Z_ZERO;
                
    temp[ Idx + 3*blockDim.x ]  = ( i < n ) ?
                v3[ i ] * w3[ i ] : MAGMA_Z_ZERO;
               
    
    __syncthreads();
    if ( Idx < 128 ){
        for( j=0; j<4; j++){
            temp[ Idx+j*blockDim.x ] += temp[ Idx+j*blockDim.x + 128 ];
        }
    }
    __syncthreads();
    if ( Idx < 64 ){
        for( j=0; j<4; j++){
            temp[ Idx+j*blockDim.x ] += temp[ Idx+j*blockDim.x + 64 ];
        }
    }
    __syncthreads();
    #if defined(PRECISION_z) || defined(PRECISION_c)
        if( Idx < 32 ){
            for( j=0; j<4; j++)
                temp[ Idx+j*blockDim.x ] += temp[ Idx+j*blockDim.x + 32 ];
                __syncthreads();
            for( j=0; j<4; j++)
                temp[ Idx+j*blockDim.x ] += temp[ Idx+j*blockDim.x + 16 ];
                __syncthreads();
            for( j=0; j<4; j++)
                temp[ Idx+j*blockDim.x ] += temp[ Idx+j*blockDim.x + 8 ];
                __syncthreads();
            for( j=0; j<4; j++)
                temp[ Idx+j*blockDim.x ] += temp[ Idx+j*blockDim.x + 4 ];
                __syncthreads();
            for( j=0; j<4; j++)
                temp[ Idx+j*blockDim.x ] += temp[ Idx+j*blockDim.x + 2 ];
                __syncthreads();
            for( j=0; j<4; j++)
                temp[ Idx+j*blockDim.x ] += temp[ Idx+j*blockDim.x + 1 ];
                __syncthreads();
        }
    #endif
    #if defined(PRECISION_d)
        if( Idx < 32 ){
            volatile double *temp2 = temp;
            for( j=0; j<4; j++){
                temp2[ Idx+j*blockDim.x ] += temp2[ Idx+j*blockDim.x + 32 ];
                temp2[ Idx+j*blockDim.x ] += temp2[ Idx+j*blockDim.x + 16 ];
                temp2[ Idx+j*blockDim.x ] += temp2[ Idx+j*blockDim.x + 8 ];
                temp2[ Idx+j*blockDim.x ] += temp2[ Idx+j*blockDim.x + 4 ];
                temp2[ Idx+j*blockDim.x ] += temp2[ Idx+j*blockDim.x + 2 ];
                temp2[ Idx+j*blockDim.x ] += temp2[ Idx+j*blockDim.x + 1 ];
            }
        }
    #endif
    #if defined(PRECISION_s)
        if( Idx < 32 ){
            volatile float *temp2 = temp;
            for( j=0; j<4; j++){
                temp2[ Idx+j*blockDim.x ] += temp2[ Idx+j*blockDim.x + 32 ];
                temp2[ Idx+j*blockDim.x ] += temp2[ Idx+j*blockDim.x + 16 ];
                temp2[ Idx+j*blockDim.x ] += temp2[ Idx+j*blockDim.x + 8 ];
                temp2[ Idx+j*blockDim.x ] += temp2[ Idx+j*blockDim.x + 4 ];
                temp2[ Idx+j*blockDim.x ] += temp2[ Idx+j*blockDim.x + 2 ];
                temp2[ Idx+j*blockDim.x ] += temp2[ Idx+j*blockDim.x + 1 ];
            }
        }
    #endif
    if ( Idx == 0 ){
        for( j=0; j<4; j++){
            vtmp[ blockIdx.x+j*n ] = temp[ j*blockDim.x ];
        }
    }
}



// block reduction for 4 vectors
__global__ void
magma_zmzdotc_kernel_2( 
    int Gs,
    int n, 
    magmaDoubleComplex * vtmp,
    magmaDoubleComplex * vtmp2 )
{
    extern __shared__ magmaDoubleComplex temp[];    
    int Idx = threadIdx.x;
    int blockSize = 128;
    int gridSize = blockSize  * 2 * gridDim.x; 
    int j;

    for( j=0; j<4; j++){
        int i = blockIdx.x * ( blockSize * 2 ) + Idx;   
        temp[Idx+j*(blockSize)] = MAGMA_Z_ZERO;
        while (i < Gs ) {
            temp[ Idx+j*(blockSize)  ] += vtmp[ i+j*n ]; 
            temp[ Idx+j*(blockSize)  ] += 
                ( i + (blockSize) < Gs ) ? vtmp[ i+j*n + (blockSize) ] 
                                                : MAGMA_Z_ZERO;
            i += gridSize;
        }
    }
    __syncthreads();
    if ( Idx < 64 ){
        for( j=0; j<4; j++){
            temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 64 ];
        }
    }
    __syncthreads();
    #if defined(PRECISION_z) || defined(PRECISION_c)
        if( Idx < 32 ){
            for( j=0; j<4; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 32 ];
                __syncthreads();
            for( j=0; j<4; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 16 ];
                __syncthreads();
            for( j=0; j<4; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 8 ];
                __syncthreads();
            for( j=0; j<4; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 4 ];
                __syncthreads();
            for( j=0; j<4; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 2 ];
                __syncthreads();
            for( j=0; j<4; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 1 ];
                __syncthreads();
        }
    #endif
    #if defined(PRECISION_d)
        if( Idx < 32 ){
            volatile double *temp2 = temp;
            for( j=0; j<4; j++){
                temp2[ Idx+j*(blockSize) ] += temp2[ Idx+j*(blockSize) + 32 ];
                temp2[ Idx+j*(blockSize) ] += temp2[ Idx+j*(blockSize) + 16 ];
                temp2[ Idx+j*(blockSize) ] += temp2[ Idx+j*(blockSize) + 8 ];
                temp2[ Idx+j*(blockSize) ] += temp2[ Idx+j*(blockSize) + 4 ];
                temp2[ Idx+j*(blockSize) ] += temp2[ Idx+j*(blockSize) + 2 ];
                temp2[ Idx+j*(blockSize) ] += temp2[ Idx+j*(blockSize) + 1 ];
            }
        }
    #endif
    #if defined(PRECISION_s)
        if( Idx < 32 ){
            volatile float *temp2 = temp;
            for( j=0; j<4; j++){
                temp2[ Idx+j*(blockSize) ] += temp2[ Idx+j*(blockSize) + 32 ];
                temp2[ Idx+j*(blockSize) ] += temp2[ Idx+j*(blockSize) + 16 ];
                temp2[ Idx+j*(blockSize) ] += temp2[ Idx+j*(blockSize) + 8 ];
                temp2[ Idx+j*(blockSize) ] += temp2[ Idx+j*(blockSize) + 4 ];
                temp2[ Idx+j*(blockSize) ] += temp2[ Idx+j*(blockSize) + 2 ];
                temp2[ Idx+j*(blockSize) ] += temp2[ Idx+j*(blockSize) + 1 ];
            }
        }
    #endif
    if ( Idx == 0 ){
        for( j=0; j<4; j++){
            vtmp2[ blockIdx.x+j*n ] = temp[ j*(blockSize) ];
        }
    }
}

/**
    Purpose
    -------

    Computes the scalar product of a set of 4 vectors such that

    skp[0,1,2,3] = [ <v_0,w_0>, <v_1,w_1>, <v_2,w_2>, <v3,w_3> ]

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
    v1          magmaDoubleComplex_ptr                 
                input vector                           
                                                       
    @param[in]                                         
    w1          magmaDoubleComplex_ptr                 
                input vector          

    @param[in]                             
    v2          magmaDoubleComplex_ptr     
                input vector               

    @param[in]                                         
    w2          magmaDoubleComplex_ptr                 
                input vector                           
                                                       
    @param[in]                                         
    v3          magmaDoubleComplex_ptr                 
                input vector                           
                                                       
    @param[in]                                         
    w3          magmaDoubleComplex_ptr                 
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
magma_zmzdotc(
    int n,  
    magmaDoubleComplex_ptr v0, 
    magmaDoubleComplex_ptr w0,
    magmaDoubleComplex_ptr v1, 
    magmaDoubleComplex_ptr w1,
    magmaDoubleComplex_ptr v2, 
    magmaDoubleComplex_ptr w2,
    magmaDoubleComplex_ptr v3, 
    magmaDoubleComplex_ptr w3,
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
    int Ms = 4 * (local_block_size) * sizeof( magmaDoubleComplex ); // 4 skp 
    magmaDoubleComplex_ptr aux1 = d1, aux2 = d2;
    int b = 1;        


    magma_zmzdotc_kernel_1<<<Gs, Bs, Ms, queue>>>
            ( Gs.x, n, v0, w0, v1, w1, v2, w2, v3, w3, d1 );

/*
    // not necessary to zero GPU mem
    magma_zmzdotc_gpumemzero<<<Gs, Bs, 0>>>( d1, n*k,1 );
    magma_zmzdotc_gpumemzero<<<Gs, Bs, 0>>>( d2, n*k,1 );
    //magmablas_zlaset( MagmaUpperLower, n, k, d1, n );
    //magmablas_zlaset( MagmaUpperLower, n, k, d2, n );
    while( Gs.x > 1 ) {
        Gs_next.x = ( Gs.x+Bs.x-1 )/ Bs.x;
        magma_zblockreduce_kernel<<< Gs_next.x, Bs.x, Ms >>> 
                                        ( Gs.x, n, k, aux1, aux2 );
        Gs.x = Gs_next.x;
        b = 1 - b;
        if ( b ) { aux1 = d1; aux2 = d2; }
        else   { aux2 = d1; aux1 = d2; }
    }
    for( int j=0; j<k; j++) {
            magma_zcopyvector( 1, aux1+j*n, 1, skp+j, 1 );
    }
*/
   
    while( Gs.x > 1 ) {
        Gs_next.x = ( Gs.x+Bs.x-1 )/ Bs.x;
        if ( Gs_next.x == 1 ) Gs_next.x = 2;
        magma_zmzdotc_kernel_2<<< Gs_next.x/2, Bs.x/2, Ms/2, queue >>> 
                    ( Gs.x, n, aux1, aux2 );
        Gs_next.x = Gs_next.x /2;
        Gs.x = Gs_next.x;
        b = 1 - b;
        if ( b ) { aux1 = d1; aux2 = d2; }
        else   { aux2 = d1; aux1 = d2; }
    }

        // copy vectors to host
    magma_zgetvector( 4 , aux1, 1, skp, 1 );
    
    magma_zprint_gpu( 4, 1, aux1, 4 );
    
    printf("the skp is: %d %d %d %d\n", skp[0], skp[1], skp[2], skp[3]);

   magmablasSetKernelStream( orig_queue );
   return MAGMA_SUCCESS;
}


