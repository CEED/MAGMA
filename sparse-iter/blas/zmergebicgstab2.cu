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

#define BLOCK_SIZE 512

#define PRECISION_z


// These routines merge multiple kernels from zmergebicgstab into one
// This is the code used for the ASHES2014 paper
// "Accelerating Krylov Subspace Solvers on Graphics Processing Units".


// accelerated reduction for one vector
__global__ void 
magma_zreduce_kernel_spmv1(    int Gs,
                               int n, 
                               magmaDoubleComplex *vtmp,
                               magmaDoubleComplex *vtmp2 ){

    extern __shared__ magmaDoubleComplex temp[];    
    int Idx = threadIdx.x;
    int blockSize = 128;
    int gridSize = blockSize  * 2 * gridDim.x; 
    temp[Idx] = MAGMA_Z_MAKE( 0.0, 0.0);
    int i = blockIdx.x * ( blockSize * 2 ) + Idx;   
    while (i < Gs ) {
        temp[ Idx  ] += vtmp[ i ]; 
        temp[ Idx  ] += ( i + blockSize < Gs ) ? vtmp[ i + blockSize ] 
                                                : MAGMA_Z_MAKE( 0.0, 0.0); 
        i += gridSize;
    }
    __syncthreads();
    if ( Idx < 64 ){
        temp[ Idx ] += temp[ Idx + 64 ];
    }
    __syncthreads();
    #if defined(PRECISION_z) || defined(PRECISION_c)
        if( Idx < 32 ){
            temp[ Idx ] += temp[ Idx + 32 ];__syncthreads();
            temp[ Idx ] += temp[ Idx + 16 ];__syncthreads();
            temp[ Idx ] += temp[ Idx + 8 ];__syncthreads();
            temp[ Idx ] += temp[ Idx + 4 ];__syncthreads();
            temp[ Idx ] += temp[ Idx + 2 ];__syncthreads();
            temp[ Idx ] += temp[ Idx + 1 ];__syncthreads();
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


__global__ void 
magma_zbicgmerge_spmv1_kernel(  
                 int n,
                 magmaDoubleComplex *d_val, 
                 int *d_rowptr, 
                 int *d_colind,
                 magmaDoubleComplex *p,
                 magmaDoubleComplex *r,
                 magmaDoubleComplex *v,
                 magmaDoubleComplex *vtmp
                                            ){

    extern __shared__ magmaDoubleComplex temp[]; 
    int Idx = threadIdx.x;   
    int i   = blockIdx.x * blockDim.x + Idx;
    int j;

    if( i<n ){
        magmaDoubleComplex dot = MAGMA_Z_ZERO;
        int start = d_rowptr[ i ];
        int end = d_rowptr[ i+1 ];
        for( j=start; j<end; j++)
            dot += d_val[ j ] * p[ d_colind[j] ];
        v[ i ] =  dot;
    }

    __syncthreads(); 

    temp[ Idx ] = ( i < n ) ? v[ i ] * r[ i ] : MAGMA_Z_MAKE( 0.0, 0.0);
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
            temp[ Idx ] += temp[ Idx + 32 ];__syncthreads();
            temp[ Idx ] += temp[ Idx + 16 ];__syncthreads();
            temp[ Idx ] += temp[ Idx + 8 ];__syncthreads();
            temp[ Idx ] += temp[ Idx + 4 ];__syncthreads();
            temp[ Idx ] += temp[ Idx + 2 ];__syncthreads();
            temp[ Idx ] += temp[ Idx + 1 ];__syncthreads();
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

__global__ void 
magma_zbicgstab_alphakernel(  
                    magmaDoubleComplex *skp ){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if( i==0 ){
        magmaDoubleComplex tmp = skp[0];
        skp[0] = skp[4]/tmp;
    }
}

/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

    Purpose
    =======

    Merges the first SpmV using CSR with the dot product 
    and the computation of alpha

    Arguments
    =========

    int n                               dimension n
    magmaDoubleComplex *d1              temporary vector
    magmaDoubleComplex *d2              temporary vector
    magmaDoubleComplex *d_val           matrix values
    int *d_rowptr                       matrix row pointer
    int *d_colind                       matrix column indices
    magmaDoubleComplex *d_p             input vector p
    magmaDoubleComplex *d_r             input vector r
    magmaDoubleComplex *d_v             output vector v
    magmaDoubleComplex *skp             array for parameters ( skp[0]=alpha )

    ========================================================================  */

extern "C" int
magma_zbicgmerge_spmv1(  int n,
                         magmaDoubleComplex *d1,
                         magmaDoubleComplex *d2,
                         magmaDoubleComplex *d_val, 
                         int *d_rowptr, 
                         int *d_colind,
                         magmaDoubleComplex *d_p,
                         magmaDoubleComplex *d_r,
                         magmaDoubleComplex *d_v,
                         magmaDoubleComplex *skp ){

    int local_block_size=256;
    dim3 Bs( local_block_size );
    dim3 Gs( (n+local_block_size-1)/local_block_size );
    dim3 Gs_next;
    int Ms =  local_block_size * sizeof( magmaDoubleComplex ); 
    magmaDoubleComplex *aux1 = d1, *aux2 = d2;
    int b = 1;        

    magma_zbicgmerge_spmv1_kernel<<<Gs, Bs, Ms>>>
                ( n, d_val, d_rowptr, d_colind, d_p, d_r, d_v, d1 );

    while( Gs.x > 1 ){
        Gs_next.x = ( Gs.x+Bs.x-1 )/ Bs.x ;
        if( Gs_next.x == 1 ) Gs_next.x = 2;
        magma_zreduce_kernel_spmv1<<< Gs_next.x/2, Bs.x/2, Ms/2 >>> 
                            ( Gs.x, n, aux1, aux2 );
        Gs_next.x = Gs_next.x /2;
        Gs.x = Gs_next.x;
        b = 1 - b;
        if( b ){ aux1 = d1; aux2 = d2; }
        else   { aux2 = d1; aux1 = d2; }
    }


    cudaMemcpy( skp, aux1, sizeof( magmaDoubleComplex ), 
                                        cudaMemcpyDeviceToDevice );
    dim3 Bs2( 2 );
    dim3 Gs2( 1 );
    magma_zbicgstab_alphakernel<<<Gs2, Bs2, 0>>>( skp );

   return MAGMA_SUCCESS;
}

/* -------------------------------------------------------------------------- */

// accelerated block reduction for multiple vectors
__global__ void 
magma_zreduce_kernel_spmv2( int Gs,
                           int n, 
                           magmaDoubleComplex *vtmp,
                           magmaDoubleComplex *vtmp2 ){

    extern __shared__ magmaDoubleComplex temp[];    
    int Idx = threadIdx.x;
    int blockSize = 128;
    int gridSize = blockSize  * 2 * gridDim.x; 
    int j;

    for( j=0; j<2; j++){
        int i = blockIdx.x * ( blockSize * 2 ) + Idx;   
        temp[Idx+j*(blockSize)] = MAGMA_Z_MAKE( 0.0, 0.0);
        while (i < Gs ) {
            temp[ Idx+j*(blockSize)  ] += vtmp[ i+j*n ]; 
            temp[ Idx+j*(blockSize)  ] += 
                ( i + (blockSize) < Gs ) ? vtmp[ i+j*n + (blockSize) ] 
                : MAGMA_Z_MAKE( 0.0, 0.0); 
            i += gridSize;
        }
    }
    __syncthreads();
    if ( Idx < 64 ){
        for( j=0; j<2; j++){
            temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 64 ];
        }
    }
    __syncthreads();
    #if defined(PRECISION_z) || defined(PRECISION_c)
        if( Idx < 32 ){
            for( j=0; j<2; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 32 ];
                __syncthreads();
            for( j=0; j<2; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 16 ];
                __syncthreads();
            for( j=0; j<2; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 8 ];
                __syncthreads();
            for( j=0; j<2; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 4 ];
                __syncthreads();
            for( j=0; j<2; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 2 ];
                __syncthreads();
            for( j=0; j<2; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 1 ];
                __syncthreads();
        }
    #endif
    #if defined(PRECISION_d)
        if( Idx < 32 ){
            volatile double *temp2 = temp;
            for( j=0; j<2; j++){
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
            for( j=0; j<2; j++){
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
        for( j=0; j<2; j++){
            vtmp2[ blockIdx.x+j*n ] = temp[ j*(blockSize) ];
        }
    }
}

__global__ void 
magma_zbicgmerge_spmv2_kernel(  
                 int n,
                 magmaDoubleComplex *d_val, 
                 int *d_rowptr, 
                 int *d_colind,
                 magmaDoubleComplex *s,
                 magmaDoubleComplex *t,
                 magmaDoubleComplex *vtmp
                                            ){

    extern __shared__ magmaDoubleComplex temp[]; 
    int Idx = threadIdx.x;   
    int i   = blockIdx.x * blockDim.x + Idx;
    int j;

    if( i<n ){
        magmaDoubleComplex dot = MAGMA_Z_ZERO;
        int start = d_rowptr[ i ];
        int end = d_rowptr[ i+1 ];
        for( j=start; j<end; j++)
            dot += d_val[ j ] * s[ d_colind[j] ];
        t[ i ] =  dot;
    }

    __syncthreads(); 

    // 2 vectors 
    if (i<n){
            magmaDoubleComplex tmp2 = t[i];
            temp[Idx] = s[i] * tmp2;
            temp[Idx+blockDim.x] = tmp2 * tmp2;
    }
    else{
        for( j=0; j<2; j++)
            temp[Idx+j*blockDim.x] =MAGMA_Z_MAKE( 0.0, 0.0);
    }
    __syncthreads();
    if ( Idx < 128 ){
        for( j=0; j<2; j++){
            temp[ Idx+j*blockDim.x ] += temp[ Idx+j*blockDim.x + 128 ];
        }
    }
    __syncthreads();
    if ( Idx < 64 ){
        for( j=0; j<2; j++){
            temp[ Idx+j*blockDim.x ] += temp[ Idx+j*blockDim.x + 64 ];
        }
    }
    __syncthreads();
    #if defined(PRECISION_z) || defined(PRECISION_c)
        if( Idx < 32 ){
            for( j=0; j<2; j++)
                temp[ Idx+j*blockDim.x ] += temp[ Idx+j*blockDim.x + 32 ];
                __syncthreads();
            for( j=0; j<2; j++)
                temp[ Idx+j*blockDim.x ] += temp[ Idx+j*blockDim.x + 16 ];
                __syncthreads();
            for( j=0; j<2; j++)
                temp[ Idx+j*blockDim.x ] += temp[ Idx+j*blockDim.x + 8 ];
                __syncthreads();
            for( j=0; j<2; j++)
                temp[ Idx+j*blockDim.x ] += temp[ Idx+j*blockDim.x + 4 ];
                __syncthreads();
            for( j=0; j<2; j++)
                temp[ Idx+j*blockDim.x ] += temp[ Idx+j*blockDim.x + 2 ];
                __syncthreads();
            for( j=0; j<2; j++)
                temp[ Idx+j*blockDim.x ] += temp[ Idx+j*blockDim.x + 1 ];
                __syncthreads();
        }
    #endif
    #if defined(PRECISION_d)
        if( Idx < 32 ){
            volatile double *temp2 = temp;
            for( j=0; j<2; j++){
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
            for( j=0; j<2; j++){
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
        for( j=0; j<2; j++){
            vtmp[ blockIdx.x+j*n ] = temp[ j*blockDim.x ];
        }
    }
}

__global__ void 
magma_zbicgstab_omegakernel(  
                    magmaDoubleComplex *skp ){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if( i==0 ){
        skp[2] = skp[6]/skp[7];
        skp[3] = skp[4];
    }
}

/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

    Purpose
    =======

    Merges the second SpmV using CSR with the dot product 
    and the computation of omega

    Arguments
    =========

    int n                               dimension n
    int n                               dimension n
    magmaDoubleComplex *d1              temporary vector
    magmaDoubleComplex *d2              temporary vector
    magmaDoubleComplex *d_val           matrix values
    int *d_rowptr                       matrix row pointer
    int *d_colind                       matrix column indices
    magmaDoubleComplex *d_s             input vector s
    magmaDoubleComplex *d_t             output vector t
    magmaDoubleComplex *skp             array for parameters

    ========================================================================  */

extern "C" int
magma_zbicgmerge_spmv2(  
                 int n,
                 magmaDoubleComplex *d1,
                 magmaDoubleComplex *d2,
                 magmaDoubleComplex *d_val, 
                 int *d_rowptr, 
                 int *d_colind,
                 magmaDoubleComplex *d_s,
                 magmaDoubleComplex *d_t,
                 magmaDoubleComplex *skp ){

    int local_block_size=256;
    dim3 Bs( local_block_size );
    dim3 Gs( (n+local_block_size-1)/local_block_size );
    dim3 Gs_next;
    int Ms =  2*local_block_size * sizeof( magmaDoubleComplex ); 
    magmaDoubleComplex *aux1 = d1, *aux2 = d2;
    int b = 1;        

    magma_zbicgmerge_spmv2_kernel<<<Gs, Bs, Ms>>>
                ( n, d_val, d_rowptr, d_colind, d_s, d_t, d1 );

    while( Gs.x > 1 ){
        Gs_next.x = ( Gs.x+Bs.x-1 )/ Bs.x ;
        if( Gs_next.x == 1 ) Gs_next.x = 2;
        magma_zreduce_kernel_spmv2<<< Gs_next.x/2, Bs.x/2, Ms/2 >>> 
                    ( Gs.x, n, aux1, aux2 );
        Gs_next.x = Gs_next.x /2;
        Gs.x = Gs_next.x;
        b = 1 - b;
        if( b ){ aux1 = d1; aux2 = d2; }
        else   { aux2 = d1; aux1 = d2; }
    }


    cudaMemcpy( skp+6, aux1, sizeof( magmaDoubleComplex ), 
                                    cudaMemcpyDeviceToDevice );
    cudaMemcpy( skp+7, aux1+n, sizeof( magmaDoubleComplex ), 
                                    cudaMemcpyDeviceToDevice );
    dim3 Bs2( 2 );
    dim3 Gs2( 1 );
    magma_zbicgstab_omegakernel<<<Gs2, Bs2, 0>>>( skp );

   return MAGMA_SUCCESS;
}

/* -------------------------------------------------------------------------- */

__global__ void 
magma_zbicgmerge_xrbeta_kernel(  
                    int n, 
                    magmaDoubleComplex *rr,
                    magmaDoubleComplex *r,
                    magmaDoubleComplex *p,
                    magmaDoubleComplex *s,
                    magmaDoubleComplex *t,
                    magmaDoubleComplex *x, 
                    magmaDoubleComplex *skp,
                    magmaDoubleComplex *vtmp
                                            ){

    extern __shared__ magmaDoubleComplex temp[]; 
    int Idx = threadIdx.x;   
    int i   = blockIdx.x * blockDim.x + Idx;
    int j;

    magmaDoubleComplex alpha=skp[0];
    magmaDoubleComplex omega=skp[2];

    if( i<n ){
        magmaDoubleComplex sl;
        sl = s[i];
        x[i] = x[i] + alpha * p[i] + omega * sl;
        r[i] = sl - omega * t[i];
    }

    __syncthreads(); 

    // 2 vectors 
    if (i<n){
            magmaDoubleComplex tmp2 = r[i];
            temp[Idx] = rr[i] * tmp2;
            temp[Idx+blockDim.x] = tmp2 * tmp2;
    }
    else{
        for( j=0; j<2; j++)
            temp[Idx+j*blockDim.x] =MAGMA_Z_MAKE( 0.0, 0.0);
    }
    __syncthreads();
    if ( Idx < 128 ){
        for( j=0; j<2; j++){
            temp[ Idx+j*blockDim.x ] += temp[ Idx+j*blockDim.x + 128 ];
        }
    }
    __syncthreads();
    if ( Idx < 64 ){
        for( j=0; j<2; j++){
            temp[ Idx+j*blockDim.x ] += temp[ Idx+j*blockDim.x + 64 ];
        }
    }
    __syncthreads();
    #if defined(PRECISION_z) || defined(PRECISION_c)
        if( Idx < 32 ){
            for( j=0; j<2; j++)
                temp[ Idx+j*blockDim.x ] += temp[ Idx+j*blockDim.x + 32 ];
                __syncthreads();
            for( j=0; j<2; j++)
                temp[ Idx+j*blockDim.x ] += temp[ Idx+j*blockDim.x + 16 ];
                __syncthreads();
            for( j=0; j<2; j++)
                temp[ Idx+j*blockDim.x ] += temp[ Idx+j*blockDim.x + 8 ];
                __syncthreads();
            for( j=0; j<2; j++)
                temp[ Idx+j*blockDim.x ] += temp[ Idx+j*blockDim.x + 4 ];
                __syncthreads();
            for( j=0; j<2; j++)
                temp[ Idx+j*blockDim.x ] += temp[ Idx+j*blockDim.x + 2 ];
                __syncthreads();
            for( j=0; j<2; j++)
                temp[ Idx+j*blockDim.x ] += temp[ Idx+j*blockDim.x + 1 ];
                __syncthreads();
        }
    #endif
    #if defined(PRECISION_d)
        if( Idx < 32 ){
            volatile double *temp2 = temp;
            for( j=0; j<2; j++){
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
            for( j=0; j<2; j++){
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
        for( j=0; j<2; j++){
            vtmp[ blockIdx.x+j*n ] = temp[ j*blockDim.x ];
        }
    }
}

__global__ void 
magma_zbicgstab_betakernel(  
                    magmaDoubleComplex *skp ){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if( i==0 ){
        magmaDoubleComplex tmp1 = skp[4]/skp[3];
        magmaDoubleComplex tmp2 = skp[0] / skp[2];
        skp[1] =  tmp1*tmp2;
    }
}

/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

    Purpose
    =======

    Merges the second SpmV using CSR with the dot product 
    and the computation of omega

    Arguments
    =========

    int n                               dimension n
    int n                               dimension n
    magmaDoubleComplex *d1              temporary vector
    magmaDoubleComplex *d2              temporary vector
    magmaDoubleComplex *d_rr            input vector rr
    magmaDoubleComplex *d_r             input/output vector r
    magmaDoubleComplex *d_p             input vector p
    magmaDoubleComplex *d_s             input vector s
    magmaDoubleComplex *d_t             input vector t
    magmaDoubleComplex *d_x             output vector x
    magmaDoubleComplex *skp             array for parameters

    ========================================================================  */

extern "C" int
magma_zbicgmerge_xrbeta(  
                 int n,
                 magmaDoubleComplex *d1,
                 magmaDoubleComplex *d2,
                 magmaDoubleComplex *rr,
                 magmaDoubleComplex *r,
                 magmaDoubleComplex *p,
                 magmaDoubleComplex *s,
                 magmaDoubleComplex *t,
                 magmaDoubleComplex *x, 
                 magmaDoubleComplex *skp ){

    int local_block_size=256;
    dim3 Bs( local_block_size );
    dim3 Gs( (n+local_block_size-1)/local_block_size );
    dim3 Gs_next;
    int Ms =  2*local_block_size * sizeof( magmaDoubleComplex ); 
    magmaDoubleComplex *aux1 = d1, *aux2 = d2;
    int b = 1;        
    magma_zbicgmerge_xrbeta_kernel<<<Gs, Bs, Ms>>>
                    ( n, rr, r, p, s, t, x, skp, d1);  

    while( Gs.x > 1 ){
        Gs_next.x = ( Gs.x+Bs.x-1 )/ Bs.x ;
        if( Gs_next.x == 1 ) Gs_next.x = 2;
        magma_zreduce_kernel_spmv2<<< Gs_next.x/2, Bs.x/2, Ms/2 >>> 
                            ( Gs.x, n, aux1, aux2 );
        Gs_next.x = Gs_next.x /2;
        Gs.x = Gs_next.x;
        b = 1 - b;
        if( b ){ aux1 = d1; aux2 = d2; }
        else   { aux2 = d1; aux1 = d2; }
    }


    cudaMemcpy( skp+4, aux1, sizeof( magmaDoubleComplex ), 
                                        cudaMemcpyDeviceToDevice );
    cudaMemcpy( skp+5, aux1+n, sizeof( magmaDoubleComplex ), 
                                        cudaMemcpyDeviceToDevice );
    dim3 Bs2( 2 );
    dim3 Gs2( 1 );
    magma_zbicgstab_betakernel<<<Gs2, Bs2, 0>>>( skp );

   return MAGMA_SUCCESS;
}

/* -------------------------------------------------------------------------- */

