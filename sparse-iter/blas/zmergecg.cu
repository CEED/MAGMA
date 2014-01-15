/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> c d s
       @author Hartwig Anzt

*/
#include "common_magma.h"

#define BLOCK_SIZE 512

#define PRECISION_z


// These routines merge multiple kernels from zmergecg into one

// accelerated reduction for one vector
__global__ void 
magma_zcgreduce_kernel_spmv1( int Gs,
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
    #if defined(PRECISION_f)
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

// computes the SpMV using CSR and the first step of the reduction
__global__ void 
magma_zcgmerge_spmvcsr_kernel(  
                 int n,
                 magmaDoubleComplex *d_val, 
                 int *d_rowptr, 
                 int *d_colind,
                 magmaDoubleComplex *d,
                 magmaDoubleComplex *z,
                 magmaDoubleComplex *vtmp
                                           ){

    extern __shared__ magmaDoubleComplex temp[]; 
    int Idx = threadIdx.x;   
    int i   = blockIdx.x * blockDim.x + Idx;
    int j;

    temp[ Idx ] = MAGMA_Z_MAKE( 0.0, 0.0);

    if( i<n ){
        magmaDoubleComplex dot = MAGMA_Z_ZERO;
        int start = d_rowptr[ i ];
        int end = d_rowptr[ i+1 ];
        for( j=start; j<end; j++)
            dot += d_val[ j ] * d[ d_colind[j] ];
        z[ i ] =  dot;
        temp[ Idx ] =  d[ i ] * dot;
    }

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

// computes the SpMV using ELLPACKT and the first step of the reduction
__global__ void 
magma_zcgmerge_spmvellpackt_kernel(  
                 int n,
                 int num_cols_per_row,
                 magmaDoubleComplex *d_val, 
                 int *d_colind,
                 magmaDoubleComplex *d,
                 magmaDoubleComplex *z,
                 magmaDoubleComplex *vtmp
                                           ){

    extern __shared__ magmaDoubleComplex temp[]; 
    int Idx = threadIdx.x;   
    int i   = blockIdx.x * blockDim.x + Idx;

    temp[ Idx ] = MAGMA_Z_MAKE( 0.0, 0.0);

    if(i < n ){
        magmaDoubleComplex dot = MAGMA_Z_MAKE(0.0, 0.0);
        for ( int k = 0; k < num_cols_per_row ; k ++){
            int col = d_colind [ n * k + i ];
            magmaDoubleComplex val = d_val [ n * k + i ];
            if( val != 0)
                dot += val * d[ col ];
        }
        z[ i ] =  dot;
        temp[ Idx ] = d[ i ] * dot;
    }

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

// computes the SpMV using ELLPACK and the first step of the reduction
__global__ void 
magma_zcgmerge_spmvellpack_kernel(  
                 int n,
                 int num_cols_per_row,
                 magmaDoubleComplex *d_val, 
                 int *d_colind,
                 magmaDoubleComplex *d,
                 magmaDoubleComplex *z,
                 magmaDoubleComplex *vtmp
                                           ){

    extern __shared__ magmaDoubleComplex temp[]; 
    int Idx = threadIdx.x;   
    int i   = blockIdx.x * blockDim.x + Idx;

    temp[ Idx ] = MAGMA_Z_MAKE( 0.0, 0.0);

    if(i < n ){
        magmaDoubleComplex dot = MAGMA_Z_MAKE(0.0, 0.0);
        for ( int k = 0; k < num_cols_per_row ; k ++){
            int col = d_colind [ num_cols_per_row * i + k ];
            magmaDoubleComplex val = d_val [ num_cols_per_row * i + k ];
            if( val != 0)
                dot += val * d[ col ];
        }
        z[ i ] =  dot;
        temp[ Idx ] = d[ i ] * dot;
    }

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

// kernel to handle scalars
__global__ void // rho = beta/tmp; gamma = beta;
magma_zcg_rhokernel(  
                    magmaDoubleComplex *skp ){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if( i==0 ){
        magmaDoubleComplex tmp = skp[1];
        skp[3] = tmp/skp[4];
        skp[2] = tmp;
    }
}

/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======

    Merges the first SpmV using different formats with the dot product 
    and the computation of rho

    Arguments
    =========

    magma_storage_t storage_t           matrix storage type
    int n                               dimension n
    int max_nnz_row                     for ELLPACK/T
    magmaDoubleComplex *d1              temporary vector
    magmaDoubleComplex *d2              temporary vector
    magmaDoubleComplex *d_val           matrix values
    int *d_rowptr                       matrix row pointer
    int *d_colind                       matrix column indices
    magmaDoubleComplex *d_d             input vector d
    magmaDoubleComplex *d_z             input vector z
    magmaDoubleComplex *skp             array for parameters ( skp[3]=rho )

    =====================================================================  */

extern "C" int
magma_zcgmerge_spmv1(  
                 magma_storage_t storage_t,
                 int n,
                 int max_nnz_row,  
                 magmaDoubleComplex *d1,
                 magmaDoubleComplex *d2,
                 magmaDoubleComplex *d_val, 
                 int *d_rowptr, 
                 int *d_colind,
                 magmaDoubleComplex *d_d,
                 magmaDoubleComplex *d_z,
                 magmaDoubleComplex *skp ){

    int local_block_size=256;
    dim3 Bs( local_block_size );
    dim3 Gs( (n+local_block_size-1)/local_block_size );
    dim3 Gs_next;
    int Ms =  local_block_size * sizeof( magmaDoubleComplex ); 
    magmaDoubleComplex *aux1 = d1, *aux2 = d2;
    int b = 1;        

    if( storage_t == Magma_CSR )
        magma_zcgmerge_spmvcsr_kernel<<<Gs, Bs, Ms>>>
        ( n, d_val, d_rowptr, d_colind, d_d, d_z, d1 );
    else if( storage_t == Magma_ELLPACK )
        magma_zcgmerge_spmvellpack_kernel<<<Gs, Bs, Ms>>>
        ( n, max_nnz_row, d_val, d_colind, d_d, d_z, d1 );
    else if( storage_t == Magma_ELLPACKT )
        magma_zcgmerge_spmvellpackt_kernel<<<Gs, Bs, Ms>>>
        ( n, max_nnz_row, d_val, d_colind, d_d, d_z, d1 );

    while( Gs.x > 1 ){
        Gs_next.x = ( Gs.x+Bs.x-1 )/ Bs.x ;
        if( Gs_next.x == 1 ) Gs_next.x = 2;
        magma_zcgreduce_kernel_spmv1<<< Gs_next.x/2, Bs.x/2, Ms/2 >>> 
                                        ( Gs.x, n, aux1, aux2 );
        Gs_next.x = Gs_next.x /2;
        Gs.x = Gs_next.x;
        b = 1 - b;
        if( b ){ aux1 = d1; aux2 = d2; }
        else   { aux2 = d1; aux1 = d2; }
    }


    cudaMemcpy( skp+4, aux1, sizeof( magmaDoubleComplex ), 
                                            cudaMemcpyDeviceToDevice );
    dim3 Bs2( 2 );
    dim3 Gs2( 1 );
    magma_zcg_rhokernel<<<Gs2, Bs2, 0>>>( skp );

   return MAGMA_SUCCESS;
}


/* -------------------------------------------------------------------------- */

// updates x and r and computes the first part of the dot product r*r
__global__ void 
magma_zcgmerge_xrbeta_kernel(  
                    int n, 
                    magmaDoubleComplex *x, 
                    magmaDoubleComplex *r,
                    magmaDoubleComplex *d,
                    magmaDoubleComplex *z,
                    magmaDoubleComplex *skp,
                    magmaDoubleComplex *vtmp
                                            ){

    extern __shared__ magmaDoubleComplex temp[]; 
    int Idx = threadIdx.x;   
    int i   = blockIdx.x * blockDim.x + Idx;

    magmaDoubleComplex rho = skp[3];
    magmaDoubleComplex mrho = MAGMA_Z_MAKE( -1.0, 0.0)*rho;

    temp[ Idx ] = MAGMA_Z_MAKE( 0.0, 0.0);

    if( i<n ){
        x[i] += rho * d[i] ;
        r[i] += mrho  * z[i];
        temp[ Idx ] = r[i] * r[i];
    }
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

// kernel to handle scalars
__global__ void //alpha = beta / gamma
magma_zcg_alphabetakernel(  
                    magmaDoubleComplex *skp ){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if( i==0 ){
        magmaDoubleComplex tmp1 = skp[1];
        skp[0] =  tmp1/skp[2];
        //printf("beta=%e\n", MAGMA_Z_REAL(tmp1));
    }
}

// update search Krylov vector d
__global__ void 
magma_zcg_d_kernel(  
                    int n, 
                    magmaDoubleComplex *skp,
                    magmaDoubleComplex *r,
                    magmaDoubleComplex *d
                                           ){
  
    int i   = blockIdx.x * blockDim.x + threadIdx.x;

    magmaDoubleComplex alpha = skp[0];

    if( i<n ){
        d[i] = r[i] + alpha * d[i];
    }

}



/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======

    Merges the update of r and x with the dot product and performs then 
    the update for the Krylov vector d

    Arguments
    =========

    int n                               dimension n
    magmaDoubleComplex *d1              temporary vector
    magmaDoubleComplex *d2              temporary vector
    magmaDoubleComplex *d_x             input vector x
    magmaDoubleComplex *d_r             input/output vector r
    magmaDoubleComplex *d_p             input vector p
    magmaDoubleComplex *d_s             input vector s
    magmaDoubleComplex *d_t             input vector t
    magmaDoubleComplex *d_x             output vector x
    magmaDoubleComplex *skp             array for parameters

    =====================================================================  */

extern "C" int
magma_zcgmerge_xrbeta(  
                 int n,
                 magmaDoubleComplex *d1,
                 magmaDoubleComplex *d2,
                 magmaDoubleComplex *d_x,
                 magmaDoubleComplex *d_r,
                 magmaDoubleComplex *d_d,
                 magmaDoubleComplex *d_z, 
                 magmaDoubleComplex *skp ){

    int local_block_size=256;
    dim3 Bs( local_block_size );
    dim3 Gs( (n+local_block_size-1)/local_block_size );
    dim3 Gs_next;
    int Ms =  2*local_block_size * sizeof( magmaDoubleComplex ); 
    magmaDoubleComplex *aux1 = d1, *aux2 = d2;
    int b = 1;        
    magma_zcgmerge_xrbeta_kernel<<<Gs, Bs, Ms>>>
                                    ( n, d_x, d_r, d_d, d_z, skp, d1);  



    while( Gs.x > 1 ){
        Gs_next.x = ( Gs.x+Bs.x-1 )/ Bs.x ;
        if( Gs_next.x == 1 ) Gs_next.x = 2;
        magma_zcgreduce_kernel_spmv1<<< Gs_next.x/2, Bs.x/2, Ms/2 >>> 
                                    ( Gs.x, n, aux1, aux2 );
        Gs_next.x = Gs_next.x /2;
        Gs.x = Gs_next.x;
        b = 1 - b;
        if( b ){ aux1 = d1; aux2 = d2; }
        else   { aux2 = d1; aux1 = d2; }
    }


    cudaMemcpy( skp+1, aux1, sizeof( magmaDoubleComplex ), 
                                                cudaMemcpyDeviceToDevice );
    dim3 Bs2( 2 );
    dim3 Gs2( 1 );
    magma_zcg_alphabetakernel<<<Gs2, Bs2, 0>>>( skp );

    dim3 Bs3( local_block_size );
    dim3 Gs3( (n+local_block_size-1)/local_block_size );
    magma_zcg_d_kernel<<<Gs3, Bs3, 0>>>( n, skp, d_r, d_d );  

   return MAGMA_SUCCESS;
}

/* -------------------------------------------------------------------------- */

