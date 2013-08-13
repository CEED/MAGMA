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

#define BLOCK_SIZE 256


__global__ void 
magma_zpipelined_correction( int n,  
                             int k,
                             magmaDoubleComplex *skp, 
                             magmaDoubleComplex *r,
                             magmaDoubleComplex *v ){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double zz= 0.0, tmp= 0.0;
    magmaDoubleComplex rr;


    extern __shared__ magmaDoubleComplex temp[];    
    
    temp[ i ] = ( i < k ) ? skp[ i ] * skp[ i ] : MAGMA_Z_MAKE( 0.0, 0.0);

    __syncthreads();

     if (i < 64) { temp[ i ] += temp[ i + 64 ]; } __syncthreads(); 

     if( i < 32 ){
        temp[ i ] += temp[ i + 32 ];__syncthreads();    
        temp[ i ] += temp[ i + 16 ];__syncthreads(); 
        temp[ i ] += temp[ i +  8 ];__syncthreads(); 
        temp[ i ] += temp[ i +  4 ];__syncthreads(); 
        temp[ i ] += temp[ i +  2 ];__syncthreads(); 
        temp[ i ] += temp[ i +  1 ];__syncthreads();      
    }
    if( i == 0 ){
        tmp = MAGMA_Z_REAL( temp[ i ] );
        zz = MAGMA_Z_REAL( skp[(k)] );
        skp[k] = MAGMA_Z_MAKE( sqrt(zz-tmp),0.0 );
    }
}

__global__ void 
magma_zpipelined_copyscale( int n,  
                             int k,
                             magmaDoubleComplex *skp, 
                             magmaDoubleComplex *r,
                             magmaDoubleComplex *v ){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    magmaDoubleComplex rr=skp[k];
    if( i<n ){
        v[i] =  r[i] * 1.0 / rr;

    }
}


/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======

    Computes the correction term of the pipelined GMRES according to P. Ghysels 
    and scales and copies the new search direction
    
    Returns the vector v = r/ ( skp[k] - (sum_i=1^k skp[i]^2) ) .

    Arguments
    =========

    int n                             legth of v_i
    int k                             # skp entries <v_i,r> ( without <r> )
    magmaDoubleComplex *r             vector of length n
    magmaDoubleComplex *v             vector of length n

    =====================================================================  */

extern "C" magma_int_t
magma_zcopyscale(   int n, 
                    int k,
                    magmaDoubleComplex *r, 
                    magmaDoubleComplex *v,
                    magmaDoubleComplex *skp ){

    
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( (k+BLOCK_SIZE+1)/BLOCK_SIZE );
    unsigned int Ms =   Bs.x * sizeof( magmaDoubleComplex ); 

    dim3 Gs2( (n+BLOCK_SIZE+1)/BLOCK_SIZE );


    magma_zpipelined_correction<<<Gs, Bs, Ms>>>( n, k, skp, r, v );
    magma_zpipelined_copyscale<<<Gs2, Bs, 0>>>( n, k, skp, r, v );



    return MAGMA_SUCCESS;
}




