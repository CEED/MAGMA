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




__global__ void 
magma_zmemreader_kernel1(  
                    int n, 
                    magmaDoubleComplex *a,
                    magmaDoubleComplex *b ){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int local_index = threadIdx.x;


    if( index<n ){
        a[index] =  a[index] + b[index];
    }

}

__global__ void 
magma_zmemreader_kernel2(  
                    int n, 
                    magmaDoubleComplex *a,
                    magmaDoubleComplex *b ){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int local_index = threadIdx.x;


    if( index<n ){
        a[index] =  a[index] + b[local_index];
    }

}


__global__ void 
magma_zmemreader_kernel3(  
                    int n, 
                    magmaDoubleComplex *a,
                    magmaDoubleComplex *b ){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int local_index = threadIdx.x;


    if( index<n ){
        if(local_index == 0 ){
            a[blockIdx.x]=b[0];
            b[0]=b[0]+MAGMA_Z_MAKE(1.0, 0.0);
        }
    }

}



extern "C" magma_int_t
magma_ztestkernel(  int n ){

    magmaDoubleComplex one  = MAGMA_Z_MAKE(1.0, 0.0);
    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
    double start, end, shared1, shared2;
    magmaDoubleComplex *a, *b, *a_host;

    printf( "# n  shared1: %.2e shared2 %.2e \n", shared1, shared2 );

    for(int nn=100000; nn<n; nn=nn+100000){

    // init DEV vectors
    magma_zmalloc( &a, nn );
    magma_zmalloc( &b, nn );

    magma_zmalloc_cpu( &a_host, nn );

    for( int i=0; i<(nn); i++){
        a_host[i] = (MAGMA_Z_MAKE(0.0, 0.0));
    }
        magma_zsetvector( nn, a_host, 1, a, 1 );
        magma_zsetvector( nn, a_host, 1, b, 1 );

    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( (nn+BLOCK_SIZE-1)/BLOCK_SIZE );

    magma_device_sync(); start = magma_wtime(); 
    magma_zmemreader_kernel1<<<Gs, Bs, 0>>>( nn, a, b );
    magma_device_sync(); end = magma_wtime(); 
    shared1 = end-start;

    magma_device_sync(); start = magma_wtime(); 
    magma_zmemreader_kernel2<<<Gs, Bs, 0>>>( nn, a, b );
    magma_device_sync(); end = magma_wtime(); 
    shared2 = end-start;

    printf( " %d  %.2e  %.2e\n", nn, shared1, shared2 );

    magma_free( a );
    magma_free( b );


    magma_free_cpu( a_host );

    }
 /*
    magma_zmemreader_kernel3<<<Gs, Bs, 0>>>( n, a, b );


    magma_zgetvector( n, a, 1, a_host, 1 );

    printf("update order using %d thread blocks:\n", Gs.x);
    for( int i=0; i<(n/BLOCK_SIZE); i++)
        printf(" %d ", (int)(MAGMA_Z_REAL(a_host[i])));
*/


   return MAGMA_SUCCESS;
}
