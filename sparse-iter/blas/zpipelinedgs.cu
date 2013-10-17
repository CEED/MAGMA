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
magma_zdotgpumem(
                        int n, 
                        magmaDoubleComplex *v,
                        magmaDoubleComplex *r,
                        magmaDoubleComplex *vtmp){

  int IDx = threadIdx.x;   
  int i   = blockIdx.x * blockDim.x + threadIdx.x; 

  extern __shared__ magmaDoubleComplex temp[];  

    temp[ IDx ] = ( i < n ) ? v[ i ] * r[ i ] : MAGMA_Z_MAKE( 0.0, 0.0);

    __syncthreads();

    if ( IDx < 128 ){ temp[ IDx ] += temp[ IDx + 128 ];__syncthreads(); }
    if ( IDx < 64 ) { temp[ IDx ] += temp[ IDx +  64 ];__syncthreads(); }

    if( IDx < 32 ){
       temp[ IDx ] += temp[ IDx + 32 ];__syncthreads();    
       temp[ IDx ] += temp[ IDx + 16 ];__syncthreads(); 
       temp[ IDx ] += temp[ IDx +  8 ];__syncthreads(); 
       temp[ IDx ] += temp[ IDx +  4 ];__syncthreads(); 
       temp[ IDx ] += temp[ IDx +  2 ];__syncthreads(); 
       temp[ IDx ] += temp[ IDx +  1 ];__syncthreads();      
   }

   if( IDx == 0)  
     vtmp[ blockIdx.x ]= temp[0];

}


__global__ void 
magma_zreducegpumem(       int vectorSize, 
                           magmaDoubleComplex *vtmp,
                           magmaDoubleComplex *vtmp2 ){

    extern __shared__ magmaDoubleComplex temp[];    

    unsigned int IDx = threadIdx.x;
    unsigned int blockSize = 128; 
    unsigned int i = blockIdx.x * ( blockSize * 2 ) + threadIdx.x;   
    unsigned int gridSize = blockSize  * 2 * gridDim.x;

    
    temp[ IDx ] = MAGMA_Z_MAKE( 0.0, 0.0); 
    while (i < vectorSize ) {
        temp[ IDx ] += vtmp[i]; 
        temp[ IDx ] += ( i + blockSize < vectorSize ) ? vtmp[ i + blockSize ] : MAGMA_Z_MAKE( 0.0, 0.0); 
        i += gridSize;
    }
   __syncthreads();

    if (IDx < 64) { temp[ IDx ] += temp[ IDx + 64 ]; } __syncthreads(); 

    if( IDx < 32 ){
       temp[ IDx ] += temp[ IDx + 32 ];__syncthreads();    
       temp[ IDx ] += temp[ IDx + 16 ];__syncthreads(); 
       temp[ IDx ] += temp[ IDx +  8 ];__syncthreads(); 
       temp[ IDx ] += temp[ IDx +  4 ];__syncthreads(); 
       temp[ IDx ] += temp[ IDx +  2 ];__syncthreads(); 
       temp[ IDx ] += temp[ IDx +  1 ];__syncthreads();      
   }

   if( IDx == 0)  
     vtmp2[ blockIdx.x ]= temp[0];

}


__global__ void 
zp1gmres_(             int  n, 
                       int  k, 
                       magmaDoubleComplex *skp, 
                       magmaDoubleComplex *v, 
                       magmaDoubleComplex *z){
    int row = blockDim.x * blockIdx.x + threadIdx.x ;
    if(row < n ){
        magmaDoubleComplex z_local;
        z_local = z[ row ] - skp[ 0 ] * v[ row ];
        for(int i=1; i<k-1; i++){
            z_local = z_local - skp[i] * v[row+i*n];
        }
        z[row] = z_local - skp[k-1] * v[ row+(k-1)*n ];
    }
}

__global__ void 
magma_zsaxpygpumem( magmaDoubleComplex *v, 
                    magmaDoubleComplex *r, 
                    magmaDoubleComplex *skp, 
                    int n, int k ){
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   magmaDoubleComplex temp = 1.0 * (*skp);
   if( i < n )
      r[ i ] =  temp * v[ i ] + r[ i ];
}


__global__ void 
magma_zsaxpygpumem_neg( magmaDoubleComplex *v, 
                    magmaDoubleComplex *r, 
                    magmaDoubleComplex *skp, 
                    int n, int k ){
   int i = blockIdx.x * blockDim.x + threadIdx.x;

   magmaDoubleComplex temp = -1.0 * (*skp);
   if( i < n )
      r[ i ] =  temp * v[ i ] + r[ i ];
}


__global__ void 
magma_zgpumemzero(  magmaDoubleComplex *d, int n ){

   int i = blockIdx.x * blockDim.x + threadIdx.x;

   if( i < n )
      d[ i ] = MAGMA_Z_MAKE( 0.0, 0.0 );
}


__global__ void 
magma_zpipelined_correction( magmaDoubleComplex *skp, 
                             magmaDoubleComplex *r,
                             magmaDoubleComplex *v,
                             int n, 
                             int ldh, 
                             int k ){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double zz= 0.0, tmp= 0.0;
   
    if( i == 0 ){
        for( int j=0; j<k; j++ ){
            tmp += MAGMA_Z_REAL( skp[j]*skp[j] );
        }
        printf("skp[k]= %f\n", skp[k]);
        zz = MAGMA_Z_REAL( skp[k] );
        printf("zz= %f\n", zz);
        skp[k] = MAGMA_Z_MAKE( sqrt(zz-tmp),0.0 );
        printf("skp[k]= %f\n", skp[k]);
        for( int j=k+1; j<ldh; j++ ){
            skp[j] = MAGMA_Z_MAKE( 0.0, 0.0 );
        }
    }
    __syncthreads();
    
    if( i<n ){
        v[i] =  r[i] * 1.0/skp[k];

    }

}




/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======

    Computes the scalar product of a set of vectors v_i such that

    skp = ( <v_0,r>, <v_1,r>, ... <v_k-1,r>, <r,r> )

    Returns the vector skp.

    Arguments
    =========

    int n                             legth of v_i
    int k                             # vectors to orthogonalize against
    magmaDoubleComplex *v                     v = (v_0 .. v_i.. v_k-1)
    magmaDoubleComplex *r                     r
    magmaDoubleComplex *skp                   vector of scalar products (<v_i,r>...<r,r>)

    =====================================================================  */

extern "C" magma_int_t
magma_zmergedgs(    int n, 
                    int ldh, 
                    int k,
                    magmaDoubleComplex *v, 
                    magmaDoubleComplex *r,
                    magmaDoubleComplex *skp ){

    
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( (n+BLOCK_SIZE+1)/BLOCK_SIZE );
    unsigned int Ms =   Bs.x * sizeof( magmaDoubleComplex ); 
    dim3 Gs_next;

    magmaDoubleComplex *d1, *d2;
    magma_zmalloc( &d1, n );
    magma_zmalloc( &d2, n );

    magmaDoubleComplex *aux1 = d1, *aux2 = d2;
    int b = 1;        
    dim3 Bs_2; Bs_2.x = (Bs.x/2);    

magmaDoubleComplex *print;
    magma_zmalloc_cpu( &print, n ); 
 printf("in loop %d\n", k);

// for the vector r
    magma_zdotgpumem<<<Gs, Bs, Ms>>>( n, r, r, d1);
        while( Gs.x > 1 ){
            Gs_next.x = (Gs.x+Bs.x-1) / Bs.x ;
            if( Gs_next.x == 1 ) Gs_next.x = 2;
            magma_zreducegpumem<<< Gs_next.x/2, Bs.x/2, Ms/2 >>> ( Gs.x, aux1, aux2 );
            Gs_next.x = Gs_next.x/2;
            Gs.x = Gs_next.x;

                    cublasGetVector( n , sizeof( magmaDoubleComplex ), d1, 1, print, 1 ); 
                    printf("skp_tmp:  :   ");
                    for( int l=0; l<5; l++)
                        printf("%f  ", print[l]);
                    printf("\n");
                    cublasGetVector( n , sizeof( magmaDoubleComplex ), d2, 1, print, 1 ); 
                    printf("toa_tmp:  :   ");
                    for( int l=0; l<5; l++)
                        printf("%f  ", print[l]);
                    printf("\n");
            b = 1 - b;
            if( b ){ aux1 = d1; aux2 = d2; }
            else   { aux2 = d1; aux1 = d2; }
        }  
    if( b == 0 )
        cudaMemcpy( skp+k, aux2, sizeof( magmaDoubleComplex ), cudaMemcpyDeviceToDevice);
    else if( b == 1 )
        cudaMemcpy( skp+k, aux1, sizeof( magmaDoubleComplex ), cudaMemcpyDeviceToDevice);
    // reset Grid-size
    Gs.x = ( (n+BLOCK_SIZE+1)/BLOCK_SIZE ); 
    b = 1;
    magma_zgpumemzero<<<Gs, Bs, 0>>>( d1, n );
    magma_zgpumemzero<<<Gs, Bs, 0>>>( d2, n );

magmaDoubleComplex *skp_h;
    magma_zmalloc_cpu( &skp_h, (ldh+1) ); 

                    cublasGetVector( ldh+1 , sizeof( magmaDoubleComplex ), skp, 1, skp_h, 1 ); 
                    printf("skp_dev_2:   ");
                    for( int i=0; i<ldh+1; i++)
                        printf("%f  ", skp_h[i]);
                    printf("\n");

    
    // for the k vectors v(i)
    for(int i=0; i<k; i++){
                
                    cublasGetVector( n , sizeof( magmaDoubleComplex ), d1, 1, print, 1 ); 
                    printf("vector v(%d)\n",i);
                    //printf("d_1:   ");
                    //for( int l=0; l<n; l++)
                    //    printf("%f  ", print[l]);
                    //printf("\n");
                    cublasGetVector( n , sizeof( magmaDoubleComplex ), d2, 1, print, 1 ); 
                    //printf("d_2:   ");
                    //for( int l=0; l<n; l++)
                    //    printf("%f  ", print[l]);
                    //printf("\n");
        magma_zdotgpumem<<<Gs, Bs, Ms>>>( n, v+(i*n), r, d1 );

                    cublasGetVector( n , sizeof( magmaDoubleComplex ), d1, 1, print, 1 ); 
                    //printf("d_1:   ");
                    //for( int l=0; l<5; l++)
                    //    printf("%f  ", print[i]);
                    //printf("\n");
                    cublasGetVector( n , sizeof( magmaDoubleComplex ), d2, 1, print, 1 ); 
                    //printf("d_2:   ");
                    //for( int l=0; l<n; l++)
                    //    printf("%f  ", print[l]);
                    //printf("\n");

        while( Gs.x > 1 ){

            //Gs_next.x = (Gs.x+Bs.x-1) / Bs.x ;
Gs_next.x = (unsigned int)ceil( (float) Gs.x / Bs.x ); 
            if( Gs_next.x == 1 ) Gs_next.x = 2;
            magma_zreducegpumem<<< Gs_next.x/2, Bs.x/2, Ms/2 >>> ( Gs.x, aux1, aux2 );
            Gs_next.x = Gs_next.x/2;
                    //printf("in loop Gs.x: %d  Gs_next.x: %d\n", Gs.x, Gs_next.x);
            Gs.x = Gs_next.x;
                    cublasGetVector( n , sizeof( magmaDoubleComplex ), d1, 1, print, 1 ); 
                    printf("skp_tmp:  :   ");
                    for( int l=0; l<5; l++)
                        printf("%f  ", print[l]);
                    printf("\n");
                    cublasGetVector( n , sizeof( magmaDoubleComplex ), d2, 1, print, 1 ); 
                    printf("toa_tmp:  :   ");
                    for( int l=0; l<5; l++)
                        printf("%f  ", print[l]);
                    printf("\n");



            b = 1 - b;
            if( b ){ aux1 = d1; aux2 = d2; }
            else   { aux2 = d1; aux1 = d2; }
        }
        if( b == 0 )
            cudaMemcpy( skp+i, aux2, sizeof( magmaDoubleComplex ), cudaMemcpyDeviceToDevice);
        else if( b == 1 )
            cudaMemcpy( skp+i, aux1, sizeof( magmaDoubleComplex ), cudaMemcpyDeviceToDevice);
        // reset Grid-size
        Gs.x = ( (n+BLOCK_SIZE+1)/BLOCK_SIZE ); 
        b = 1;





magmaDoubleComplex *skp_h;
    magma_zmalloc_cpu( &skp_h, (ldh+1) ); 

                    cublasGetVector( ldh+1 , sizeof( magmaDoubleComplex ), skp, 1, skp_h, 1 ); 
                    printf("skp_dev:   ");
                    for( int j=0; j<ldh+1; j++)
                        printf("%f  ", skp_h[j]);
                    printf("\n");
                    magma_device_sync();

                    cublasGetVector( n , sizeof( magmaDoubleComplex ), r, 1, print, 1 ); 
                    printf("r:  :   ");
                    for( int l=0; l<5; l++)
                        printf("%f  ", print[l]);
                    printf("\n");
                    cublasGetVector( n , sizeof( magmaDoubleComplex ), v, 1, print, 1 ); 
                    printf("v:  :   ");
                    for( int l=0; l<5; l++)
                        printf("%f  ", print[l]);
                    printf("\n");
     //       magma_zaxpy(n, -skp_h[i], r, 1, v, 1); 
magma_zsaxpygpumem_neg<<<Gs, Bs, 0>>>( v, r, &skp[i], n, i );
printf("skp[i]=%f\n",skp[i]);
                    cublasGetVector( n , sizeof( magmaDoubleComplex ), r, 1, print, 1 ); 
                    printf("r:  :   ");
                    for( int l=0; l<5; l++)
                        printf("%f  ", print[l]);
                    printf("\n");

        

    }
                    magma_device_sync();
    

    magma_zpipelined_correction<<<Gs, Bs, 0>>>( skp, r, v+(k*n), n, ldh, k );

               printf("zz correction   ");

                    cublasGetVector( ldh+1 , sizeof( magmaDoubleComplex ), skp, 1, skp_h, 1 ); 
                    printf("skp_dev_3:   ");
                    for( int i=0; i<ldh+1; i++)
                        printf("%f  ", skp_h[i]);
                    printf("\n");

    magma_free(d1);
    magma_free(d2);

   return MAGMA_SUCCESS;
}




