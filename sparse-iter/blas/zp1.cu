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

#if (GPUSHMEM < 200)
   #define BLOCK_SIZE 128
#else
   #define BLOCK_SIZE 512
#endif




__global__ void 
zp1gmres_mgs(          int  n, 
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



/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======

    Orthogonalizes the vector z agains all vectors v_0...v_k
    using the scalar products in skp. This is part of classical Gram-Schmidt.

    z = z - sum skp(i) * v_i = z - sum <v_i,z> v_i

    Returns the vector z.

    Arguments
    =========

    magma_int_t n                             legth of v_i
    magma_int_t k                             # vectors to orthogonalize against
    magmaDoubleComplex *v                     v = (v_0 .. v_i.. v_k-1)
    magmaDoubleComplex *z                     vector to orthogonalize

    =====================================================================  */

extern "C" magma_int_t
magma_zp1gmres_mgs(    magma_int_t  n, 
                       magma_int_t  k, 
                       magmaDoubleComplex *skp, 
                       magmaDoubleComplex *v, 
                       magmaDoubleComplex *z){

   dim3 grid( (n+BLOCK_SIZE-1)/BLOCK_SIZE, 1, 1);

   zp1gmres_mgs<<< grid, BLOCK_SIZE, 0 >>>( n, k, skp, v, z );

   return MAGMA_SUCCESS;
}




