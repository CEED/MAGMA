/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s

*/

#include "common_magma.h"
#include "../include/magmasparse_z.h"
#include "../../include/magma.h"


#define PRECISION_z
#define BLOCKSIZE 256


__global__ void 
magma_zbajac_csr_ls_kernel(int localiters, int n, 
                            magmaDoubleComplex *valD, 
                            magma_index_t *rowD, 
                            magma_index_t *colD, 
                            magmaDoubleComplex *valR, 
                            magma_index_t *rowR,
                            magma_index_t *colR, 
                            magmaDoubleComplex *b,                                
                            magmaDoubleComplex *x ){

    int ind_diag =  blockIdx.x*blockDim.x;
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    int i, j, start, end;   

    if(index<n){
    
        start=rowR[index];
        end  =rowR[index+1];
        
        magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
        magmaDoubleComplex tmp = zero, v = zero; 

        #pragma unroll
        for( i=start; i<end; i++ )
             v += valR[i] * x[ colR[i] ];

        start=rowD[index];
        end  =rowD[index+1];

        #pragma unroll
        for( i=start; i<end; i++ )
            tmp += valD[i] * x[ colD[i] ];

        v =  b[index] - v;

        /* add more local iterations */           
        __shared__ magmaDoubleComplex local_x[ BLOCKSIZE ];
        local_x[threadIdx.x] = x[index] + ( v - tmp) / (valD[start]);
        __syncthreads();

        #pragma unroll
        for( j=0; j<localiters; j++ )
        {
            tmp = zero;
            #pragma unroll
            for( i=start; i<end; i++ )
                tmp += valD[i] * local_x[ colD[i] - ind_diag];

            local_x[threadIdx.x] +=  ( v - tmp) / (valD[start]);
        }
        x[index] = local_x[threadIdx.x];
    }
}



__global__ void 
magma_zbajac_csr_kernel(    int n, 
                            magmaDoubleComplex *valD, 
                            magma_index_t *rowD, 
                            magma_index_t *colD, 
                            magmaDoubleComplex *valR, 
                            magma_index_t *rowR,
                            magma_index_t *colR, 
                            magmaDoubleComplex *b,                                
                            magmaDoubleComplex *x ){

    int ind_diag =  blockIdx.x*blockDim.x;
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    int i, j, start, end;   

    if(index<n){
        
        magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
        magmaDoubleComplex tmp = zero, v = zero; 

        start=rowR[index];
        end  =rowR[index+1];

        #pragma unroll
        for( i=start; i<end; i++ )
             v += valR[i] * x[ colR[i] ];

        start=rowD[index];
        end  =rowD[index+1];

        #pragma unroll
        for( i=start; i<end; i++ )
            v += valD[i] * x[ colD[i] ];

        x[index] = ( b[index] - v ) / (valD[start]); 
    }
}









/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

    Purpose
    =======
    
    This routine is a block-asynchronous Jacobi iteration performing s
    local Jacobi-updates within the block. Input format is two CSR matrices,
    one containing the diagonal blocks, one containing the rest.

    Arguments
    =========

    magma_int_t localiters              number of local Jacobi-like updates
    magma_z_sparse_matrix D             input matrix with diagonal blocks
    magma_z_sparse_matrix R             input matrix with non-diagonal parts
    magma_z_vector b                    RHS
    magma_z_vector *x                   iterate/solution
    
    ======================================================================    */

extern "C" magma_int_t
magma_zbajac_csr(   magma_int_t localiters,
                    magma_z_sparse_matrix D,
                    magma_z_sparse_matrix R,
                    magma_z_vector b,
                    magma_z_vector *x ){

    int blocksize1 = BLOCKSIZE;
    int blocksize2 = 1;

    int dimgrid1 = ( D.num_rows + blocksize1 -1 ) / blocksize1;
    int dimgrid2 = 1;
    int dimgrid3 = 1;

    dim3 grid( dimgrid1, dimgrid2, dimgrid3 );
    dim3 block( blocksize1, blocksize2, 1 );

    if( localiters == 1 )
    magma_zbajac_csr_kernel<<< grid, block, 0, magma_stream >>>
        ( D.num_rows, D.val, D.row, D.col, 
                        R.val, R.row, R.col, b.val, x->val );
    else
        magma_zbajac_csr_ls_kernel<<< grid, block, 0, magma_stream >>>
        ( localiters, D.num_rows, D.val, D.row, D.col, 
                        R.val, R.row, R.col, b.val, x->val );

    return MAGMA_SUCCESS;
}



