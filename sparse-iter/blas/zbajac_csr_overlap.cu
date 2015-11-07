/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s

*/
#include "common_magmasparse.h"

#define PRECISION_z
#define BLOCKSIZE 256

__global__ void magma_zk_testLocking(unsigned int* locks, int n) {
    int id = threadIdx.x % n;
    bool leaveLoop = false;
    while (!leaveLoop) {
        if (atomicExch(&(locks[id]), 1u) == 0u) {
            //critical section
            leaveLoop = true;
            atomicExch(&(locks[id]),0u);
        }
    } 
}

/*
__global__ void
magma_zbajac_csr_o_ls_kernel(int localiters, int n, 
                             int matrices, int overlap, 
                             magma_z_matrix *D, magma_z_matrix *R,
                             const magmaDoubleComplex *  __restrict__ b,                            
                             magmaDoubleComplex * x )
{
   // int inddiag =  blockIdx.x*(blockDim.x - overlap) - overlap;
   // int index   =  blockIdx.x*(blockDim.x - overlap) - overlap + threadIdx.x;
        int inddiag =  blockIdx.x*blockDim.x/2-blockDim.x/2;
    int index   = blockIdx.x*blockDim.x/2+threadIdx.x-blockDim.x/2;
    int i, j, start, end;
    
     __shared__ magmaDoubleComplex local_x[ BLOCKSIZE ];
    
    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
    magmaDoubleComplex bl, tmp = zero, v = zero; 
    magmaDoubleComplex *valR, *valD;
    magma_index_t *colR, *rowR, *colD, *rowD;
    
    //valR = R[ (1+blockIdx.x-1)%matrices ].dval;
    //colR = R[ (1+blockIdx.x-1)%matrices ].dcol;
    //rowR = R[ (1+blockIdx.x-1)%matrices ].drow;
    //valD = D[ (1+blockIdx.x-1)%matrices ].dval;
    //colD = D[ (1+blockIdx.x-1)%matrices ].dcol;
    //rowD = D[ (1+blockIdx.x-1)%matrices ].drow;
    
        if( blockIdx.x%2==1 ){
        valR = R[0].dval;
        valD = D[0].dval;
        colR = R[0].dcol;
        rowR = R[0].drow;
        colD = D[0].dcol;
        rowD = D[0].drow;
    }else{
        valR = R[1].dval;
        valD = D[1].dval;
        colR = R[1].dcol;
        rowR = R[1].drow;
        colD = D[1].dcol;
        rowD = D[1].drow;
    }

    if ( index>-1 && index < n ) {
        start = rowR[index];
        end   = rowR[index+1];
printf("bdx:%d idx:%d  start:%d  end:%d\n", blockIdx.x, threadIdx.x, start, end);

#if (__CUDA_ARCH__ >= 350) && (defined(PRECISION_d) || defined(PRECISION_s))
        bl = __ldg( b+index );
#else
        bl = b[index];
#endif


     #pragma unroll
     for( i=start; i<end; i++ )
          v += valR[i] * x[ colR[i] ];

     start = rowD[index];
     end   = rowD[index+1];

     #pragma unroll
     for( i=start; i<end; i++ )
         tmp += valD[i] * x[ colD[i] ];

     v =  bl - v;

     // add more local iterations            

     local_x[threadIdx.x] = x[index] ;//+ ( v - tmp);// / (valD[start]);
   __syncthreads();

     #pragma unroll
     for( j=0; j<localiters-1; j++ )
     {
         tmp = zero;
         #pragma unroll
         for( i=start; i<end; i++ )
             tmp += valD[i] * local_x[ colD[i] - inddiag];
     
         local_x[threadIdx.x] +=  ( v - tmp) / (valD[start]);
     }
     if( threadIdx.x > overlap ) { // RAS
         x[index] = local_x[threadIdx.x];
     }
    }   
}

*/
__global__ void
magma_zbajac_csr_o_ls_kernel(int localiters, int n, 
                             int matrices, int overlap, 
                            magmaDoubleComplex * valD1, 
                            magma_index_t * rowD1, 
                            magma_index_t * colD1, 
                            magmaDoubleComplex * valR1, 
                            magma_index_t * rowR1,
                            magma_index_t * colR1, 
                            magmaDoubleComplex * valD2, 
                            magma_index_t * rowD2, 
                            magma_index_t * colD2, 
                            magmaDoubleComplex * valR2, 
                            magma_index_t * rowR2,
                            magma_index_t * colR2, 
                            const magmaDoubleComplex *  __restrict__ b,                            
                            magmaDoubleComplex * x )
{
    int inddiag =  blockIdx.x*blockDim.x/2-blockDim.x/2;
    int index   = blockIdx.x*blockDim.x/2+threadIdx.x-blockDim.x/2;
    int i, j, start, end;
    //bool leaveLoop = false;
    
    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
    magmaDoubleComplex bl, tmp = zero, v = zero; 
    magmaDoubleComplex *valR, *valD;
    magma_index_t *colR, *rowR, *colD, *rowD;
    
    if( blockIdx.x%2==1 ){
        valR = valR1;
        valD = valD1;
        colR = colR1;
        rowR = rowR1;
        colD = colD1;
        rowD = rowD1;
    }else{
        valR = valR2; 
        valD = valD2;
        colR = colR2;
        rowR = rowR2;
        colD = colD2;
        rowD = rowD2;
    }

    if ( index>-1 && index < n ) {
        start = rowR[index];
        end   = rowR[index+1];


#if (__CUDA_ARCH__ >= 350) && (defined(PRECISION_d) || defined(PRECISION_s))
        bl = __ldg( b+index );
#else
        bl = b[index];
#endif


        #pragma unroll
        for( i=start; i<end; i++ )
             v += valR[i] * x[ colR[i] ];

        start = rowD[index];
        end   = rowD[index+1];

        #pragma unroll
        for( i=start; i<end; i++ )
            tmp += valD[i] * x[ colD[i] ];

        v =  bl - v;

        // add more local iterations            
        __shared__ magmaDoubleComplex local_x[ BLOCKSIZE ];
        local_x[threadIdx.x] = x[index] + ( v - tmp) / (valD[start]);
        __syncthreads();

        #pragma unroll
        for( j=0; j<localiters-1; j++ )
        {
            tmp = zero;
            #pragma unroll
            for( i=start; i<end; i++ )
                tmp += valD[i] * local_x[ colD[i] - inddiag];
        
            local_x[threadIdx.x] +=  ( v - tmp) / (valD[start]);
        }
        if( threadIdx.x > overlap ) { // only write back the lower subdomain
            x[index] = local_x[threadIdx.x];
        }
    }   
}





__global__ void
magma_zbajac_csr_o_ls_kernel1(int localiters, int n, 
                             int matrices, int overlap, 
                            magmaDoubleComplex * valD1, 
                            magma_index_t * rowD1, 
                            magma_index_t * colD1, 
                            magmaDoubleComplex * valR1, 
                            magma_index_t * rowR1,
                            magma_index_t * colR1, 
                            const magmaDoubleComplex *  __restrict__ b,                            
                            magmaDoubleComplex * x )
{
    int inddiag =  blockIdx.x*blockDim.x;
    int index   = blockIdx.x*blockDim.x+threadIdx.x;
    int i, j, start, end;
    //bool leaveLoop = false;
    
    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
    magmaDoubleComplex bl, tmp = zero, v = zero; 
    magmaDoubleComplex *valR, *valD;
    magma_index_t *colR, *rowR, *colD, *rowD;
    
    valR = valR1;
    valD = valD1;
    colR = colR1;
    rowR = rowR1;
    colD = colD1;
    rowD = rowD1;

    if ( index>-1 && index < n ) {
        start = rowR[index];
        end   = rowR[index+1];


#if (__CUDA_ARCH__ >= 350) && (defined(PRECISION_d) || defined(PRECISION_s))
        bl = __ldg( b+index );
#else
        bl = b[index];
#endif


        #pragma unroll
        for( i=start; i<end; i++ )
             v += valR[i] * x[ colR[i] ];

        start = rowD[index];
        end   = rowD[index+1];

        #pragma unroll
        for( i=start; i<end; i++ )
            tmp += valD[i] * x[ colD[i] ];

        v =  bl - v;

        // add more local iterations            
        __shared__ magmaDoubleComplex local_x[ BLOCKSIZE ];
        local_x[threadIdx.x] = x[index] + ( v - tmp) / (valD[start]);
        __syncthreads();

        #pragma unroll
        for( j=0; j<localiters-1; j++ )
        {
            tmp = zero;
            #pragma unroll
            for( i=start; i<end; i++ )
                tmp += valD[i] * local_x[ colD[i] - inddiag];
        
            local_x[threadIdx.x] +=  ( v - tmp) / (valD[start]);
        }
        if( threadIdx.x > overlap ) { // only write back the lower subdomain
            x[index] = local_x[threadIdx.x];
        }
    }   
}





/**
    Purpose
    -------
    
    This routine is a block-asynchronous Jacobi iteration 
    with directed restricted additive Schwarz overlap (top-down) performing s
    local Jacobi-updates within the block. Input format is two CSR matrices,
    one containing the diagonal blocks, one containing the rest.

    Arguments
    ---------

    @param[in]
    localiters  magma_int_t
                number of local Jacobi-like updates

    @param[in]
    D1          magma_z_matrix
                input matrix with diagonal blocks

    @param[in]
    R1          magma_z_matrix
                input matrix with non-diagonal parts
                
    @param[in]
    D2          magma_z_matrix
                input matrix with diagonal blocks

    @param[in]
    R2          magma_z_matrix
                input matrix with non-diagonal parts

    @param[in]
    b           magma_z_matrix
                RHS

    @param[in]
    x           magma_z_matrix*
                iterate/solution

    
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" magma_int_t
magma_zbajac_csr_overlap(
    magma_int_t localiters,
    magma_int_t matrices,
    magma_int_t overlap,
    magma_z_matrix *D,
    magma_z_matrix *R,
    magma_z_matrix b,
    magma_z_matrix *x,
    magma_queue_t queue )
{
    
    
    int blocksize1 = BLOCKSIZE;
    int blocksize2 = 1;
    int size = D[0].num_rows;
    int min_nnz=100;
    

    
    for(int i=0; i<matrices; i++){
       min_nnz = min(min_nnz, R[i].nnz);   
    }
    
    if( min_nnz > 0 ){
        if( matrices == 1 ){
            int dimgrid1 = magma_ceildiv( size  , blocksize1 );
            int dimgrid2 = 1;
            int dimgrid3 = 1;
            dim3 grid( dimgrid1, dimgrid2, dimgrid3 );
            dim3 block( blocksize1, blocksize2, 1 );
            magma_zbajac_csr_o_ls_kernel1<<< grid, block, 0, queue >>>
            ( localiters, size, matrices, overlap,
            D[0].dval, D[0].drow, D[0].dcol, R[0].dval, R[0].drow, R[0].dcol, 
            b.dval, x->dval );  
        } else if (matrices == 2){
            int dimgrid1 = magma_ceildiv( size * blocksize1/(blocksize1-overlap) , blocksize1 );
            int dimgrid2 = 1;
            int dimgrid3 = 1;
            dim3 grid( dimgrid1, dimgrid2, dimgrid3 );
            dim3 block( blocksize1, blocksize2, 1 );
            magma_zbajac_csr_o_ls_kernel<<< grid, block, 0, queue >>>
                ( localiters, size, matrices, overlap,
                    D[0].dval, D[0].drow, D[0].dcol, R[0].dval, R[0].drow, R[0].dcol, 
                    D[1].dval, D[1].drow, D[1].dcol, R[1].dval, R[1].drow, R[1].dcol,
                    b.dval, x->dval );  
        } else
           printf("error: invalid matrix count.\n");
          // magma_zbajac_csr_o_ls_kernel<<< grid, block, 0, queue >>>
            //    ( localiters, size, matrices, overlap, D, R, b.dval, x->dval );


    }
    else {
            printf("error: all elements in diagonal block.\n");
    }
    return MAGMA_SUCCESS;
}
