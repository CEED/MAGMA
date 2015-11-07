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
        if( threadIdx.x >= overlap ) { // only write back the lower subdomain
            x[index] = local_x[threadIdx.x];
        }
    }   
}


__global__ void
magma_zbajac_csr_o_ls_kernel8(int localiters, int n, 
                             int matrices, int overlap, 
                            magmaDoubleComplex * valD0, magma_index_t * rowD0, magma_index_t * colD0, magmaDoubleComplex * valR0, magma_index_t * rowR0, magma_index_t * colR0, 
                            magmaDoubleComplex * valD1, magma_index_t * rowD1, magma_index_t * colD1, magmaDoubleComplex * valR1, magma_index_t * rowR1, magma_index_t * colR1, 
                            magmaDoubleComplex * valD2, magma_index_t * rowD2, magma_index_t * colD2, magmaDoubleComplex * valR2, magma_index_t * rowR2, magma_index_t * colR2, 
                            magmaDoubleComplex * valD3, magma_index_t * rowD3, magma_index_t * colD3, magmaDoubleComplex * valR3, magma_index_t * rowR3, magma_index_t * colR3, 
                            magmaDoubleComplex * valD4, magma_index_t * rowD4, magma_index_t * colD4, magmaDoubleComplex * valR4, magma_index_t * rowR4, magma_index_t * colR4, 
                            magmaDoubleComplex * valD5, magma_index_t * rowD5, magma_index_t * colD5, magmaDoubleComplex * valR5, magma_index_t * rowR5, magma_index_t * colR5, 
                            magmaDoubleComplex * valD6, magma_index_t * rowD6, magma_index_t * colD6, magmaDoubleComplex * valR6, magma_index_t * rowR6, magma_index_t * colR6, 
                            magmaDoubleComplex * valD7, magma_index_t * rowD7, magma_index_t * colD7, magmaDoubleComplex * valR7, magma_index_t * rowR7, magma_index_t * colR7, 
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
    
    if( blockIdx.x%matrices==0 ){
        valR = valR7; valD = valD7; colR = colR7; rowR = rowR7; colD = colD7; rowD = rowD7;
    }else if ( blockIdx.x%matrices==1 ) {
        valR = valR6; valD = valD6; colR = colR6; rowR = rowR6; colD = colD6; rowD = rowD6;
    }else if ( blockIdx.x%matrices==2 ) {
        valR = valR5; valD = valD5; colR = colR5; rowR = rowR5; colD = colD5; rowD = rowD5;
    }else if ( blockIdx.x%matrices==3 ) {
        valR = valR4; valD = valD4; colR = colR4; rowR = rowR4; colD = colD4; rowD = rowD4;
    }else if ( blockIdx.x%matrices==4 ) {
        valR = valR3; valD = valD3; colR = colR3; rowR = rowR3; colD = colD3; rowD = rowD3;
    }else if ( blockIdx.x%matrices==5 ) {
        valR = valR2; valD = valD2; colR = colR2; rowR = rowR2; colD = colD2; rowD = rowD2;
    }else if ( blockIdx.x%matrices==6 ) {
        valR = valR1; valD = valD1; colR = colR1; rowR = rowR1; colD = colD1; rowD = rowD1;
    }else if ( blockIdx.x%matrices==7 ) {
        valR = valR0; valD = valD0; colR = colR0; rowR = rowR0; colD = colD0; rowD = rowD0;
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
        if( threadIdx.x >= overlap ) { // only write back the lower subdomain
            x[index] = local_x[threadIdx.x];
        }
    }   
}





__global__ void
magma_zbajac_csr_o_ls_kernel1(int localiters, int n, 
                             int matrices, int overlap, 
                            magmaDoubleComplex * valD, 
                            magma_index_t * rowD, 
                            magma_index_t * colD, 
                            magmaDoubleComplex * valR, 
                            magma_index_t * rowR,
                            magma_index_t * colR, 
                            const magmaDoubleComplex *  __restrict__ b,                            
                            magmaDoubleComplex * x )
{
    int inddiag =  blockIdx.x*blockDim.x;
    int index   =  blockIdx.x*blockDim.x+threadIdx.x;
    int i, j, start, end;
    //bool leaveLoop = false;
    

    if ( index>-1 && index < n ) {
        start = rowR[index];
        end   = rowR[index+1];
       // printf("block:%d  index:%d  n:%d start:%d end:%d\n", blockIdx.x, index, n, start, end);

        magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
        magmaDoubleComplex bl, tmp = zero, v = zero; 


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
        if( threadIdx.x >= overlap ) { // only write back the lower subdomain
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
        } else if (matrices == 8){
            int dimgrid1 = magma_ceildiv( size * blocksize1/(blocksize1-overlap) , blocksize1 );
            int dimgrid2 = 1;
            int dimgrid3 = 1;
            dim3 grid( dimgrid1, dimgrid2, dimgrid3 );
            dim3 block( blocksize1, blocksize2, 1 );
            magma_zbajac_csr_o_ls_kernel8<<< grid, block, 0, queue >>>
                ( localiters, size, matrices, overlap,
                    D[0].dval, D[0].drow, D[0].dcol, R[0].dval, R[0].drow, R[0].dcol, 
                    D[1].dval, D[1].drow, D[1].dcol, R[1].dval, R[1].drow, R[1].dcol,
                    D[2].dval, D[2].drow, D[2].dcol, R[2].dval, R[2].drow, R[2].dcol,
                    D[3].dval, D[3].drow, D[3].dcol, R[3].dval, R[3].drow, R[3].dcol,
                    D[4].dval, D[4].drow, D[4].dcol, R[4].dval, R[4].drow, R[4].dcol,
                    D[5].dval, D[5].drow, D[5].dcol, R[5].dval, R[5].drow, R[5].dcol,
                    D[6].dval, D[6].drow, D[6].dcol, R[6].dval, R[6].drow, R[6].dcol,
                    D[7].dval, D[7].drow, D[7].dcol, R[7].dval, R[7].drow, R[7].dcol,
                    b.dval, x->dval );  
               //magma_zbajac_csr_o_ls_kernel<<< grid, block, 0, queue >>>
               // ( localiters, size, matrices, overlap, D, R, b.dval, x->dval );
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
               //magma_zbajac_csr_o_ls_kernel<<< grid, block, 0, queue >>>
               // ( localiters, size, matrices, overlap, D, R, b.dval, x->dval );
        } else{
           printf("error: invalid matrix count.\n");
        }


    }
    else {
            printf("error: all elements in diagonal block.\n");
    }
    return MAGMA_SUCCESS;
}
