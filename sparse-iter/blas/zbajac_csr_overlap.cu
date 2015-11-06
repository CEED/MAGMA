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


__global__ void
magma_zbajac_csr_o_ls_kernel(int localiters, int n, 
                            unsigned int* locks,
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
                            magmaDoubleComplex * x,
                            magmaDoubleComplex *y )
{
    int inddiag =  blockIdx.x*blockDim.x/2-blockDim.x/2;
    int index = blockIdx.x*blockDim.x/2+threadIdx.x-blockDim.x/2;
    int i, j, start, end;
    //bool leaveLoop = false;
    
    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
    magmaDoubleComplex bl, tmp = zero, v = zero; 
    magmaDoubleComplex *valR, *valD;
    magma_index_t *colR, *rowR, *colD, *rowD;
    
    if( blockIdx.x%2==0 ){
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

    if ( index>-1 && index < n && threadIdx.x > 127 ) {
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
                tmp += valD[i] * local_x[ colD[i] - inddiag];

            local_x[threadIdx.x] +=  ( v - tmp) / (valD[start]);
        }
        if( threadIdx.x > 127 ) { // only write back the lower subdomain
            y[index] = local_x[threadIdx.x];
        }
    }   
}



__global__ void
magma_zbajac_csr_o_kernel(    
    int n, 
    magmaDoubleComplex * valD, 
    magma_index_t * rowD, 
    magma_index_t * colD, 
    magmaDoubleComplex * valR, 
    magma_index_t * rowR,
    magma_index_t * colR, 
    magmaDoubleComplex * b,                                
    magmaDoubleComplex * x )
{
    int index = blockIdx.x*blockDim.x/2+threadIdx.x;
    int i, start, end;   

    if (index < n) {
        magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
        magmaDoubleComplex bl, tmp = zero, v = zero; 

#if (__CUDA_ARCH__ >= 350) && (defined(PRECISION_d) || defined(PRECISION_s))
        bl = __ldg( b+index );
#else
        bl = b[index];
#endif

        start = rowR[index];
        end   = rowR[index+1];

        #pragma unroll
        for( i=start; i<end; i++ )
             v += valR[i] * x[ colR[i] ];

        v =  bl - v;

        start = rowD[index];
        end   = rowD[index+1];

        #pragma unroll
        for( i=start; i<end; i++ )
            tmp += valD[i] * x[ colD[i] ];

        x[index] = x[index] + ( v - tmp ) / (valD[start]); 
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
    magma_z_matrix D1,
    magma_z_matrix R1,
    magma_z_matrix D2,
    magma_z_matrix R2,
    magma_z_matrix b,
    magma_z_matrix *x,
    magma_queue_t queue )
{
    
    //initialize the locks array on the GPU to (0...0)
    unsigned int* locks;
    //unsigned int zeros[D1.num_rows]; for (int i = 0; i < D1.num_rows; i++) {zeros[D1.num_rows] = 0u;}
    //cudaMalloc((void**)&locks, sizeof(unsigned int)*D1.num_rows);
    //cudaMemcpy(locks, zeros, sizeof(unsigned int)*D1.num_rows, cudaMemcpyHostToDevice);
    magmaDoubleComplex *y, *tmp;
    cudaMalloc((void**)&y, sizeof(magmaDoubleComplex)*D1.num_rows);
    
    int blocksize1 = BLOCKSIZE;
    int blocksize2 = 1;

    int dimgrid1 = magma_ceildiv(  2*D1.num_rows, blocksize1 );
    int dimgrid2 = 1;
    int dimgrid3 = 1;
    
    dim3 grid( dimgrid1, dimgrid2, dimgrid3 );
    dim3 block( blocksize1, blocksize2, 1 );
    if ( R1.nnz > 0 && R2.nnz > 0 ) { 
        magma_zbajac_csr_o_ls_kernel<<< grid, block, 0, queue >>>
            ( localiters, D1.num_rows, locks,
                D1.dval, D1.drow, D1.dcol, R1.dval, R1.drow, R1.dcol,
                D2.dval, D2.drow, D2.dcol, R2.dval, R2.drow, R2.dcol, 
                b.dval, x->dval, y );    
            
        tmp = x->dval;
        x->dval = y;
        y = tmp;
    }
    else {
        printf("error: all elements in diagonal block.\n");
    }
    cudaFree(y);
    //cudaFree(locks);
    return MAGMA_SUCCESS;
}
