/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @author Tingxing Dong
       @author Azzam Haidar

       @precisions normal z -> s d c
*/


#ifndef MAGMABLAS_TRSV_TEMPLATE_H
#define MAGMABLAS_TRSV_TEMPLATE_H

#include "gemm_template_device_defs.cuh"// use make_FloatingPoint

#define PRECISION_z

#include "gemv_template_kernel_batched.cuh"





#define A(i, j)  (A + (i) + (j)*lda)   // A(i, j) means at i row, j column

extern __shared__ magmaDoubleComplex shared_data[];


//==============================================================================
/*
   used in upper nontranspose and lower transpose
*/
template<magma_trans_t transA, magma_diag_t diag> 
static __device__ void 
ztrsv_backwards_tri_device( int n,
    const magmaDoubleComplex * __restrict__ A, int lda, 
    magmaDoubleComplex       * __restrict__ b, int incb,
    magmaDoubleComplex *sx)

{
/*
   assume sx is in shared memory
*/
    int tx = threadIdx.x; 
    magmaDoubleComplex a; 

    for(int step=0; step<n; step++)
    {
        if(tx < n) 
        {

            if(transA == MagmaNoTrans)
            {
                 a = A[ (n-1) + (n-1) * lda - tx - step * lda];//rowwise access data in a coalesced way
            }
            else if(transA == MagmaTrans)
            {    
                 a = A[ (n-1) + (n-1) * lda - tx * lda  - step];//columwise access data, not in a coalesced way
            }
            else
            {    
                 a = MAGMA_Z_CNJG(A[ (n-1) + (n-1) * lda - tx * lda  - step]);//columwise access data, not in a coalesced way
            }


            if(tx == step)
            {
                if(diag == MagmaUnit)
                {
                    sx[n-1-tx] = (b[n-1-tx] - sx[n-1-tx]); 
                }
                else
                {
                    sx[n-1-tx] = (b[n-1-tx] - sx[n-1-tx])/a; 
                }
            }
         }
         __syncthreads(); //there should be a sych here but can be avoided if BLOCK_SIZE =32

         if(tx < n) 
         {
            if(tx > step)
            {
                sx[n-1-tx] += a * sx[n-1-step];
            }

        }
    }        

}

//==============================================================================
/*
   used in lower nontranspose and upper transpose
*/

template<magma_trans_t transA, magma_diag_t diag> 
static __device__ void 
ztrsv_forwards_tri_device(int n,
    const magmaDoubleComplex * __restrict__ A, int lda, 
    magmaDoubleComplex       * __restrict__ b, int incb,
    magmaDoubleComplex *sx)

{
/*
   assume sx is in shared memory
*/
    int tx = threadIdx.x; 
    magmaDoubleComplex a; 

    for(int step=0; step<n; step++)
    {
        if(tx < n) //hard code to BLOCK_SIZE and test divisible case only make 1Gflop/s difference
        {

            if(transA == MagmaNoTrans)
            {
                 a = A[tx + step * lda];//rowwise access data in a coalesced way
            }
            else  if(transA == MagmaTrans)
            {    
                 a = A[ tx * lda  + step];//columwise access data, not in a coalesced way
            }
            else
            {    
                 a = MAGMA_Z_CNJG(A[ tx * lda  + step]);//columwise access data, not in a coalesced way
            }


            if(tx == step)
            {
                if(diag == MagmaUnit)
                {
                    sx[tx] = (b[tx] - sx[tx]); 
                }
                else
                {
                    sx[tx] = (b[tx] - sx[tx])/a; 
                }
            }
        }
        __syncthreads(); //there should be a sych here but can be avoided if BLOCK_SIZE =32

        if(tx < n) 
        {
            if(tx > step)
            {
                sx[tx] += a * sx[step];
            }

        }
    }        

}
//==============================================================================
template<const int BLOCK_SIZE, const int BLK_X, const int BLK_Y,  const int TILE_SIZE, const int flag, const magma_uplo_t uplo, const magma_trans_t trans, const magma_diag_t diag> 
static __device__ void
ztrsv_notrans_device(
    int n, 
    const magmaDoubleComplex * __restrict__ A, int lda,
    magmaDoubleComplex *b, int incb, 
    magmaDoubleComplex *x)
{


    int tx = threadIdx.x;
    int col = n;
    magmaDoubleComplex *sx = (magmaDoubleComplex*)shared_data;

    if(flag == 0)
    {
        for( int j = tx; j < n; j += BLOCK_SIZE )
        { 
            sx[j] = MAGMA_Z_ZERO;
        }
    }
    else
    {
        for( int j = tx; j < n; j += BLOCK_SIZE )
        { 
            sx[j] = x[j]; 
        }
    }
    __syncthreads();


    if (uplo == MagmaUpper)
    {
        for(int i=0; i<n; i+=BLOCK_SIZE)
        {    
            int jb = min(BLOCK_SIZE, n-i);
            col -= jb;

            gemvn_template_device<magmaDoubleComplex, BLK_X, BLK_Y, TILE_SIZE>(jb, i, MAGMA_Z_ONE, A(col, col+jb), lda, sx+col+jb, 1, MAGMA_Z_ONE, sx+col, 1);
            __syncthreads();

            ztrsv_backwards_tri_device<trans, diag>(jb, A(col, col), lda, b+col, incb, sx+col);
            __syncthreads();
        }
    }
    else 
    {
        for(int i=0; i<n; i+=BLOCK_SIZE)
        {          
            int jb = min(BLOCK_SIZE, n-i);
            col = i;

            gemvn_template_device<magmaDoubleComplex, BLK_X, BLK_Y, TILE_SIZE>(jb, i, MAGMA_Z_ONE, A(col, 0), lda, sx, 1, MAGMA_Z_ONE, sx+col, 1);
            __syncthreads();

            ztrsv_forwards_tri_device<trans, diag>( jb, A(col, col), lda, b+col, incb, sx+col);
            __syncthreads();
        }
    }


    for( int j = tx; j < n; j += BLOCK_SIZE )
    { 
        x[j] = sx[j]; // write to x in reverse order
    }
    __syncthreads();

}



//==============================================================================

template<const int BLOCK_SIZE, const int BLK_X, const int BLK_Y,  const int TILE_SIZE, const int flag, const magma_uplo_t uplo, const magma_trans_t trans, const magma_diag_t diag > 
static __device__ void
ztrsv_trans_device(
    int n, 
    const magmaDoubleComplex * __restrict__ A, int lda,
    magmaDoubleComplex *b, int incb, 
    magmaDoubleComplex *x)
{
    int tx = threadIdx.x;
    int col = n;
    magmaDoubleComplex *sx = (magmaDoubleComplex*)shared_data;

    if(flag == 0)
    {
        for( int j = tx; j < n; j += BLOCK_SIZE )
        { 
            sx[j] = MAGMA_Z_ZERO;
        }
    }
    else
    {
        for( int j = tx; j < n; j += BLOCK_SIZE )
        { 
            sx[j] = x[j]; 
        }
    }
    __syncthreads();
 
    if (uplo == MagmaLower)
    {
        for(int i=0; i<n; i+=BLOCK_SIZE)
        {    
            int jb = min(BLOCK_SIZE, n-i);
            col -= jb;
            // zgemvc is used in tranposed cose
            gemvc_template_device<magmaDoubleComplex, BLK_X, BLK_Y, TILE_SIZE, trans>(i, jb, MAGMA_Z_ONE, A(col+jb, col), lda, sx+col+jb, 1, MAGMA_Z_ONE, sx+col, 1);
            __syncthreads();

            ztrsv_backwards_tri_device<trans, diag>(jb, A(col, col), lda, b+col, incb, sx+col);
            __syncthreads();
        }
    }
    else
    {
        for(int i=0; i<n; i+=BLOCK_SIZE)
        {    
            int jb = min(BLOCK_SIZE, n-i);
            col = i;

            gemvc_template_device<magmaDoubleComplex, BLK_X, BLK_Y, TILE_SIZE, trans>(i, jb, MAGMA_Z_ONE, A(0, col), lda, sx, 1, MAGMA_Z_ONE, sx+col, 1);
            __syncthreads();   

            ztrsv_forwards_tri_device<trans, diag>(jb, A(col, col), lda, b+col, incb, sx+col);
            __syncthreads();
        }
    }


    for( int j = tx; j < n; j += BLOCK_SIZE )
    { 
        x[j] = sx[j]; // write to x in reverse order
    }
    __syncthreads();

}


#endif //MAGMABLAS_TRSV_TEMPLATE_H
