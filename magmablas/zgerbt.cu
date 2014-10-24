/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c


       @author Adrien REMY
*/
#include "common_magma.h"


#define block_height  32
#define block_width  4
#define block_length 256


__global__ void 
magmablas_zelementary_multiplication(
    magmaDoubleComplex *dA, magmaDoubleComplex *du, 
    magmaDoubleComplex *dv, magma_int_t lda, magma_int_t N)
{    
    magma_int_t idx, idy;

    idx = blockIdx.x * blockDim.x + threadIdx.x;
    idy = blockIdx.y * blockDim.y + threadIdx.y;

    if ((idx < N/2)&&(idy < N/2)){
    
    dA += idx + idy * lda;

    magmaDoubleComplex a00, a10, a01, a11, b1, b2, b3, b4;
    __shared__ magmaDoubleComplex u1[block_height], u2[block_height], v1[block_width], v2[block_width];

    du += idx;
    dv += idy;

    u1[threadIdx.x]=du[0];
    u2[threadIdx.x]=du[N/2];
    v1[threadIdx.y]=dv[0];
    v2[threadIdx.y]=dv[N/2];

    __syncthreads();
 
    a00 = dA[0];
    a01 = dA[lda*N/2];
    a10 = dA[N/2];
    a11 = dA[lda*N/2+N/2];

    b1 = a00 + a01;
    b2 = a10 + a11;
    b3 = a00 - a01;
    b4 = a10 - a11;

    dA[0] = u1[threadIdx.x] * v1[threadIdx.y] * (b1 + b2);
    dA[lda*N/2] = u1[threadIdx.x] * v2[threadIdx.y] * (b3 + b4);
    dA[N/2] = u2[threadIdx.x] * v1[threadIdx.y] * (b1 - b2);
    dA[lda*N/2+N/2] = u2[threadIdx.x] * v2[threadIdx.y] *(b3 - b4);
    }
}

 __global__ void 
magmablas_zapply_vector(
    magmaDoubleComplex * du, magmaDoubleComplex * db, magma_int_t N)
{
    magma_int_t idx;

    idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N/2){
    
    du += idx;
    db += idx;

    magmaDoubleComplex a1,a2;
 
    a1 = du[0]*db[0];
    a2 = du[N/2]*db[N/2];

    db[0] = a1 + a2;
    db[N/2] = a1 -a2;
    }
}

__global__ void 
magmablas_zapply_transpose_vector(
    magmaDoubleComplex * du,magmaDoubleComplex * db, magma_int_t N)
{
   magma_int_t idx;

    idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N/2){
    
    du += idx;
    db += idx;

    magmaDoubleComplex a1,a2;
 
    a1 = db[0] + db[N/2];
    a2 = db[0] - db[N/2];

    db[0] = du[0]*a1;
    db[N/2] = du[N/2]*a2;
    }
}



/**
    Purpose
    -------
    ZPRBT_MVT compute B = UTB to randomize B
    
    Arguments
    ---------
    @param[in]
    N       INTEGER
            The number of values of d_b.  N >= 0.
    @param[in]
    d_u     COMPLEX_16 array, dimension (N,2)
            The 2*N vector representing the random butterfly matrix V
  
    @param[in,out]
    d_b     COMPLEX_16 array, dimension (N)
            The N vector d_b computed by ZGESV_NOPIV_GPU
            On exit d_b = d_u*d_b

   
 ********************************************************************/


extern "C" void
magmablas_zprbt_mtv(
    magma_int_t N, magmaDoubleComplex *d_u, magmaDoubleComplex *d_b)
{
/*

*/
    magma_int_t threadsPerBlock = block_length;
    magma_int_t blocksPerGrid = N/(4*block_length) + ((N%(4*block_length))!=0);

    magmablas_zapply_transpose_vector<<<blocksPerGrid, threadsPerBlock>>>(d_u+N,    d_b,N/2);
    magmablas_zapply_transpose_vector<<<blocksPerGrid, threadsPerBlock>>>(d_u+N+N/2,d_b+N/2,N/2);

    threadsPerBlock = block_length;
    blocksPerGrid = N/(2*block_length) + ((N%(2*block_length))!=0);
    magmablas_zapply_transpose_vector<<<blocksPerGrid, threadsPerBlock>>>(d_u,d_b,N);
}



/**
    Purpose
    -------
    ZPRBT_MV compute B = VB to obtain the non randomized solution
    
    Arguments
    ---------
    @param[in]
    N       INTEGER
            The number of values of d_b.  N >= 0.
    
    @param[in,out]
    d_b     COMPLEX_16 array, dimension (N)
            The N vector d_b computed by ZGESV_NOPIV_GPU
            On exit d_b = d_v*d_b

    @param[in]
    d_v     COMPLEX_16 array, dimension (N,2)
            The 2*N vector representing the random butterfly matrix V

 ********************************************************************/




extern "C" void
magmablas_zprbt_mv(
    magma_int_t N, magmaDoubleComplex *d_v, magmaDoubleComplex *d_b)
{

    magma_int_t threadsPerBlock = block_length;
    magma_int_t blocksPerGrid = N/(2*block_length) + ((N%(2*block_length))!=0);

    magmablas_zapply_vector<<<blocksPerGrid, threadsPerBlock>>>(d_v,d_b,N);


    threadsPerBlock = block_length;
    blocksPerGrid = N/(4*block_length) + ((N%(4*block_length))!=0);

    magmablas_zapply_vector<<<blocksPerGrid, threadsPerBlock>>>(d_v+N,    d_b,N/2);
    magmablas_zapply_vector<<<blocksPerGrid, threadsPerBlock>>>(d_v+N+N/2,d_b+N/2,N/2);


}


/**
    Purpose
    -------
    ZPRBT randomize a square general matrix using partial randomized transformation
    
    Arguments
    ---------
    @param[in]
    N       INTEGER
            The number of columns and rows of the matrix d_A.  N >= 0.
    
    @param[in,out]
    d_A     COMPLEX_16 array, dimension (N,lda)
            The N-by-N matrix d_A
            On exit d_A = d_uT*d_A*d_V

    @param[in]
    lda    INTEGER
            The leading dimension of the array d_A.  LDA >= max(1,N).

    @param[in]
    d_u     COMPLEX_16 array, dimension (N,2)
            The 2*N vector representing the random butterfly matrix U
 
    @param[in]
    d_v     COMPLEX_16 array, dimension (N,2)
            The 2*N vector representing the random butterfly matrix V

 ********************************************************************/




extern "C" void 
magmablas_zprbt(
    magma_int_t N, magmaDoubleComplex *d_A, magma_int_t lda, magmaDoubleComplex*d_u, magmaDoubleComplex *d_v)
{
/*

*/
    d_u += lda;
    d_v += lda;

    dim3 threadsPerBlock(block_height, block_width);
    dim3 blocksPerGrid(N/(4*block_height) + ((N%(4*block_height))!=0), 
                       N/(4*block_width)  + ((N%(4*block_width))!=0));

    magmablas_zelementary_multiplication<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_u, d_v, lda, N/2);
    magmablas_zelementary_multiplication<<<blocksPerGrid, threadsPerBlock>>>(d_A+lda*N/2, d_u, d_v+N/2, lda, N/2);
    magmablas_zelementary_multiplication<<<blocksPerGrid, threadsPerBlock>>>(d_A+N/2, d_u+N/2, d_v, lda, N/2);
    magmablas_zelementary_multiplication<<<blocksPerGrid, threadsPerBlock>>>(d_A+lda*N/2+N/2, d_u+N/2, d_v+N/2, lda, N/2);

    dim3 threadsPerBlock2(block_height, block_width);
    dim3 blocksPerGrid2(N/(2*block_height) + ((N%(2*block_height))!=0), 
                        N/(2*block_width)  + ((N%(2*block_width))!=0));
    magmablas_zelementary_multiplication<<<blocksPerGrid2, threadsPerBlock2>>>(d_A, d_u-lda, d_v-lda, lda, N);
}


__global__ void
magmablas_zbackward_error(
    magma_int_t N, magmaDoubleComplex *d_A, magma_int_t lda, 
    magmaDoubleComplex *d_x, magmaDoubleComplex *d_b, magmaDoubleComplex *d_be)
{
    magma_int_t idx, k;
  magmaDoubleComplex  temp = MAGMA_Z_ZERO;

    idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    d_be += idx;
    d_A += idx; 
    d_b += idx;

    d_be[0] = d_b[0];
    for (k=0; k<N; k++){
        d_be[0] -= d_A[lda*k]*d_x[k];
    }
    for (k=0; k<N; k++){
         temp += fabs(d_A[lda*k])*fabs(d_b[k]);
    }
    temp += fabs(d_b[0]);
    d_be[0] = fabs(d_be[0])/temp;   
}

extern "C" void
magmablas_zprbt_backward_error(
    magma_int_t N, magmaDoubleComplex *d_A, magma_int_t lda, 
    magmaDoubleComplex *d_x, magmaDoubleComplex *d_b, magmaDoubleComplex *d_be)
{
/*

*/
   magma_int_t threadsPerBlock = block_length;
   magma_int_t blocksPerGrid = N/(block_length) + ((N%(block_length))!=0);

    magmablas_zbackward_error<<<blocksPerGrid, threadsPerBlock>>>(N,d_A,lda,d_x,d_b,d_be);
}
