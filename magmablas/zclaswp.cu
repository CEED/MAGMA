/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions mixed zc -> ds

*/
#include "common_magma.h"

#define NB 64

__global__ void
zclaswp_kernel(int n, magmaDoubleComplex *A, int lda, magmaFloatComplex *SA, int m, const magma_int_t *ipiv)
{
    int ind = blockIdx.x*NB + threadIdx.x;
    int newind;
    magmaFloatComplex res;
    
    if (ind < m) {
        SA   += ind;
        ipiv += ind;
        
        newind = ipiv[0];
        
        for(int i=0; i < n; i++) {
            res = MAGMA_C_MAKE( (float)cuCreal(A[newind+i*lda]),
                                (float)cuCimag(A[newind+i*lda]) );
            SA[i*lda] = res; 
        }
    }
}

__global__ void
zclaswp_inv_kernel(int n, magmaDoubleComplex *A, int lda, magmaFloatComplex *SA, int m, const magma_int_t *ipiv)
{
    int ind = blockIdx.x*NB + threadIdx.x;
    int newind;
    magmaDoubleComplex res;

    if (ind < m) {
        A    += ind;
        ipiv += ind;

        newind = ipiv[0];

        for(int i=0; i < n; i++) {
            res = MAGMA_Z_MAKE( (double)cuCrealf(SA[newind+i*lda]),
                                (double)cuCimagf(SA[newind+i*lda]) );
            A[i*lda] = res;
        }
    }
}


/**
    Purpose
    -------
    Row i of A is cast to single precision in row ipiv[i] of SA, for 0 <= i < M.

    @param[in]
    n       INTEGER.
            On entry, N specifies the number of columns of the matrix A.
    
    @param[in]
    A       DOUBLE PRECISION array on the GPU, dimension (LDA,N)
            On entry, the M-by-N matrix to which the row interchanges will be applied.
    
    @param[in]
    lda     INTEGER.
            LDA specifies the leading dimension of A.
    
    @param[out]
    SA      REAL array on the GPU, dimension (LDA,N)
            On exit, the single precision, permuted matrix.
        
    @param[in]
    m       The number of rows to be interchanged.
    
    @param[in]
    ipiv    INTEGER array on the GPU, dimension (M)
            The vector of pivot indices. Row i of A is cast to single 
            precision in row ipiv[i] of SA, for 0 <= i < m. 
    
    @param[in]
    incx    INTEGER
            If IPIV is negative, the pivots are applied in reverse 
            order, otherwise in straight-forward order.

    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void
magmablas_zclaswp( magma_int_t n, magmaDoubleComplex *A, magma_int_t lda,
                   magmaFloatComplex *SA, magma_int_t m,
                   const magma_int_t *ipiv, magma_int_t incx )
{
    int blocks = (m - 1)/NB + 1;
    dim3 grid(blocks, 1, 1);
    dim3 threads(NB, 1, 1);

    if (incx >= 0)
        zclaswp_kernel<<< grid, threads, 0, magma_stream >>>(n, A, lda, SA, m, ipiv);
    else
        zclaswp_inv_kernel<<< grid, threads, 0, magma_stream >>>(n, A, lda, SA, m, ipiv);
}
