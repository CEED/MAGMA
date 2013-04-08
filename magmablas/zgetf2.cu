/*

    -- MAGMA (version 1.1) --
    Univ. of Tennessee, Knoxville
    Univ. of California, Berkeley
    Univ. of Colorado, Denver
    November 2011

    @precisions normal z -> s d c
*/

#include "stdio.h"
#include "common_magma.h"

#define CUBLAS_V2 

#ifdef CUBLAS_V2
   #include "cublas_v2.h"
#else
   #include "cublas.h"
#endif


    
#define A(i, j)  (A +(i) + (j)*lda)   // A(i, j) means at i row, j column 

void  magma_swap(int n, cuDoubleComplex *x, int i, int j, int lda);

void magma_scal_ger(int, int, cuDoubleComplex *, int);


int magma_zgetf2(int m, int n, cuDoubleComplex *A, int lda, int *ipiv, int* info)
{
   
/*  ZGETF2 computes an LU factorization of a general m-by-n matrix A */
/*  using partial pivoting with row interchanges. */

/*  The factorization has the form */
/*     A = P * L * U */
/*  where P is a permutation matrix, L is lower triangular with unit */
/*  diagonal elements (lower trapezoidal if m > n), and U is upper */
/*  triangular (upper trapezoidal if m < n). */

/*  This is the right-looking Level 2 BLAS version of the algorithm. */

/*  Arguments */
/*  ========= */

/*  M       (input) INTEGER */
/*          The number of rows of the matrix A.  M >= 0. */

/*  N       (input) INTEGER */
/*          The number of columns of the matrix A.  N >= 0. */

/*  A       (input/output) cuDoubleComplex PRECISION array, dimension (LDA,N) */
/*          On entry, the m by n matrix to be factored. */
/*          On exit, the factors L and U from the factorization */
/*          A = P*L*U; the unit diagonal elements of L are not stored. */

/*  LDA     (input) INTEGER */
/*          The leading dimension of the array A.  LDA >= max(1,M). */

/*  IPIV    (output) INTEGER array, dimension (min(M,N)) */
/*          The pivot indices; for 1 <= i <= min(M,N), row i of the */
/*          matrix was interchanged with row IPIV(i). */

/*  INFO    (output) INTEGER */
/*          = 0: successful exit */
/*          < 0: if INFO = -k, the k-th argument had an illegal value */
/*          > 0: if INFO = k, U(k,k) is exactly zero. The factorization */
/*               has been completed, but the factor U is exactly */
/*               singular, and division by zero will occur if it is used */
/*               to solve a system of equations. */

/*  ===================================================================== */




//    Quick return if possible 

    if (m == 0 || n == 0) {
    return 0;
    }

    if(n>1024)
    {
       printf("N = %d > 1024 is not supported. This routine targets on slim matrix\n", n);
       return 1;
    }

//     Compute machine safe minimum 

#ifdef CUBLAS_V2
    cublasHandle_t handle;

    cublasCreate(&handle);

#else
    cublasInit();

#endif

    int i__1 = min(m, n);

    
    for (int j = 0; j < i__1; j++) {

//        Find pivot and test for singularity. 

        //int i__2 = m - j;

        int jp; 

        cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);    
    
#ifdef CUBLAS_V2
        cublasIzamax(handle,  m-j, (const cuDoubleComplex*)A(j,j), 1, &jp); 
#else
        jp = cublasIzamax(m-j, (const cuDoubleComplex*)A(j,j), 1); 
#endif

        jp = jp -1 + j;
    

        ipiv[j] = jp + 1;
    
    //if ( A(jp, j) != 0.0)
//    {
//           Apply the interchange to columns 1:N. 

         cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

         if (jp != j) 
             {
                magma_swap(n, A, j, jp, lda);
           
          }   

//           Compute elements J+1:M of J-th column.  
           
          if (j < m) 
          {        

              magma_scal_ger(m-j, n-j, A(j, j), lda);

          }
 
          else if (*info == 0) 
          {
              *info = j;
          }

   }

#ifdef CUBLAS_V2
    cublasDestroy(handle);

#else
    cublasShutdown();

#endif

    return 0;

} 


#define swap_bs 64
#define ger_bs 1024


__global__ void kernel_swap(int n, cuDoubleComplex *x, int i, int j, int lda)
{
    int id = blockIdx.x * swap_bs + threadIdx.x;

    if(id < n)
    {    
        cuDoubleComplex res = x[i + lda * id]; 

        x[i + lda * id] = x[j + lda*id];

        x[j + lda*id] = res;

    }

}


void  magma_swap(int n, cuDoubleComplex *x, int i, int j, int lda)
{

    dim3 threads(swap_bs, 1, 1);
    
    int num_blocks = (n - 1)/swap_bs + 1;

    dim3 grid(num_blocks,1);

    kernel_swap<<< grid, threads, 0, magma_stream >>>(n, x, i, j, lda);

}


extern __shared__  cuDoubleComplex shared_data[];

__global__ void
kernel_scal_ger(int m, int n, cuDoubleComplex *A, int lda)
{

    cuDoubleComplex *shared_y = (cuDoubleComplex *)shared_data;

    int tid = blockIdx.x * ger_bs + threadIdx.x;
        
    cuDoubleComplex reg = MAGMA_Z_ZERO;
        
    
    if(threadIdx.x < n)
    {    
        shared_y[threadIdx.x] = A[lda * threadIdx.x];    
    }
    
    __syncthreads();

    if(tid<m && tid>0)
    {
        reg = A[tid];

        reg *= MAGMA_Z_DIV(MAGMA_Z_ONE, shared_y[0]);

        A[tid] = reg;

        #pragma unroll
        for(int i=1; i<n; i++)
        {
            A[tid + i*lda] += (MAGMA_Z_NEG_ONE) * shared_y[i] * reg;
        }

    }
} 


void  magma_scal_ger(int m, int n, cuDoubleComplex *A, int lda)
{


    dim3 threads(ger_bs, 1, 1);
    
    int num_blocks = (m - 1)/ger_bs + 1;

    dim3 grid(num_blocks,1);
    
    size_t shared_size = sizeof(cuDoubleComplex)*(n);

    kernel_scal_ger<<< grid, threads, shared_size, magma_stream>>>(m, n, A, lda);


}



