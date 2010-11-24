/*
    -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

       @precisions mixed zc -> ds

*/
#include <stdio.h>
#include <cublas.h>
#include "magma.h"

__device__ int flag = 0; 

static __global__ void 
zlag2c_generic(const cuDoubleComplex *A, cuFloatComplex *SA, int M, int N, int lda,
               int LDSA, double RMAX ) 
{
    double mRMAX = - RMAX;
    int ibx = blockIdx.x * 64;
    int tx  = threadIdx.x;
    int ty  = threadIdx.y;
    int idt = ty * 16 + tx;
    if( ( ibx + idt) >= M) {
        A  += (M-1);
        SA += (M-1);
    }
    else {
        A  += ibx+idt;
        SA += ibx+idt;
    }
    const cuDoubleComplex *Aend = A+lda*N;
    cuDoubleComplex Ap[1]={A[0]};
    do{
        A+=lda ; 

        if( (cuCreal(Ap[0]) < mRMAX) || (cuCreal(Ap[0]) < mRMAX)
#if defined(PRECISION_z) || defined(PRECISION_c)
            || (cuCimag(Ap[0]) < mRMAX) || (cuCimag(Ap[0]) < mRMAX) 
#endif
            )
            {
                flag = 1; 
            }
        SA[0] = cuComplexDoubleToFloat( Ap[0] );
        Ap[0] = A[0];	
        SA += LDSA;
		
    }while( A < Aend  ) ; 

    if( (cuCreal(Ap[0]) < mRMAX) || (cuCreal(Ap[0]) < mRMAX)
#if defined(PRECISION_z) || defined(PRECISION_c)
        || (cuCimag(Ap[0]) < mRMAX) || (cuCimag(Ap[0]) < mRMAX) 
#endif
        )
        {
            flag = 1;
        }
    SA[0] = cuComplexDoubleToFloat( Ap[0] );

}

static  __global__ void 
zlag2c_special(const cuDoubleComplex *A, cuFloatComplex *SA, int M, int N, int lda,int LDSA, 
               double RMAX) 
{
    double mRMAX = - RMAX;
    int ibx = blockIdx.x * 64;
    int tx  = threadIdx.x;
    int ty  = threadIdx.y;
    int idt = ty * 16 + tx;
    if( ( ibx + idt) >= M) {
        A  += (M-1);
        SA += (M-1);
    }
    else {
        A  += ibx+idt;
        SA += ibx+idt;
    }
    cuDoubleComplex Ap[1] = {A[0]};

    if( (cuCreal(Ap[0]) < mRMAX) || (cuCreal(Ap[0]) < mRMAX)
#if defined(PRECISION_z) || defined(PRECISION_c)
        || (cuCimag(Ap[0]) < mRMAX) || (cuCimag(Ap[0]) < mRMAX) 
#endif
        )
        {
            flag = 1;
        }
    SA[0] = cuComplexDoubleToFloat( Ap[0] );
}

static void 
magmablas_zlag2c_64_64_16_4_v2(int M, int N , 
                               const cuDoubleComplex *A, int lda, 
                               cuFloatComplex *SA, int LDSA, 
                               double RMAX, magma_int_t *info ) 
{    
    if( M==0 || N==0 ){
        printf("One of the Matrix Dimension is 0\n");
        exit(-1);
    }

    dim3 threads( 16, 4 );
    dim3 grid(M/64+(M%64!=0),1);
    if( N > 1){ 
        zlag2c_generic<<< grid, threads >>> ( A, SA, M, N, lda, LDSA, RMAX ) ;
    }
    else{
        zlag2c_special<<< grid, threads >>> ( A, SA, M, N, lda, LDSA, RMAX ) ;
    }
    *info = flag;
}           

extern "C" void 
magmablas_zlag2c( int M, int N , 
                  const cuDoubleComplex *A, int lda, 
                  cuFloatComplex *SA, int LDSA, 
                  magma_int_t *info ) 
{    
/*
  Note
  ====
	- We have to provide INFO at the end that zlag2c isn't doable now. 
	- Transfer a single value TO/FROM CPU/GPU
	- SLAMCH that's needed is called from underlying BLAS
	- Only used in iterative refinement
	- Do we want to provide this in the release?
  Purpose
  =======

  ZLAG2C converts a DOUBLE PRECISION matrix, SA, to a SINGLE
  PRECISION matrix, A.

  RMAX is the overflow for the SINGLE PRECISION arithmetic
  ZLAG2C checks that all the entries of A are between -RMAX and
  RMAX. If not the convertion is aborted and a flag is raised.

  This is an auxiliary routine so there is no argument checking.

  Arguments
  =========

  M       (input) INTEGER
          The number of lines of the matrix A.  M >= 0.

  N       (input) INTEGER
          The number of columns of the matrix A.  N >= 0.

  A       (input) DOUBLE PRECISION array, dimension (LDA,N)
          On entry, the M-by-N coefficient matrix A.

  LDA     (input) INTEGER
          The leading dimension of the array A.  LDA >= max(1,M).

  SA      (output) REAL array, dimension (LDSA,N)
          On exit, if INFO=0, the M-by-N coefficient matrix SA; if
          INFO>0, the content of SA is unspecified.

  LDSA    (input) INTEGER
          The leading dimension of the array SA.  LDSA >= max(1,M).

  INFO    (output) INTEGER
          = 0:  successful exit.
          = 1:  an entry of the matrix A is greater than the SINGLE
                PRECISION overflow threshold, in this case, the content
                of SA in exit is unspecified.

  ===========================================================================  */

    double RMAX = (double)lapackf77_slamch("O");
    magmablas_zlag2c_64_64_16_4_v2(M, N, A, lda, SA, LDSA, RMAX, info ) ; 

}
