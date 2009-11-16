/*
    -- MAGMA (version 0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2009
*/

	/*
		How all the threads will be synchronized?
		Does ( float ) typecast work?
		Which one is slower comparing to CPU ?
	*/

#include <stdio.h>
__device__ int flag = 1 ; 
extern "C" __global__ void dlag2s_generic(const double *A, float *SA, int M, int N, int lda,int LDSA, double RMAX) {
	int ibx = blockIdx.x * 64;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int idt = ty * 16 + tx;
        if( ( ibx + idt) >= M) {
		A+= (M-1);
		SA+= (M-1);
	}
	else {
		A+= ibx+idt;
   		SA+=ibx+idt;
	}
	const double *Aend = A+lda*N;
	double RMAX_ = -1.0*RMAX ; 
	double Ap[1]={A[0]};
	do{
		A+=lda ; 
		if( Ap[0] < RMAX_ || Ap[0] > RMAX ) 
			flag = 1 ; 
		SA[0] = (float)Ap[0];
		Ap[0]=A[0];	
		SA+=LDSA;
		
        }while( A < Aend  ) ; 

	if( Ap[0] < RMAX_ || Ap[0] > RMAX ) 
		flag = 0 ; 
	SA[0] =(float) Ap[0];

}
extern "C" __global__ void dlag2s_special(const double *A, float *SA, int M, int N, int lda,int LDSA, double RMAX) {
	int ibx = blockIdx.x * 64;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int idt = ty * 16 + tx;
        if( ( ibx + idt) >= M) {
		A+= (M-1);
		SA+= (M-1);
	}
	else {
		A+= ibx+idt;
   		SA+=ibx+idt;
	}
	double RMAX_ = -1.0*RMAX ; 
	double Ap[1]={A[0]};
	if( Ap[0] < RMAX_ || Ap[0] > RMAX ) 
		flag = 0 ; 
	SA[0] =(float) Ap[0];
}
extern "C" void magmablas_dlag2s_64_64_16_4_v2(int M, int N , const double *A, int lda, float *SA , int LDSA, float RMAX ) 
{    
	if( M==0 || N==0 ){
		printf("One of the Matrix Dimension is 0\n");
		exit(-1);
	}
        //float RMAXX = slamch_("O");
        dim3 threads( 16, 4 );
        dim3 grid(M/64+(M%64!=0),1);
	if( N > 1){ 
	        dlag2s_generic<<< grid, threads >>> (  A, SA, M, N,lda , LDSA , RMAX) ;
	}
	else{
	        dlag2s_special<<< grid, threads >>> (  A, SA, M, N,lda , LDSA , RMAX) ;
	}
}           

extern "C" void magma_dlag2s(int M, int N , const double *A, int lda, float *SA , int LDSA, float RMAX ) 
{    
/*
  Note
  ====
	- We have to provide INFO at the end that dlag2s isn't doable now. 
	- Transfer a single value TO/FROM CPU/GPU
	- SLAMCH that's needed is called from underlying BLAS
	- Only used in iterative refinement
	- Do we want to provide this in the release?
  Purpose
  =======

  DLAG2S converts a DOUBLE PRECISION matrix, SA, to a SINGLE
  PRECISION matrix, A.

  RMAX is the overflow for the SINGLE PRECISION arithmetic
  DLAG2S checks that all the entries of A are between -RMAX and
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

  =========
*/
	magmablas_dlag2s_64_64_16_4_v2(M , N , A , lda , SA , LDSA , RMAX ) ; 
}
