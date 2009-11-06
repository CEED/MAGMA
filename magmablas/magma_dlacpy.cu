/*
    -- MAGMA (version 0.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       June 2009
*/

#include <stdio.h>

extern "C" __global__ void dlacpy_generic(int M, int N, double *SA, int LDSA , double *A , int LDA, int *INFO ) { 
	int ibx = blockIdx.x * 64;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int idt = ty * 16 + tx;

	if( (ibx+idt) >=M ){
		SA+= M-1;
   		A+=  M-1;
	}
	else{
		SA+= ibx+idt;
   		A+=ibx+idt;
	}
	const double * SAend = SA+LDSA*N;
	double Ap[1]={SA[0]};
	do {
		SA+=LDSA;
		A[0] = Ap[0];
		Ap[0]=SA[0];	
		A+=LDA;
	} while (SA < SAend);
	A[0] = Ap[0];
}

extern "C" __global__ void dlacpy_special(int M, int N, double *SA, int LDSA , double *A , int LDA, int *INFO ) { 
	int ibx = blockIdx.x * 64;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int idt = ty * 16 + tx;
	if( (ibx+idt) >=M ){
		SA+= M-1;
   		A+=  M-1;
	}
	else{
		SA+= ibx+idt;
   		A+=ibx+idt;
	}
	double Ap[1]={SA[0]};
	A[0] = Ap[0];
}

extern "C" void magmablas_dlacpy_64_64_16_4_v2(int M, int N, double *SA, int LDSA , double *A , int LDA, int *INFO){
        dim3 threads( 16, 4 );
        dim3 grid(M/64+(M%64!=0),1);
	if( N == 1 ) 
	        dlacpy_special<<< grid, threads >>> (  M, N,SA, LDSA ,A,LDA,INFO) ;
	else	
  	      dlacpy_generic<<< grid, threads >>> (  M, N,SA, LDSA ,A,LDA,INFO) ;
}
extern "C" void magma_dlacpy(int M, int N, double *SA, int LDSA , double *A , int LDA, int *INFO){
/*
  Note
  ====
  - UPLO Parameter is disabled
  - Do we want to provide a generic function to the user with all the options?

  Purpose
  =======

  DLACPY copies all or part of a two-dimensional matrix A to another
  matrix B.

  Arguments
  =========

  UPLO    (input) CHARACTER*1
          Specifies the part of the matrix A to be copied to B.
          = 'U':      Upper triangular part
          = 'L':      Lower triangular part
          Otherwise:  All of the matrix A

  M       (input) INTEGER
          The number of rows of the matrix A.  M >= 0.

  N       (input) INTEGER
          The number of columns of the matrix A.  N >= 0.

  A       (input) DOUBLE PRECISION array, dimension (LDA,N)
          The m by n matrix A.  If UPLO = 'U', only the upper triangle
          or trapezoid is accessed; if UPLO = 'L', only the lower
          triangle or trapezoid is accessed.

  LDA     (input) INTEGER
          The leading dimension of the array A.  LDA >= max(1,M).

  B       (output) DOUBLE PRECISION array, dimension (LDB,N)
          On exit, B = A in the locations specified by UPLO.

  LDB     (input) INTEGER
          The leading dimension of the array B.  LDB >= max(1,M).

  =====================================================================
*/
	magmablas_dlacpy_64_64_16_4_v2(M,N,SA,LDSA,A,LDA,INFO);
}
