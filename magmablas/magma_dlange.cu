/*
    -- MAGMA (version 0.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       June 2009
*/
#include <stdio.h>
#include "cuda.h"
#include "cublas.h"
extern "C" __global__ void magma_dlange_special(const double *A, double *C, int M, int N, int lda) {

	int ibx = blockIdx.x * 64;

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int idt = ty * 16 + tx;

	double Cb[4] = {0,0,0,0};

	A+= ibx+idt ;
	const double * Aend = A+lda*N;

	
	double Ap[4]={A[0],A[lda],A[2*lda],A[3*lda]};
	
   	C+=ibx+idt;
	__shared__ double Cbb[64];;
		A+=4*lda;
	do {
		Cb[0]+=fabs(Ap[0]);
		Ap[0]=A[0];	
		Cb[1]+=fabs(Ap[1]);	
		Ap[1]=A[lda];	
		Cb[2]+=fabs(Ap[2]);	
		Ap[2]=A[2*lda];	
		Cb[3]+=fabs(Ap[3]);	
		Ap[3]=A[3*lda];
		A+=4*lda;
	
	} while (A < Aend);

	Cb[0]+=fabs(Ap[0]);
	Cb[1]+=fabs(Ap[1]);	
	Cb[2]+=fabs(Ap[2]);	
	Cb[3]+=fabs(Ap[3]);	

	Cbb[idt]=Cb[0]+Cb[1]+Cb[2]+Cb[3];
	C[0]= Cbb[idt];
}
/*

	Now do the rest of the parts in CPU ( getting the maximum )  Hybrid .. wow  
*/

/*
This Kernel Will be called when
                               M,N %64 != 0 
*/

extern "C" __global__ void magma_dlange_generic(const double *A, double *C, int M, int N, int lda , int N_mod_4) {

	int ibx = blockIdx.x * 64;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int idt = ty * 16 + tx;
	double Cb[4] = {0,0,0,0};

	/*
		Rouding up along row.
	*/
	if( (ibx + idt) >= M )
		A+= (M-1); 
	else
		A+= ibx+idt ;


	double Cbb;

   	C+=ibx+idt;
	/*
		Where to update. In generic case one place will be update more than once. 
		What about skipping it ?
			-- Another level of optimization is required. 
	*/
	if( N >= 8 ) {

		const double * Aend = A+lda*N   ;
		double Ap[4]={A[0],A[lda],A[2*lda],A[3*lda]};	
	
		A+=4*lda;

		do {
			Cb[0]+=fabs(Ap[0]);
			Ap[0]=A[0];	
			Cb[1]+=fabs(Ap[1]);	
			Ap[1]=A[lda];	
			Cb[2]+=fabs(Ap[2]);	
			Ap[2]=A[2*lda];	
			Cb[3]+=fabs(Ap[3]);	
			Ap[3]=A[3*lda];
			A+=4*lda;	
		}while (A < Aend);

		Cb[0]+=fabs(Ap[0]);
		Cb[1]+=fabs(Ap[1]);	
		Cb[2]+=fabs(Ap[2]);	
		Cb[3]+=fabs(Ap[3]);	
	}

	else{
		 if(N >= 4){
			Cb[0]+=fabs(A[0]);
			Cb[1]+=fabs(A[lda]);	
			Cb[2]+=fabs(A[2*lda]);	
			Cb[3]+=fabs(A[3*lda]);
			A+= 4*lda ;	
		 }
	}

	/*
		Clean up Code .......................... e.g. N  = 1,2,3, 513, 514, 515 etc. 
	*/
	switch(N_mod_4){

			case 0:
			break;

			case 1:
			Cb[0]+=fabs(A[0]);
			break;

			case 2:
			Cb[0]+=fabs(A[0]);
			Cb[1]+=fabs(A[lda]);	
			break;

			case 3:
			Cb[0]+=fabs(A[0]);
			Cb[1]+=fabs(A[lda]);	
			Cb[2]+=fabs(A[2*lda]);	
			break;
       	}

	/*Computing Final Result*/
	Cbb=Cb[0]+Cb[1]+Cb[2]+Cb[3];
	C[0]= Cbb;

}

extern "C" void
magmablas_dlange_64_64_16_4(const double *A, double *C, int M, int N, int lda,int tree_depth){

        dim3 threads( 16, 4 );
        dim3 grid(M/64+(M%64!=0),1);
	if( M %64 == 0  && N %64 == 0 ){ 
        	magma_dlange_special<<< grid, threads >>> ( A, C , M , N , lda);
	}
	else{
        	int N_mod_4 = N % 4 ;
		N = N - N_mod_4 ;  
        	magma_dlange_generic<<< grid, threads >>> ( A, C , M , N , lda , N_mod_4);
	}
}

extern "C" 
double  magma_dlange( char norm, int M, int N , double *A, int LDA , double * WORK ){
/*
  !!!!!!!!!!!!!!                
        -- Curreltly it returns NORM = 'I' only.
		This is needed for Iterative Refinement     
        -- Most probably this will be some internal utility function
        -- Right now the kernel requires M and N divisible by 64
	-- Implemented Generic Case
	-- Stan is there any function to get a single value from GPU which is superfast
  !!!!!!!!!!!!!!
 

  Purpose
  =======

  DLANGE  returns the value of the one norm,  or the Frobenius norm, or
  the  infinity norm,  or the  element of  largest absolute value  of a
  real matrix A.

 

  Description
  ===========

  DLANGE returns the value

     DLANGE = ( max(abs(A(i,j))), NORM = 'M' or 'm'
              (
              ( norm1(A),         NORM = '1', 'O' or 'o'
              (
              ( normI(A),         NORM = 'I' or 'i'
              (
              ( normF(A),         NORM = 'F', 'f', 'E' or 'e'

  where  norm1  denotes the  one norm of a matrix (maximum column sum),
  normI  denotes the  infinity norm  of a matrix  (maximum row sum) and
  normF  denotes the  Frobenius norm of a matrix (square root of sum of
  squares).  Note that  max(abs(A(i,j)))  is not a consistent matrix norm.

  Arguments
  =========

  NORM    (input) CHARACTER*1
          Specifies the value to be returned in DLANGE as described
          above.

  M       (input) INTEGER
          The number of rows of the matrix A.  M >= 0.  When M = 0,
          DLANGE is set to zero.

  N       (input) INTEGER
          The number of columns of the matrix A.  N >= 0.  When N = 0,
          DLANGE is set to zero.

  A       (input) DOUBLE PRECISION array, dimension (LDA,N)
          The m by n matrix A.
          A is in GPU memory. 

  LDA     (input) INTEGER
          The leading dimension of the array A.  LDA >= max(M,1).

  WORK    (workspace) DOUBLE PRECISION array, dimension (MAX(1,LWORK)),
          where LWORK >= M when NORM = 'I'; otherwise, WORK is not
          referenced.
          WORK is in GPU memory. 

 =====================================================================
*/
  if( norm !='I' && norm!='i') {
      printf("Only normI(A) is provided in this release!");
     exit(-1);
  }
  magmablas_dlange_64_64_16_4( A, WORK , M , N ,  LDA , 6 );
  int val = cublasIdamax(N,WORK,1);
  double retVal[1];
  cublasGetMatrix( 1, 1, sizeof( double ), WORK+val-1, 1, retVal, 1 ) ;
  return retVal[0];
}
