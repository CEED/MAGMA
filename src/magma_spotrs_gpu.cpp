#include <stdio.h>
#include <math.h>
#include "magmablas.h"
#include "magma.h"
#include "cublas.h"
#include "cuda.h"
/*
   -- MAGMA (version 0.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       August 2009

  Purpose
  =======

  SPOTRS solves a system of linear equations A*X = B with a symmetric
  positive definite matrix A using the Cholesky factorization
  A = U**T*U or A = L*L**T computed by SPOTRF.

  Arguments
  =========

  UPLO    (input) CHARACTER*1
          = 'U':  Upper triangle of A is stored;
          = 'L':  Lower triangle of A is stored.

  N       (input) INTEGER
          The order of the matrix A.  N >= 0.

  NRHS    (input) INTEGER
          The number of right hand sides, i.e., the number of columns
          of the matrix B.  NRHS >= 0.

  A       (input) REAL array, dimension (LDA,N)
          The triangular factor U or L from the Cholesky factorization
          A = U**T*U or A = L*L**T, as computed by SPOTRF.

  LDA     (input) INTEGER
          The leading dimension of the array A.  LDA >= max(1,N).

  B       (input/output) REAL array, dimension (LDB,NRHS)
          On entry, the right hand side matrix B.
          On exit, the solution matrix X.

  LDB     (input) INTEGER
          The leading dimension of the array B.  LDB >= max(1,N).

  INFO    (output) INTEGER
          = 0:  successful exit
          < 0:  if INFO = -i, the i-th argument had an illegal value

  =====================================================================
*/
int MAX( int a, int b){
 return a>b ? a: b ;
}

void magma_spotrs_gpu( char *UPLO , int N , int NRHS, float *A , int LDA ,float *B, int LDB, int *INFO){
		*INFO = 0 ; 
		if( *UPLO !='U' && *UPLO !='u' && *UPLO !='L' && *UPLO!='l')
			*INFO = - 1 ; 
		if( N < 0 )
			*INFO = -2 ; 
		if( NRHS < 0) 
			*INFO = -3 ; 
		if ( LDA < MAX(1,N))
			*INFO = -5; 
		if ( LDB < MAX(1,N))
			*INFO = -7;
                if( *INFO != 0 ) 
		    magma_xerbla("magma_spotrs_gpu", INFO); 
		if( N==0 || NRHS ==0) 
			return;	
                if( *UPLO =='U' || *UPLO=='u'){
                         magmablas_strsm('L','U','T','N', N , NRHS,   A , LDA , B , LDB );
                         magmablas_strsm('L','U','N','N', N , NRHS,   A , LDA , B , LDB );
                }
                else{
                         magmablas_strsm('L','L','N','N', N , NRHS,  A , LDA , B , LDB );
                         magmablas_strsm('L','L','T','N', N , NRHS,  A , LDA , B , LDB );
                }
}
