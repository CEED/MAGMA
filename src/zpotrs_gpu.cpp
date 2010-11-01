/*
    -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

       @precisions normal z -> s d c

*/

#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cublas.h>
#include "magma.h"
#include "magmablas.h"

extern "C" magma_int_t
magma_zpotrs_gpu(char UPLO_, magma_int_t N , magma_int_t NRHS, double2 *A , magma_int_t LDA,
		 double2 *B, magma_int_t LDB, magma_int_t *INFO)
{
/*  -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010
 
    Purpose
    =======

    ZPOTRS solves a system of linear equations A*X = B with a hemmetric
    positive definite matrix A using the Cholesky factorization
    A = U\*\*H*U or A = L*L\*\*H computed by ZPOTRF.

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

    A       (input) COMPLEX_16 array, dimension (LDA,N)
            The triangular factor U or L from the Cholesky factorization
            A = U\*\*H*U or A = L*L\*\*H, as computed by ZPOTRF.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    B       (input/output) COMPLEX_16 array, dimension (LDB,NRHS)
            On entry, the right hand side matrix B.
            On exit, the solution matrix X.

    LDB     (input) INTEGER
            The leading dimension of the array B.  LDB >= max(1,N).

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
    =====================================================================   */
    #define MAX(a,b)       (((a)>(b))?(a):(b))

    char UPLO[2] = {UPLO_, 0};

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
    if( *INFO != 0 ){ 
      magma_xerbla("magma_zpotrs_gpu", INFO); 
      return 0;
    }
    if( N==0 || NRHS ==0) 
      return 0;	
    if( *UPLO =='U' || *UPLO=='u'){
      cublasZtrsm('L','U','T','N', N , NRHS, 1.0, A , LDA , B , LDB );
      cublasZtrsm('L','U','N','N', N , NRHS, 1.0, A , LDA , B , LDB );
    }
    else{
      cublasZtrsm('L','L','N','N', N , NRHS, 1.0, A , LDA , B , LDB );
      cublasZtrsm('L','L','T','N', N , NRHS, 1.0, A , LDA , B , LDB );
    }

    return 0;
}
#undef MAX
