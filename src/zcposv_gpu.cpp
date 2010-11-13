/*
    -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

       @precisions mixed zc -> ds

*/
#include <stdio.h>
#include <math.h>
#include "magmablas.h"
#include "magma.h"
#include "cublas.h"
#include "cuda.h"

#define ITERMAX 30
#define BWDMAX 1.0

extern "C" magma_int_t
magma_zcposv_gpu(char UPLO, magma_int_t N, magma_int_t NRHS, double2 *A, magma_int_t LDA, double2 *B, 
                 magma_int_t LDB, double2 *X, magma_int_t LDX, double2 *WORK, float2 *SWORK,
                 magma_int_t *ITER, magma_int_t *INFO, float2 *H_SWORK, double2 *H_WORK)
{
/*  -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

    Purpose
    =======

    ZCPOSV computes the solution to a real system of linear equations
       A * X = B,
    where A is an N-by-N symmetric positive definite matrix and X and B
    are N-by-NRHS matrices.

    ZCPOSV first attempts to factorize the matrix in SINGLE PRECISION
    and use this factorization within an iterative refinement procedure
    to produce a solution with DOUBLE PRECISION norm-wise backward error
    quality (see below). If the approach fails the method switches to a
    DOUBLE PRECISION factorization and solve.

    The iterative refinement is not going to be a winning strategy if
    the ratio SINGLE PRECISION performance over DOUBLE PRECISION
    performance is too small. A reasonable strategy should take the
    number of right-hand sides and the size of the matrix into account.
    This might be done with a call to ILAENV in the future. Up to now, we
    always try iterative refinement.

    The iterative refinement process is stopped if
        ITER > ITERMAX
    or for all the RHS we have:
        RNRM < SQRT(N)*XNRM*ANRM*EPS*BWDMAX
    where
        o ITER is the number of the current iteration in the iterative
          refinement process
        o RNRM is the infinity-norm of the residual
        o XNRM is the infinity-norm of the solution
        o ANRM is the infinity-operator-norm of the matrix A
        o EPS is the machine epsilon returned by DLAMCH('Epsilon')
    The value ITERMAX and BWDMAX are fixed to 30 and 1.0D+00 respectively.

    Arguments
    =========

    UPLO    (input) CHARACTER
            = 'U':  Upper triangle of A is stored;
            = 'L':  Lower triangle of A is stored.

    N       (input) INTEGER
            The number of linear equations, i.e., the order of the
            matrix A.  N >= 0.

    NRHS    (input) INTEGER
            The number of right hand sides, i.e., the number of columns
            of the matrix B.  NRHS >= 0.

    A       (input or input/output) DOUBLE PRECISION array, dimension (LDA,N)
            On entry, the symmetric matrix A.  If UPLO = 'U', the leading
            N-by-N upper triangular part of A contains the upper
            triangular part of the matrix A, and the strictly lower
            triangular part of A is not referenced.  If UPLO = 'L', the
            leading N-by-N lower triangular part of A contains the lower
            triangular part of the matrix A, and the strictly upper
            triangular part of A is not referenced.
            On exit, if iterative refinement has been successfully used
            (INFO.EQ.0 and ITER.GE.0, see description below), then A is
            unchanged, if double2 precision factorization has been used
            (INFO.EQ.0 and ITER.LT.0, see description below), then the
            array A contains the factor U or L from the Cholesky
            factorization A = U**T*U or A = L*L**T.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    B       (input) DOUBLE PRECISION array, dimension (LDB,NRHS)
            The N-by-NRHS right hand side matrix B.

    LDB     (input) INTEGER
            The leading dimension of the array B.  LDB >= max(1,N).

    X       (output) DOUBLE PRECISION array, dimension (LDX,NRHS)
            If INFO = 0, the N-by-NRHS solution matrix X.

    LDX     (input) INTEGER
            The leading dimension of the array X.  LDX >= max(1,N).

    WORK    (workspace) DOUBLE PRECISION array, dimension (N*NRHS)
            This array is used to hold the residual vectors.

    SWORK   (workspace) REAL array, dimension (N*(N+NRHS))
            This array is used to use the single precision matrix and the
            right-hand sides or solutions in single precision.

    ITER    (output) INTEGER
            < 0: iterative refinement has failed, double2 precision
                 factorization has been performed
                 -1 : the routine fell back to full precision for
                      implementation- or machine-specific reasons
                 -2 : narrowing the precision induced an overflow,
                      the routine fell back to full precision
                 -3 : failure of SPOTRF
                 -31: stop the iterative refinement after the 30th
                      iterations
            > 0: iterative refinement has been successfully used.
                 Returns the number of iterations

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
            > 0:  if INFO = i, the leading minor of order i of (DOUBLE
                  PRECISION) A is not positive definite, so the
                  factorization could not be completed, and the solution
                  has not been computed.

    H_SWORK    (workspace) REAL array, dimension at least (nb, nb)
            where nb can be obtained through magma_get_cpotrf_nb(*n)
            Work array allocated with cudaMallocHost.

    H_WORK   (workspace) DOUBLE array, dimension at least (nb, nb)
             where nb can be obtained through magma_get_zpotrf_nb(*n)
             Work array allocated with cudaMallocHost.

    =====================================================================    */

  #define max(a,b)       (((a)>(b))?(a):(b))

  double2 c_neg_one = MAGMA_Z_NEG_ONE;
  double2 c_one = MAGMA_Z_ONE;
  int c_ione = 1;

  *ITER = 0 ;
  *INFO = 0 ; 

  if ( N <0)
    *INFO = -1;
  else if(NRHS<0)
    *INFO =-2;
  else if(LDA < max(1,N))
    *INFO =-4;
  else if( LDB < max(1,N))
    *INFO =-7;
  else if( LDX < max(1,N))
    *INFO =-9;
   
  if(*INFO!=0){
    magma_xerbla("magma_zcposv",INFO);
  }

  if( N == 0 || NRHS == 0 ) 
    return 0;

  double ANRM , CTE , EPS;
  EPS  = lapackf77_dlamch("Epsilon");
  ANRM = magma_zlansy(  'I',  UPLO , N ,A, LDA ,WORK);
  CTE = ANRM * EPS *  pow((double)N,0.5) * BWDMAX ;  

  int PTSA  = N*NRHS;
  int PTSX  = 0 ;  
  float RMAX = lapackf77_slamch("O");
  int IITER ;
  double2 alpha = c_neg_one;
  double2 beta = c_one; 
  double2 XNRM[1] , RNRM[1]; 
  float2 RMAX_cplx;
  MAGMA_Z_SET2REAL( RMAX_cplx, RMAX );
 
  magmablas_zlag2c(N , NRHS , B , LDB , SWORK+PTSX, N , RMAX_cplx );
  if(*INFO !=0){
    *ITER = -2 ;
    goto L40;
  } 
  
  magma_zlat2c(UPLO ,  N ,  A , LDA , SWORK+PTSA, N , INFO ); 
  if(*INFO !=0){
    *ITER = -2 ;
    goto L40;
  }   
  magma_cpotrf_gpu(UPLO, N, SWORK+ PTSA, LDA, INFO);
  if(INFO[0] !=0){
    *ITER = -3 ;
    goto L40;
  }
  magma_cpotrs_gpu(UPLO, N , NRHS, SWORK+PTSA, LDA, SWORK+PTSX, LDB, INFO);
  magmablas_clag2z(N, NRHS, SWORK+PTSX, N, X, LDX, INFO);

  magma_zlacpy(N, NRHS, B, LDB, WORK, N);

  if( NRHS == 1 )
    magmablas_zcymv(UPLO, N, c_neg_one, A, LDA, X, 1, c_one, WORK, 1);
  else
    cublasZsymm('L', UPLO, N, NRHS, c_neg_one, A, LDA, X, LDX, c_one, WORK, N);
  
  int i, j;
  for(i=0; i<NRHS; i++){
    j = cublasIzamax(N, X+i*N, 1); 
    cublasGetMatrix(1, 1, sizeof(double2), X+i*N+j-1, 1,XNRM, 1);
    MAGMA_Z_SET2REAL( XNRM[0], lapackf77_zlange( "F", &c_ione, &c_ione, XNRM, &c_ione, XNRM ) );
    j = cublasIzamax (N, WORK+i*N, 1); 
    cublasGetMatrix(1, 1, sizeof(double2), WORK+i*N+j-1, 1, RNRM, 1);
    MAGMA_Z_SET2REAL( RNRM[0], lapackf77_zlange( "F", &c_ione, &c_ione, RNRM, &c_ione, RNRM ) );
    if( MAGMA_Z_GET_X( RNRM[0] ) > MAGMA_Z_GET_X( XNRM[0] ) *CTE ){
      goto L10;
    } 
  }
  *ITER =0; 
  return 0;
  
 L10:
  ;

  for(IITER=1;IITER<ITERMAX;){
    *INFO = 0 ; 
    magmablas_zlag2c(N, NRHS, WORK, N, SWORK+PTSX, N, RMAX_cplx);
    if(*INFO !=0){
      *ITER = -2 ;
      goto L40;
    } 
    magma_cpotrs_gpu(UPLO, N, NRHS, SWORK+PTSA, LDA, SWORK+PTSX, LDB, INFO);

    for(i=0;i<NRHS;i++){
      magmablas_zcaxpycp(SWORK+i*N, X+i*N, N, N, LDA, B+i*N,WORK+i*N) ;
    }

    if( NRHS == 1 )
      magmablas_zcymv(UPLO, N, alpha, A, LDA, X, 1, beta, WORK, 1);
    else 
     cublasZsymm('L', UPLO, N, NRHS, alpha, A, LDA, X, LDX, beta, WORK, N);

    for(i=0; i<NRHS; i++){
      int j;
      j = cublasIzamax( N , X+i*N  , 1) ; 
      cublasGetMatrix( 1, 1, sizeof(double2), X+i*N+j-1, 1, XNRM, 1 ) ;
      MAGMA_Z_SET2REAL( XNRM[0], lapackf77_zlange( "F", &c_ione, &c_ione, XNRM, &c_ione, XNRM ) );
      j = cublasIzamax ( N ,WORK+i*N , 1 ) ; 
      cublasGetMatrix( 1, 1, sizeof(double2), WORK+i*N+j-1, 1, RNRM, 1 ) ;
      MAGMA_Z_SET2REAL( RNRM[0], lapackf77_zlange( "F", &c_ione, &c_ione, RNRM, &c_ione, RNRM ) );
      if( MAGMA_Z_GET_X( RNRM[0] ) > MAGMA_Z_GET_X( XNRM[0] ) *CTE ){
        goto L20;
      } 
    }
    *ITER = IITER ;  
    return 0;
  L20:
    IITER++ ;
  }
  *ITER = -ITERMAX - 1 ; 

 L40:
  magma_zpotrf_gpu(UPLO, N, A, LDA, INFO);
  if( *INFO != 0 ){
    return 0;
  }
  magma_zlacpy(N, NRHS, B , LDB, X, N);
  magma_zpotrs_gpu(UPLO, N, NRHS, A, LDA, X, LDB, INFO);
  return 0;
}

#undef max
