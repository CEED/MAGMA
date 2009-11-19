#include <stdio.h>
#include <math.h>
#include "magmablas.h"
#include "magma.h"
#include "cublas.h"
#include "cuda.h"

#define cublasDtrsm magmablas_dtrsm
#define cublasDgemm magmablasDgemm
#define ITERMAX 30
#define BWDMAX 1.0

int
magma_dsposv_gpu(char UPLO, int N, int NRHS, double *A,int LDA, double *B, 
                 int LDB, double *X, int LDX, double *WORK, float *SWORK,
                 int *ITER, int *INFO, float *H_SWORK, double *H_WORK)
{
/*  -- MAGMA (version 0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2009

    Purpose
    =======

    DSPOSV computes the solution to a real system of linear equations
       A * X = B,
    where A is an N-by-N symmetric positive definite matrix and X and B
    are N-by-NRHS matrices.

    DSPOSV first attempts to factorize the matrix in SINGLE PRECISION
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

    A       (input or input/output) DOUBLE PRECISION array,
            dimension (LDA,N)
            On entry, the symmetric matrix A.  If UPLO = 'U', the leading
            N-by-N upper triangular part of A contains the upper
            triangular part of the matrix A, and the strictly lower
            triangular part of A is not referenced.  If UPLO = 'L', the
            leading N-by-N lower triangular part of A contains the lower
            triangular part of the matrix A, and the strictly upper
            triangular part of A is not referenced.
            On exit, if iterative refinement has been successfully used
            (INFO.EQ.0 and ITER.GE.0, see description below), then A is
            unchanged, if double precision factorization has been used
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
            < 0: iterative refinement has failed, double precision
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
            where nb can be obtained through magma_get_spotrf_nb(*n)
            Work array allocated with cudaMallocHost.

    H_WORK   (workspace) DOUBLE array, dimension at least (nb, nb)
             where nb can be obtained through magma_get_dpotrf_nb(*n)
             Work array allocated with cudaMallocHost.

    =====================================================================    */

  #define MAX(a,b)       (((a)>(b))?(a):(b))

  *ITER = 0 ;
  *INFO = 0 ; 

  if ( N <0)
    *INFO = -1;
  else if(NRHS<0)
    *INFO =-2;
  else if(LDA < MAX(1,N))
    *INFO =-4;
  else if( LDB < MAX(1,N))
    *INFO =-7;
  else if( LDX < MAX(1,N))
    *INFO =-9;
   
  if(*INFO!=0){
    magma_xerbla("magma_dsposv",INFO);
  }

  if( N == 0 || NRHS == 0 ) 
    return 0;

  double ANRM , CTE , EPS;
  EPS  = dlamch_("Epsilon");
  ANRM = magma_dlansy(  'I',  UPLO , N ,A, LDA ,WORK);
  CTE = ANRM * EPS *  pow((double)N,0.5) * BWDMAX ;  

  int PTSA  = N*NRHS;
  int PTSX  = 0 ;  
  int status ; 
  float RMAX = slamch_("O");
  int IITER ;
  double alpha = -1.0;
  double beta = 1 ; 
  double XNRM[1] , RNRM[1]; 
  int i1,j1,ii;
 
  magma_dlag2s(N , NRHS , B , LDB , SWORK+PTSX, N , RMAX );
  if(*INFO !=0){
    *ITER = -2 ;
    goto L40;
  } 
  
  magma_dlat2s(UPLO ,  N ,  A , LDA , SWORK+PTSA, N , INFO ); 
  if(*INFO !=0){
    *ITER = -2 ;
    goto L40;
  }   
  magma_spotrf_gpu(&UPLO, &N, SWORK+ PTSA, &LDA, H_SWORK, INFO);
  if(INFO[0] !=0){
    *ITER = -3 ;
    goto L40;
  }
  magma_spotrs_gpu( &UPLO,N ,NRHS, SWORK+PTSA , LDA ,SWORK+PTSX,LDB,INFO);
  magmablas_slag2d(N , NRHS , SWORK+PTSX, N , X , LDX , INFO );

  magma_dlacpy(N, NRHS, B , LDB, WORK, N);

  if( NRHS == 1 )
  magma_dsymv( UPLO , N ,  -1.0,  A , LDA , X ,  1 , 1.0 ,  WORK , 1 );
  else
  cublasDsymm('L', UPLO , N , NRHS , -1.0,  A , LDA , X , LDX , 1 , WORK , N );

  int i,j ;
  for(i=0;i<NRHS;i++){
    j = cublasIdamax( N ,X+i*N, 1) ; 
    cublasGetMatrix( 1, 1, sizeof(double), X+i*N+j-1, 1,XNRM, 1 ) ;
    XNRM[0]= fabs( XNRM[0]);
    j = cublasIdamax ( N , WORK+i*N  , 1 ) ; 
    cublasGetMatrix( 1, 1, sizeof(double), WORK+i*N+j-1, 1, RNRM, 1 ) ;
    RNRM[0] =fabs( RNRM[0]); 
    if( RNRM[0] > XNRM[0]*CTE ){
      goto L10;
    } 
  }
  *ITER =0; 
  return 0;

 L10:
  ;

  for(IITER=1;IITER<ITERMAX;){
    *INFO = 0 ; 
    magma_dlag2s(N , NRHS , WORK , N , SWORK+PTSX , N ,RMAX ) ;
    if(*INFO !=0){
      *ITER = -2 ;
      goto L40;
    } 
    magma_spotrs_gpu( &UPLO,N ,NRHS, SWORK+PTSA , LDA ,SWORK+PTSX,LDB,INFO);

    for(i=0;i<NRHS;i++){
        magmablas_sdaxpycp(SWORK+i*N, X+i*N, N, N, LDA, B+i*N,WORK+i*N) ;
    }

    if( NRHS == 1 )
      magma_dsymv( UPLO , N , alpha,  A , LDA , X ,  1 , beta ,  WORK , 1 );
    else 
     cublasDsymm('L', UPLO , N , NRHS , alpha,  A , LDA , X , LDX , beta , WORK , N );

    for(i=0;i<NRHS;i++){
      int j,inc=1 ;
      j = cublasIdamax( N , X+i*N  , 1) ; 
      cublasGetMatrix( 1, 1, sizeof(double), X+i*N+j-1, 1, XNRM, 1 ) ;
      XNRM[0] =  fabs (XNRM[0]) ;
      j = cublasIdamax ( N ,WORK+i*N , 1 ) ; 
      cublasGetMatrix( 1, 1, sizeof(double), WORK+i*N+j-1, 1, RNRM, 1 ) ;
      RNRM[0] =  fabs (RNRM[0]); 
      if( RNRM[0] > (XNRM[0]*CTE) ){
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
  magma_dpotrf_gpu(&UPLO, &N, A, &LDA, H_WORK, INFO);
  if( *INFO != 0 ){
    return 0;
  }
  magma_dlacpy(N, NRHS, B , LDB, X, N);
  magma_dpotrs_gpu(&UPLO,N ,NRHS, A  , LDA ,X,LDB,INFO);
  return 0;
}

#undef MAX
