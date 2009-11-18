#include <stdio.h>
#include <math.h>
#include "magmablas.h"
#include "magma.h"
#include "cublas.h"
#include "cuda.h"

#define cublasDtrsm magmablas_dtrsm
#define cublasDgemm magmablasDgemm
#define BWDMAX 1.0
#define ITERMAX 30

int
magma_dsgesv_gpu(int N, int NRHS, double *A, int LDA, int *IPIV, double *B, 
		 int LDB, double *X, int LDX, double *WORK, float *SWORK,
		 int *ITER, int *INFO, float *H_SWORK, double *H_WORK,
		 int *DIPIV)
{
/*  -- MAGMA (version 0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2009


    Purpose
    =======

    DSGESV computes the solution to a real system of linear equations
       A * X = B,
    where A is an N-by-N matrix and X and B are N-by-NRHS matrices.

    DSGESV first attempts to factorize the matrix in SINGLE PRECISION
    and use this factorization within an iterative refinement procedure
    to produce a solution with DOUBLE PRECISION normwise backward error
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

    N       (input) INTEGER
            The number of linear equations, i.e., the order of the
            matrix A.  N >= 0.

    NRHS    (input) INTEGER
            The number of right hand sides, i.e., the number of columns
            of the matrix B.  NRHS >= 0.

    A       (input or input/ouptut) DOUBLE PRECISION array,
            dimension (LDA,N)
            On entry, the N-by-N coefficient matrix A.
            On exit, if iterative refinement has been successfully used
            (INFO.EQ.0 and ITER.GE.0, see description below), then A is
            unchanged, if double precision factorization has been used
            (INFO.EQ.0 and ITER.LT.0, see description below), then the
            array A contains the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    IPIV    (output) INTEGER array, dimension (N)
            The pivot indices that define the permutation matrix P;
            row i of the matrix was interchanged with row IPIV(i).
            Corresponds either to the single precision factorization
            (if INFO.EQ.0 and ITER.GE.0) or the double precision
            factorization (if INFO.EQ.0 and ITER.LT.0).

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
                 -3 : failure of SGETRF
                 -31: stop the iterative refinement after the 30th
                      iterations
            > 0: iterative refinement has been sucessfully used.
                 Returns the number of iterations
 
    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
            > 0:  if INFO = i, U(i,i) computed in DOUBLE PRECISION is
                  exactly zero.  The factorization has been completed,
                  but the factor U is exactly singular, so the solution
                  could not be computed.

    H_SWORK (workspace) REAL array, dimension at least (nb, nb)
            where nb can be obtained through magma_get_spotrf_nb(*n)
            Work array allocated with cudaMallocHost.

    H_WORK  (workspace) DOUBLE array, dimension at least (nb, nb)
            where nb can be obtained through magma_get_dpotrf_nb(*n)
            Work array allocated with cudaMallocHost.

    DIPIV   (output) INTEGER array on the GPU, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was moved to row IPIV(i).

    =====================================================================    */

  #define MAX(a,b)       (((a)>(b))?(a):(b))

  /*
    Check The Parameters. 
  */
  *ITER = 0 ;
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
    magma_xerbla("magma_dsgesv",INFO) ;
  }

  if( N == 0 || NRHS == 0 )
    return 0;

  double ANRM , CTE , EPS;
  EPS  = dlamch_("Epsilon");
  ANRM = magma_dlange('I', N, N , A, LDA , WORK );
  CTE = ANRM * EPS *  pow((double)N,0.5) * BWDMAX ;

  int PTSA  = N*NRHS;
  int status ;
  float RMAX = slamch_("O");
  int IITER ;
  double alpha = -1.0;
  double beta = 1 ;
  int DLDA =  ( N / 32 ) * 32 ; 
  if ( DLDA < N ) 
	DLDA += 32 ;  
  magma_dlag2s(N , NRHS , B , LDB , SWORK, N , RMAX );
  if(*INFO !=0){
    *ITER = -2 ;
    printf("magmablas_dlag2s\n");
    goto L40;
  }
  magma_dlag2s(N , N , A , LDA , SWORK+PTSA, LDA , RMAX); // Merge with DLANGE /
  if(*INFO !=0){
    *ITER = -2 ;
    printf("magmablas_dlag2s\n");
    goto L40;
  }
  double XNRM[1] , RNRM[1] ;
  magma_sgetrf_gpu2(&N, &N,SWORK+PTSA, &LDA,IPIV, DIPIV, H_SWORK, INFO);
  if(INFO[0] !=0){
    *ITER = -3 ;
    goto L40;
  }
  magma_sdgetrs_gpu(&N,&NRHS,SWORK+PTSA,&LDA,DIPIV,SWORK, B,&LDB, INFO);
  int i,j ;
  magmablas_slag2d(N , NRHS , SWORK, N , X , LDX , INFO );
  magma_dlacpy(N, NRHS, B , LDB, WORK, N);
  if( NRHS == 1 )
   magmablas_magma_dgemv_MLU(N,N,A,LDA,X,WORK);
  else
   cublasDgemm( 'N', 'N', N, NRHS, N, -1.0, A, LDA, X, LDX, 1.0, WORK, N);


  for(i=0;i<NRHS;i++){
    j = cublasIdamax( N ,X+i*N, 1) ;
    cublasGetMatrix( 1, 1, sizeof(double), X+i*N+j-1, 1,XNRM, 1 ) ;
    XNRM[0]= fabs( XNRM[0]);
    j = cublasIdamax ( N , WORK+i*N  , 1 ) ;
    cublasGetMatrix( 1, 1, sizeof(double), WORK+i*N+j-1, 1, RNRM, 1 ) ;
    RNRM[0] =fabs( RNRM[0]);
    // printf("\n\t\t--   %lf  %lf --\n", RNRM[0] , XNRM[0]*CTE ); 
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
/*
        Convert R (in WORK) from double precision to single precision
        and store the result in SX.
        Solve the system SA*SX = SR.
        -- These two Tasks are merged here. 
*/
    magma_sdgetrs_gpu(&N,&NRHS,SWORK+PTSA,&LDA,DIPIV, SWORK,WORK, &LDB, INFO);
    if(INFO[0] !=0){
      *ITER = -3 ;
      goto L40;
    }
    for(i=0;i<NRHS;i++){
       magmablas_sdaxpycp(SWORK+i*N,X+i*N,N,N,LDA,B+i*N,WORK+i*N) ;
    }
/*
unnecessary may be*/
    magma_dlacpy(N, NRHS, B , LDB, WORK, N);
    if( NRHS == 1 )
        magmablas_magma_dgemv_MLU(N,N, A,LDA,X,WORK);
    else
        cublasDgemm( 'N', 'N', N, NRHS, N, alpha, A, LDA, X, LDX, beta, WORK, N);

/*
        Check whether the NRHS normwise backward errors satisfy the
        stopping criterion. If yes, set ITER=IITER>0 and return.
*/
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
/*
        If we are here, the NRHS normwise backward errors satisfy the
        stopping criterion, we are good to exit.
*/

    *ITER = IITER ;
    return 0;
    L20:
    IITER++ ;
  }
  /*
     If we are at this place of the code, this is because we have
     performed ITER=ITERMAX iterations and never satisified the
     stopping criterion, set up the ITER flag accordingly and follow up
     on double precision routine.
  */
  *ITER = -ITERMAX - 1 ;

  L40:
  /*
     Single-precision iterative refinement failed to converge to a
     satisfactory solution, so we resort to double precision.  
  */
  if( *INFO != 0 ){
    return 0;
  }

  magma_dgetrf_gpu(&N, &N, A, &LDA, IPIV, H_WORK, INFO);
  magma_dlacpy(N, NRHS, B , LDB, X, N);
  magma_dgetrs_gpu("N",N ,NRHS, A ,LDA,IPIV, X,N,INFO,H_WORK);
  return 0;
}

#undef MAX
