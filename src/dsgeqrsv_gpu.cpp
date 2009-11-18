#include <stdio.h>
#include <math.h>
#include "magmablas.h"
#include "magma.h"
#include "cublas.h"
#include "cuda.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cublas.h"
#include "magma.h"


#define MAX(a,b)       (((a)>(b))?(a):(b))

#define BWDMAX 1.0
#define ITERMAX 30

extern "C" double dlamch_(char *);
extern "C" float slamch_(char *);

void magma_dsgeqrsv
(
int M , int N ,int NRHS, 
double *A,int LDA ,double *B,int LDB,double *X,int LDX, double *WORK, 
float *SWORK,int *ITER,int *INFO, 
float *tau , int lwork , float *h_work , float *d_work ,
double *tau_d , int lwork_d , double *h_work_d , double *d_work_d 
){
  /*
    Check The Parameters. 
  */
  *ITER = 0 ;
  *INFO = 0 ;
  if ( N <0)
    *INFO = -1;
  else if(NRHS<0)
    *INFO =-3;
  else if(LDA < MAX(1,N))
    *INFO =-5;
  else if( LDB < MAX(1,N))
    *INFO =-7;
  else if( LDX < MAX(1,N))
    *INFO =-9;

  if(*INFO!=0){
    printf("%d %d %d\n", M , N , NRHS);
    magma_xerbla("magma_dsgeqrsv_gpu",INFO) ;
  }

  if( N == 0 || NRHS == 0 )
    return;

  double ANRM , CTE , EPS;

  EPS  = dlamch_("Epsilon");
  ANRM = magma_dlange('I', N, N , A, LDA , WORK );
  CTE = ANRM * EPS *  pow((double)N,0.5) * BWDMAX ;
  int PTSA  = N*NRHS;
  float RMAX = slamch_("O");
  int IITER ;
  double alpha = -1.0;
  double beta = 1 ;
  magma_dlag2s(N , NRHS , B , LDB , SWORK, N , RMAX );
  if(*INFO !=0){
    *ITER = -2 ;
    printf("magmablas_dlag2s\n");
    goto L40;
  }
  magma_dlag2s(N , N , A , LDA , SWORK+PTSA, N , RMAX); // Merge with DLANGE /
  if(*INFO !=0){
    *ITER = -2 ;
    printf("magmablas_dlag2s\n");
    goto L40;
  }
  double XNRM[1] , RNRM[1] ;

  /*
   In an ideal version these variables should come from 
   user. 
  */
  magma_sgeqrf_gpu2(&M, &N,  SWORK+PTSA , &N, tau, h_work, &lwork, d_work, INFO);
/*
*/
  // magma_sgetrf_gpu2(&N, &N, SWORK+PTSA, &N,IPIV, DIPIV, H_SWORK,        INFO);
  if(INFO[0] !=0){
    *ITER = -3 ;
    goto L40;
  }
  // SWORK = B 
  magma_sgeqrs_gpu(&M, &N, &NRHS,  SWORK+PTSA  ,       &N,  tau,      SWORK , &M, h_work, &lwork, d_work, INFO );
  //magma_sdgetrs_gpu(&N,        &NRHS, SWORK+PTSA,&LDA,DIPIV,SWORK, B, &LDB                       , INFO);
 // SWORK = X in SP 

  magmablas_slag2d(N , NRHS , SWORK, N , X , LDX , INFO );
 // X = X in DP 
  magma_dlacpy(N, NRHS, B , LDB, WORK, N);
// WORK = B in DP 

  if( NRHS == 1 )
   magmablas_magma_dgemv_MLU(N,N,A,LDA,X,WORK);
  else
   cublasDgemm( 'N', 'N', N, NRHS, N, -1.0, A, LDA, X, LDX, 1.0, WORK, N);
// WORK contains the residula .. 

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
  return ;



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
// make SWORK = WORK ... residuals... 
 //   magma_sdgetrs_gpu(&N,&NRHS,SWORK+PTSA,&LDA,DIPIV, SWORK,WORK, &LDB, INFO);

  magma_dlag2s(N , NRHS , WORK , LDB , SWORK, N , RMAX );
  magma_sgeqrs_gpu(&M, &N, &NRHS,  SWORK+PTSA  ,       &N,  tau,      SWORK , &M, h_work, &lwork, d_work, INFO );
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
    return ;
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
 magma_dgeqrf_gpu2(&M, &N, A, &N, tau_d, h_work_d, &lwork_d, d_work_d, INFO);
  if( *INFO != 0 ){
    return ;
  }
 magma_dlacpy(N, NRHS, B , LDB, X, N);
 magma_dgeqrs_gpu(&M, &N, &NRHS, A, &N, tau_d,
                       X, &M, h_work_d, &lwork_d, d_work_d, INFO);
  return ;
}

#undef MAX
