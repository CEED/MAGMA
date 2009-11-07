#include <stdio.h>
#include <math.h>
#include "magmablas.h"
#include "magma.h"
#include "cublas.h"
#include "cuda.h"
#define ITERMAX 30
#define BWDMAX 1.0

void magma_spotrs_gpu( char *UPLO , int N , int NRHS, float *A , int LDA ,float *B, int LDB, int *INFO);
void magma_dpotrs_gpu( char *UPLO , int N , int NRHS, double *A , int LDA ,double *B, int LDB, int *INFO);

int MAX( int a, int b){
 return a>b ? a: b ;
}



void magma_dsposv(
               char UPLO, 
	       int N ,
	       int NRHS, 
	       double *A, 
	       int LDA ,
	       double *B, 
	       int LDB,
	       double *X,
	       int LDX,
	       double *WORK,
	       float *SWORK,
	       int *ITER,
	       int *INFO
	       ,
	       float *h_work,
	       double *h_work2 
		){

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
    printf("ERROR in PARAMETER NUMBER %d\n",*INFO);
  }


  if( N == 0 || NRHS == 0 ) 
    return;

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
    printf("magmablas_dlag2s\n");
    goto L40;
  } 
  
  magma_dlat2s(UPLO ,  N ,  A , LDA , SWORK+PTSA, N , INFO ); 
  if(*INFO !=0){
    *ITER = -2 ;
    printf("magmablas_dlag2s\n");
    goto L40;
  }   
  magma_spotrf_gpu(&UPLO, &N, SWORK+ PTSA, &LDA, h_work, INFO);
  if(INFO[0] !=0){
    *ITER = -3 ;
    goto L40;
  }
  magma_spotrs_gpu( &UPLO,N ,NRHS, SWORK+PTSA , LDA ,SWORK+PTSX,LDB,INFO);
  magmablas_slag2d(N , NRHS , SWORK+PTSX, N , X , LDX , INFO );

  magma_dlacpy(N, NRHS, B , LDB, WORK, N);
  /*
  if( NRHS == 1 )
   magmablas_magma_dgemv_MLU(N,N,A,LDA,X,WORK);
  else 
   cublasDgemm( 'N', 'N', N, NRHS, N, -1.0, A, LDA, X, LDX, 1.0, WORK, N);
  */
  //cublasDsymm('L', UPLO , N , NRHS , -1.0,  A , LDA , X , LDX , 1 , WORK , N );
  magma_dsymv('L', UPLO , N ,  -1.0,  A , LDA , X ,  1 , 1.0 ,  WORK , 1 );
//extern "C" void mdsymv (char side , char uplo , int m , double alpha ,  double *A , int lda ,
// double *X , int incx , double beta , double *Y , int incy )

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
    magma_dlag2s(N , NRHS , WORK , N , SWORK+PTSX , N ,RMAX ) ;
    if(*INFO !=0){
      *ITER = -2 ;
      goto L40;
    } 
    magma_spotrs_gpu( &UPLO,N ,NRHS, SWORK+PTSA , LDA ,SWORK+PTSX,LDB,INFO);

    //magmablas_slag2d_64_64_16_4_v2(N , NRHS , SWORK+PTSX, N ,WORK , N , INFO );
    //magmablas_sdaxpycp(SWORK,X,N,N,LDA,B,WORK) ;
    for(i=0;i<NRHS;i++){
        //cublasDaxpy(N, 1.0, WORK+i*N, 1 , X+i*N, 1);
        //magmablas_sdaxpy(SWORK+i*N,X+i*N,N,N,LDA) ;
        magmablas_sdaxpycp(SWORK+i*N, X+i*N, N, N, LDA, B+i*N,WORK+i*N) ;
    }
    //magmablas_dlacpy_64_64_16_4_v2(N, NRHS, B , LDB, WORK, N);
/*
    if( NRHS == 1 )
        magmablas_magma_dgemv_MLU(N,N, A,LDA,X,WORK);
    else 
        cublasDgemm( 'N', 'N', N, NRHS, N, alpha, A, LDA, X, LDX, beta, WORK, N);
*/    

      magma_dsymv('L', UPLO , N , alpha,  A , LDA , X ,  1 , beta ,  WORK , 1 );
//    cublasDsymm('L', UPLO , N , NRHS , alpha,  A , LDA , X , LDX , beta , WORK , N );

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
    return ;
    L20:
    IITER++ ;
  }
  *ITER = -ITERMAX - 1 ; 

  L40:
  magma_dpotrf_gpu(&UPLO, &N, A, &LDA, h_work2, INFO);
  if( *INFO != 0 ){
    return ;
  }
  magma_dlacpy(N, NRHS, B , LDB, X, N);
  magma_dpotrs_gpu(&UPLO,N ,NRHS, A  , LDA ,X,LDB,INFO);
  return ;
}


void magma_spotrs_gpu( char *UPLO , int N , int NRHS, float *A , int LDA ,float *B, int LDB, int *INFO){
                if( *UPLO =='U'){
                         magmablas_strsm('L','U','T','N', N , NRHS,   A , LDA , B , LDB );
                         magmablas_strsm('L','U','N','N', N , NRHS,   A , LDA , B , LDB );
                }
                else{
                         magmablas_strsm('L','L','N','N', N , NRHS,  A , LDA , B , LDB );
                         magmablas_strsm('L','L','T','N', N , NRHS,  A , LDA , B , LDB );
                }
}
void magma_dpotrs_gpu( char *UPLO , int N , int NRHS, double *A , int LDA ,double *B, int LDB, int *INFO){
                if( *UPLO =='U'){
                         magmablas_dtrsm('L','U','T','N', N , NRHS,   A , LDA , B , LDB );
                         magmablas_dtrsm('L','U','N','N', N , NRHS,   A , LDA , B , LDB );
                }
                else{
                         magmablas_dtrsm('L','L','N','N', N , NRHS,  A , LDA , B , LDB );
                         magmablas_dtrsm('L','L','T','N', N , NRHS,  A , LDA , B , LDB );
                }
}


