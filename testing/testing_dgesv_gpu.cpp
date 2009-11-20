#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cublas.h"
#include "magma.h"

int init_matrix(void *A, int size , int elem_size){
  double *AD; 
  AD = (double*)A ; 
  int j ; 
  
  for(j = 0; j < size; j++)
    AD[j] = (rand()) / (double)RAND_MAX;
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing dgesv
*/
int main(int argc , char **argv)
{
    cuInit( 0 );
    cublasInit( );

    int printall = 0 ;
    printout_devices( );

    int i, INFO[1], NRHS = 1, N = 0;
    int size[10] = {1024,2048,3072,4032,5184,6016,7040,8064,9088,10112};

    if (argc != 1){
      for(i = 1; i<argc; i++){
	if (strcmp("-N", argv[i])==0)
          N = atoi(argv[++i]);
      }
      if (N>0) size[0] = size[9] = N;
      else exit(1);
    }
    else {
      printf("\nUsage: \n");
      printf("  testing_dgesv -N %d\n\n", 1024);
    }

    N = size[9];
    
    TimeStruct start, end;
    printf("\n\n");
    printf("  N            GPU GFlop/s      || b-Ax || / ||A||\n");
    printf("========================================================\n");
    
    int LDA, LDB, LDX;
    int maxnb = magma_get_dgetrf_nb(N);

    int lwork = N*maxnb;

    LDB = LDX = LDA = N ;
    int status ;
    double *d_A , * d_B , *d_X ; 
    double *h_work_M_S;
    int *IPIV ;
    double *A , *B, *X; 
 
    status = cublasAlloc((N+32)*(N+32) + 32*maxnb + lwork+2*maxnb*maxnb, 
			 sizeof(double), (void**)&d_A ) ;
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (d_A)\n");
      exit(1);
    }
    status = cublasAlloc(LDB*NRHS, sizeof(double), (void**)&d_B ) ;
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (d_B)\n");
      exit(1);
    }
    status = cublasAlloc(LDB*NRHS, sizeof(double), (void**)&d_X ) ;
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (d_X)\n");
      exit(1);
    }

    status = cudaMallocHost( (void**)&h_work_M_S, (lwork+32*maxnb)*sizeof(double) );
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (h_work_M_S)\n");
      exit(1);
    }
    
    A = ( double *) malloc ( sizeof(double) * LDA*N);
    if( A == NULL ) {
      printf("Allocation Error\n");
      exit(1);
    }	    	
    B = ( double *) malloc ( sizeof(double) * LDB*NRHS);
    if( B == NULL ){
      printf("Allocation Error\n");
      exit(1);
    }	    	
    X = ( double *) malloc ( sizeof(double) *LDB*NRHS);
    if( X == NULL ) {
      printf("Allocation Error\n");
      exit(1);
    }	    	
    
    IPIV = ( int *) malloc ( sizeof (int) * N ) ;
    if( IPIV == NULL ) {
      printf("Allocation Error\n");
      exit(1);
    }	    	    	
    
    for(i=0; i<10; i++){
      N = size[i];
      LDB = LDX = LDA = N ;

      int dlda = (N/32)*32;
      if (dlda<N) dlda+=32;

      init_matrix(A, LDA*N, sizeof(double));
      init_matrix(B, LDB * NRHS, sizeof(double));
      
      double perf;
      
      printf("%5d",N); 
      
      cublasSetMatrix( N, N, sizeof( double ), A, N, d_A, dlda ) ;
      cublasSetMatrix( N, NRHS, sizeof( double ), B, N, d_B, N ) ;

      //=====================================================================
      //              SP - GPU 
      //=====================================================================
      start = get_current_time();
      magma_dgetrf_gpu(&N, &N, d_A, &dlda, IPIV, h_work_M_S, INFO);
      magma_dgetrs_gpu("N", N, NRHS, d_A, dlda, IPIV, d_B, LDB, INFO, h_work_M_S);
      end = get_current_time();
      perf = (2.*N*N*N/3.+2.*N*N)/(1000000*GetTimerValue(start,end));
      printf("             %6.2f", perf);
      cublasGetMatrix( N, NRHS, sizeof( double ), d_B , N, X ,N) ;
      
      //=====================================================================
      //              ERROR DP vs MIXED  - GPU 
      //=====================================================================
      double Rnorm, Anorm;   
      double *worke = (double *)malloc(N*sizeof(double));
      Anorm = dlange_("I", &N, &N, A, &LDA, worke);
      double ONE = -1.0 , NEGONE = 1.0 ;
      dgemm_( "No Transpose", "No Transpose", &N, &NRHS, &N, &NEGONE, A, &LDA, X, &LDX, &ONE, B, &N);
      Rnorm=dlange_("I", &N, &NRHS, B, &LDB, worke); 
      free(worke);
      
      printf("        %e", Rnorm/Anorm);
      
      printf("\n");
      if (argc != 1)
        break;
    }

    free(IPIV);
    free(X);
    free(B);
    free(A);
    cublasFree(h_work_M_S);
    cublasFree(d_X);
    cublasFree(d_B);
    cublasFree(d_A);

    cublasShutdown();
}
