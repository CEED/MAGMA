#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cublas.h"
#include "magma.h"

int init_matrix(void *A, int size , int elem_size){
  float *AD; 
  AD = (float*)A ; 
  int j ; 
  
  for(j = 0; j < size; j++)
    AD[j] = (rand()) / (float)RAND_MAX;
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing sgesv
*/
int main(int argc , char **argv)
{
    cuInit( 0 );
    cublasInit( );

    int printall = 0 ;
    printout_devices( );

    int i, INFO[1], NRHS = 100, N = 0;
    int size[10] = {1024,2048,3072,4032,5184,6016,7040,8064,9088,10112};
    int num_problems = 10;

    if (argc != 1){
      for(i = 1; i<argc; i++){
	if (strcmp("-N", argv[i])==0)
          N = atoi(argv[++i]);
	else if (strcmp("-nrhs", argv[i])==0)
          NRHS = atoi(argv[++i]);
      }
      if (N>0) {
	size[0] = size[9] = N;
	num_problems = 1;
      }
    }

    printf("\nUsage: \n");
    printf("  testing_sgesv -nrhs %d -N %d\n\n", NRHS, 1024);

    N = size[9];
    
    TimeStruct start, end;
    printf("\n\n");
    printf("  N     NRHS       GPU GFlop/s      || b-Ax || / ||A||\n");
    printf("========================================================\n");
    
    int LDA, LDB, LDX;
    int maxnb = magma_get_sgetrf_nb(N);

    int lwork = N*maxnb;

    if (NRHS > maxnb)
      lwork = N * NRHS;

    LDB = LDX = LDA = N ;
    int status ;
    float *d_A , * d_B , *d_X ; 
    float *h_work_M_S;
    int *IPIV ;
    float *A , *B, *X; 
 
    status = cublasAlloc((N+32)*(N+32) + 32*maxnb + lwork+2*maxnb*maxnb, 
			 sizeof(float), (void**)&d_A ) ;
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (d_A)\n");
      exit(1);
    }
    status = cublasAlloc(LDB*NRHS, sizeof(float), (void**)&d_B ) ;
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (d_B)\n");
      exit(1);
    }
    status = cublasAlloc(LDB*NRHS, sizeof(float), (void**)&d_X ) ;
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (d_X)\n");
      exit(1);
    }

    status = cudaMallocHost( (void**)&h_work_M_S, (lwork+32*maxnb)*sizeof(float) );
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (h_work_M_S)\n");
      exit(1);
    }
    
    A = ( float *) malloc ( sizeof(float) * LDA*N);
    if( A == NULL ) {
      printf("Allocation Error\n");
      exit(1);
    }	    	
    B = ( float *) malloc ( sizeof(float) * LDB*NRHS);
    if( B == NULL ){
      printf("Allocation Error\n");
      exit(1);
    }	    	
    X = ( float *) malloc ( sizeof(float) *LDB*NRHS);
    if( X == NULL ) {
      printf("Allocation Error\n");
      exit(1);
    }	    	
    
    IPIV = ( int *) malloc ( sizeof (int) * N ) ;
    if( IPIV == NULL ) {
      printf("Allocation Error\n");
      exit(1);
    }	    	    	
    
    for(i=0; i<num_problems; i++){
      N = size[i];
      LDB = LDX = LDA = N ;

      int dlda = (N/32)*32;
      if (dlda<N) dlda+=32;

      init_matrix(A, LDA*N, sizeof(float));
      init_matrix(B, LDB * NRHS, sizeof(float));
      
      float perf;
      
      printf("%5d  %4d",N, NRHS); 
      
      cublasSetMatrix( N, N, sizeof( float ), A, N, d_A, dlda ) ;
      cublasSetMatrix( N, NRHS, sizeof( float ), B, N, d_B, N ) ;

      //=====================================================================
      // Solve Ax = b through an LU factorization
      //=====================================================================
      start = get_current_time();
      magma_sgetrf_gpu(&N, &N, d_A, &dlda, IPIV, INFO);
      magma_sgetrs_gpu("N", N, NRHS, d_A, dlda, IPIV, d_B, LDB, INFO, h_work_M_S);
      end = get_current_time();
      perf = (2.*N*N*N/3.+2.*NRHS*N*N)/(1000000*GetTimerValue(start,end));
      printf("             %6.2f", perf);
      cublasGetMatrix( N, NRHS, sizeof( float ), d_B , LDB, X ,LDX) ;
      
      //=====================================================================
      // ERROR
      //=====================================================================
      float Rnorm, Anorm, Bnorm;   
      float *worke = (float *)malloc(N*sizeof(float));

      Anorm = slange_("I", &N, &N, A, &LDA, worke);
      Bnorm = slange_("I", &N, &NRHS, B, &LDB, worke);

      float ONE = -1.0 , NEGONE = 1.0 ;
      sgemm_( "N", "N", &N, &NRHS, &N, &NEGONE, A, &LDA, X, &LDX, &ONE, B, &N);
      Rnorm=slange_("I", &N, &NRHS, B, &LDB, worke); 
      free(worke);
      
      printf("        %e\n", Rnorm/(Anorm*Bnorm));
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
