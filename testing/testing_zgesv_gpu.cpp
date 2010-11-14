/*
 *  -- MAGMA (version 1.0) --
 *     Univ. of Tennessee, Knoxville
 *     Univ. of California, Berkeley
 *     Univ. of Colorado, Denver
 *     November 2010
 *
 * @precisions normal z -> c d s
 *
 **/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cublas.h"
#include "magma.h"

#define COMPLEX
#undef REAL

static int init_matrix(double2 *A, int size, int elem_size){
  int j ; 
  
  for(j = 0; j < size; j++) {
    A[j] = ( rand() / (double)RAND_MAX );
#if defined(COMPLEX)
    A[j] += I*( rand() / (double)RAND_MAX );
#endif
  }

  return 0;
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing sgesv
*/
int main(int argc , char **argv)
{
    cuInit( 0 );
    cublasInit( );

    printout_devices( );

    int i, info, NRHS = 100, N = 0;
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
    printf("  testing_zgesv -nrhs %d -N %d\n\n", NRHS, 1024);

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
    double2 *d_A , * d_B , *d_X ; 
    double2 *h_work_M_S;
    int *IPIV ;
    double2 *A , *B, *X; 
 
    status = cublasAlloc((N+32)*(N+32) + 32*maxnb + lwork+2*maxnb*maxnb, 
			 sizeof(double2), (void**)&d_A ) ;
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (d_A)\n");
      exit(1);
    }
    status = cublasAlloc(LDB*NRHS, sizeof(double2), (void**)&d_B ) ;
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (d_B)\n");
      exit(1);
    }
    status = cublasAlloc(LDB*NRHS, sizeof(double2), (void**)&d_X ) ;
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (d_X)\n");
      exit(1);
    }

    status = cudaMallocHost( (void**)&h_work_M_S, (lwork+32*maxnb)*sizeof(double2) );
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (h_work_M_S)\n");
      exit(1);
    }
    
    A = ( double2 *) malloc ( sizeof(double2) * LDA*N);
    if( A == NULL ) {
      printf("Allocation Error\n");
      exit(1);
    }	    	
    B = ( double2 *) malloc ( sizeof(double2) * LDB*NRHS);
    if( B == NULL ){
      printf("Allocation Error\n");
      exit(1);
    }	    	
    X = ( double2 *) malloc ( sizeof(double2) *LDB*NRHS);
    if( X == NULL ) {
      printf("Allocation Error\n");
      exit(1);
    }	    	
    
    IPIV = ( magma_int_t *) malloc ( sizeof(magma_int_t) * N ) ;
    if( IPIV == NULL ) {
      printf("Allocation Error\n");
      exit(1);
    }	    	    	
    
    for(i=0; i<num_problems; i++){
      N = size[i];
      LDB = LDX = LDA = N ;

      int dlda = (N/32)*32;
      if (dlda<N) dlda+=32;

      init_matrix(A, LDA*N,    sizeof(double2));
      init_matrix(B, LDB*NRHS, sizeof(double2));
      
      double perf;
      
      printf("%5d  %4d",N, NRHS); 
      
      cublasSetMatrix( N, N,    sizeof( double2 ), A, N, d_A, dlda ) ;
      cublasSetMatrix( N, NRHS, sizeof( double2 ), B, N, d_B, N    ) ;

      //=====================================================================
      // Solve Ax = b through an LU factorization
      //=====================================================================
      start = get_current_time();
      magma_zgetrf_gpu( N, N, d_A, dlda, IPIV, &info);
      magma_zgetrs_gpu('N', N, NRHS, d_A, dlda, IPIV, d_B, LDB, &info, h_work_M_S);
      end = get_current_time();
      perf = (2.*N*N*N/3.+2.*NRHS*N*N)/(1000000*GetTimerValue(start,end));
      printf("             %6.2f", perf);
      cublasGetMatrix( N, NRHS, sizeof( double2 ), d_B , LDB, X ,LDX) ;
      
      //=====================================================================
      // ERROR
      //=====================================================================
      double Rnorm, Anorm, Bnorm;   
      double *worke = (double *)malloc(N*sizeof(double));

      Anorm = lapackf77_zlange("I", &N, &N, A, &LDA, worke);
      Bnorm = lapackf77_zlange("I", &N, &NRHS, B, &LDB, worke);

      double2 ONE = -1.0 , NEGONE = 1.0 ;
      blasf77_zgemm( "N", "N", &N, &NRHS, &N, &NEGONE, A, &LDA, X, &LDX, &ONE, B, &N);
      Rnorm=lapackf77_zlange("I", &N, &NRHS, B, &LDB, worke); 
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
