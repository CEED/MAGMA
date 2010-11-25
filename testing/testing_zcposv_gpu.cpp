/*
    -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

       @precisions mixed zc -> ds
*/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include "magma.h"
#include "magmablas.h"

#define PRECISION_z

int main(int argc, char **argv)
{
#if defined(PRECISION_z) && (GPUSHMEM < 200)
  fprintf(stderr, "This functionnality is not available in MAGMA for this precisions actually\n");
  return EXIT_SUCCESS;
#else
  printf("Iterative Refinement- Cholesky \n");
  printf("\n");
  cuInit( 0 );
  cublasInit( );
    
  printout_devices( );


  printf("\nUsage:\n\t\t ./testing_dsposv -N 1024");
  printf("\n\nEpsilon(CuDoubleComplex): %10.20lf \nEpsilon(Single): %10.20lf\n", 
	 lapackf77_dlamch("Epsilon"), lapackf77_slamch("Epsilon"));

  TimeStruct start, end;

  int i ;
  int startN= 1024 ;
  int count = 8;
  int step = 1024;  
  int N = count * step ;
  int NRHS=1 ;
  int once  = 0 ;  

  N =startN+(count-1) * step ;

  if( argc == 3) { 
      N  = atoi( argv[2]);
      once = N ; 
  }
  int sizetest[10] = {1024,2048,3072,4032,5184,6016,7040,8064,9088,10112};

  printf("  N   DP-Factor  DP-Solve  SP-Factor  SP-Solve  MP-Solve  ||b-Ax||/||A||  NumIter\n");
  printf("==================================================================================\n");

  int size; 
  int LDA, LDB, LDX;
  int ITER, INFO;

  LDB = LDX = LDA = N ;

  int status ;
   
  cuDoubleComplex *M_WORK;
  cuFloatComplex  *M_SWORK;
  cuDoubleComplex *As, *Bs, *Xs;
  cuDoubleComplex *res_ ; 
  cuDoubleComplex *CACHE ;  
  int CACHE_L  = 10000 ;
  cuDoubleComplex *d_A , * d_B , *d_X;
  int ione     = 1;
  int ISEED[4] = {0,0,0,1};

  size =N*N;
  size+= N*NRHS;
  status = cublasAlloc( size, sizeof(cuFloatComplex), (void**)&M_SWORK ) ;
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (M_SWORK)\n");
    goto FREE2;
  }

  size= N*NRHS;
  status = cublasAlloc( size, sizeof(cuDoubleComplex), (void**)&M_WORK ) ;
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (M_WORK)\n");
    goto FREE2_1;
  }
  
  size =LDA * N ;
  As = ( cuDoubleComplex *) malloc ( sizeof ( cuDoubleComplex ) * size);
  if( As == NULL ){
    printf("Allocation A\n");
    goto FREE3;
  }

  size = NRHS * N ;
  Bs = ( cuDoubleComplex *) malloc ( sizeof ( cuDoubleComplex ) * size);
  if( Bs == NULL ){
    printf("Allocation A\n");
    goto FREE4;
  }

  Xs = ( cuDoubleComplex *) malloc ( sizeof ( cuDoubleComplex ) * size);
  if( Xs == NULL ){
    printf("Allocation A\n");
    goto FREE5;
  }
 
  size = N*NRHS ;
  res_ = ( cuDoubleComplex *) malloc ( sizeof(cuDoubleComplex)*size);
  if( res_ == NULL ){
    printf("Allocation A\n");
    goto FREE6;
  }

  size = CACHE_L * CACHE_L ;
  CACHE = ( cuDoubleComplex *) malloc ( sizeof( cuDoubleComplex) * size ) ; 
  if( CACHE == NULL ){
    printf("Allocation A\n");
    goto FREE7;
  }
  
  size = N * N ;
  status = cublasAlloc( size, sizeof(cuDoubleComplex), (void**)&d_A ) ;
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (dipiv)\n");
    goto FREE8;
  }

  size = LDB * NRHS ;
  d_B = ( cuDoubleComplex *) malloc ( sizeof ( cuDoubleComplex ) * size);
  status = cublasAlloc( size, sizeof(cuDoubleComplex), (void**)&d_B ) ;
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (dipiv)\n");
    goto FREE9;
  }

  d_X = ( cuDoubleComplex *) malloc ( sizeof ( cuDoubleComplex ) * size);
  status = cublasAlloc( size, sizeof(cuDoubleComplex), (void**)&d_X ) ;
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (dipiv)\n");
    goto FREE10;
  }

  for(i=0;i<count;i++){
      NRHS =  1 ; 
      N = step*(i)+startN;
      if( once == 0 ) 
          N = sizetest[i] ;
      else N  = once ;
  
      LDB = LDX = LDA = N ;

      size = LDA * N ;
      /* Initialize the matrix */
      lapackf77_zlarnv( &ione, ISEED, &size, As );
      /* Symmetrize and increase the diagonal */
      { 
        int i, j;
        for(i=0; i<N; i++) {
          As[i*LDA+i] = MAGMA_Z_MAKE( (MAGMA_Z_GET_X(As[i*LDA+i]) + 2000.), 0. );
          
          for(j=0; j<i; j++)
            As[i*LDA+j] = As[j*LDA+i];
        }
      }
      
      size = LDB * NRHS ;
      lapackf77_zlarnv( &ione, ISEED, &size, Bs );
      
      cublasSetMatrix( N, N,    sizeof( cuDoubleComplex ), As, N, d_A, N ) ;
      cublasSetMatrix( N, NRHS, sizeof( cuDoubleComplex ), Bs, N, d_B, N ) ;
    
      double perf ;  

      printf("%5d  ", N); 
      fflush(stdout);

      char uplo = 'L';

      //=====================================================================
      //              Mixed Precision Iterative Refinement - GPU 
      //=====================================================================
      start = get_current_time();
      magma_zcposv_gpu(uplo, N, NRHS, d_A, LDA, d_B, LDA, d_X, LDA, M_WORK, 
                       M_SWORK, &ITER, &INFO);
      end = get_current_time();
      perf = (1.*N*N*N/3.+2.*N*N)/(1000000*GetTimerValue(start,end));
      cublasGetMatrix( N, NRHS, sizeof( cuDoubleComplex), d_X, N,res_, N ) ;
      double lperf = perf ; 

      //=====================================================================
      //                 Error Computation 
      //=====================================================================
      char norm='I';
      char side ='L';
      cuDoubleComplex ONE = MAGMA_Z_NEG_ONE;
      cuDoubleComplex NEGONE = MAGMA_Z_ONE ;

      double Rnorm, Anorm;
      double *worke = (double *)malloc(N*sizeof(double));
      Anorm = lapackf77_zlanhe( &norm, &uplo,  &N, As, &N, worke);
      blasf77_zhemm( &side, &uplo, &N, &NRHS, &NEGONE, As, &LDA, res_, &LDX, &ONE, Bs, &N);
      Rnorm = lapackf77_zlange("I", &N, &NRHS, Bs, &LDB, worke);
      free(worke);

      //=====================================================================
      //                 Double Precision Factor 
      //=====================================================================
      start = get_current_time();
      magma_zpotrf_gpu(uplo, N, d_A, LDA, &INFO);
      end = get_current_time();
      perf = (1.*N*N*N/3.)/(1000000*GetTimerValue(start,end));
      printf("%6.2f    ", perf);
      fflush(stdout);

      //=====================================================================
      //                 Double Precision Solve 
      //=====================================================================
      start = get_current_time();
      magma_zpotrf_gpu(uplo, N, d_A, LDA, &INFO);
      magma_zpotrs_gpu('L', N, NRHS, d_A, LDA, d_B, LDB, &INFO);
      end = get_current_time();
      perf = (1.*N*N*N/3.+2.*N*N)/(1000000*GetTimerValue(start,end));
      printf("%6.2f    ", perf);
      fflush(stdout);

      //=====================================================================
      //                 Single Precision Factor 
      //=====================================================================
      start = get_current_time();
      magma_cpotrf_gpu(uplo, N, M_SWORK+N*NRHS, LDA, &INFO);
      end = get_current_time();
      perf = (1.*N*N*N/3.)/(1000000*GetTimerValue(start,end));
      printf("%6.2f     ", perf);
      fflush(stdout);

      //=====================================================================
      //                 Single Precision Solve 
      //=====================================================================
      start = get_current_time();
      magma_cpotrf_gpu(uplo, N, M_SWORK+N*NRHS, LDA, &INFO);
      magma_cpotrs_gpu('L', N, NRHS, M_SWORK+N*NRHS, LDA, M_SWORK, LDB, &INFO);
      end = get_current_time();
      perf = (1.*N*N*N/3.+2.*N*N)/(1000000*GetTimerValue(start,end));
      printf("%6.2f    ", perf);
      fflush(stdout);

      printf("%6.2f     ", lperf);
      printf("%e    %3d", Rnorm/Anorm, ITER);

      fflush(stdout);

      printf("\n");

      if( once != 0 ){
	break;
      } 
  }

   cublasFree(d_X);
FREE10: 
   cublasFree(d_B);
FREE9: 
   cublasFree(d_A);
FREE8: 
    free(CACHE);
FREE7:
    free(res_);
FREE6:
    free(Xs);
FREE5:
    free(Bs);
FREE4:
    free(As);
FREE3:
    cublasFree(M_WORK);  
FREE2_1:
    cublasFree(M_SWORK);  
FREE2:
    cublasShutdown();

#endif /*defined(PRECISION_z) && (GPUSHMEM < 200)*/
}
