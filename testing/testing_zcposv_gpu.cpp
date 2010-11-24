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
#include "magmablas.h"
#include "magma.h"

int init_matrix_sym(void *A, int size , int elem_size, int lda){
    int i ;
    int j ;
    if( elem_size==sizeof(cuDoubleComplex)){

	cuDoubleComplex *AD; 

	AD = (cuDoubleComplex*)A ; 

        for(i = 0; i< size*size ; i++) 
           AD[i]= (rand()) / (cuFloatComplex)RAND_MAX;
        for(j=0; j<size*size; j+=(lda+1)){
	   AD[j]+=2000; 
	}

        for(i = 0; i< size ; i++) 
        for(j = 0; j <i; j++){
           AD[j*lda+i]= AD[i*lda+j];
        } 

        for(i = 0; i< size ; i++) 
        for(j = 0; j <size; j++){
           if( AD[j*lda+i]!= AD[i*lda+j]) exit(1);
        } 
    }

    else if( elem_size==sizeof(cuFloatComplex)){
	cuFloatComplex *AD; 
	AD = (cuFloatComplex*)A ; 
        for(i = 0; i< size*size ; i++)
           AD[i]= (rand()) / (cuFloatComplex)RAND_MAX;
        for(j=0; j<size*size; j+=(lda+1)){
           AD[j]+=2000;
        }

        for(i = 0; i< size ; i++)
        for(j = 0; j <i; j++){
           AD[j*lda+i]= AD[i*lda+j];
        }

        for(i = 0; i< size ; i++)
        for(j = 0; j <size; j++){
           if( AD[j*lda+i]!= AD[i*lda+j]) exit(1);
        }
    }
}

int init_matrix(void *A, int size , int elem_size){
    if( elem_size==sizeof(cuDoubleComplex)){
	cuDoubleComplex *AD; 
	AD = (cuDoubleComplex*)A ; 
	int j ; 
	
        for(j = 0; j < size; j++)
           AD[j] = (rand()) / (cuDoubleComplex)RAND_MAX;
    }
    else if( elem_size==sizeof(cuFloatComplex)){
	cuFloatComplex *AD; 
	AD = (cuFloatComplex*)A ; 
	int j ; 
        for(j = 0; j < size; j++)
             AD[j] = (rand()) / (cuFloatComplex)RAND_MAX;
    }
}

int copy_matrix(void *S, void *D,int size , int elem_size){
    if( elem_size==sizeof(cuDoubleComplex)){
	cuDoubleComplex *SD, *DD; 
	SD = (cuDoubleComplex*)S ; 
	DD = (cuDoubleComplex*)D ; 
	int j ; 
        for(j = 0; j < size; j++)
	    DD[j]=SD[j];
    }
    else if( elem_size==sizeof(cuFloatComplex)){
	cuFloatComplex *SD, *DD; 
	SD = (cuFloatComplex*)S ; 
	DD = (cuFloatComplex*)D ; 
	int j ; 
        for(j = 0; j < size; j++)
	    DD[j]=SD[j];
    }
}

void die(char *message){
  printf("Error in %s\n",message);
  fflush(stdout);
  exit(-1);	
}

int main(int argc, char **argv)
{
  int  printall =0 ; 
  
  printf("Iterative Refinement- Cholesky \n");
  printf("\n");
  cuInit( 0 );
  cublasInit( );
    
  printout_devices( );


  printf("\nUsage:\n\t\t ./testing_dsposv -N 1024");
  printf("\n\nEpsilon(CuDoubleComplex): %10.20lf \nEpsilon(Single): %10.20lf\n", 
	 lapackf77_dlamch("Epsilon"), lapackf77_slamch("Epsilon"));

  TimeStruct start, end;

  int LEVEL=1;
  int i ;
  int startN= 1024 ;
  int count = 8;
  int step = 1024;  
  int N = count * step ;
  int NRHS=1 ;
  int error = 0 ;
  int once  = 0 ;  

  N =startN+(count-1) * step ;

  if( argc == 3) { 
      N  = atoi( argv[2]);
      once = N ; 
  }
  int sizetest[10] = {1024,2048,3072,4032,5184,6016,7040,8064,9088,10112};

  printf("  N   DP-Factor  DP-Solve  SP-Factor  SP-Solve  MP-Solve  ||b-Ax||/||A||  NumIter\n");
  printf("==================================================================================\n");

  int size ; 
  int LDA ;
  int LDB ;
  int LDX ;
  int ITER;

  LDB = LDX = LDA = N ;

  int status ;
   
  cuDoubleComplex *h_work_M_D;
  cuFloatComplex *h_work_M_S;
  int maxNB ;
  cuDoubleComplex *M_WORK ; 
  cuFloatComplex *M_SWORK ; 
  cuDoubleComplex *As, *Bs, *Xs;
  cuDoubleComplex *res_ ; 
  cuDoubleComplex *CACHE ;  
  int CACHE_L  = 10000 ;
  cuDoubleComplex *d_A , * d_B , *d_X;

  maxNB =  magma_get_spotrf_nb(N);
  size = maxNB * maxNB ;     
  status = cudaMallocHost( (void**)&h_work_M_S,  size*sizeof(cuFloatComplex) );
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (dipiv)\n");
    goto FREE1;
  }
  maxNB = magma_get_dpotrf_nb(N);
  size = maxNB * maxNB ;     
  status = cudaMallocHost( (void**)&h_work_M_D,  size*sizeof(cuDoubleComplex) );
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (dipiv)\n");
    goto FREE1_1;
  }


  size =N*N;
  size+= N*NRHS;
  status = cublasAlloc( size, sizeof(cuFloatComplex), (void**)&M_SWORK ) ;
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (dipiv)\n");
    goto FREE2;
  }
  size= N*NRHS;
  status = cublasAlloc( size, sizeof(cuDoubleComplex), (void**)&M_WORK ) ;
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (dipiv)\n");
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
  
    int N1 = N;
    
    int INFO[1];

    LDB = LDX = LDA = N ;

    size = LDA * N ;
    init_matrix_sym( As, N , sizeof(cuDoubleComplex), N );

    size = LDB * NRHS ;
    init_matrix(Bs, size, sizeof(cuDoubleComplex));

    cublasSetMatrix( N, N, sizeof( cuDoubleComplex ), As, N, d_A , N ) ;
    cublasSetMatrix( N, NRHS, sizeof( cuDoubleComplex ), Bs, N,d_B, N ) ;

    double perf ;  

    printf("%5d  ", N); 
    fflush(stdout);

    char uplo = 'L';
    int IPIV[N];
    int iter_CPU = 0 ;

    //=====================================================================
    //              Mixed Precision Iterative Refinement - GPU 
    //=====================================================================
    start = get_current_time();
    magma_zcposv_gpu(uplo, N, NRHS, d_A, LDA, d_B, LDA, d_X, LDA, M_WORK, 
		     M_SWORK, &ITER, INFO, h_work_M_S, h_work_M_D);
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
    lapackf77_zhemm( &side, &uplo, &N, &NRHS, &NEGONE, As, &LDA, res_, &LDX, &ONE, Bs, &N);
    Rnorm = lapackf77_zlange("I", &N, &NRHS, Bs, &LDB, worke);
    free(worke);

    //=====================================================================
    //                 Double Precision Factor 
    //=====================================================================
    start = get_current_time();
    magma_zpotrf_gpu(uplo, N, d_A, LDA, INFO);
    end = get_current_time();
    perf = (1.*N*N*N/3.)/(1000000*GetTimerValue(start,end));
    printf("%6.2f    ", perf);
    fflush(stdout);

    //=====================================================================
    //                 Double Precision Solve 
    //=====================================================================
    start = get_current_time();
    magma_zpotrf_gpu(uplo, N, d_A, LDA, INFO);
    magma_zpotrs_gpu('L', N, NRHS, d_A, LDA, d_B, LDB, INFO);
    end = get_current_time();
    perf = (1.*N*N*N/3.+2.*N*N)/(1000000*GetTimerValue(start,end));
    printf("%6.2f    ", perf);
    fflush(stdout);

    //=====================================================================
    //                 Single Precision Factor 
    //=====================================================================
    start = get_current_time();
    magma_cpotrf_gpu(uplo, N, M_SWORK+N*NRHS, LDA, INFO);
    end = get_current_time();
    perf = (1.*N*N*N/3.)/(1000000*GetTimerValue(start,end));
    printf("%6.2f     ", perf);
    fflush(stdout);

    //=====================================================================
    //                 Single Precision Solve 
    //=====================================================================
    start = get_current_time();
    magma_cpotrf_gpu(uplo, N, M_SWORK+N*NRHS, LDA, INFO);
    magma_cpotrs_gpu('L', N, NRHS, M_SWORK+N*NRHS, LDA, M_SWORK, LDB, INFO);
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
    cublasFree(h_work_M_D);
FREE1_1:
    cublasFree(h_work_M_S);
FREE1:
    cublasShutdown();
}
