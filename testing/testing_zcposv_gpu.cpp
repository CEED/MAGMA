/*
    -- MAGMA (version 0.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

       @precisions mixed zc -> ds

*/
#include <stdio.h>
#include<stdlib.h>
#include<math.h>
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cublas.h"
#include "magmablas.h"
#include "magma.h"


/*
*/

int init_matrix_sym(void *A, int size , int elem_size, int lda){
    int i ;
    int j ;
    if( elem_size==sizeof(double2)){

	double2 *AD; 

	AD = (double2*)A ; 

        for(i = 0; i< size*size ; i++) 
           AD[i]= (rand()) / (float2)RAND_MAX;
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

    else if( elem_size==sizeof(float2)){
	float2 *AD; 
	AD = (float2*)A ; 
        for(i = 0; i< size*size ; i++)
           AD[i]= (rand()) / (float2)RAND_MAX;
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
    if( elem_size==sizeof(double2)){
	double2 *AD; 
	AD = (double2*)A ; 
	int j ; 
	
        for(j = 0; j < size; j++)
           AD[j] = (rand()) / (double2)RAND_MAX;
    }
    else if( elem_size==sizeof(float2)){
	float2 *AD; 
	AD = (float2*)A ; 
	int j ; 
        for(j = 0; j < size; j++)
             AD[j] = (rand()) / (float2)RAND_MAX;
    }
}

int copy_matrix(void *S, void *D,int size , int elem_size){
    if( elem_size==sizeof(double2)){
	double2 *SD, *DD; 
	SD = (double2*)S ; 
	DD = (double2*)D ; 
	int j ; 
        for(j = 0; j < size; j++)
	    DD[j]=SD[j];
    }
    else if( elem_size==sizeof(float2)){
	float2 *SD, *DD; 
	SD = (float2*)S ; 
	DD = (float2*)D ; 
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


int main(int argc , char **argv){
 int  printall =0 ; 
// FILE *fp ;
// fp = fopen("results_zcposv.txt","w");
// if( fp == NULL ) return 1;
  
 printf("Iterative Refinement- Cholesky \n");
// fprintf(fp, "Iterative Refinement- Cholesky \n");
    printf("\n");
    cuInit( 0 );
    cublasInit( );

    printout_devices( );


    printf("\nUsage:\n\t\t ./testing_zcposv -N 1024");
  //  fprintf(fp, "\nUsage:\n\t\t ./testing_zcposv -N 1024");

 printf("\n\nEpsilon(Double): %10.20lf \nEpsilon(Single): %10.20lf\n", lapackf77_dlamch("Epsilon"), lapackf77_slamch("Epsilon"));
 //fprintf(fp, "Epsilon(Double): %10.20lf \nEpsilon(Single): %10.20lf\n", lapackf77_dlamch("Epsilon"), lapackf77_slamch("Epsilon"));

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

  printf("\n\nN\tDouble-Factor\tDouble-Solve\tSingle-Factor\tSigle-Solve\tMixed Precision Solver\t || b-Ax || / ||A||  \t NumIter\n");
  //fprintf(fp, "\n\nN\tDouble-Factor\tDouble-Solve\tSingle-Factor\tSigle-Solve\tMixed Precision Solver\t || b-Ax || / ||A||\t NumIter\n");

      printf("===============================================================================================================================================================================\n");
      //fprintf(fp,"==============================================================================================================================================================================\n");


  int size ; 
  int LDA ;
  int LDB ;
  int LDX ;
  int ITER;

  LDB = LDX = LDA = N ;

  int status ;
   
    double2 *h_work_M_D;
    float2 *h_work_M_S;
    int maxNB ;
    double2 *M_WORK ; 
    float2 *M_SWORK ; 
    double2 *As, *Bs, *Xs;
    double2 *res_ ; 
    double2 *CACHE ;  
    int CACHE_L  = 10000 ;
    double2 *d_A , * d_B , *d_X;
//printf("%d %d %d %d\n", magma_get_cpotrf_nb(N) ,  magma_get_cpotrf_nb(1024) ,  magma_get_cpotrf_nb(4032) ,  magma_get_zpotrf_nb(1024) ,  magma_get_zpotrf_nb(4032) ); exit(1);
    maxNB =  magma_get_cpotrf_nb(N);
    size = maxNB * maxNB ;     
    status = cudaMallocHost( (void**)&h_work_M_S,  size*sizeof(float2) );
    if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (dipiv)\n");
    goto FREE1;
    }
    maxNB = magma_get_zpotrf_nb(N);
    size = maxNB * maxNB ;     
    status = cudaMallocHost( (void**)&h_work_M_D,  size*sizeof(double2) );
    if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (dipiv)\n");
    goto FREE1_1;
    }


    size =N*N;
    size+= N*NRHS;
    status = cublasAlloc( size, sizeof(float2), (void**)&M_SWORK ) ;
    if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (dipiv)\n");
    goto FREE2;
    }
    size= N*NRHS;
    status = cublasAlloc( size, sizeof(double2), (void**)&M_WORK ) ;
    if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (dipiv)\n");
    goto FREE2_1;
    }

    size =LDA * N ;
    As = ( double2 *) malloc ( sizeof ( double2 ) * size);
    if( As == NULL ){
		 printf("Allocation A\n");
		 goto FREE3;
    }
    size = NRHS * N ;
    Bs = ( double2 *) malloc ( sizeof ( double2 ) * size);
    if( Bs == NULL ){
		 printf("Allocation A\n");
		 goto FREE4;
    }
    Xs = ( double2 *) malloc ( sizeof ( double2 ) * size);
    if( Xs == NULL ){
		 printf("Allocation A\n");
		 goto FREE5;
    }



    size = N*NRHS ;
    res_ = ( double2 *) malloc ( sizeof(double2)*size);
    if( res_ == NULL ){
		 printf("Allocation A\n");
		 goto FREE6;
    }
   size = CACHE_L * CACHE_L ;
   CACHE = ( double2 *) malloc ( sizeof( double2) * size ) ; 
   if( CACHE == NULL ){
		 printf("Allocation A\n");
		 goto FREE7;
   }

  size = N * N ;
  status = cublasAlloc( size, sizeof(double2), (void**)&d_A ) ;
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (dipiv)\n");
    goto FREE8;
  }

  size = LDB * NRHS ;
  d_B = ( double2 *) malloc ( sizeof ( double2 ) * size);
  status = cublasAlloc( size, sizeof(double2), (void**)&d_B ) ;
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (dipiv)\n");
    goto FREE9;
  }

  d_X = ( double2 *) malloc ( sizeof ( double2 ) * size);
  status = cublasAlloc( size, sizeof(double2), (void**)&d_X ) ;
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
    init_matrix_sym( As, N , sizeof(double2), N );

    size = LDB * NRHS ;
    init_matrix(Bs, size, sizeof(double2));

    cublasSetMatrix( N, N, sizeof( double2 ), As, N, d_A , N ) ;
    cublasSetMatrix( N, NRHS, sizeof( double2 ), Bs, N,d_B, N ) ;

    double2 perf ;  

    printf("%5d ",N); 
    //fprintf(fp, "%10d ",N); 
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
    //magma_zpotrf_gpu(&uplo, &N,d_A, &LDA, h_work_M_D, INFO);
    //magma_zpotrs_gpu( "L",N ,NRHS, d_A  , LDA ,d_B,LDB,INFO);
    end = get_current_time();
    perf = (1.*N*N*N/3.+2.*N*N)/(1000000*GetTimerValue(start,end));
    cublasGetMatrix( N, NRHS, sizeof( double2), d_X, N,res_, N ) ;
    double2 lperf = perf ; 

    //=====================================================================
    //                 Error Computation 
    //=====================================================================

      char norm='I';
      char side ='L';
      double2 ONE = -1.0 , NEGONE = 1.0 ;
/*
    double2 Rnorm, Anorm, Xnorm, Bnorm;

    double2 *worke = (double2 *)malloc(N*sizeof(double2));
    Anorm = lapackf77_zlansy(&norm, &uplo,  &N, As, &N, worke);
 //   Anorm = lapackf77_zlange("I", &N, &N, As, &N, worke);

    Xnorm = lapackf77_zlange(&norm, &N, &NRHS, res_, &LDB, worke);
    Bnorm = lapackf77_zlange(&norm, &N, &NRHS, Bs, &LDB, worke);
    blasf77_zsymm( &side, &uplo, &N, &NRHS, &NEGONE, As, &LDA, res_, &LDX, &ONE, Bs, &N);
    Rnorm=lapackf77_zlange(&norm, &N, &NRHS, Bs, &LDB, worke);
    double2 eps1 = lapackf77_dlamch("Epsilon");
    printf("\t--   %e %e --\t", Rnorm ,  Anorm);
    free(worke);
*/
      double2 Rnorm, Anorm;
      double2 *worke = (double2 *)malloc(N*sizeof(double2));
      Anorm = lapackf77_zlansy( &norm, &uplo,  &N, As, &N, worke);
      blasf77_zsymm( &side, &uplo, &N, &NRHS, &NEGONE, As, &LDA, res_, &LDX, &ONE, Bs, &N);
      Rnorm=lapackf77_zlange("I", &N, &NRHS, Bs, &LDB, worke);
      free(worke);

    //=====================================================================
    //                 Double Precision Factor 
    //=====================================================================
    start = get_current_time();
    magma_zpotrf_gpu(uplo, N, d_A, LDA, INFO);
    end = get_current_time();
    perf = (1.*N*N*N/3.)/(1000000*GetTimerValue(start,end));
    printf("\t%6.2f", perf);
    //fprintf(fp,"\t%6.2f", perf);
    fflush(stdout);
    //=====================================================================
    //                 Double Precision Solve 
    //=====================================================================

    start = get_current_time();
    magma_zpotrf_gpu(uplo, N, d_A, LDA, INFO);
    magma_zpotrs_gpu('L', N, NRHS, d_A, LDA, d_B, LDB, INFO);
    end = get_current_time();
    perf = (1.*N*N*N/3.+2.*N*N)/(1000000*GetTimerValue(start,end));
    printf("\t\t%6.2f", perf);
    //fprintf(fp,"\t\t%6.2f", perf);
    fflush(stdout);


    //=====================================================================
    //                 Single Precision Factor 
    //=====================================================================

    start = get_current_time();
    magma_cpotrf_gpu(uplo, N, M_SWORK+N*NRHS, LDA, INFO);
    end = get_current_time();
    perf = (1.*N*N*N/3.)/(1000000*GetTimerValue(start,end));
    printf("\t\t%6.2f ", perf);
    //fprintf(fp,"\t\t%6.2f", perf);
    fflush(stdout);

    //=====================================================================
    //                 Single Precision Solve 
    //=====================================================================

    start = get_current_time();
    magma_cpotrf_gpu(uplo, N, M_SWORK+N*NRHS, LDA, INFO);
    magma_cpotrs_gpu('L', N, NRHS, M_SWORK+N*NRHS, LDA, M_SWORK, LDB, INFO);
    end = get_current_time();
    perf = (1.*N*N*N/3.+2.*N*N)/(1000000*GetTimerValue(start,end));
    printf("\t\t%6.2f", perf);
    //fprintf(fp,"\t\t%6.2f", perf);
    fflush(stdout);


    printf("\t\t%6.2f", lperf);
    //fprintf(fp,"\t\t%6.2f", lperf);
//    printf("\t\t\t%e\t%17d",Rnorm/((Anorm*Xnorm+Bnorm)*N*eps1), ITER);
 //   fprintf(fp, "\t\t\t%e\t%17d",Rnorm/((Anorm*Xnorm+Bnorm)*N*eps1), ITER);
    printf("\t\t\t%e\t%3d", Rnorm/Anorm, ITER);
    //fprintf(fp, "\t\t\t%e\t%3d", Rnorm/Anorm, ITER);


    fflush(stdout);

    printf("\n");
    //fprintf(fp,"\n");

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
   // fclose(fp);
    cublasShutdown();
}
