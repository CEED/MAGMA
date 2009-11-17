/*
    -- MAGMA (version 0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2009
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
    if( elem_size==sizeof(double)){

	double *AD; 

	AD = (double*)A ; 

        for(i = 0; i< size*size ; i++) 
           AD[i]= (rand()) / (float)RAND_MAX;
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

    else if( elem_size==sizeof(float)){
	float *AD; 
	AD = (float*)A ; 
        for(i = 0; i< size*size ; i++)
           AD[i]= (rand()) / (float)RAND_MAX;
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
    if( elem_size==sizeof(double)){
	double *AD; 
	AD = (double*)A ; 
	int j ; 
	
        for(j = 0; j < size; j++)
           AD[j] = (rand()) / (double)RAND_MAX;
    }
    else if( elem_size==sizeof(float)){
	float *AD; 
	AD = (float*)A ; 
	int j ; 
        for(j = 0; j < size; j++)
             AD[j] = (rand()) / (float)RAND_MAX;
    }
}

int copy_matrix(void *S, void *D,int size , int elem_size){
    if( elem_size==sizeof(double)){
	double *SD, *DD; 
	SD = (double*)S ; 
	DD = (double*)D ; 
	int j ; 
        for(j = 0; j < size; j++)
	    DD[j]=SD[j];
    }
    else if( elem_size==sizeof(float)){
	float *SD, *DD; 
	SD = (float*)S ; 
	DD = (float*)D ; 
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
 FILE *fp ;
 fp = fopen("results_dsposv.txt","w");
 if( fp == NULL ) return 1;
  
 printf("Iterative Refinement- Cholesky \n");
 fprintf(fp, "Iterative Refinement- Cholesky \n");
    printf("\n");
    cuInit( 0 );
    cublasInit( );

    printout_devices( );


    printf("\nUsage:\n\t\t ./testing_dsposv N");
    fprintf(fp, "\nUsage:\n\t\t ./testing_dsposv N");

 printf("\n\nEpsilon(Double): %10.20lf \nEpsilon(Single): %10.20lf\n", dlamch_("Epsilon"), slamch_("Epsilon"));
 fprintf(fp, "Epsilon(Double): %10.20lf \nEpsilon(Single): %10.20lf\n", dlamch_("Epsilon"), slamch_("Epsilon"));

  TimeStruct start, end;

  int LEVEL=1;
  int i ;
  int startN=64 + rand()%32 ;
  int count = 8;
  int step = 512;  
  int N = count * step ;
  int NRHS=1 ;
//512 1 16 64 1
  int error = 0 ; 
  if( argc == 6) { 
      step  = atoi ( argv[1]);
      NRHS  = atoi( argv[2]);
      count  = atoi ( argv[3]);
      startN = atoi( argv[4]); 
      error = atoi(argv[5]);
  }

  N =startN+(count-1) * step ;

 printf("\n\n\tN\tDouble-Factor\tDouble-Solve\t\tSingle-Factor\tSigle-Solve\t   Mixed Precision Solver \t||Ax-B||_oo/((||A||_oo||x||_oo+||B||_oo).N.eps)\tNumIter\n");
  fprintf(fp, "\n\n\tN\tDouble-Factor\tDouble-Solve\t\tSingle-Factor\tSigle-Solve\t   Mixed Precision Solver \t||Ax-B||_oo/((||A||_oo||x||_oo+||B||_oo).N.eps)\tNumIter\n");
      printf("===============================================================================================================================================================================\n");
      fprintf(fp,"==============================================================================================================================================================================\n");


  int size ; 
  int LDA ;
  int LDB ;
  int LDX ;
  int ITER;

  LDB = LDX = LDA = N ;

  int status ;
   
    double *h_work_M_D;
    float *h_work_M_S;
    int maxNB ;
    double *M_WORK ; 
    float *M_SWORK ; 
    double *As, *Bs, *Xs;
    double *res_ ; 
    double *CACHE ;  
    int CACHE_L  = 10000 ;
    double *d_A , * d_B , *d_X;

    maxNB = magma_get_spotrf_nb(N);
    size = maxNB * maxNB ;     
    status = cudaMallocHost( (void**)&h_work_M_S,  size*sizeof(float) );
    if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (dipiv)\n");
    goto FREE1;
    }
    maxNB = magma_get_dpotrf_nb(N);
    size = maxNB * maxNB ;     
    status = cudaMallocHost( (void**)&h_work_M_D,  size*sizeof(double) );
    if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (dipiv)\n");
    goto FREE1_1;
    }


    size =N*N;
    size+= N*NRHS;
    status = cublasAlloc( size, sizeof(float), (void**)&M_SWORK ) ;
    if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (dipiv)\n");
    goto FREE2;
    }
    size= N*NRHS;
    status = cublasAlloc( size, sizeof(double), (void**)&M_WORK ) ;
    if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (dipiv)\n");
    goto FREE2_1;
    }

    size =LDA * N ;
    As = ( double *) malloc ( sizeof ( double ) * size);
    if( As == NULL ){
		 printf("Allocation A\n");
		 goto FREE3;
    }
    size = NRHS * N ;
    Bs = ( double *) malloc ( sizeof ( double ) * size);
    if( Bs == NULL ){
		 printf("Allocation A\n");
		 goto FREE4;
    }
    Xs = ( double *) malloc ( sizeof ( double ) * size);
    if( Xs == NULL ){
		 printf("Allocation A\n");
		 goto FREE5;
    }



    size = N*NRHS ;
    res_ = ( double *) malloc ( sizeof(double)*size);
    if( res_ == NULL ){
		 printf("Allocation A\n");
		 goto FREE6;
    }
   size = CACHE_L * CACHE_L ;
   CACHE = ( double *) malloc ( sizeof( double) * size ) ; 
   if( CACHE == NULL ){
		 printf("Allocation A\n");
		 goto FREE7;
   }

  size = N * N ;
  status = cublasAlloc( size, sizeof(double), (void**)&d_A ) ;
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (dipiv)\n");
    goto FREE8;
  }

  size = LDB * NRHS ;
  d_B = ( double *) malloc ( sizeof ( double ) * size);
  status = cublasAlloc( size, sizeof(double), (void**)&d_B ) ;
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (dipiv)\n");
    goto FREE9;
  }

  d_X = ( double *) malloc ( sizeof ( double ) * size);
  status = cublasAlloc( size, sizeof(double), (void**)&d_X ) ;
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (dipiv)\n");
    goto FREE10;
  }



 
  for(i=0;i<count;i++){
    NRHS =  1 ; 
    N = step*(i)+startN;
    int N1 = N;
    
    int INFO[1];

    LDB = LDX = LDA = N ;

    size = LDA * N ;
    init_matrix_sym( As, N , sizeof(double), N );

    size = LDB * NRHS ;
    init_matrix(Bs, size, sizeof(double));

    cublasSetMatrix( N, N, sizeof( double ), As, N, d_A , N ) ;
    cublasSetMatrix( N, NRHS, sizeof( double ), Bs, N,d_B, N ) ;

    double perf ;  
    int maxnb = magma_get_sgetrf_nb(N) ;
    int PTSX = 0 , PTSA = maxnb*N*NRHS ;

    printf("%10d ",N); 
    fprintf(fp, "%10d ",N); 
    fflush(stdout);

    char uplo = 'L';
    int IPIV[N];
    int iter_CPU = 0 ;

    //=====================================================================
    //              Mixed Precision Iterative Refinement - GPU 
    //=====================================================================

    start = get_current_time();
    magma_dsposv(uplo, N , NRHS , d_A , LDA , d_B , LDA , d_X , LDA , M_WORK , M_SWORK , &ITER , INFO , h_work_M_S , h_work_M_D ) ;
    end = get_current_time();
    perf = (1.*N*N*N/3.+2.*N*N)/(1000000*GetTimerValue(start,end));
    cublasGetMatrix( N, NRHS, sizeof( double), d_X, N,res_, N ) ;
    double lperf = perf ; 

    //=====================================================================
    //                 Error Computation 
    //=====================================================================
    double Rnorm, Anorm, Xnorm, Bnorm;
    char norm='I';
    double *worke = (double *)malloc(N*sizeof(double));
    Xnorm = dlange_(&norm, &N, &NRHS, res_, &LDB, worke);
    Anorm = dlansy_(&uplo, &norm,  &N, As, &LDA, worke);
    Bnorm = dlange_(&norm, &N, &NRHS, Bs, &LDB, worke);
    double ONE = -1.0 , NEGONE = 1.0 ;
    char side ='L';
    dsymm_( &side, &uplo, &N, &NRHS, &NEGONE, As, &LDA, res_, &LDX, &ONE, Bs, &N);
    Rnorm=dlange_(&norm, &N, &NRHS, Bs, &LDB, worke);
    double eps1 = dlamch_("Epsilon");
    free(worke);



    //=====================================================================
    //                 Double Precision Factor 
    //=====================================================================
    start = get_current_time();
    magma_dpotrf_gpu(&uplo, &N,d_A, &LDA, h_work_M_D, INFO);
    end = get_current_time();
    perf = (1.*N*N*N/3.)/(1000000*GetTimerValue(start,end));
    printf("\t%6.2f", perf);
    fprintf(fp,"\t%6.2f", perf);
    fflush(stdout);
    //=====================================================================
    //                 Double Precision Solve 
    //=====================================================================

    start = get_current_time();
    magma_dpotrf_gpu(&uplo, &N,d_A, &LDA, h_work_M_D, INFO);
    magma_dpotrs_gpu( "L",N ,NRHS, d_A  , LDA ,d_B,LDB,INFO);
    end = get_current_time();
    perf = (1.*N*N*N/3.+2.*N*N)/(1000000*GetTimerValue(start,end));
    printf("\t\t%6.2f", perf);
    fprintf(fp,"\t\t%6.2f", perf);
    fflush(stdout);


    //=====================================================================
    //                 Single Precision Factor 
    //=====================================================================

    start = get_current_time();
    magma_spotrf_gpu(&uplo, &N, M_SWORK+N*NRHS, &LDA, h_work_M_S, INFO);
    end = get_current_time();
    perf = (1.*N*N*N/3.)/(1000000*GetTimerValue(start,end));
    printf("\t\t\t%6.2f ", perf);
    fprintf(fp,"\t\t\t%6.2f", perf);
    fflush(stdout);

    //=====================================================================
    //                 Single Precision Solve 
    //=====================================================================

    start = get_current_time();
    magma_spotrf_gpu(&uplo, &N, M_SWORK+N*NRHS, &LDA, h_work_M_S, INFO);
    magma_spotrs_gpu( "L",N ,NRHS, M_SWORK+N*NRHS  , LDA ,M_SWORK,LDB,INFO);
    end = get_current_time();
    perf = (1.*N*N*N/3.+2.*N*N)/(1000000*GetTimerValue(start,end));
    printf("\t\t%6.2f", perf);
    fprintf(fp,"\t\t%6.2f", perf);
    fflush(stdout);




    printf("\t\t\t%6.2f", lperf);
    fprintf(fp,"\t\t\t%6.2f", lperf);
    printf("\t\t\t\t%e\t%27d",Rnorm/((Anorm*Xnorm+Bnorm)*N*eps1), ITER);
    fprintf(fp, "\t\t\t\t%e\t%27d",Rnorm/((Anorm*Xnorm+Bnorm)*N*eps1), ITER);
    fflush(stdout);

    printf("\n");
    fprintf(fp,"\n");
    LEVEL = 1 ; 
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
    fclose(fp);
    cublasShutdown();
}
