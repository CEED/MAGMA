#include <stdio.h>
#include<stdlib.h>
#include<math.h>
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cublas.h"
#include "magmablas.h"
#include "magma.h"
#define ITERMAX 30
#define BWDMAX 1.0
int init_matrix(void *A, int size , int elem_size){
    if( elem_size==sizeof(float)){
	float *AD; 
	AD = (float*)A ; 
	int j ; 
	
        for(j = 0; j < size; j++)
           AD[j] = (rand()) / (float)RAND_MAX;
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
    if( elem_size==sizeof(float)){
	float *SD, *DD; 
	SD = (float*)S ; 
	DD = (float*)D ; 
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

void cache_flush( float * CACHE , int length ) {
     int i = 0 ; 
     for( i=0;i<length ;i++){
          CACHE[i]=CACHE[i]+0.1;
     }
}


int main(int argc , char **argv){
    cuInit( 0 );
    cublasInit( );




 int printall = 0 ;
 FILE *fp ;
 fp = fopen("results_sgesv.txt","w");
 if( fp == NULL ) return 1;
 printf("SP Solver LU \n");
 fprintf(fp, "SP Solver LU \n");
    printf("\n");
    printout_devices( );
 
    printf("\nUsage:\n\t\t ./testing_sgesv N");
    fprintf(fp, "\nUsage:\n\t\t ./testing_sgesv N");
  //printf("Iterative Refinement\n");
  //fprintf(fp,"Iterative Refinement\n");
  
 printf("\n\nEpsilon(Double): %10.20lf \nEpsilon(Single): %10.20lf\n", dlamch_("Epsilon"), slamch_("Epsilon"));
 fprintf(fp, "\nEpsilon(Double): %10.20lf \nEpsilon(Single): %10.20lf\n", dlamch_("Epsilon"), slamch_("Epsilon"));

  TimeStruct start, end;
 int LEVEL=1;
  printf("\n\nN\tSigle-Solve\t||Ax-B||_oo/((||A||_oo||x||_oo+||B||_oo).N.eps)\n");
  fprintf(fp, "\n\nN\tSigle-Solve\t||Ax-B||_oo/((||A||_oo||x||_oo+||B||_oo).N.eps)\n");
  printf("===========================================================================================================\n"); 
  fprintf(fp,"============================================================================================================\n"); 

  int i ;
  int startN=64  + 0+ 0 *     rand() %20   ;
  int count = 14;
  int step = 512 ;  
  int N = count * step ;
  int NRHS=1 ;
  N =startN+(count-1) * step ;
  int once = 0 ; 
  if( argc == 3) {
      once = 1 ;  
    startN =   N  = atoi ( argv[2]);
  }

  
int size ; 
  int LDA ;
  int LDB ;
  int LDX ;
  int ITER;
  int maxnb = magma_get_sgetrf_nb(N) ;
  int maxnb_d = magma_get_sgetrf_nb(N) ;
  maxnb_d = maxnb ; 
  int lwork = N*maxnb;
  int lwork_d = N*maxnb_d;


  LDB = LDX = LDA = N ;
  int status ;
  float *d_A , * d_B , *d_X ; 
  float *h_work_M_S;
  int *IPIV ;
  float *A , *B, *X ;
  float *As, *Bs, *Xs;
  float *res_ ; 
  float *CACHE ;  
  int CACHE_L  = 10000 ;

  size = (N+32)*(N+32) + 32*maxnb + lwork+2*maxnb*maxnb; 
  status = cublasAlloc( size, sizeof(float), (void**)&d_A ) ;
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (dipiv)\n");
    goto FREE1;
  }
  size = LDB * NRHS ;
  status = cublasAlloc( size, sizeof(float), (void**)&d_B ) ;
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (dipiv)\n");
    goto FREE2;
  }

  d_X = ( float *) malloc ( sizeof ( float ) * size);
  status = cublasAlloc( size, sizeof(float), (void**)&d_X ) ;
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (dipiv)\n");
    goto FREE3;
  }
    size =  (lwork+32*maxnb);     
    status = cudaMallocHost( (void**)&h_work_M_S,  size*sizeof(float) );
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (dipiv)\n");
    goto FREE4;
  }

    size = N*NRHS ; 
    size = (N+32)*(N+32) + 32*maxnb + lwork+2*maxnb*maxnb;
    size += maxnb*N*NRHS;

    size =3* LDA * N ;
    A = ( float *) malloc ( sizeof ( float ) * size);
    if( A == NULL )
    {
	 printf("Allocation Error\n");
         goto FREE8;
    }	    	


    size = LDB * NRHS ;
    B = ( float *) malloc ( sizeof ( float ) * size);
    if( B == NULL )
    {
	 printf("Allocation Error\n");
         goto FREE10;
    }	    	

    X = ( float *) malloc ( sizeof ( float ) * size);
    if( X == NULL ) 
    {
	 printf("Allocation Error\n");
         goto FREE11;
    }	    	

    IPIV = ( int *) malloc ( sizeof (int) * N ) ;
    if( IPIV == NULL ) 
    {
	 printf("Allocation Error\n");
         goto FREE12;
    }	    	

    size =3*LDA * N ;
    As = ( float *) malloc ( sizeof ( float ) * size);
    if( As == NULL )
    {
	 printf("Allocation Error\n");
         goto FREE13;
    }	    	
    size = NRHS * N ;
    Bs = ( float *) malloc ( sizeof ( float ) * size);
    if( Bs == NULL )
    {
	 printf("Allocation Error\n");
         goto FREE14;
    }	    	
    Xs = ( float *) malloc ( sizeof ( float ) * size);
    if( Xs == NULL )
    {
	 printf("Allocation Error\n");
         goto FREE15;
    }	    	


        
    size = NRHS * N ;	
    size += ( N * N ) ; 

    size = N*NRHS ;
    res_ = ( float *) malloc ( sizeof(float)*size);
    if( res_ == NULL ) 
    {
	 printf("Allocation Error\n");
         goto FREE18;
    }	    	


   size = CACHE_L * CACHE_L ;
   CACHE = ( float *) malloc ( sizeof( float) * size ) ; 
   if( CACHE == NULL ) 
    {
	 printf("Allocation Error\n");
         goto FREE19;
    }	    	

  for(i=0;i<count;i++){

    N = step*(i)+startN    ;
    int N1 = N ;
    
    int INFO[1];

    float *WORK ;
    float  *SWORK ;
    float *L_A , *L_B , *L_X ,*L_WORK1, *L_SWORK1 ;	


    LDB = LDX = LDA = N ;



    maxnb = magma_get_sgetrf_nb(N) ;
    maxnb_d = magma_get_sgetrf_nb(N) ;
    maxnb = maxnb > maxnb_d ? maxnb : maxnb_d ;
    maxnb_d = maxnb ; 
    lwork = N1*maxnb;
    lwork_d = N1*maxnb_d;
 

    size = LDA * N ;
    init_matrix(A, size, sizeof(float));
    size = LDB * NRHS ;
    init_matrix(B, size, sizeof(float));
    
    float perf ;  
    if( LEVEL == 0 ) printf("DIM  ");
    int maxnb = magma_get_sgetrf_nb(N) ;
    int PTSX = 0 , PTSA = maxnb*N*NRHS ;

    //magma_sgetrs("N",N ,NRHS, M_SWORK+PTSA ,N,IPIV, M_SWORK+PTSX, N,INFO );
    printf("%5d",N); 
    fprintf(fp,"%5d",N); 


    cublasSetMatrix( N, N, sizeof( float ), A, N, d_A, N ) ;
    cublasSetMatrix( N, NRHS, sizeof( float ), X, N, d_X, N ) ;
    cublasSetMatrix( N, NRHS, sizeof( float ), B, N, d_B, N ) ;

    slacpy_("All", &N, &NRHS, B , &LDB, X, &N);

    int iter_CPU = ITER ;
    slacpy_("All", &N, &NRHS, X , &LDB, res_, &N);




    //=====================================================================
    //              SP - GPU 
    //=====================================================================
    start = get_current_time();
    magma_sgetrf_gpu(&N, &N,d_A, &N,IPIV, h_work_M_S, INFO);
    magma_sgetrs_v2("N", N,NRHS,d_A,LDA,IPIV,d_B,LDB, INFO, h_work_M_S);
    end = get_current_time();
    perf = (2.*N*N*N/3.+2.*N*N)/(1000000*GetTimerValue(start,end));
    printf("\t\t%6.2f", perf);
    fprintf(fp,"\t\t%6.2f", perf);
    cublasGetMatrix( N, NRHS, sizeof( float ), d_B , N, X , N ) ;

    //=====================================================================
    //              ERROR DP vs MIXED  - GPU 
    //=====================================================================
    float Rnorm, Anorm, Xnorm, Bnorm;
    char norm='I';      
    float *worke = (float *)malloc(N*sizeof(float));
    Xnorm = slange_(&norm, &N, &NRHS, X, &LDB, worke);
    Anorm = slange_(&norm, &N, &N, A, &LDA, worke);
    Bnorm = slange_(&norm, &N, &NRHS, B, &LDB, worke);
    float ONE = -1.0 , NEGONE = 1.0 ;
    sgemm_( "No Transpose", "No Transpose", &N, &NRHS, &N, &NEGONE, A, &LDA, X, &LDX, &ONE, B, &N);
    Rnorm=slange_(&norm, &N, &NRHS, B, &LDB, worke);
    float eps1 = slamch_("Epsilon"); 
    free(worke);

    printf("\t\t%e",Rnorm/((Anorm*Xnorm+Bnorm)*N*eps1));
    fprintf(fp, "\t\t%e",Rnorm/((Anorm*Xnorm+Bnorm)*N*eps1));



 printf("\n");
 fprintf(fp,"\n");
    LEVEL = 1 ;
if ( once == 1 ) break ;  
  }






    free(CACHE);
FREE19:
    free(res_);
FREE18:
FREE17:
FREE16:
    free(Xs);
FREE15:
    free(Bs);
FREE14:
    free(As);
FREE13:
    free(IPIV);
FREE12:
    free(X);
FREE11:
    free(B);
FREE10:
FREE9:
    free(A);
FREE8:
FREE7:
FREE6:
FREE5:
    cublasFree(h_work_M_S);
FREE4:
    cublasFree(d_X);
FREE3:
    cublasFree(d_B);
FREE2:
    cublasFree(d_A);
FREE1:
    fclose(fp);
    cublasShutdown();
}
