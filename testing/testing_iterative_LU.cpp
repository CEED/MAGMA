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

void cache_flush( double * CACHE , int length ) {
     int i = 0 ; 
     for( i=0;i<length ;i++){
          CACHE[i]=CACHE[i]+0.1;
     }
}


int main(int argc , char **argv){
 int printall = 0 ;
 FILE *fp ;
 fp = fopen("results-LU-Solve-MP.txt","w");
 if( fp == NULL ) return 1;
 
  printf("Iterative Refinement\n");
  fprintf(fp,"Iterative Refinement\n");

 printf("DP-Eps: %10.20lf \nSP-Eps: %10.20lf\n", dlamch_("Epsilon"), slamch_("Epsilon"));
 fprintf(fp, "DP-Eps: %10.20lf \nSP-Eps: %10.20lf\n", dlamch_("Epsilon"), slamch_("Epsilon"));
  TimeStruct start, end;

  int LEVEL=1;
    printf("\n____________________________________________________________________\n"); 
    fprintf(fp,"\n____________________________________________________________________\n"); 
  printf("DIM\tMP-CPU\tDP-CPU\tSP-CPU\tMP-GPU\tDP-GPU\tSP-CPU\tERROR\tRES\tRation\tNRHS\tIt-CPU\tIt-GPU\n");
  fprintf(fp,"DIM\tMP-CPU\tDP-CPU\tSP-CPU\tMP-GPU\tDP-GPU\tSP-CPU\tERROR\tRES\tRation\tNRHS\tIt-CPU\tIt-GPU\n");
  int i ;
  int startN=64 ;
  int count = 18;
  int step = 512 ;  
  int N = count * step ;
  int NRHS=1 ;
  if( argc == 5) { 
      step  = atoi ( argv[1]);
      NRHS  = atoi( argv[2]);
      count  = atoi ( argv[3]);
      startN = atoi( argv[4]); 
  }
  N =startN+(count-1) * step ;

  int size ; 
  int LDA ;
  int LDB ;
  int LDX ;
  int ITER;
  int maxnb = magma_get_sgetrf_nb(N) ;
  int maxnb_d = magma_get_dgetrf_nb(N) ;
  maxnb_d = maxnb ; 
  int lwork = N*maxnb;
  int lwork_d = N*maxnb_d;


  LDB = LDX = LDA = N ;
  int status ;
  double *d_A , * d_B , *d_X ; 
  float *h_work_M_S;
  double *h_work_M_D ; 
  float *M_SWORK ; 
  double *M_WORK ;	
  int *IPIV ;
  double *A , *B, *X ;
  int *DIPIV;
  float *As, *Bs, *Xs;
  double *L_WORK ;	
  float  * L_SWORK ;
  double *res_ ; 
  double *CACHE ;  
  int CACHE_L  = 10000 ;

  size = (N+32)*(N+32) + 32*maxnb + lwork+2*maxnb*maxnb; 
  status = cublasAlloc( size, sizeof(double), (void**)&d_A ) ;
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (dipiv)\n");
    goto FREE1;
  }
	
  size = LDB * NRHS ;
  d_B = ( double *) malloc ( sizeof ( double ) * size);
  status = cublasAlloc( size, sizeof(double), (void**)&d_B ) ;
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (dipiv)\n");
    goto FREE2;
  }

  d_X = ( double *) malloc ( sizeof ( double ) * size);
  status = cublasAlloc( size, sizeof(double), (void**)&d_X ) ;
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

   status = cudaMallocHost( (void**)&h_work_M_D,  size*sizeof(double) );
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (dipiv)\n");
    goto FREE5;
  }
    size = N*NRHS ; 
   status =  cublasAlloc( size, sizeof(double), (void**)&M_WORK ) ;
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (dipiv)\n");
    goto FREE6;
  }
    size = (N+32)*(N+32) + 32*maxnb + lwork+2*maxnb*maxnb;
    size += maxnb*N*NRHS;
   status = cublasAlloc( size, sizeof(float), (void**)&M_SWORK ) ;
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (dipiv)\n");
    goto FREE7;
  }

    size =3* LDA * N ;
    A = ( double *) malloc ( sizeof ( double ) * size);
    if( A == NULL )
    {
	 printf("Allocation Error\n");
         goto FREE8;
    }	    	

 status = cublasAlloc(N,sizeof(int), (void**)&DIPIV);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (dipiv)\n");
    goto FREE9;
  }

    size = LDB * NRHS ;
    B = ( double *) malloc ( sizeof ( double ) * size);
    if( B == NULL )
    {
	 printf("Allocation Error\n");
         goto FREE10;
    }	    	

    X = ( double *) malloc ( sizeof ( double ) * size);
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
    L_WORK = ( double*) malloc ( sizeof ( double ) * size ) ;
    if(L_WORK==NULL)
    {
	 printf("Allocation Error\n");
         goto FREE16;
    }	    	
    size += ( N * N ) ; 
    L_SWORK = (float*) malloc ( sizeof (float) * size ) ;
    if(L_SWORK==NULL)
    {
	 printf("Allocation Error\n");
         goto FREE17;
    }	    	

    size = N*NRHS ;
    res_ = ( double *) malloc ( sizeof(double)*size);
    if( res_ == NULL ) 
    {
	 printf("Allocation Error\n");
         goto FREE18;
    }	    	


   size = CACHE_L * CACHE_L ;
   CACHE = ( double *) malloc ( sizeof( double) * size ) ; 
   if( CACHE == NULL ) 
    {
	 printf("Allocation Error\n");
         goto FREE19;
    }	    	

  for(i=0;i<count;i++){

    N = step*(i)+startN;
    int N1 = N;
    
    int INFO[1];

    double *WORK ;
    float  *SWORK ;
    double *L_A , *L_B , *L_X ,*L_WORK1, *L_SWORK1 ;	


    LDB = LDX = LDA = N ;



    maxnb = magma_get_sgetrf_nb(N) ;
    maxnb_d = magma_get_dgetrf_nb(N) ;
    maxnb = maxnb > maxnb_d ? maxnb : maxnb_d ;
    maxnb_d = maxnb ; 
    lwork = N1*maxnb;
    lwork_d = N1*maxnb_d;
 

    size = LDA * N ;
    init_matrix(A, size, sizeof(double));
    size = LDB * NRHS ;
    init_matrix(B, size, sizeof(double));
    
    double perf ;  
    if( LEVEL == 0 ) printf("DIM  ");
    int maxnb = magma_get_sgetrf_nb(N) ;
    int PTSX = 0 , PTSA = maxnb*N*NRHS ;

    //magma_sgetrs("N",N ,NRHS, M_SWORK+PTSA ,N,IPIV, M_SWORK+PTSX, N,INFO );
    printf("%4d ,",N); 
    fprintf(fp,"%4d ,",N); 
/*
    //double alpha = 1 , beta=1; 
    float alpha = 1 , beta=1; 
    cache_flush( CACHE, CACHE_L*CACHE_L);
    start = get_current_time();
    sgemm_("N","N", &N, &N,&N,&alpha, As, &LDA,As+N*N, &N,&beta, As+N*N*2,&N);
    end = get_current_time();
    perf = (2.*N*N*N)/(1000000*GetTimerValue(start,end));
    printf("%6.2f \n", perf);
    fflush( stdout);
    continue ;
  */  
   //=====================================================================
    //              SP - CPU 
    //=====================================================================
    //cache_flush( CACHE, CACHE_L*CACHE_L);
    start = get_current_time();
    sgetrf_( &N, &N, As, &LDA, IPIV,  INFO);
    end = get_current_time();
    perf = (2.*N*N*N/3.)/(1000000*GetTimerValue(start,end));
if( printall == 1 ) 
    printf("%6.2f ,", perf);
    fprintf(fp,"%6.2f ,", perf);
//    continue ;
    //=====================================================================
    //              MIXED - CPU 
    //=====================================================================
    cublasSetMatrix( N, N, sizeof( double ), A, N, d_A, N ) ;
    cublasSetMatrix( N, NRHS, sizeof( double ), X, N, d_X, N ) ;
    cublasSetMatrix( N, NRHS, sizeof( double ), B, N, d_B, N ) ;

    dlacpy_("All", &N, &NRHS, B , &LDB, X, &N);

    cache_flush( CACHE, CACHE_L*CACHE_L);
    start = get_current_time();
    dsgesv_( &N, &NRHS, A, &LDA, IPIV, B,&LDB, X, &LDX, L_WORK,L_SWORK, &ITER, INFO);
    //dgesv_( &N, &NRHS, A, &LDA, IPIV, X, &LDB, INFO);
    end = get_current_time();

    //printf("%d ",INFO[0]);
    //printf("%d ",ITER);
    int iter_CPU = ITER ;
    perf = (2.*N*N*N/3.+2.*N*N)/(1000000*GetTimerValue(start,end));
if( printall == 1 ) 
    printf("%6.2f ,", perf);
    fprintf(fp, "%6.2f ,", perf);
    //dsgesv_l( N, NRHS, A, LDA, IPIV, B,LDB, X, LDX, L_WORK,L_SWORK, &ITER, INFO);
    dlacpy_("All", &N, &NRHS, X , &LDB, res_, &N);

/*
    //=====================================================================
    //              DP - CPU 
    //=====================================================================

    cublasGetMatrix( N, N, sizeof( double ), d_A, N, A, N ) ;
    cublasGetMatrix( N, NRHS, sizeof( double ), d_X, N, X, N ) ;
    cublasGetMatrix( N, NRHS, sizeof( double ), d_B, N, B, N ) ;
    dlag2s_(&N,&N,A, &LDA, As, &LDA , INFO);
    dlag2s_(&N,&NRHS,B, &LDA, Bs, &LDA , INFO);
    dlag2s_(&N,&NRHS,X, &LDA, Xs, &LDA , INFO);
    cache_flush( CACHE, CACHE_L*CACHE_L);
    start = get_current_time();
    dgesv_( &N, &NRHS, A, &LDA, IPIV, B, &LDB, INFO);
    end = get_current_time();
    perf = (2.*N*N*N/3.+2.*N*N)/(1000000*GetTimerValue(start,end));
if( printall == 1 ) 
    printf("%6.2f ,", perf);
    fprintf(fp, "%6.2f ,", perf);
 
   //=====================================================================
    //              SP - CPU 
    //=====================================================================
    cache_flush( CACHE, CACHE_L*CACHE_L);
    start = get_current_time();
    sgesv_( &N, &NRHS, As, &LDA, IPIV, Bs, &LDB, INFO);
    end = get_current_time();
    perf = (2.*N*N*N/3.+2.*N*N)/(1000000*GetTimerValue(start,end));
if( printall == 1 ) 
    printf("%6.2f ,", perf);
    fprintf(fp, "%6.2f ,", perf);
*/
    //=====================================================================
    //              MIXED - GPU 
    //=====================================================================
    cublasGetMatrix( N, N, sizeof( double ), d_A, N, A, N ) ;
    perf = 0.0;
    cache_flush( CACHE, CACHE_L*CACHE_L);
    start = get_current_time();
    //cublasDgemm( 'N', 'N', N, N, NRHS, -1.0, d_A, LDA, d_B, LDX, 1.0, d_X, N);
    magma_dsgesv(  N , NRHS,d_A,LDA,IPIV,d_B,LDB,d_X,LDX,M_WORK,M_SWORK,&ITER,INFO, h_work_M_S, h_work_M_D, DIPIV);
    end = get_current_time();
    //printf("%d ",INFO[0]);
    printf("%d ",ITER);
    int iter_GPU = ITER ;
    perf = (2.*N*N*N/3.+2.*N*N)/(1000000*GetTimerValue(start,end));
    printf("%6.2f ", perf);
    fprintf(fp,"%6.2f ,", perf);
    cublasGetMatrix( N, NRHS, sizeof( double ), d_X, N, X, N ) ;


    //=====================================================================
    //              DP - GPU 
    //=====================================================================
    cublasSetMatrix( N, N, sizeof( double ), A, N, d_A, N ) ;
    float RMAX = slamch_("O");
   // magmablas_dlag2s_64_64_16_4_v2(N , N , d_A , LDA , M_SWORK+PTSA, N, RMAX); // Merge with DLANGE /
   // magmablas_dlag2s_64_64_16_4_v2(N , NRHS , d_B , LDA , M_SWORK+PTSX, N ,RMAX); // Merge with DLANGE /
    cache_flush( CACHE, CACHE_L*CACHE_L);
    start = get_current_time();
    magma_dgetrf_gpu(&N, &N, d_A, &N, IPIV, h_work_M_D, INFO);
    //magma_dgetrs_v2("N",N ,NRHS, d_A ,N,IPIV, d_B, N,INFO, h_work_M_D );
    end = get_current_time();
    perf = (2.*N*N*N/3.)/(1000000*GetTimerValue(start,end));
    printf("%6.2f ", perf);
    fprintf(fp, "%6.2f ,", perf);
    cublasGetMatrix( N, NRHS, sizeof( double ), d_B, N, res_, N ) ;



    cublasSetMatrix( N, N, sizeof( double ), A, N, d_A, N ) ;
    RMAX = slamch_("O");
   // magmablas_dlag2s_64_64_16_4_v2(N , N , d_A , LDA , M_SWORK+PTSA, N, RMAX); // Merge with DLANGE /
   // magmablas_dlag2s_64_64_16_4_v2(N , NRHS , d_B , LDA , M_SWORK+PTSX, N ,RMAX); // Merge with DLANGE /
    cache_flush( CACHE, CACHE_L*CACHE_L);
    start = get_current_time();
    magma_dgetrf_gpu(&N, &N, d_A, &N, IPIV, h_work_M_D, INFO);
    magma_dgetrs_v2("N",N ,NRHS, d_A ,N,IPIV, d_B, N,INFO, h_work_M_D );
    end = get_current_time();
    perf = (2.*N*N*N/3.+2.*N*N)/(1000000*GetTimerValue(start,end));
    printf("%6.2f ", perf);
    fprintf(fp, "%6.2f ,", perf);
    cublasGetMatrix( N, NRHS, sizeof( double ), d_B, N, res_, N ) ;

    //=====================================================================
    //              SP - GPU 
    //=====================================================================
    cache_flush( CACHE, CACHE_L*CACHE_L);
    start = get_current_time();
    magma_sgetrf_gpu2(&N, &N,M_SWORK+PTSA, &N,IPIV, DIPIV, h_work_M_S, INFO);
    magma_sdgetrs_gpu(&N,&NRHS,M_SWORK+PTSA,&LDA,DIPIV,M_SWORK,d_B ,&LDB, INFO);
    //magma_sgetrf_gpu(&N, &N, M_SWORK+PTSA, &N, IPIV, h_work_M_S, INFO);
    //magma_sgetrs_v2("N",N ,NRHS, M_SWORK+PTSA ,N,IPIV, M_SWORK+PTSX, N,INFO, h_work_M_S );
    end = get_current_time();
    perf = (2.*N*N*N/3.+2.*N*N)/(1000000*GetTimerValue(start,end));
    printf("%6.2f ", perf);
    fprintf(fp, "%6.2f ,", perf);

    cache_flush( CACHE, CACHE_L*CACHE_L);
    start = get_current_time();
    magma_sgetrf_gpu(&N, &N, M_SWORK+PTSA, &N, IPIV, h_work_M_S, INFO);
    end = get_current_time();
    perf = (2.*N*N*N/3.)/(1000000*GetTimerValue(start,end));
    printf("%6.2f ", perf);
    fprintf(fp, "%6.2f ,", perf);
    //=====================================================================
    //              ERROR DP vs MIXED  - GPU 
    //=====================================================================

   double res ;
   double exact_res = 0.;
   double max_norm_x = 0.0 , max_norm_res = 0.0;
   int it =0,jt=0;
   for(it=0; it<N; it++) {
   for(jt=0; jt<NRHS; jt++) {
       res = res_[it+jt*N] - X[it+jt*N];
       exact_res += res*res;
       if (max_norm_res < fabs(res)) max_norm_res = fabs(res);
       if (max_norm_x < fabs(X[it+jt*N])) max_norm_x = fabs(X[it+jt*N]);
   }
   }
if( printall == 1 ) 
   printf("   %10.20lf ,",exact_res);
   fprintf(fp, "   %10.20lf ,",exact_res);

   exact_res = sqrt(exact_res);

   double matnorm = dlange_("I", &N, &NRHS,  X, &N, res_);
 /*
    printf("Final residual = %e\n\n", exact_res);
    printf("                 %e / (%e * %e) = %e\n",
            max_norm_res, matnorm, max_norm_x,
            max_norm_res/(matnorm*max_norm_x));

*/
if( printall == 1 ) 
    printf("%e ,", exact_res);
    fprintf(fp, "%e ,", exact_res);
if( printall == 1 ) 
    printf("%e ,",
            max_norm_res/(matnorm*max_norm_x));
    fprintf(fp, "%e ,",
            max_norm_res/(matnorm*max_norm_x));

if( printall == 0 )  printf("\n");


    LEVEL = 1 ; 
if( printall == 1 ) 
    printf(" %4d , %2d , %2d\n",NRHS,iter_CPU, iter_GPU); 
    fprintf(fp, " %4d , %2d , %2d\n",NRHS,iter_CPU, iter_GPU); 
  }

    free(CACHE);
FREE19:
    free(res_);
FREE18:
    free(L_SWORK);
FREE17:
    free(L_WORK);
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
    cublasFree(DIPIV);
FREE9:
    free(A);
FREE8:
    cublasFree(M_SWORK);  
FREE7:
    cublasFree(M_WORK);  
FREE6:
    cublasFree(h_work_M_D);
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
