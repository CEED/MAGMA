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
#define ITERMAX 30
#define BWDMAX 1.0
void init_matrix(void *A, int size , int elem_size){
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

void copy_matrix(void *S, void *D,int size , int elem_size){
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

void cache_flush( double2 * CACHE , int length ) {
    int i = 0 ;
    for( i=0;i<length ;i++){
        CACHE[i]=CACHE[i]+0.1;
    }
}


int main(int argc , char **argv){
    cuInit( 0 );
    cublasInit( );

    //FILE *fp ;
    //fp = fopen("results_zcgesv.txt","w");
    //if( fp == NULL ) return 1;
    printf("Iterative Refinement- LU \n");
    //fprintf(fp, "Iterative Refinement- LU \n");
    printf("\n");
    printout_devices( );

    printf("\nUsage:\n\t\t ./testing_zcgesv N");
    //fprintf(fp, "\nUsage:\n\t\t ./testing_zcgesv N");
    //printf("Iterative Refinement\n");
    //fprintf(fp,"Iterative Refinement\n");

    printf("\n\nEpsilon(Double): %10.20lf \nEpsilon(Single): %10.20lf\n", lapackf77_dlamch("Epsilon"), lapackf77_slamch("Epsilon"));
    //fprintf(fp, "\nEpsilon(Double): %10.20lf \nEpsilon(Single): %10.20lf\n", lapackf77_dlamch("Epsilon"), lapackf77_slamch("Epsilon"));

    TimeStruct start, end;
    int LEVEL=1;
    printf("\n\nN\tDouble-Factor\tDouble-Solve\tSingle-Factor\tSigle-Solve\tMixed Precision Solver\t || b-Ax || / ||A||  \t NumIter\n");
    //fprintf(fp, "\n\nN\tDouble-Factor\tDouble-Solve\tSingle-Factor\tSigle-Solve\tMixed Precision Solver\t || b-Ax || / ||A||\t NumIter\n");
    printf("===========================================================================================================================================================\n");
    //fprintf(fp,"===========================================================================================================================================================\n");

    int i ;
    int sizetest[10] = {1024,2048,3072,4032,5184,6016,7040,8064,9088,10112};


    int startN=1024    ;
    int count = 8;
    int step = 1024 ;
    int N = count * step ;
    int NRHS=1 ;
    N =startN+(count-1) * step + 32 ;
    int once = 0 ;
    if( argc == 3) {
        N  = atoi( argv[2]) ;
        once = N ;
        startN = N ;
    }

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

    /*
      This is crucial for generic matrix size
      Keep in mind to give some bigger amount of memory.
    */
    LDB = LDX = LDA = N ;
    LDA = ( N / 32 ) * 32 ;
    if ( LDA < N ) LDA += 32 ;

    int status ;
    double2 *d_A , * d_B , *d_X ;
    float2 *h_work_M_S;
    double2 *h_work_M_D ;
    float2 *M_SWORK ;
    double2 *M_WORK ;
    int *IPIV ;
    double2 *A , *B, *X ;
    int *DIPIV;
    float2 *As, *Bs, *Xs;
    double2 *L_WORK ;
    float2  * L_SWORK ;
    double2 *res_ ;
    double2 *CACHE ;
    int CACHE_L  = 10000 ;

    size = (N+32)*(N+32) + 32*maxnb + lwork+2*maxnb*maxnb;
    status = cublasAlloc( size, sizeof(double2), (void**)&d_A ) ;
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! device memory allocation error (dipiv)\n");
        goto FREE1;
    }

    size = LDB * NRHS ;
    d_B = ( double2 *) malloc ( sizeof ( double2 ) * size);
    status = cublasAlloc( size, sizeof(double2), (void**)&d_B ) ;
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! device memory allocation error (dipiv)\n");
        goto FREE2;
    }

    d_X = ( double2 *) malloc ( sizeof ( double2 ) * size);
    status = cublasAlloc( size, sizeof(double2), (void**)&d_X ) ;
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! device memory allocation error (dipiv)\n");
        goto FREE3;
    }
    size =  (lwork+32*maxnb);
    status = cudaMallocHost( (void**)&h_work_M_S,  size*sizeof(float2) );
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! device memory allocation error (dipiv)\n");
        goto FREE4;
    }

    status = cudaMallocHost( (void**)&h_work_M_D,  size*sizeof(double2) );
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! device memory allocation error (dipiv)\n");
        goto FREE5;
    }
    size = N*NRHS ;
    status =  cublasAlloc( size, sizeof(double2), (void**)&M_WORK ) ;
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! device memory allocation error (dipiv)\n");
        goto FREE6;
    }
    size = (N+32)*(N+32) + 32*maxnb + lwork+2*maxnb*maxnb;
    size += maxnb*N*NRHS;
    status = cublasAlloc( size, sizeof(float2), (void**)&M_SWORK ) ;
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! device memory allocation error (dipiv)\n");
        goto FREE7;
    }

    size =3* LDA * N ;
    A = ( double2 *) malloc ( sizeof ( double2 ) * size);
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
    B = ( double2 *) malloc ( sizeof ( double2 ) * size);
    if( B == NULL )
        {
            printf("Allocation Error\n");
            goto FREE10;
        }

    X = ( double2 *) malloc ( sizeof ( double2 ) * size);
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
    As = ( float2 *) malloc ( sizeof ( float2 ) * size);
    if( As == NULL )
        {
            printf("Allocation Error\n");
            goto FREE13;
        }
    size = NRHS * N ;
    Bs = ( float2 *) malloc ( sizeof ( float2 ) * size);
    if( Bs == NULL )
        {
            printf("Allocation Error\n");
            goto FREE14;
        }
    Xs = ( float2 *) malloc ( sizeof ( float2 ) * size);
    if( Xs == NULL )
        {
            printf("Allocation Error\n");
            goto FREE15;
        }



    size = NRHS * N ;
    L_WORK = ( double2*) malloc ( sizeof ( double2 ) * size ) ;
    if(L_WORK==NULL)
        {
            printf("Allocation Error\n");
            goto FREE16;
        }
    size += ( N * N ) ;
    L_SWORK = (float2*) malloc ( sizeof (float2) * size ) ;
    if(L_SWORK==NULL)
        {
            printf("Allocation Error\n");
            goto FREE17;
        }

    size = N*NRHS ;
    res_ = ( double2 *) malloc ( sizeof(double2)*size);
    if( res_ == NULL )
        {
            printf("Allocation Error\n");
            goto FREE18;
        }


    size = CACHE_L * CACHE_L ;
    CACHE = ( double2 *) malloc ( sizeof( double2) * size ) ;
    if( CACHE == NULL )
        {
            printf("Allocation Error\n");
            goto FREE19;
        }

    for(i=0;i<count;i++){

//    N = step*(i)+startN - 32;

        if( once == 0 )
            N = sizetest[i];
        else N =  once ;

        int N1 = N ;

        int INFO[1];

        LDB = LDX = LDA = N ;
        /*
          This is crucial for LU factorization
          the LDA should be divisible by 32.
        */
        LDA = ( N / 32 ) * 32 ;
        if ( LDA < N ) LDA += 32 ;

        maxnb = magma_get_sgetrf_nb(N) ;
        maxnb_d = magma_get_dgetrf_nb(N) ;
        maxnb = maxnb > maxnb_d ? maxnb : maxnb_d ;
        maxnb_d = maxnb ;
        lwork = N1*maxnb;
        lwork_d = N1*maxnb_d;

        size = LDA * N ;
        init_matrix(A, size, sizeof(double2));
        size = LDB * NRHS ;
        init_matrix(B, size, sizeof(double2));

        double2 perf ;
        if( LEVEL == 0 ) printf("DIM  ");
        int maxnb = magma_get_sgetrf_nb(N) ;
        int PTSA = maxnb*N*NRHS ;

        printf("%5d",N);
        //fprintf(fp,"%5d",N);


        cublasSetMatrix( N, N,    sizeof( double2 ), A, N, d_A, LDA );
        cublasSetMatrix( N, NRHS, sizeof( double2 ), X, N, d_X, N   );
        cublasSetMatrix( N, NRHS, sizeof( double2 ), B, N, d_B, N   );

        lapackf77_zlacpy("All", &N, &NRHS, B , &LDB, X,    &N);
        lapackf77_zlacpy("All", &N, &NRHS, X , &LDB, res_, &N);


        //=====================================================================
        //              MIXED - GPU
        //=====================================================================

        *INFO = 0 ;
        perf = 0.0;
        start = get_current_time();
        magma_zcgesv_gpu(N, NRHS, d_A,LDA, IPIV, d_B,LDB, d_X, LDX, M_WORK,
                         M_SWORK, &ITER, INFO, h_work_M_S, h_work_M_D, DIPIV);
        end = get_current_time();
        int iter_GPU = ITER ;
        perf = (2.*N*N*N/3.+2.*N*N)/(1000000*GetTimerValue(start,end));
        double2 lperf = perf ;
        cublasGetMatrix( N, NRHS, sizeof( double2 ), d_X, N, X, N ) ;

        //=====================================================================
        //              ERROR DP vs MIXED  - GPU
        //=====================================================================
        double2 Rnorm, Anorm;
        double2 *worke = (double2 *)malloc(N*sizeof(double2));
        Anorm = lapackf77_zlange("I", &N, &N, A, &N, worke);
        double2 ONE = -1.0 , NEGONE = 1.0 ;
        blasf77_zgemm( "No Transpose", "No Transpose", &N, &NRHS, &N, &NEGONE, A, &N, X, &LDX, &ONE, B, &N);
        Rnorm=lapackf77_zlange("I", &N, &NRHS, B, &LDB, worke);
        free(worke);
        //=====================================================================
        //              DP - GPU
        //=====================================================================
        cublasSetMatrix( N, N, sizeof( double2 ), A, N, d_A, N ) ;
        float2 RMAX = lapackf77_slamch("O");
        start = get_current_time();
        magma_zgetrf_gpu(N, N, d_A, N, IPIV, INFO);
        end = get_current_time();
        perf = (2.*N*N*N/3.)/(1000000*GetTimerValue(start,end));
        printf("\t%6.2f", perf);
        //fprintf(fp,"\t%6.2f", perf);
        cublasGetMatrix( N, NRHS, sizeof( double2 ), d_B, N, res_, N ) ;



        cublasSetMatrix( N, N, sizeof( double2 ), A, N, d_A, N ) ;
        RMAX = lapackf77_slamch("O");
        start = get_current_time();
        magma_zgetrf_gpu(N, N, d_A, N, IPIV, INFO);
        magma_zgetrs_gpu('N', N, NRHS, d_A ,N,IPIV, d_B, N,INFO, h_work_M_D );
        end = get_current_time();
        perf = (2.*N*N*N/3.+2.*N*N)/(1000000*GetTimerValue(start,end));
        printf("\t\t%6.2f", perf);
        //fprintf(fp,"\t\t%6.2f", perf);
        cublasGetMatrix( N, NRHS, sizeof( double2 ), d_B, N, res_, N ) ;

        //=====================================================================
        //              SP - GPU
        //=====================================================================

        start = get_current_time();
        magma_cgetrf_gpu(N, N, M_SWORK+PTSA, LDA, IPIV, INFO);
        //magma_cgetrf_gpu(&N, &N, M_SWORK+PTSA, &N, IPIV, h_work_M_S, INFO);
        end = get_current_time();
        perf = (2.*N*N*N/3.)/(1000000*GetTimerValue(start,end));
        printf("\t\t%6.2f", perf);
        //fprintf(fp,"\t\t%6.2f", perf);

        start = get_current_time();
        magma_cgetrf_gpu(N, N, M_SWORK+PTSA, LDA, IPIV, INFO);
        magma_zcgetrs_gpu(N, NRHS, M_SWORK+PTSA, LDA, DIPIV, M_SWORK, d_B , LDB, INFO);
        end = get_current_time();
        perf = (2.*N*N*N/3.+2.*N*N)/(1000000*GetTimerValue(start,end));
        printf("\t\t%6.2f", perf);
        //fprintf(fp,"\t\t%6.2f", perf);


        printf("\t\t\t%6.2f", lperf);
        //fprintf(fp,"\t\t\t%6.2f", lperf);

        printf("\t\t\t%e\t%3d", Rnorm/Anorm, iter_GPU);
        //fprintf(fp, "\t\t\t%e\t%3d", Rnorm/Anorm, iter_GPU);

        printf("\n");
        //fprintf(fp,"\n");

        if( once != 0 ) break ;

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
    //fclose(fp);
    cublasShutdown();
}
