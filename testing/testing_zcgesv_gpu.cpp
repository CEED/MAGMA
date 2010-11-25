/*
  -- MAGMA (version 0.1) --
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

int main(int argc , char **argv)
{
#if defined(PRECISION_z) && (GPUSHMEM < 200)
    fprintf(stderr, "This functionnality is not available in MAGMA for this precisions actually\n");
    return EXIT_SUCCESS;
#else
    cuInit( 0 );
    cublasInit( );

    printf("Iterative Refinement- LU \n");
    printf("\n");
    printout_devices( );

    printf("\nUsage:\n\t\t ./testing_zcgesv N");
    printf("\n\nEpsilon(Double): %10.20lf \nEpsilon(Single): %10.20lf\n", lapackf77_dlamch("Epsilon"), lapackf77_slamch("Epsilon"));

    TimeStruct start, end;
    int LEVEL=1;
    printf("\n\nN\tDouble-Factor\tDouble-Solve\tSingle-Factor\tSigle-Solve\tMixed Precision Solver\t || b-Ax || / ||A||  \t NumIter\n");
    printf("===========================================================================================================================================================\n");

    int i ;
    int sizetest[10] = {1024,2048,3072,4032,5184,6016,7040,8064,9088,10112};
    int startN = 1024;
    int count  = 8;
    int step   = 1024;
    int N      = count * step ;
    int NRHS   = 1;
    N = startN+(count-1) * step + 32 ;
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
    int maxnb   = magma_get_cgetrf_nb(N) ;
    int maxnb_d = magma_get_zgetrf_nb(N) ;
    maxnb_d = maxnb ;
    int lwork = N*maxnb;
    int lwork_d = N*maxnb_d;
    int ione     = 1;
    int ISEED[4] = {0,0,0,1};

    /*
      This is crucial for generic matrix size
      Keep in mind to give some bigger amount of memory.
    */
    LDB = LDX = LDA = N ;
    LDA = ( N / 32 ) * 32 ;
    if ( LDA < N ) LDA += 32 ;
    LDB = LDA;

    int status ;
    double perf, lperf;
    cuDoubleComplex *dA, *dB, *dX;
    cuDoubleComplex *A,  *B,  *X;
    cuDoubleComplex *M_WORK;
    cuFloatComplex  *M_SWORK;
    int *IPIV, *DIPIV;

    cuDoubleComplex *res_ ;
    cuDoubleComplex ONE    = MAGMA_Z_NEG_ONE;
    cuDoubleComplex NEGONE = MAGMA_Z_ONE;

    size = (N+32)*(N+32) + 32*maxnb + lwork+2*maxnb*maxnb;
    status = cublasAlloc( size, sizeof(cuDoubleComplex), (void**)&dA ) ;
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! device memory allocation error (dA)\n");
        goto FREE1;
    }

    size = LDB * NRHS ;
    dB = ( cuDoubleComplex *) malloc ( sizeof ( cuDoubleComplex ) * size);
    status = cublasAlloc( size, sizeof(cuDoubleComplex), (void**)&dB ) ;
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! device memory allocation error (dB)\n");
        goto FREE2;
    }

    dX = ( cuDoubleComplex *) malloc ( sizeof ( cuDoubleComplex ) * size);
    status = cublasAlloc( size, sizeof(cuDoubleComplex), (void**)&dX ) ;
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! device memory allocation error (dX)\n");
        goto FREE3;
    }

    size = N*NRHS ;
    status = cublasAlloc( size, sizeof(cuDoubleComplex), (void**)&M_WORK ) ;
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! device memory allocation error (M_WORK)\n");
        goto FREE6;
    }
    size = (N+32)*(N+32) + 32*maxnb + lwork+2*maxnb*maxnb;
    size += maxnb*N*NRHS;
    status = cublasAlloc( size, sizeof(cuFloatComplex), (void**)&M_SWORK ) ;
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! device memory allocation error (M_SWORK)\n");
        goto FREE7;
    }

    size =3* LDA * N ;
    A = ( cuDoubleComplex *) malloc ( sizeof ( cuDoubleComplex ) * size);
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
    B = ( cuDoubleComplex *) malloc ( sizeof ( cuDoubleComplex ) * size);
    if( B == NULL )
        {
            printf("Allocation Error\n");
            goto FREE10;
        }

    X = ( cuDoubleComplex *) malloc ( sizeof ( cuDoubleComplex ) * size);
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

    size = N*NRHS ;
    res_ = ( cuDoubleComplex *) malloc ( sizeof(cuDoubleComplex)*size);
    if( res_ == NULL )
        {
            printf("Allocation Error\n");
            goto FREE18;
        }


    for(i=0;i<count;i++){
        if( once == 0 )
            N = sizetest[i];
        else 
            N =  once ;
        
        int N1 = N ;
        int INFO[1];

        LDB = LDX = LDA = N ;
        /*
          This is crucial for LU factorization
          the LDA should be divisible by 32.
        */
        LDA = ( N / 32 ) * 32 ;
        if ( LDA < N ) LDA += 32 ;
        LDB = LDA;
        
        maxnb   = magma_get_cgetrf_nb(N) ;
        maxnb_d = magma_get_zgetrf_nb(N) ;
        maxnb   = maxnb > maxnb_d ? maxnb : maxnb_d ;
        maxnb_d = maxnb ;
        lwork   = N1*maxnb;
        lwork_d = N1*maxnb_d;

        size = LDA * N ;
        lapackf77_zlarnv( &ione, ISEED, &size, A );
        size = LDB * NRHS ;
        lapackf77_zlarnv( &ione, ISEED, &size, B );

        if( LEVEL == 0 ) printf("DIM  ");
        int PTSA = maxnb*N*NRHS ;

        printf("%5d",N);

        cublasSetMatrix( N, N,    sizeof( cuDoubleComplex ), A, N, dA, LDA );
        cublasSetMatrix( N, NRHS, sizeof( cuDoubleComplex ), X, N, dX, LDB );
        cublasSetMatrix( N, NRHS, sizeof( cuDoubleComplex ), B, N, dB, LDB );

        lapackf77_zlacpy( MagmaUpperLowerStr, &N, &NRHS, B, &LDB, X,    &N);
        lapackf77_zlacpy( MagmaUpperLowerStr, &N, &NRHS, X, &LDB, res_, &N);

        //=====================================================================
        //              MIXED - GPU
        //=====================================================================

        *INFO = 0 ;
        perf  = 0.0;
        start = get_current_time();
        magma_zcgesv_gpu(N, NRHS, 
                         dA, LDA, IPIV, DIPIV, 
                         dB, LDB, dX, LDX, 
                         M_WORK, M_SWORK, &ITER, INFO);
        
        end = get_current_time();
        int iter_GPU = ITER ;
        lperf = (2.*N*N*N/3.+2.*N*N)/(1000000*GetTimerValue(start,end));
        cublasGetMatrix( N, NRHS, sizeof( cuDoubleComplex ), dX, N, X, LDB );

        //=====================================================================
        //              ERROR DP vs MIXED  - GPU
        //=====================================================================
        double Rnorm, Anorm;
        double *worke = (double *)malloc(N*sizeof(double));

        Anorm = lapackf77_zlange("I", &N, &N, A, &N, worke);

        blasf77_zgemm( MagmaNoTransStr, MagmaNoTransStr, 
                       &N, &NRHS, &N, 
                       &NEGONE, A, &N, 
                                X, &LDX, 
                       &ONE,    B, &N);
        Rnorm = lapackf77_zlange("I", &N, &NRHS, B, &LDB, worke);
        free(worke);

        //=====================================================================
        //              DP - GPU
        //=====================================================================

        cublasSetMatrix( N, N, sizeof( cuDoubleComplex ), A, N, dA, LDA );
        
        start = get_current_time();
        magma_zgetrf_gpu(N, N, dA, N, IPIV, INFO);
        end = get_current_time();
        perf = (2.*N*N*N/3.)/(1000000*GetTimerValue(start,end));
        printf("\t%6.2f", perf);

        cublasGetMatrix( N, NRHS, sizeof( cuDoubleComplex ), dB, LDB, res_, N   );
        cublasSetMatrix( N, N,    sizeof( cuDoubleComplex ), A,  N,   dA,   LDA );

        start = get_current_time();
        magma_zgetrf_gpu(N, N, dA, N, IPIV, INFO);
        magma_zgetrs_gpu( MagmaNoTrans, N, NRHS, dA, LDA, IPIV, dB, LDB, INFO );
        end = get_current_time();

        perf = (2.*N*N*N/3.+2.*N*N)/(1000000*GetTimerValue(start,end));
        printf("\t\t%6.2f", perf);

        cublasGetMatrix( N, NRHS, sizeof( cuDoubleComplex ), dB, LDB, res_, N ) ;

        //=====================================================================
        //              SP - GPU
        //=====================================================================

        start = get_current_time();
        magma_cgetrf_gpu(N, N, M_SWORK+PTSA, LDA, IPIV, INFO);
        end = get_current_time();
        perf = (2.*N*N*N/3.)/(1000000*GetTimerValue(start,end));
        printf("\t\t%6.2f", perf);

        start = get_current_time();
        magma_cgetrf_gpu( N, N,    M_SWORK+PTSA, LDA, IPIV, INFO);
        magma_zcgetrs_gpu(N, NRHS, M_SWORK+PTSA, LDA, DIPIV, M_SWORK, dB, LDB, INFO);
        end = get_current_time();
        perf = (2.*N*N*N/3.+2.*N*N)/(1000000*GetTimerValue(start,end));

        printf("\t\t%6.2f", perf);
        printf("\t\t\t%6.2f", lperf);
        printf("\t\t\t%e\t%3d", Rnorm / Anorm, iter_GPU);
        printf("\n");

        if( once != 0 ) break ;
        
    }

    free(res_);
  FREE18:
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
    cublasFree(dX);
  FREE3:
    cublasFree(dB);
  FREE2:
    cublasFree(dA);
  FREE1:
    //fclose(fp);
    cublasShutdown();

#endif /*defined(PRECISION_z) && (GPUSHMEM < 200)*/
}
