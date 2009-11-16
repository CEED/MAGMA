/*
    -- MAGMA (version 0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2009
   --  Rajib Nath
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cublas.h"
#include "magma.h"
#include "magmablas.h"

void fill( double *A, int n, int maxi ){
        for( int j = 0; j < n; j++ )
                A[j] = double( (rand()%(maxi*2+1)) - maxi ) / ( maxi + 1.f );
}

int M , N , K , LDA, LDB , LDC ;

double verifyResult(const double *mat, const double *mat_ref) {
    double norm = 0.0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
               if (fabs((double)mat[i+j * M ] - (double)mat_ref[i+j * M ]) > norm){
                   norm = fabs((double)mat[i +j*M] - (double)mat_ref[i+j*M]);
                }
            }
    }
    return norm;
}


int main( int argc, char** argv) 
{
   FILE *fp ;
   fp = fopen("results_dgemm.txt", "w");


    printf("This is an Experimental Release of GEMM Routine without Padding\n\n");
    fprintf(fp, "This is an Experimental Release of GEMM Routine without Padding\n\n");

    int oneTime =512;
    int step = 512 ;
    int count = 10;   
    int flag = 1 ; 
if ( argc == 2 ) {
     oneTime = atoi(argv[1] ) ;
     step =  1000  ;
     count = 1;   
     flag = 0 ; 
    
}

    TimeStruct start, end;
    double cuda_perf , magma_perf ; 

    cuInit( 0 );
    cublasInit( );
    printout_devices( );
    printf("\n");
    fprintf(fp, "\n");
   
printf("Usage:\n\t\t./testing_dgemm N\n");
fprintf(fp, "Usage:\n\t\t./testing_dgemm N\n");

    char TRANSA = 'N' ;
    char TRANSB = 'N' ;
    printf("\n");
    fprintf(fp, "\n");
    printf("    N\t\tmagmablas0.2 GFLops/s\tcudablas-2.3 GFlops/s    error\n");
    fprintf(fp, "    N\t\tmagmablas0.2 GFLops/s\tcudablas-2.3 GFlops/s    error\n");
    printf("=============================================================================\n");
    fprintf(fp, "=============================================================================\n");
    for(int i=oneTime;i<=(oneTime+(count-1)*step);i+=step){
    for( int ops = 0 ; ops <1 + flag ; ops ++){

    double *A, *B, * C ; 
    double ALPHA=1 ,BETA = 1;  

 
    M = N = K = LDA = LDB = LDC =i+ops;
    int size_A1 , size_B1, size_C1 ;

    if( TRANSA == 'N') 	
       size_A1 = LDA * K ;
    else  	
       size_A1 = LDA * M ;
    if( TRANSB == 'N') 	
       size_B1 = LDB * N ;
    else	 
       size_B1 = LDB * K ;

    size_C1 = LDC * N ; 
    double *h_A = (double* ) malloc(sizeof(double) * size_A1);
    double *h_B = (double* ) malloc(sizeof(double) * size_B1);
    double *h_C_m = (double* ) malloc(sizeof(double) * size_C1);
    double *h_C_c = (double* ) malloc(sizeof(double) * size_C1);
    if( h_A == NULL ||  h_B == NULL ||  h_C_m == NULL ||  h_C_c == NULL )  {
      fprintf (stderr, "!!!! host memory allocation error\n");
      exit(1);
    }


    fill( h_A, size_A1, 31 );
    fill( h_B, size_B1, 31 );
    fill( h_C_m, size_C1, 31 );
    memcpy(h_C_c, h_C_m, sizeof(double) * size_C1);

      /* =====================================================================
         Performs operation using MAGMA-BLAS
         =================================================================== */
    double *d_A_m , *d_B_m , *d_C_m; 
    cublasAlloc( size_A1, sizeof(double), (void**)&d_A_m );
    cublasAlloc( size_B1, sizeof(double), (void**)&d_B_m ) ;
    cublasAlloc( size_C1, sizeof(double), (void**)&d_C_m ) ;
    if(TRANSA=='N')
    cublasSetMatrix( M, K, sizeof( double ), h_A,   LDA, d_A_m, LDA ) ;
    else
    cublasSetMatrix( K, M, sizeof( double ), h_A,   LDA, d_A_m, LDA ) ;
    if(TRANSB=='N')
    cublasSetMatrix( K, N, sizeof( double ), h_B,   LDB, d_B_m, LDB ) ;
    else
    cublasSetMatrix( N, K, sizeof( double ), h_B,   LDB, d_B_m, LDB ) ;
    cublasSetMatrix( M, N, sizeof( double ), h_C_m, LDC, d_C_m, LDC ) ;


    start = get_current_time();
    magmablasDgemm( TRANSA, TRANSB, M, N, K, ALPHA, d_A_m, LDA, d_B_m, LDB, BETA, d_C_m, LDC );
    end = get_current_time();
   // magmablasDgemm( TRANSA, TRANSB, M, N, K, ALPHA, d_A_m, LDA, d_B_m, LDB, BETA, d_C_m, LDC );
    cublasGetMatrix( M, N, sizeof( double ), d_C_m, LDC, h_C_m, LDC ) ;
    magma_perf = 2.*M*N*K/(GetTimerValue(start,end))/1e6 ;
    cublasFree(d_A_m);
    cublasFree(d_B_m);
    cublasFree(d_C_m);
      /* =====================================================================
         Performs operation using CUDA-BLAS
         =================================================================== */
    double *d_A_c , *d_B_c , *d_C_c;
    cublasAlloc( size_A1, sizeof(double), (void**)&d_A_c );
    cublasAlloc( size_B1, sizeof(double), (void**)&d_B_c ) ;
    cublasAlloc( size_C1, sizeof(double), (void**)&d_C_c ) ;
    if(TRANSA=='N')
    cublasSetMatrix( M, K, sizeof( double ), h_A,   LDA, d_A_c, LDA ) ;
    else	
    cublasSetMatrix( K, M, sizeof( double ), h_A,   LDA, d_A_c, LDA ) ;
    if(TRANSB=='N')
    cublasSetMatrix( K, N, sizeof( double ), h_B,   LDB, d_B_c, LDB ) ;
    else
    cublasSetMatrix( N, K, sizeof( double ), h_B,   LDB, d_B_c, LDB ) ;

    cublasSetMatrix( M, N, sizeof( double ), h_C_c, LDC, d_C_c, LDC ) ;
    start = get_current_time();
    cublasDgemm( TRANSA, TRANSB, M, N, K, ALPHA, d_A_c, LDA, d_B_c, LDB, BETA, d_C_c, LDC );
    end = get_current_time();
   // cublasDgemm( TRANSA, TRANSB, M, N, K, ALPHA, d_A_c, LDA, d_B_c, LDB, BETA, d_C_c, LDC );
    cublasGetMatrix( M, N, sizeof( double ), d_C_c, LDC, h_C_c, LDC ) ;
    cuda_perf = 2.*M*N*K/(GetTimerValue(start,end))/1e6 ;

    // * Memory clean up * /
    cublasFree(d_A_c);
    cublasFree(d_B_c);
    cublasFree(d_C_c);

      /* =====================================================================
         Error Computation and Performance Compariosn
         =================================================================== */


    double error = verifyResult(h_C_m, h_C_c);

   fprintf(fp, "%5d\t\t%6.12f\t\t%6.12f\t\t%e\n",
             M,magma_perf, cuda_perf, error);
   printf("%5d\t\t%6.12f\t\t%6.12f\t\t%e\n",
             M,magma_perf, cuda_perf, error);

    free(h_A);
    free(h_B);
    free(h_C_m);
    free(h_C_c);
}  
  } 
    cublasShutdown();
    fclose(fp);

}
