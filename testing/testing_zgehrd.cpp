/*
    -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

       @precisions normal z -> s d c

*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "testings.h"

// Flops formula
#define PRECISION_z
#define CHECK_ERROR
#if defined(PRECISION_z) || defined(PRECISION_c)
  #define FLOPS(n) ( 4.*10. * n * n * n / 3. )
#else
  #define FLOPS(n) (    10. * n * n * n / 3. )
#endif


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgehrd2
*/
int main( int argc, char** argv)
{
    TESTING_CUDA_INIT();

    TimeStruct start, end;
    cuDoubleComplex *h_A, *h_R, *h_work, *tau, *dT;
    double gpu_perf, cpu_perf, eps;

    /* Matrix size */
    int N=0, n2, lda;
    int size[10] = {1024,2048,3072,4032,5184,6016,7040,8064,9088,10112};

    cublasStatus status;
    int i, j, info;
    int ione     = 1;
    int ISEED[4] = {0,0,0,1};
    
    if (argc != 1){
        for(i = 1; i<argc; i++){
            if (strcmp("-N", argv[i])==0)
                N = atoi(argv[++i]);
        }
        if (N>0) size[0] = size[9] = N;
        else exit(1);
    }
    else {
        printf("\nUsage: \n");
        printf("  testing_zgehrd2 -N %d\n\n", 1024);
    }

    eps = lapackf77_dlamch( "E" );

    lda = N;
    n2 = size[9] * size[9];

    int nb = magma_get_zgehrd_nb(size[9]);
    int lwork = size[9]*nb;
    
    TESTING_MALLOC   ( h_A   , cuDoubleComplex, n2        );
    TESTING_MALLOC   ( tau   , cuDoubleComplex, size[9]   );
    TESTING_HOSTALLOC( h_R   , cuDoubleComplex, n2        );
    TESTING_HOSTALLOC( h_work, cuDoubleComplex, lwork     );
    TESTING_DEVALLOC ( dT    , cuDoubleComplex, nb*size[9]);

    printf("\n\n");
    printf("  N    CPU GFlop/s    GPU GFlop/s   |A-QHQ'|/N|A|  |I-QQ'|/N \n");
    printf("=============================================================\n");
    for(i=0; i<10; i++){
        N = lda = size[i];
        n2 = N*N;

        /* Initialize the matrices */
        lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
        lapackf77_zlacpy( MagmaUpperLowerStr, &N, &N, h_A, &N, h_R, &N );

        /* ====================================================================
           Performs operation using MAGMA
           =================================================================== */
        start = get_current_time();
        // magma_zgehrd2( N, ione, N, h_R, N, tau, h_work, &lwork, &info);
	magma_zgehrd ( N, ione, N, h_R, N, tau, h_work, &lwork, dT, &info);
        end = get_current_time();

        gpu_perf = FLOPS(N)/(1000000.*GetTimerValue(start,end));
        //printf("GPU Processing time: %f (s) \n", GetTimerValue(start,end)/1000.);

        /* =====================================================================
           Check the factorization
           =================================================================== */
        double result[2] = {0., 0.};

#ifdef CHECK_ERROR
        cuDoubleComplex *hwork_Q, *twork;

	TESTING_HOSTALLOC( hwork_Q, cuDoubleComplex, N*N );
        twork   = (cuDoubleComplex*)malloc( 2* N * N * sizeof(cuDoubleComplex));
	
        int ltwork = 2*N*N;

        for(j=0; j<n2; j++)
            hwork_Q[j] = h_R[j];
 
        for(j=0; j<N-1; j++)
            for(int i=j+2; i<N; i++)
                h_R[i+j*N] = MAGMA_Z_ZERO;

	//lapackf77_zunghr(&N, &ione, &N, hwork_Q, &N, tau, h_work, &lwork, &info);
	nb = magma_get_zgehrd_nb(N);
	magma_zunghr(N, ione, N, hwork_Q, N, tau, dT, nb, &info);

#if defined(PRECISION_z) || defined(PRECISION_c) 
        double *rwork   = (double*)malloc( N * sizeof(double));
        lapackf77_zhst01(&N, &ione, &N, h_A, &N, h_R, &N, hwork_Q, &N,
                         twork, &ltwork, rwork, result);
        free(rwork);
#else
        lapackf77_zhst01(&N, &ione, &N, h_A, &N, h_R, &N, hwork_Q, &N,
                         twork, &ltwork, result);
#endif

	TESTING_HOSTFREE( hwork_Q );
        free(twork);
        /* =====================================================================
           Performs operation using LAPACK
           =================================================================== */
        start = get_current_time();
        lapackf77_zgehrd(&N, &ione, &N, h_R, &lda, tau, h_work, &lwork, &info);
        end = get_current_time();
        if (info < 0)
            printf("Argument %d of zgehrd had an illegal value.\n", -info);
#endif

        cpu_perf = FLOPS(N)/(1000000.*GetTimerValue(start,end));
        //printf("CPU Processing time: %f (s) \n", GetTimerValue(start,end)/1000.);

        /* =====================================================================
           Print performance and error.
           =================================================================== */
        printf("%5d    %6.2f         %6.2f      %e %e\n",
               size[i], cpu_perf, gpu_perf,
               result[0]*eps, result[1]*eps);

        if (argc != 1)
            break;
    }

    /* Memory clean up */
    TESTING_FREE    ( h_A  );
    TESTING_FREE    ( tau  );
    TESTING_HOSTFREE(h_work);
    TESTING_HOSTFREE( h_R  );
    TESTING_DEVFREE ( dT   );

    /* Shutdown */
    TESTING_CUDA_FINALIZE();
    return EXIT_SUCCESS;
}
