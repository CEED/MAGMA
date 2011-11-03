/*
 *  -- MAGMA (version 1.0) --
 *     Univ. of Tennessee, Knoxville
 *     Univ. of California, Berkeley
 *     Univ. of Colorado, Denver
 *     November 2010
 *
 * @precisions normal z -> c d s
 *
 **/
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
#include "magma_lapack.h"
#include "testings.h"

// Flops formula
#define PRECISION_z
#if defined(PRECISION_z) || defined(PRECISION_c)
#define FLOPS(n) ( 6.*FMULS_GETRF(n,n) + 2.*FADDS_GETRF(n,n) \
                 + 6.*FMULS_GETRI(n)   + 2.*FADDS_GETRI(n)   )
#else
#define FLOPS(n) (    FMULS_GETRF(n,n) +    FADDS_GETRF(n,n) \
                 +    FMULS_GETRI(n)   +    FADDS_GETRI(n)   )
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgetrf
*/
int main( int argc, char** argv) 
{
    TESTING_CUDA_INIT();

    magma_timestr_t  start, end;
    double      flops, gpu_perf, cpu_perf;
    cuDoubleComplex *h_A, *h_R, *h_A1, *h_A2;
    cuDoubleComplex *d_A;
    magma_int_t N = 0, n2, lda, ldda;
    magma_int_t size[10] = {1024,2048,3072,4032,5184,6048,7200,8064,8928,10240};
    
    magma_int_t i, info;
    cuDoubleComplex c_zero    = MAGMA_Z_ZERO;
    cuDoubleComplex c_one     = MAGMA_Z_ONE;
    cuDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    cuDoubleComplex *work;
    cuDoubleComplex tmp;
    double rwork[1];
    magma_int_t *ipiv;
    
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
        printf("  testing_zgetri_gpu -N %d\n\n", 1024);
    }
    
    // query for Lapack workspace size */
    magma_int_t lwork;
    N     = size[9];
    lda   = N;
    work  = &tmp;
    lwork = -1;
    lapackf77_zgetri( &N, h_A, &lda, ipiv, work, &lwork, &info );
    lwork = int( MAGMA_Z_REAL( *work ));
    printf( "lwork %d\n", lwork );

    /* Allocate host memory for the matrix */
    n2   = N * N;
    ldda = ((N+31)/32) * 32;
    TESTING_MALLOC(   ipiv, magma_int_t,     N );
    TESTING_MALLOC(   work, cuDoubleComplex, lwork );
    TESTING_MALLOC(    h_A, cuDoubleComplex, n2);
    TESTING_MALLOC(   h_A1, cuDoubleComplex, n2);
    TESTING_MALLOC(   h_A2, cuDoubleComplex, n2);
    TESTING_HOSTALLOC( h_R, cuDoubleComplex, n2);
    TESTING_DEVALLOC(  d_A, cuDoubleComplex, ldda*N );

    printf("\n\n");
    printf("  N    CPU GFlop/s    GPU GFlop/s    ||R||_F / ||A||_F    |I - AAinv|_F / n |A|_F |Ainv|_F epsilon\n");
    printf("========================================================\n");
    for(i=0; i<10; i++){
        N   = size[i];
        lda = N; 
        n2  = lda*N;
        flops = FLOPS( (double)N ) / 1000000;
        
        ldda = ((N+31)/32)*32;

        /* Initialize the matrix */
        lapackf77_zlarnv( &ione, ISEED, &n2, h_A );

        /* ====================================================================
           Performs operation using MAGMA 
           =================================================================== */
        cublasSetMatrix( N, N, sizeof(cuDoubleComplex), h_A, lda, d_A, ldda);
        start = get_current_time();
        magma_zgetrf_gpu( N, N, d_A, ldda, ipiv, &info );
        magma_zgetri_gpu( N,    d_A, ldda, ipiv, &info );
        end = get_current_time();
        if (info != 0)
            printf( "An error occured in magma_zgetri, info=%d\n", info );

        gpu_perf = flops / GetTimerValue(start, end);
        
        cublasGetMatrix( N, N, sizeof(cuDoubleComplex), d_A, ldda, h_A1, lda);
        
        /* =====================================================================
           Performs operation using LAPACK 
           =================================================================== */
        lapackf77_zlacpy( MagmaUpperLowerStr, &N, &N, h_A, &lda, h_A2, &lda );
        start = get_current_time();
        lapackf77_zgetrf( &N, &N, h_A2, &lda, ipiv, &info );
        lapackf77_zgetri( &N,     h_A2, &lda, ipiv, work, &lwork, &info );
        end = get_current_time();
        if (info != 0)
            printf( "An error occured in zgetri, info=%d\n", info );
        
        cpu_perf = flops / GetTimerValue(start, end);
      
        /* =====================================================================
           Check the result | I - A Ainv | / n |A| |Ainv| e
           =================================================================== */
        lapackf77_zlaset( "Full", &N, &N, &c_zero, &c_one, h_R, &lda );  // identity
        blasf77_zgemm( "No", "No", &N, &N, &N,
                       &c_neg_one, h_A,  &lda,
                                   h_A1, &lda,
                       &c_one,     h_R,  &lda );
        double I_AAinv_norm = lapackf77_zlange( "Fro", &N, &N, h_R,  &lda, rwork );
        double A_norm       = lapackf77_zlange( "F",   &N, &N, h_A,  &lda, rwork );
        double Ainv_norm    = lapackf77_zlange( "Fro", &N, &N, h_A1, &lda, rwork );
        double eps = lapackf77_dlamch( "Epsilon" );
        
        /* =====================================================================
           Check the result compared to LAPACK
           =================================================================== */
        blasf77_zaxpy(&n2, &c_neg_one, h_A1, &ione, h_A2, &ione);
        double A1_A2_norm = lapackf77_zlange("f", &N, &N, h_R, &lda, rwork);
        
        printf( "%5d    %6.2f         %6.2f        %e        %e\n", 
                N, cpu_perf, gpu_perf,
                A1_A2_norm / (N*A_norm),
                I_AAinv_norm / (N*A_norm*Ainv_norm*eps) );
        
        if (argc != 1)
            break;
    }

    /* Memory clean up */
    TESTING_FREE( ipiv );
    TESTING_FREE( h_A  );
    TESTING_FREE( h_A2 );
    TESTING_HOSTFREE( h_R );
    TESTING_DEVFREE( d_A );

    /* Shutdown */
    TESTING_CUDA_FINALIZE();
}
