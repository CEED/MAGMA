/*
 *  -- MAGMA (version 1.1) --
 *     Univ. of Tennessee, Knoxville
 *     Univ. of California, Berkeley
 *     Univ. of Colorado, Denver
 *     November 2011
 *
 * @precisions normal z -> c d s
 *
 **/
// includes, system
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <unistd.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zposv_gpu
*/
int main( int argc, char** argv)
{
    TESTING_CUDA_INIT();

    real_Double_t   gflops, gpu_perf, gpu_time;
    double          Rnorm, Anorm, Xnorm, *work;
    cuDoubleComplex c_one     = MAGMA_Z_ONE;
    cuDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    cuDoubleComplex *h_A, *h_B, *h_X;
    cuDoubleComplex *d_A, *d_B;
    const char  *uplo     = MagmaLowerStr;
    magma_int_t lda, ldb, N;
    magma_int_t ldda, lddb;
    magma_int_t i, info, szeA, szeB;
    magma_int_t ione     = 1;
    magma_int_t NRHS     = 100;
    magma_int_t ISEED[4] = {0,0,0,1};
    const int MAXTESTS   = 10;
    magma_int_t size[MAXTESTS] = {1024,2048,3072,4032,5184,6016,7040,8064,9088,10112};

    // process command line arguments
    printf( "\nUsage:\n" );
    printf( "  %s -N <matrix size> -R <right hand sides>\n", argv[0] );
    printf( "  -N can be repeated up to %d times\n", MAXTESTS );
    int ntest = 0;
    int ch;
    while( (ch = getopt( argc, argv, "N:R:" )) != -1 ) {
        switch( ch ) {
            case 'N':
                if ( ntest == MAXTESTS ) {
                    printf( "error: -N exceeded maximum %d tests\n", MAXTESTS );
                    exit(1);
                }
                else {
                    size[ntest] = atoi( optarg );
                    if ( size[ntest] <= 0 ) {
                        printf( "error: -N value %d <= 0\n", size[ntest] );
                        exit(1);
                    }
                    ntest++;
                }
                break;
            case 'R':
                NRHS = atoi( optarg );
                break;
            case '?':
            default:
                exit(1);
        }
    }
    argc -= optind;
    argv += optind;
    if ( ntest == 0 ) {
        ntest = MAXTESTS;
    }
    
    // allocate maximum amount of memory required
    N = 0;
    for( i = 0; i < ntest; ++i ) {
        N = max( N, size[i] );
    }
    lda = ldb = N;
    lddb = ldda = ((N+31)/32)*32;
    
    TESTING_MALLOC( h_A, cuDoubleComplex, lda*N    );
    TESTING_MALLOC( h_B, cuDoubleComplex, ldb*NRHS );
    TESTING_MALLOC( h_X, cuDoubleComplex, ldb*NRHS );
    TESTING_MALLOC( work, double,         N        );

    TESTING_DEVALLOC( d_A, cuDoubleComplex, ldda*N    );
    TESTING_DEVALLOC( d_B, cuDoubleComplex, lddb*NRHS );

    printf("\n");
    printf("    N   NRHS   GPU GFlop/s (sec)   ||B - AX|| / ||A||*||X||\n");
    printf("===========================================================\n");
    
    for( i = 0; i < ntest; ++i ) {
        N   = size[i];
        lda = ldb = N;
        ldda = ((N+31)/32)*32;
        lddb = ldda;
        gflops = ( FLOPS_ZPOTRF( (double)N ) +
                   FLOPS_ZPOTRS( (double)N, (double)NRHS ) ) / 1e9;

        /* ====================================================================
           Initialize the matrix
           =================================================================== */
        szeA = lda*N;
        szeB = ldb*NRHS;
        lapackf77_zlarnv( &ione, ISEED, &szeA, h_A );
        lapackf77_zlarnv( &ione, ISEED, &szeB, h_B );
        /* Symmetrize and increase the diagonal */
        {
            magma_int_t i, j;
            for(i=0; i<N; i++) {
                MAGMA_Z_SET2REAL( h_A[i*lda+i], ( MAGMA_Z_REAL(h_A[i*lda+i]) + 1.*N ) );
                for(j=0; j<i; j++)
                    h_A[i*lda+j] = cuConj(h_A[j*lda+i]);
            }
        }
        
        cublasSetMatrix( N, N,    sizeof( cuDoubleComplex ), h_A, N, d_A, ldda );
        cublasSetMatrix( N, NRHS, sizeof( cuDoubleComplex ), h_B, N, d_B, lddb );

        /* ====================================================================
           Performs operation using MAGMA
           =================================================================== */
        gpu_time = magma_wtime();
        magma_zposv_gpu( uplo[0], N, NRHS, d_A, ldda, d_B, lddb, &info );
        gpu_time = magma_wtime() - gpu_time;
        if (info != 0)
            printf("magma_zpotrf_gpu returned error %d.\n", info);

        gpu_perf = gflops / gpu_time;

        /* =====================================================================
           Residual
           =================================================================== */
        cublasGetMatrix( N, NRHS, sizeof( cuDoubleComplex ), d_B, lddb, h_X, ldb );
        
        Anorm = lapackf77_zlange("I", &N, &N,    h_A, &lda, work);
        Xnorm = lapackf77_zlange("I", &N, &NRHS, h_X, &ldb, work);

        blasf77_zgemm( MagmaNoTransStr, MagmaNoTransStr, &N, &NRHS, &N,
                       &c_one,     h_A, &lda,
                                   h_X, &ldb,
                       &c_neg_one, h_B, &ldb );
        
        Rnorm = lapackf77_zlange("I", &N, &NRHS, h_B, &ldb, work);

        printf( "%5d  %5d   %7.2f (%7.2f)   %8.2e\n",
                N, NRHS, gpu_perf, gpu_time, Rnorm/(Anorm*Xnorm) );
    }

    /* Memory clean up */
    TESTING_FREE( h_A );
    TESTING_FREE( h_B );
    TESTING_FREE( h_X );
    TESTING_FREE( work );

    TESTING_DEVFREE( d_A );
    TESTING_DEVFREE( d_B );

    TESTING_CUDA_FINALIZE();
}
