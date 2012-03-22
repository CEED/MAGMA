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
#include "magma_lapack.h"
#include "testings.h"

// Flops formula
#define PRECISION_z
#if defined(PRECISION_z) || defined(PRECISION_c)
#define FLOPS(n) ( 6. * FMULS_HETRD(n) + 2. * FADDS_HETRD(n))
#else
#define FLOPS(n) (      FMULS_HETRD(n) +      FADDS_HETRD(n))
#endif

extern "C" magma_int_t
magma_zhebbd(char uplo, magma_int_t n,
             cuDoubleComplex *a, magma_int_t lda,
             cuDoubleComplex *tau,
             cuDoubleComplex *work, magma_int_t lwork,
             magma_int_t *info);

extern "C" magma_int_t
magma_dsbtrd( int THREADS, char uplo, int n, int nb, 
                   double *A, int LDA, double *D, double *E);
extern "C" void cmp_vals(int n, double *wr1, double *wr2, double *nrmI, double *nrm1, double *nrm2);

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zhebbd
*/
int main( int argc, char** argv)
{
    TESTING_CUDA_INIT();

    magma_timestr_t       start, end;
    double           eps, flops, gpu_perf, gpu_time;
    cuDoubleComplex *h_A, *h_R, *h_work, *D, *E;
    cuDoubleComplex *tau;

    /* Matrix size */
    magma_int_t N = 0, n2, lda, lwork;
    magma_int_t size[10] = {1024,2048,3072,4032,5184,6016,7040,8064,9088,10112};

    magma_int_t i, j, k, info, nb, THREADS, checkres, once = 0;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    char *uplo = (char *)MagmaLowerStr;

    THREADS=1;
    if (argc != 1){
        for(i = 1; i<argc; i++){
            if (strcmp("-N", argv[i])==0) {
                N = atoi(argv[++i]);
                once = 1;
            }
            else if (strcmp("-threads", argv[i])==0) {
                THREADS = atoi(argv[++i]);
            }
            else if (strcmp("-U", argv[i])==0)
                uplo = (char *)MagmaUpperStr;
            else if (strcmp("-L", argv[i])==0)
                uplo = (char *)MagmaLowerStr;
        }
        if ( N > 0 )
            printf("  testing_zhebbd -L|U -N %d\n\n", N);
        else
        {
            printf("\nUsage: \n");
            printf("  testing_zhebbd -L|U -N %d\n\n", 1024);
            exit(1);
        }
    }
    else {
        printf("\nUsage: \n");
        printf("  testing_zhebbd -L|U -N %d\n\n", 1024);
        N = size[9];
    }
        
    checkres  = 0;//getenv("MAGMA_TESTINGS_CHECK") != NULL;
 
    eps = lapackf77_dlamch( "E" );
    lda = N;
    n2  = lda * N; 
    nb  = 64; //magma_get_zhebbd_nb(N);
    /* We suppose the magma nb is bigger than lapack nb */
    lwork = N*nb; 

    /* Allocate host memory for the matrix */
    TESTING_MALLOC(    h_A,    cuDoubleComplex, lda*N );
    TESTING_HOSTALLOC( h_R,    cuDoubleComplex, lda*N );
    TESTING_HOSTALLOC( h_work, cuDoubleComplex, lwork );
    TESTING_MALLOC(    tau,    cuDoubleComplex, N-1   );
    TESTING_HOSTALLOC( D,    cuDoubleComplex, N );
    TESTING_HOSTALLOC( E,    cuDoubleComplex, N );

    printf("\n\n");
    printf("  N    GPU GFlop/s   \n");
    printf("=====================\n");
    for(i=0; i<10; i++){
        if ( !once ) {
            N = size[i];
        }
        lda  = N;
        n2   = N*lda;
        flops = FLOPS( (double)N ) / 1e6;

        /* ====================================================================
           Initialize the matrix
           =================================================================== */
        lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
        /* Make the matrix hermitian */
        {
            magma_int_t i, j;
            for(i=0; i<N; i++) {
                MAGMA_Z_SET2REAL( h_A[i*lda+i], ( MAGMA_Z_GET_X(h_A[i*lda+i]) ) );
                for(j=0; j<i; j++)
                    h_A[i*lda+j] = cuConj(h_A[j*lda+i]);
            }
        }
        lapackf77_zlacpy( MagmaUpperLowerStr, &N, &N, h_A, &lda, h_R, &lda );

        /* ====================================================================
           Performs operation using MAGMA
           =================================================================== */
        start = get_current_time();
        magma_zhebbd(uplo[0], N, h_R, lda, tau, h_work, lwork, &info);
        magma_dsbtrd(THREADS, uplo[0], N, nb, h_R, lda, D, E);

        end = get_current_time();
        if ( info < 0 )
            printf("Argument %d of magma_zhebbd had an illegal value\n", -info);

        gpu_perf = flops / GetTimerValue(start,end);
        gpu_time = GetTimerValue(start,end) / 1000.;

        /* =====================================================================
           Check the factorization
           =================================================================== */
        /*
        if ( checkres ) {
            FILE        *fp ;

            printf("Writing input matrix in matlab_i_mat.txt ...\n");
            fp = fopen ("matlab_i_mat.txt", "w") ;
            if( fp == NULL ){ printf("Couldn't open output file\n"); exit(1);}

            for(j=0; j<N; j++)
                for(k=0; k<N; k++)
                    {
                        #if defined(PRECISION_z) || defined(PRECISION_c)
                        fprintf(fp, "%5d %5d %11.8f %11.8f\n", k+1, j+1, 
                                h_A[k+j*lda].x, h_A[k+j*lda].y);
                        #else
                        fprintf(fp, "%5d %5d %11.8f\n", k+1, j+1, h_A[k+j*lda]);
                        #endif
                    }
            fclose( fp ) ;

          printf("Writing output matrix in matlab_o_mat.txt ...\n");
          fp = fopen ("matlab_o_mat.txt", "w") ;
          if( fp == NULL ){ printf("Couldn't open output file\n"); exit(1);}

          for(j=0; j<N; j++)
            for(k=0; k<N; k++)
              {
                #if defined(PRECISION_z) || defined(PRECISION_c)
                fprintf(fp, "%5d %5d %11.8f %11.8f\n", k+1, j+1,
                        h_R[k+j*lda].x, h_R[k+j*lda].y);
                #else
                fprintf(fp, "%5d %5d %11.8f\n", k+1, j+1, h_R[k+j*lda]);
                #endif
              } 
          fclose( fp ) ;

        }*/

        /* =====================================================================
           Print performance and error.
           =================================================================== */
        if ( checkres ) {
            printf("%5d   %6.2f  %6.2f seconds\n", N, gpu_perf, gpu_time );
            double nrmI=0.0, nrm1=0.0, nrm2=0.0;
            int    lwork2 = 256*N;
            double *work2     = (double *) malloc (lwork2*sizeof(double));
            double *D2          = (double *) malloc (N*sizeof(double));
            dsyev_( "N", "L", &N, h_A, &lda, D2, work2, &lwork2, &info );
            /* call eigensolver for our tridiag */
            dsterf_( &N, D, E, &info); 
            /* compare result */
            cmp_vals(N, D2, D, &nrmI, &nrm1, &nrm2);
            printf("===================================================================================================================\n");
            printf(" Hello here are the norm  Infinite (max)=%e  norm one (sum)=%e   norm2(sqrt)=%e \n",nrmI, nrm1, nrm2);
            printf("===================================================================================================================\n");


        } else {
            printf("%5d   %6.2f  %6.2f seconds\n", N, gpu_perf, gpu_time );
        }

        if ( once )
            break;
    }

    /* Memory clean up */
    TESTING_FREE( h_A );
    TESTING_FREE( tau );
    TESTING_HOSTFREE( h_R );
    TESTING_HOSTFREE( h_work );

    /* Shutdown */
    TESTING_CUDA_FINALIZE();
    return EXIT_SUCCESS;
}
