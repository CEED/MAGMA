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
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
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
#define FLOPS(n) ( 6. * FMULS_POTRF(n) + 2. * FADDS_POTRF(n) )
#else
#define FLOPS(n) (      FMULS_POTRF(n) +      FADDS_POTRF(n) )
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zpotrf
*/
int main( int argc, char** argv)
{
    TESTING_CUDA_INIT();

    magma_timestr_t       start, end;
    double           flops, gpu_perf, cpu_perf;
    cuDoubleComplex *h_A, *h_R;
    magma_int_t      N=0, n2, lda;
    magma_int_t      size[13] = {1024,2048,3072,4032,5184,6048,7200,8064,8928,10240,20000,30000,40000};
    magma_int_t      size_n = 13;

    magma_int_t  i, info, flag = 0;
    const char  *uplo     = MagmaLowerStr;
    cuDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magma_int_t  ione     = 1;
    magma_int_t  ISEED[4] = {0,0,0,1};
    double       work[1];  // not referenced for lange norm 'f'
    double       matnorm;

    if (argc != 1){
        for(i = 1; i<argc; i++){
            if (strcmp("-N", argv[i])==0) {
                flag = 1;
                size[0] = size[size_n-1] = atoi(argv[++i]);
             } if (strcmp("-UPLO", argv[i])==0)
              if (strcmp("U", argv[++i])) uplo = MagmaUpperStr;
        }
    }
    N = size[size_n-1];
    if (N<=0) {
      printf( "N=%d\n", (int) N );
      exit(1);
    } else {
        printf("\nUsage: \n");
        if (strcmp(MagmaLowerStr,uplo) )
            printf("  testing_zpotrf -UPLO L -N %d:%d\n\n", (int) size[0], (int) size[size_n-1]);
        else
            printf("  testing_zpotrf -UPLO U -N %d:%d\n\n", (int) size[0], (int) size[size_n-1]);
    }

    /* Allocate host memory for the matrix */
    n2 = N * N;
    TESTING_MALLOC(    h_A, cuDoubleComplex, n2);
    TESTING_HOSTALLOC( h_R, cuDoubleComplex, n2);

    printf("  N    CPU GFlop/s    GPU GFlop/s    ||R_magma - R_lapack||_F / ||R_lapack||_F\n");
    printf("========================================================\n");
    for(i=0; i<size_n; i++){
        N     = size[i];
        lda   = N;
        n2    = lda*N;
        flops = FLOPS( (double)N ) / 1000000;

        /* ====================================================================
           Initialize the matrix
           =================================================================== */
        lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
        /* Symmetrize and increase the diagonal */
        {
            magma_int_t i, j;
            for(i=0; i<N; i++) {
                MAGMA_Z_SET2REAL( h_A[i*lda+i], ( MAGMA_Z_REAL(h_A[i*lda+i]) + 1.*N ) );
                for(j=0; j<i; j++)
                    h_A[i*lda+j] = cuConj(h_A[j*lda+i]);
            }
        }
        lapackf77_zlacpy( MagmaUpperLowerStr, &N, &N, h_A, &lda, h_R, &lda );

        /* ====================================================================
           Performs operation using MAGMA
           =================================================================== */
        //magma_zpotrf(uplo[0], N, h_R, lda, &info);
        //lapackf77_zlacpy( MagmaUpperLowerStr, &N, &N, h_A, &lda, h_R, &lda );

        start = get_current_time();
        magma_zpotrf(uplo[0], N, h_R, lda, &info);
        end = get_current_time();
        if (info != 0)
            printf("Argument %d of magma_zpotrf had an illegal value.\n", (int) -info);

        gpu_perf = flops / GetTimerValue(start, end);

        /* =====================================================================
           Performs operation using LAPACK
           =================================================================== */
        start = get_current_time();
        lapackf77_zpotrf(uplo, &N, h_A, &lda, &info);
        end = get_current_time();
        if (info != 0)
            printf("Argument %d of lapack_zpotrf had an illegal value.\n", (int) -info);

        cpu_perf = flops / GetTimerValue(start, end);

        /* =====================================================================
           Check the result compared to LAPACK
           =================================================================== */
        matnorm = lapackf77_zlange("f", &N, &N, h_A, &N, work);
        blasf77_zaxpy(&n2, &c_neg_one, h_A, &ione, h_R, &ione);
        printf("%5d    %6.2f         %6.2f        %e\n",
               (int) size[i], cpu_perf, gpu_perf,
               lapackf77_zlange("f", &N, &N, h_R, &N, work) / matnorm );

        if (flag == 1)
            break;
    }

    /* Memory clean up */
    TESTING_FREE( h_A );
    TESTING_HOSTFREE( h_R );

    TESTING_CUDA_FINALIZE();
}
