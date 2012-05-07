/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

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

#define lapackf77_zgeqp3   FORTRAN_NAME( zgeqp3, ZGEQP3)
extern "C" void    magma_zgeqp3(magma_int_t *m, magma_int_t *n,
                         cuDoubleComplex *a, magma_int_t *lda, magma_int_t *jpvt, cuDoubleComplex *tau,
                         cuDoubleComplex *work, magma_int_t *lwork, double *rwork, magma_int_t *info);
extern "C" void    lapackf77_zgeqp3(magma_int_t *m, magma_int_t *n,
                         cuDoubleComplex *a, magma_int_t *lda, magma_int_t *jpvt, cuDoubleComplex *tau,
                         cuDoubleComplex *work, magma_int_t *lwork, double *rwork, magma_int_t *info);

// Flops formula
#define PRECISION_z
#if defined(PRECISION_z) || defined(PRECISION_c)
#define FLOPS(m, n) ( 6.*FMULS_GEQRF(m, n) + 2.*FADDS_GEQRF(m, n) )
#else
#define FLOPS(m, n) (    FMULS_GEQRF(m, n) +    FADDS_GEQRF(m, n) )
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgeqrf
*/
int main( int argc, char** argv) 
{
    TESTING_CUDA_INIT();

    magma_timestr_t       start, end;
    double           flops, gpu_perf, cpu_perf;
    double           matnorm, work[1];
    cuDoubleComplex  c_neg_one = MAGMA_Z_NEG_ONE;
    cuDoubleComplex *h_A, *h_R, *tau, *h_work, tmp[1];

    double *rwork;
    magma_int_t *jpvt;

    /* Matrix size */
    magma_int_t M = 0, N = 0, n2, lda, lwork;
    magma_int_t size[10] = {1024,2048,3072,4032,5184,6016,7040,8064,9088,9984};

    magma_int_t i, j, info, min_mn, nb;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};

    if (argc != 1){
        for(i = 1; i<argc; i++){
            if (strcmp("-N", argv[i])==0)
                N = atoi(argv[++i]);
            else if (strcmp("-M", argv[i])==0)
                M = atoi(argv[++i]);
        }
        if ( M == 0 ) {
            M = N;
        }
        if ( N == 0 ) {
            N = M;
        }
        if (N>0 && M>0)
            printf("  testing_zgeqrf -M %d -N %d\n\n", M, N);
        else
            {
                printf("\nUsage: \n");
                printf("  testing_zgeqrf -M %d -N %d\n\n", M, N);
                exit(1);
            }
    }
    else {
        printf("\nUsage: \n");
        printf("  testing_zgeqrf -M %d -N %d\n\n", 1024, 1024);
        M = N = size[9];
    }

    n2  = M * N;
    min_mn = min(M, N);
    nb = magma_get_zgeqrf_nb(M);

    TESTING_MALLOC(    jpvt, magma_int_t,     N );

    TESTING_MALLOC(    tau,  cuDoubleComplex, min_mn);
    TESTING_MALLOC(    rwork,double, 2*N);
    TESTING_MALLOC(    h_A,  cuDoubleComplex, n2 );
    //TESTING_HOSTALLOC( h_R,  cuDoubleComplex, n2 );
    TESTING_MALLOC( h_R,  cuDoubleComplex, n2   );

    lwork = -1;
    //lapackf77_zgeqp3(&M, &N, h_A, &M, jpvt, tau, tmp, &lwork, &info);
    //lwork = (magma_int_t)MAGMA_Z_REAL( tmp[0] );
    //lwork = max( lwork, N*nb);

    lwork = 2*N+( N+1 )*nb;

    TESTING_MALLOC( h_work, cuDoubleComplex, lwork );

    printf("\n\n");
    printf("  M     N   CPU GFlop/s   GPU GFlop/s    ||R||_F / ||A||_F\n");
    printf("==========================================================\n");
    for(i=0; i<10; i++){
         if (argc == 1){
            M = N = size[i];
        }
        min_mn= min(M, N);
        lda   = M;
        n2    = lda*N;
        flops = FLOPS( (double)M, (double)N ) / 1000000;

        /* Initialize the matrix */
        lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
        lapackf77_zlacpy( MagmaUpperLowerStr, &M, &N, h_A, &lda, h_R, &lda );

        /* ====================================================================
           Performs operation using MAGMA
           =================================================================== */

        for (j = 0; j < N; j++) {
            jpvt[j] = 0;
        }

        start = get_current_time();
        //magma_zgeqrf(M, N, h_R, lda, tau, h_work, lwork, &info);
        magma_zgeqp3(&M, &N, h_R, &lda, jpvt, tau, h_work, &lwork, rwork, &info);
        end = get_current_time();
        if (info < 0)
            printf("Argument %d of magma_zgeqrf had an illegal value.\n", -info);
        
        gpu_perf = flops / GetTimerValue(start, end);

        /* =====================================================================
           Performs operation using LAPACK
           =================================================================== */

        for (j = 0; j < N; j++) {
            jpvt[j] = 0;
        }

        start = get_current_time();
        //lapackf77_zgeqrf(&M, &N, h_A, &lda, tau, h_work, &lwork, &info);
        //lapackf77_zgeqp3(&M, &N, h_A, &lda, jpvt, tau, h_work, &lwork, &info);
        lapackf77_zgeqp3(&M, &N, h_A, &lda, jpvt, tau, h_work, &lwork, rwork, &info);
        end = get_current_time();
        if (info < 0)
            printf("Argument %d of lapack_zgeqrf had an illegal value.\n", -info);

        cpu_perf = flops / GetTimerValue(start, end);

        /* =====================================================================
           Check the result compared to LAPACK
           =================================================================== */
        matnorm = lapackf77_zlange("f", &M, &N, h_A, &lda, work);
        blasf77_zaxpy(&n2, &c_neg_one, h_A, &ione, h_R, &ione);

        printf("%5d %5d  %6.2f         %6.2f        %e\n",
               M, N, cpu_perf, gpu_perf,
               lapackf77_zlange("f", &M, &N, h_R, &lda, work) / matnorm);

        if (argc != 1)
            break;
    }

    /* Memory clean up */
    TESTING_FREE( jpvt );
    TESTING_FREE( tau );
    TESTING_FREE( rwork );
    TESTING_FREE( h_A );
    //TESTING_HOSTFREE( h_R );
    TESTING_FREE( h_R );
    TESTING_FREE( h_work );

    /* Shutdown */
    TESTING_CUDA_FINALIZE();
}
