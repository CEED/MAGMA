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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas.h>

#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

#define PRECISION_z
// Flops formula
#if defined(PRECISION_z) || defined(PRECISION_c)
#define FLOPS_GETRF(m, n   ) ( 6.*FMULS_GETRF(m, n   ) + 2.*FADDS_GETRF(m, n   ) )
#define FLOPS_GETRS(m, nrhs) ( 6.*FMULS_GETRS(m, nrhs) + 2.*FADDS_GETRS(m, nrhs) )
#else
#define FLOPS_GETRF(m, n   ) (    FMULS_GETRF(m, n   ) +    FADDS_GETRF(m, n   ) )
#define FLOPS_GETRS(m, nrhs) (    FMULS_GETRS(m, nrhs) +    FADDS_GETRS(m, nrhs) )
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgesv
*/
int main(int argc , char **argv)
{
    TESTING_CUDA_INIT();

    magma_timestr_t  start, end;
    double      flops, gpu_perf;
    double      Rnorm, Anorm, Bnorm, *work;
    cuDoubleComplex zone  = MAGMA_Z_ONE;
    cuDoubleComplex mzone = MAGMA_Z_NEG_ONE;
    cuDoubleComplex *h_A, *h_LU, *h_B, *h_X;
    magma_int_t *ipiv;
    magma_int_t lda, ldb;
    magma_int_t i, info, szeA, szeB;
    magma_int_t N        = 0;
    magma_int_t ione     = 1;
    magma_int_t NRHS     = 100;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t size[10] = {1024,2048,3072,4032,5184,6016,7040,8064,9088,10112};
        
    if (argc != 1){
        for(i = 1; i<argc; i++){
            if (strcmp("-N", argv[i])==0)
                N = atoi(argv[++i]);
            else if (strcmp("-nrhs", argv[i])==0)
                NRHS = atoi(argv[++i]);
        }
        if ( N > 0 ) 
            size[0] = size[9] = N;
    }
    else {
        printf("\nUsage: \n");
        printf("  testing_zgesv -nrhs %d -N %d\n\n", NRHS, 1024);
    }

    N = size[9];
    ldb = lda = N ;
    
    TESTING_MALLOC( h_A,  cuDoubleComplex, lda*N    );
    TESTING_MALLOC( h_LU, cuDoubleComplex, lda*N    );
    TESTING_MALLOC( h_B,  cuDoubleComplex, ldb*NRHS );
    TESTING_MALLOC( h_X,  cuDoubleComplex, ldb*NRHS );
    TESTING_MALLOC( work, double,          N        );
    TESTING_MALLOC( ipiv, magma_int_t,     N        );

    printf("\n\n");
    printf("  N     NRHS       GPU GFlop/s      || b-Ax || / ||A||\n");
    printf("========================================================\n");

    for(i=0; i<10; i++){
        N   = size[i];
        lda = ldb = N;
        flops = ( FLOPS_GETRF( (double)N, (double)N ) +
                  FLOPS_GETRS( (double)N, (double)NRHS ) ) / 1e6;

        /* Initialize the matrices */
        szeA = lda*N;
        szeB = ldb*NRHS;
        lapackf77_zlarnv( &ione, ISEED, &szeA, h_A );
        lapackf77_zlarnv( &ione, ISEED, &szeB, h_B );
        
        // copy A to LU and B to X; save A and B for residual
        lapackf77_zlacpy( "F", &N, &N,    h_A, &lda, h_LU, &lda );
        lapackf77_zlacpy( "F", &N, &NRHS, h_B, &ldb, h_X,  &ldb );

        //=====================================================================
        // Solve Ax = b through an LU factorization
        //=====================================================================
        start = get_current_time();
        magma_zgesv( N, NRHS, h_LU, lda, ipiv, h_X, ldb, &info );
        end = get_current_time();
        if (info < 0)
            printf("Argument %d of magma_zgesv had an illegal value.\n", -info);

        gpu_perf = flops / GetTimerValue(start, end);

        //=====================================================================
        // ERROR
        //=====================================================================

        Anorm = lapackf77_zlange("I", &N, &N,    h_A, &lda, work);
        Bnorm = lapackf77_zlange("I", &N, &NRHS, h_B, &ldb, work);

        blasf77_zgemm( MagmaNoTransStr, MagmaNoTransStr, &N, &NRHS, &N, 
                       &zone,  h_A, &lda, 
                               h_X, &ldb, 
                       &mzone, h_B, &ldb);
        
        Rnorm = lapackf77_zlange("I", &N, &NRHS, h_B, &ldb, work);

        printf("%5d  %4d             %6.2f        %e\n",
               N, NRHS, gpu_perf, Rnorm/(Anorm*Bnorm) );

        if (argc != 1)
          break;
    }

    /* Memory clean up */
    TESTING_FREE( h_A  );
    TESTING_FREE( h_LU );
    TESTING_FREE( h_B  );
    TESTING_FREE( h_X  );
    TESTING_FREE( work );
    TESTING_FREE( ipiv );

    /* Shutdown */
    TESTING_CUDA_FINALIZE();
}
