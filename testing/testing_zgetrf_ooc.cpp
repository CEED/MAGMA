/*
 *  -- MAGMA (version 1.0) --
 *     Univ. of Tennessee, Knoxville
 *     Univ. of California, Berkeley
 *     Univ. of Colorado, Denver
 *     November 2010
 *
 * @precisions normal z -> s d c
 *
 **/
/* includes, system */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas.h>

/* includes, project */
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

/* Flops formula */
#define PRECISION_z
#if defined(PRECISION_z) || defined(PRECISION_c)
#define FLOPS(m, n) ( 6. * FMULS_GETRF(m, n) + 2. * FADDS_GETRF(m, n) )
#else
#define FLOPS(m, n) (      FMULS_GETRF(m, n) +      FADDS_GETRF(m, n) )
#endif


extern "C" magma_int_t
magma_zgetrf_ooc(magma_int_t m, magma_int_t n, cuDoubleComplex *a, magma_int_t lda,
		 magma_int_t *ipiv, magma_int_t *info);

extern "C" magma_int_t
magma_zgetrf_piv(magma_int_t m, magma_int_t n, cuDoubleComplex *a, magma_int_t lda,
		         magma_int_t *ipiv, magma_int_t *info);

double get_LU_error(magma_int_t M, magma_int_t N, 
		    cuDoubleComplex *A,  magma_int_t lda, 
		    cuDoubleComplex *LU, magma_int_t *IPIV)
{
    magma_int_t min_mn = min(M,N);
    magma_int_t ione   = 1;
    magma_int_t i, j;
    cuDoubleComplex alpha = MAGMA_Z_ONE;
    cuDoubleComplex beta  = MAGMA_Z_ZERO;
    cuDoubleComplex *L, *U;
    double work[1], matnorm, residual;
                       
    TESTING_MALLOC( L, cuDoubleComplex, M*min_mn);
    TESTING_MALLOC( U, cuDoubleComplex, min_mn*N);
    memset( L, 0, M*min_mn*sizeof(cuDoubleComplex) );
    memset( U, 0, min_mn*N*sizeof(cuDoubleComplex) );

    lapackf77_zlaswp( &N, A, &lda, &ione, &min_mn, IPIV, &ione);
    lapackf77_zlacpy( MagmaLowerStr, &M, &min_mn, LU, &lda, L, &M      );
    lapackf77_zlacpy( MagmaUpperStr, &min_mn, &N, LU, &lda, U, &min_mn );

    for(j=0; j<min_mn; j++)
        L[j+j*M] = MAGMA_Z_MAKE( 1., 0. );
    
    matnorm = lapackf77_zlange("f", &M, &N, A, &lda, work);

    blasf77_zgemm("N", "N", &M, &N, &min_mn,
                  &alpha, L, &M, U, &min_mn, &beta, LU, &lda);

    for( j = 0; j < N; j++ ) {
        for( i = 0; i < M; i++ ) {
            LU[i+j*lda] = MAGMA_Z_SUB( LU[i+j*lda], A[i+j*lda] );
	}
    }
    residual = lapackf77_zlange("f", &M, &N, LU, &lda, work);

    TESTING_FREE(L);
    TESTING_FREE(U);

    return residual / (matnorm * N);
}


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgetrf_ooc
*/
int main( int argc, char** argv)
{
    TESTING_CUDA_INIT();

    magma_timestr_t	start, end, start2, end2;
    double			flops, gpu_perf, gpu_perf2, cpu_perf, error;
    cuDoubleComplex	*h_A, *h_R;
    magma_int_t		*ipiv;
    magma_int_t		i, info, min_mn;
    magma_int_t		ione     = 1;
    magma_int_t		ISEED[4] = {0,0,0,1};

    /* Matrix size */
    magma_int_t M = 0, N = 0, n2, lda, ldda;
    magma_int_t size[12] = {1024,2048,3072,4032,5184,6016,7040,8064,9088,10112,20224,30000};
	magma_int_t size_n = 12;


    if (argc != 1){
        for(i = 1; i<argc; i++){
            if (strcmp("-N", argv[i])==0)
                N = atoi(argv[++i]);
            else if (strcmp("-M", argv[i])==0)
                M = atoi(argv[++i]);
        }
        if (M>0 && N>0)
            printf("  testing_zgetrf_ooc -M %d -N %d\n\n", M, N);
        else
            {
                printf("\nUsage: \n");
                printf("  testing_zgetrf_ooc -M %d -N %d\n\n", 1024, 1024);
                exit(1);
            }
    } else {
        M = N = size[size_n-1];
        printf("  default: M=%d N=%d\n", M, N);
    }
    if( N <= 0 || M <= 0 ) {
		printf( " exiting because M=%d and N=%d\n",M,N );
		exit(1);
	}
    ldda   = ((M+31)/32)*32;
    n2     = M * N;
    min_mn = min(M, N);
    if( n2 < min_mn ) {
		printf( " exiting because n2=%dx%d overflow (n2=%d)\n",M,N,n2 );
		exit(1);
	}

    /* Allocate host memory for the matrix */
    TESTING_MALLOC(ipiv, magma_int_t, min_mn);
    TESTING_MALLOC(    h_A, cuDoubleComplex, n2     );
    TESTING_HOSTALLOC( h_R, cuDoubleComplex, n2     );

    printf("\n\n");
    printf("  M     N   CPU GFlop/s    GPU GFlop/s   ||PA-LU||/(||A||*N)    Time (sec.) \n");
    printf("============================================================================\n");
    for(i=0; i<size_n; i++){
        if (argc == 1){
	      M = N = size[i];
        }
        min_mn= min(M, N);
        lda   = M;
        n2    = lda*N;
        ldda  = ((M+31)/32)*32;
        flops = FLOPS( (double)M, (double)N ) / 1000000;

        /* Initialize the matrix:             *
		 * generated into h_A and copy to h_R */
        lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
        lapackf77_zlacpy( MagmaUpperLowerStr, &M, &N, h_A, &lda, h_R, &lda );
#ifdef  CHECK_TESTING_ZGETRF_OOC
        printf( "   1) matrix is generated (M=%d,N=%d,lda=%d)\n",M,N,lda );
		fflush(stdout);
#endif

        /* =====================================================================
           Performs operation using LAPACK
           =================================================================== */
#ifndef  SKIP_LAPACK
		/* h_A (instead of h_R) is used since no CUDA */
        start = get_current_time();
        lapackf77_zgetrf(&M, &N, h_A, &lda, ipiv, &info);
        end = get_current_time();
        cpu_perf = flops / GetTimerValue(start, end);

		/* restoring h_A to be the orginal form */
        lapackf77_zlacpy( MagmaUpperLowerStr, &M, &N, h_R, &lda, h_A, &lda );

#ifdef CHECK_TESTING_ZGETRF_OOC
		printf( "   2) scalapack is done\n" );
#endif
        if (info < 0) {
            printf("Argument %d of zgetrf had an illegal value.\n", -info);
            break;
		}
#elif defined(CHECK_TESTING_ZGETRF_OOC)
		printf( "   2) scalapack is skiped\n" );
#endif
		fflush(stdout);

        /* ====================================================================
           Performs operation using MAGMA
           =================================================================== */
        start = get_current_time();
        magma_zgetrf_ooc( M, N, h_R, lda, ipiv, &info);
        end = get_current_time();
	    gpu_perf = flops / GetTimerValue(start, end);

#ifdef CHECK_TESTING_ZGETRF_OOC
		printf( "   3) magma is done with %d (%e sec)\n",info,GetTimerValue(start,end)/1000 );
#endif
        if (info < 0) {
            printf("Argument %d of zgetrf_ooc had an illegal value.\n", -info);
            break;
		}
		fflush(stdout);

		/* appling pivots to previous big-panels */
        start2 = get_current_time();
        magma_zgetrf_piv( M, N, h_R, lda, ipiv, &info);
        end2 = get_current_time();
	    gpu_perf2 = flops / (GetTimerValue(start2, end2) + GetTimerValue(start, end));
#ifdef CHECK_TESTING_ZGETRF_OOC
		printf( "   3) pivoting-backward is done (%e sec)\n",GetTimerValue(start2,end2)/1000 );
#endif


        /* =====================================================================
           Check the factorization
           =================================================================== */
        error = get_LU_error(M, N, h_A, lda, h_R, ipiv);
#ifndef  SKIP_LAPACK
        printf("%5d %5d  %6.2f         %6.2f         %e         %e\n",
               M, N, cpu_perf, gpu_perf, error, GetTimerValue(start,end)/1000);
        printf("   with pivoting            %6.2f                             +%e\n",
                               gpu_perf2,       GetTimerValue(start2,end2)/1000);
#else
        printf("%5d %5d   *****         %6.2f         %e         %e\n",
               M, N,           gpu_perf, error, GetTimerValue(start,end)/1000);
        printf("   with pivoting            %6.2f                             +%e\n",
                               gpu_perf2,       GetTimerValue(start2,end2)/1000);
#endif
		fflush(stdout);

        if (argc != 1)
            break;
    }

    /* Memory clean up */
    TESTING_FREE( ipiv );
    TESTING_FREE( h_A );
    TESTING_HOSTFREE( h_R );

    /* Shutdown */
    TESTING_CUDA_FINALIZE();
}
