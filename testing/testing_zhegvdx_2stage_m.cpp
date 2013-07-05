/*
    -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

    @author Raffaele Solca
    @author Azzam Haidar

    @precisions normal z -> c d s

*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <cublas.h>

// includes, project
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"
#include "magma_zbulge.h"
#include "magma_threadsetting.h"

#define PRECISION_z

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zhegvdx
*/
int main( int argc, char** argv)
{

    TESTING_INIT_MGPU();

    real_Double_t   mgpu_time;
    magmaDoubleComplex *h_A, *h_Ainit, *h_B, *h_Binit, *h_work;

#if defined(PRECISION_z) || defined(PRECISION_c)
    double *rwork;
    magma_int_t lrwork;
#endif

    double *w1, result;
    magma_int_t *iwork;
    magma_int_t N, n2, info, lwork, liwork;
    magmaDoubleComplex c_zero    = MAGMA_Z_ZERO;
    magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};

    magma_timestr_t start, end;

    magma_opts opts;
    parse_opts( argc, argv, &opts );

    char jobz = opts.jobz;
    int checkres = opts.check;

    char range = 'A';
    char uplo = opts.uplo;
    magma_int_t itype = opts.itype;

    double f = opts.fraction;

    if (f != 1)
        range='I';

    if ( checkres && jobz == MagmaNoVec ) {
        fprintf( stderr, "checking results requires vectors; setting jobz=V (option -JV)\n" );
        jobz = MagmaVec;
    }

    printf("using: nrgpu = %d, itype = %d, jobz = %c, range = %c, uplo = %c, checkres = %d, fraction = %6.4f\n",
           (int) opts.ngpu, (int) itype, jobz, range, uplo, (int) checkres, f);
    
    printf("  N     M   nr GPU     MGPU Time(s) \n");
    printf("====================================\n");
    magma_int_t threads = magma_get_numthreads();
    for( int i = 0; i < opts.ntest; ++i ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[i];
            n2     = N*N;
#if defined(PRECISION_z) || defined(PRECISION_c)
            lwork  = magma_zbulge_get_lq2(N, threads) + 2*N + N*N;
            lrwork = 1 + 5*N +2*N*N;
#else
            lwork  = magma_zbulge_get_lq2(N, threads) + 1 + 6*N + 2*N*N;
#endif
            liwork = 3 + 5*N;


            //magma_int_t NB = 96;//magma_bulge_get_nb(N);
            //magma_int_t sizvblg = magma_zbulge_get_lq2(N, threads);        
            //magma_int_t siz = max(sizvblg,n2)+2*(N*NB+N)+24*N; 
            /* Allocate host memory for the matrix */
            TESTING_HOSTALLOC(   h_A, magmaDoubleComplex, n2);
            TESTING_HOSTALLOC(   h_B, magmaDoubleComplex, n2);
            TESTING_MALLOC(    w1, double         ,  N);
            TESTING_HOSTALLOC(h_work, magmaDoubleComplex,  lwork);
#if defined(PRECISION_z) || defined(PRECISION_c)
            TESTING_HOSTALLOC( rwork,          double, lrwork);
#endif
            TESTING_MALLOC(    iwork,     magma_int_t, liwork);

            /* Initialize the matrix */
            lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
            lapackf77_zlarnv( &ione, ISEED, &n2, h_B );
            /* increase the diagonal */
            {
                for(int i=0; i<N; i++) {
                    MAGMA_Z_SET2REAL( h_B[i*N+i], ( MAGMA_Z_REAL(h_B[i*N+i]) + 1.*N ) );
                    MAGMA_Z_SET2REAL( h_A[i*N+i], MAGMA_Z_REAL(h_A[i*N+i]) );
                }
            }

            if((opts.warmup)||( checkres )){
                TESTING_MALLOC(h_Ainit, magmaDoubleComplex, n2);
                TESTING_MALLOC(h_Binit, magmaDoubleComplex, n2);
                lapackf77_zlacpy( MagmaUpperLowerStr, &N, &N, h_A, &N, h_Ainit, &N );
                lapackf77_zlacpy( MagmaUpperLowerStr, &N, &N, h_B, &N, h_Binit, &N );
            }



            magma_int_t m1 = 0;
            double vl = 0;
            double vu = 0;
            magma_int_t il = 0;
            magma_int_t iu = 0;

            if (range == 'I'){
                il = 1;
                iu = (int) (f*N);
            }

            if(opts.warmup){

                // ==================================================================
                // Warmup using MAGMA. I prefer to use smalltest to warmup A-
                // ==================================================================
                magma_zhegvdx_2stage_m(opts.ngpu, itype, jobz, range, uplo,
                                       N, h_A, N, h_B, N, vl, vu, il, iu, &m1, w1,
                                       h_work, lwork,
#if defined(PRECISION_z) || defined(PRECISION_c)
                                       rwork, lrwork,
#endif
                                       iwork, liwork,
                                       &info);
                lapackf77_zlacpy( MagmaUpperLowerStr, &N, &N, h_Ainit, &N, h_A, &N );
                lapackf77_zlacpy( MagmaUpperLowerStr, &N, &N, h_Binit, &N, h_B, &N );
            }

            // ===================================================================
            // Performs operation using MAGMA
            // ===================================================================

            start = get_current_time();
            magma_zhegvdx_2stage_m(opts.ngpu, itype, jobz, range, uplo,
                                   N, h_A, N, h_B, N, vl, vu, il, iu, &m1, w1,
                                   h_work, lwork,
#if defined(PRECISION_z) || defined(PRECISION_c)
                                   rwork, lrwork,
#endif
                                   iwork, liwork,
                                   &info);
            end = get_current_time();

            mgpu_time = GetTimerValue(start,end)/1000.;

            if ( checkres ) {
                // ===================================================================
                // Check the results following the LAPACK's [zc]hegvdx routine.
                // A x = lambda B x is solved
                // and the following 3 tests computed:
                // (1)    | A Z - B Z D | / ( |A||Z| N )  (itype = 1)
                // | A B Z - Z D | / ( |A||Z| N )  (itype = 2)
                // | B A Z - Z D | / ( |A||Z| N )  (itype = 3)
                // ===================================================================
#if defined(PRECISION_d) || defined(PRECISION_s)
                double *rwork = h_work + N*N;
#endif
                result = 1.;
                result /= lapackf77_zlanhe("1",&uplo, &N, h_Ainit, &N, rwork);
                result /= lapackf77_zlange("1",&N , &m1, h_A, &N, rwork);

                if (itype == 1){
                    blasf77_zhemm("L", &uplo, &N, &m1, &c_one, h_Ainit, &N, h_A, &N, &c_zero, h_work, &N);
                    for(int i=0; i<m1; ++i)
                        blasf77_zdscal(&N, &w1[i], &h_A[i*N], &ione);
                    blasf77_zhemm("L", &uplo, &N, &m1, &c_neg_one, h_Binit, &N, h_A, &N, &c_one, h_work, &N);
                    result *= lapackf77_zlange("1", &N, &m1, h_work, &N, rwork)/N;
                }
                else if (itype == 2){
                    blasf77_zhemm("L", &uplo, &N, &m1, &c_one, h_Binit, &N, h_A, &N, &c_zero, h_work, &N);
                    for(int i=0; i<m1; ++i)
                        blasf77_zdscal(&N, &w1[i], &h_A[i*N], &ione);
                    blasf77_zhemm("L", &uplo, &N, &m1, &c_one, h_Ainit, &N, h_work, &N, &c_neg_one, h_A, &N);
                    result *= lapackf77_zlange("1", &N, &m1, h_A, &N, rwork)/N;
                }
                else if (itype == 3){
                    blasf77_zhemm("L", &uplo, &N, &m1, &c_one, h_Ainit, &N, h_A, &N, &c_zero, h_work, &N);
                    for(int i=0; i<m1; ++i)
                        blasf77_zdscal(&N, &w1[i], &h_A[i*N], &ione);
                    blasf77_zhemm("L", &uplo, &N, &m1, &c_one, h_Binit, &N, h_work, &N, &c_neg_one, h_A, &N);
                    result *= lapackf77_zlange("1", &N, &m1, h_A, &N, rwork)/N;
                }
            }

            // ===================================================================
            // Print execution time
            // ===================================================================
            printf("%5d %5d %2d    %6.2f\n",
                   (int) N, (int) m1, (int) opts.ngpu, mgpu_time);
            if ( checkres ){
                printf("Testing the eigenvalues and eigenvectors for correctness:\n");
                if(itype==1)
                    printf("(1)    | A Z - B Z D | / (|A| |Z| N) = %e\n", result);
                else if(itype==2)
                    printf("(1)    | A B Z - Z D | / (|A| |Z| N) = %e\n", result);
                else if(itype==3)
                    printf("(1)    | B A Z - Z D | / (|A| |Z| N) = %e\n", result);
            }

            TESTING_HOSTFREE(       h_A);
            TESTING_HOSTFREE(       h_B);
            TESTING_FREE(        w1);
#if defined(PRECISION_z) || defined(PRECISION_c)
            TESTING_HOSTFREE( rwork);
#endif
            TESTING_FREE(     iwork);
            TESTING_HOSTFREE(h_work);
            if((opts.warmup)||( checkres )){
                TESTING_FREE(   h_Ainit);
                TESTING_FREE(   h_Binit);
            }
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    /* Shutdown */
    TESTING_FINALIZE_MGPU();

    return 0;
}
