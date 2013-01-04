/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    @precisions normal d -> s

    @author Stan Tomov
    @author Raffaele Solca

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

#define absv(v1) ((v1)>0? (v1): -(v1))

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing dsygvd
*/
int main( int argc, char** argv)
{
    TESTING_CUDA_INIT();

    real_Double_t   gpu_time, cpu_time;
    double *h_A, *h_R, *h_B, *h_S, *h_work;
    double *w1, *w2;
    magma_int_t *iwork;
    magma_int_t N, n2, nb, lwork, liwork, info;
    double result[4];

    double c_one     = MAGMA_D_ONE;
    double c_neg_one = MAGMA_D_NEG_ONE;

    double d_zero        =  0.;
    double d_one         =  1.;
    double d_neg_one     = -1.;
    //magma_int_t izero    = 0;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};

    magma_opts opts;
    parse_opts( argc, argv, &opts );
    
    if ( opts.check && opts.jobz == MagmaNoVectors ) {
        fprintf( stderr, "checking results requires vectors; setting jobz=V (option -JV)\n" );
        opts.jobz = MagmaVectors;
    }
    
    printf("  N     CPU Time(s)    GPU Time(s) \n");
    printf("===================================\n");
    for( int i = 0; i < opts.ntest; ++i ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[i];
            n2     = N*N;
            nb     = magma_get_dsytrd_nb(N);
            lwork  = 1 + 6*N*nb + 2* N*N;
            liwork = 3 + 5*N;

            TESTING_MALLOC(   h_A, double, n2);
            TESTING_MALLOC(   h_B, double, n2);
            TESTING_MALLOC(    w1, double,  N);
            TESTING_MALLOC(    w2, double,  N);
            TESTING_HOSTALLOC(h_R, double, n2);
            TESTING_HOSTALLOC(h_S, double, n2);
            TESTING_HOSTALLOC(h_work, double,  lwork);
            TESTING_MALLOC(    iwork, magma_int_t, liwork);
            
            /* Initialize the matrix */
            lapackf77_dlarnv( &ione, ISEED, &n2, h_A );
            lapackf77_dlarnv( &ione, ISEED, &n2, h_B );
            magma_dhpd( N, h_B, N );
            lapackf77_dlacpy( MagmaUpperLowerStr, &N, &N, h_A, &N, h_R, &N );
            lapackf77_dlacpy( MagmaUpperLowerStr, &N, &N, h_B, &N, h_S, &N );
            
            /* warmup */
            magma_dsygvd( opts.itype, opts.jobz, opts.uplo,
                          N, h_R, N, h_S, N, w1,
                          h_work, lwork,
                          iwork, liwork,
                          &info );
            if (info != 0)
                printf("magma_dsygvd returned error %d.\n", (int) info);
            
            lapackf77_dlacpy( MagmaUpperLowerStr, &N, &N, h_A, &N, h_R, &N );
            lapackf77_dlacpy( MagmaUpperLowerStr, &N, &N, h_B, &N, h_S, &N );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            magma_dsygvd( opts.itype, opts.jobz, opts.uplo,
                          N, h_R, N, h_S, N, w1,
                          h_work, lwork,
                          iwork, liwork,
                          &info );
            gpu_time = magma_wtime() - gpu_time;
            if (info != 0)
                printf("magma_dsygvd returned error %d.\n", (int) info);
            
            if ( opts.check ) {
                /* =====================================================================
                   Check the results following the LAPACK's [zc]hegvd routine.
                   A x = lambda B x is solved
                   and the following 3 tests computed:
                   (1)    | A Z - B Z D | / ( |A||Z| N )   (itype = 1)
                          | A B Z - Z D | / ( |A||Z| N )   (itype = 2)
                          | B A Z - Z D | / ( |A||Z| N )   (itype = 3)
                   (2)    | I - V V' B | / ( N )           (itype = 1,2)
                          | B - V V' | / ( |B| N )         (itype = 3)
                   (3)    | S(with V) - S(w/o V) | / | S |
                   =================================================================== */
                double temp1, temp2;
                //double *tau;
                
                if ( opts.itype == 1 || opts.itype == 2 ) {
                    lapackf77_dlaset( "A", &N, &N, &d_zero, &c_one, h_S, &N);
                    blasf77_dgemm("N", "C", &N, &N, &N, &c_one, h_R, &N, h_R, &N, &d_zero, h_work, &N);
                    blasf77_dsymm("R", &opts.uplo, &N, &N, &c_neg_one, h_B, &N, h_work, &N, &c_one, h_S, &N);
                    result[1]= lapackf77_dlange("1", &N, &N, h_S, &N, h_work) / N;
                }
                else if ( opts.itype == 3 ) {
                    lapackf77_dlacpy( MagmaUpperLowerStr, &N, &N, h_B, &N, h_S, &N);
                    blasf77_dsyrk(&opts.uplo, "N", &N, &N, &d_neg_one, h_R, &N, &d_one, h_S, &N);
                    result[1]= lapackf77_dlansy("1", &opts.uplo, &N, h_S, &N, h_work) / N /
                    lapackf77_dlansy("1", &opts.uplo, &N, h_B, &N, h_work);
                }
                
                result[0] = 1.;
                result[0] /= lapackf77_dlansy("1", &opts.uplo, &N, h_A, &N, h_work);
                result[0] /= lapackf77_dlange("1", &N , &N, h_R, &N, h_work);
                
                if ( opts.itype == 1 ) {
                    blasf77_dsymm("L", &opts.uplo, &N, &N, &c_one, h_A, &N, h_R, &N, &d_zero, h_work, &N);
                    for(int i=0; i<N; ++i)
                        blasf77_ddscal(&N, &w1[i], &h_R[i*N], &ione);
                    blasf77_dsymm("L", &opts.uplo, &N, &N, &c_neg_one, h_B, &N, h_R, &N, &c_one, h_work, &N);
                    result[0] *= lapackf77_dlange("1", &N, &N, h_work, &N, &temp1)/N;
                }
                else if ( opts.itype == 2 ) {
                    blasf77_dsymm("L", &opts.uplo, &N, &N, &c_one, h_B, &N, h_R, &N, &d_zero, h_work, &N);
                    for(int i=0; i<N; ++i)
                        blasf77_ddscal(&N, &w1[i], &h_R[i*N], &ione);
                    blasf77_dsymm("L", &opts.uplo, &N, &N, &c_one, h_A, &N, h_work, &N, &c_neg_one, h_R, &N);
                    result[0] *= lapackf77_dlange("1", &N, &N, h_R, &N, &temp1)/N;
                }
                else if ( opts.itype == 3 ) {
                    blasf77_dsymm("L", &opts.uplo, &N, &N, &c_one, h_A, &N, h_R, &N, &d_zero, h_work, &N);
                    for(int i=0; i<N; ++i)
                        blasf77_ddscal(&N, &w1[i], &h_R[i*N], &ione);
                    blasf77_dsymm("L", &opts.uplo, &N, &N, &c_one, h_B, &N, h_work, &N, &c_neg_one, h_R, &N);
                    result[0] *= lapackf77_dlange("1", &N, &N, h_R, &N, &temp1)/N;
                }
                
                /*
                lapackf77_dsyt21(&ione, &opts.uplo, &N, &izero,
                                 h_A, &N,
                                 w1, w1,
                                 h_R, &N,
                                 h_R, &N,
                                 tau, h_work, rwork, &result[0]);
                */
                
                lapackf77_dlacpy( MagmaUpperLowerStr, &N, &N, h_A, &N, h_R, &N );
                lapackf77_dlacpy( MagmaUpperLowerStr, &N, &N, h_B, &N, h_S, &N );
                
                magma_dsygvd( opts.itype, MagmaNoVectors, opts.uplo,
                              N, h_R, N, h_S, N, w2,
                              h_work, lwork,
                              iwork, liwork,
                              &info );
                if (info != 0)
                    printf("magma_dsygvd returned error %d.\n", (int) info);
                
                temp1 = temp2 = 0;
                for(int j=0; j<N; j++) {
                    temp1 = max(temp1, absv(w1[j]));
                    temp1 = max(temp1, absv(w2[j]));
                    temp2 = max(temp2, absv(w1[j]-w2[j]));
                }
                result[2] = temp2 / temp1;
            }
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            cpu_time = magma_wtime();
            lapackf77_dsygvd( &opts.itype, &opts.jobz, &opts.uplo,
                              &N, h_A, &N, h_B, &N, w2,
                              h_work, &lwork,
                              iwork, &liwork,
                              &info );
            cpu_time = magma_wtime() - cpu_time;
            if (info != 0)
                printf("lapackf77_dsygvd returned error %d.\n", (int) info);
            
            /* =====================================================================
               Print execution time
               =================================================================== */
            printf("%5d     %7.2f         %7.2f\n",
                   (int)N, cpu_time, gpu_time);
            if ( opts.check ) {
                printf("Testing the eigenvalues and eigenvectors for correctness:\n");
                if ( opts.itype==1 )
                    printf("(1)    | A Z - B Z D | / (|A| |Z| N) = %8.2e\n", result[0]);
                else if ( opts.itype==2 )
                    printf("(1)    | A B Z - Z D | / (|A| |Z| N) = %8.2e\n", result[0]);
                else if ( opts.itype==3 )
                    printf("(1)    | B A Z - Z D | / (|A| |Z| N) = %8.2e\n", result[0]);
                if ( opts.itype==1 || opts.itype==2 )
                    printf("(2)    | I -   Z Z' B | /  N         = %8.2e\n", result[1]);
                else
                    printf("(2)    | B -  Z Z' | / (|B| N)       = %8.2e\n", result[1]);
                printf("(3)    | D(w/ Z)-D(w/o Z)|/ |D|      = %8.2e\n\n", result[2]);
            }
            
            TESTING_FREE(       h_A);
            TESTING_FREE(       h_B);
            TESTING_FREE(        w1);
            TESTING_FREE(        w2);
            TESTING_FREE(     iwork);
            TESTING_HOSTFREE(h_work);
            TESTING_HOSTFREE(   h_R);
            TESTING_HOSTFREE(   h_S);
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }
    
    TESTING_CUDA_FINALIZE();
    return 0;
}
