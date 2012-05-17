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
#include <cuda.h>
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

    double *h_A, *h_R, *h_B, *h_S, *h_work;
    double *w1, *w2;
    magma_int_t *iwork;
    double gpu_time, cpu_time;

    magma_timestr_t start, end;

    /* Matrix size */
    magma_int_t N=0, n2;
    magma_int_t size[4] = {1024,2048,4100,6001};

    magma_int_t i, itype, info;
    magma_int_t ione = 1, izero = 0;
    magma_int_t five = 5;

    double c_one     = MAGMA_D_ONE;
    double c_neg_one = MAGMA_D_NEG_ONE;

    double d_zero    =  0.;
    double d_one     =  1.;
    double d_neg_one = -1.;
    double dten      = 10.;
    magma_int_t ISEED[4] = {0,0,0,1};

    //const char *uplo = MagmaLowerStr;
    char *uplo = (char*)MagmaLowerStr;
    //char *uplo = (char*)MagmaUpperStr;
    char *jobz = (char*)MagmaVectorsStr;
    itype = 1;

    magma_int_t checkres;
    double result[4];

    int flagN = 0;

    if (argc != 1){
        for(i = 1; i<argc; i++){
            if (strcmp("-N", argv[i])==0){
                N = atoi(argv[++i]);
                if (N>0){
                   printf("  testing_dsygvd -N %d\n\n", N);
                   flagN=1;
                }
                else {
                   printf("\nUsage: \n");
                   printf("  testing_dsygvd -N %d\n\n", N);
                   exit(1);
                }
            }
            if (strcmp("-itype", argv[i])==0){
                itype = atoi(argv[++i]);
                if (itype>0 && itype <= 3){
                   printf("  testing_dsygvd -itype %d\n\n", itype);
                }
                else {
                   printf("\nUsage: \n");
                   printf("  testing_dsygvd -itype %d\n\n", itype);
                   exit(1);
                }
            }
            if (strcmp("-L", argv[i])==0){
              uplo = (char*)MagmaLowerStr;
              printf("  testing_dsygvd -L");
            }
            if (strcmp("-U", argv[i])==0){
              uplo = (char*)MagmaUpperStr;
              printf("  testing_dsygvd -U");              
            }
          
        }
      
    } else {
        printf("\nUsage: \n");
        printf("  testing_dsygvd -L/U -N %d -itype %d\n\n", 1024, 1);
    }

    if(!flagN)
        N = size[3];

    checkres  = getenv("MAGMA_TESTINGS_CHECK") != NULL;

    n2  = N * N;

    /* Allocate host memory for the matrix */
    TESTING_MALLOC(   h_A, double, n2);
    TESTING_MALLOC(   h_B, double, n2);
    TESTING_MALLOC(    w1, double         ,  N);
    TESTING_MALLOC(    w2, double         ,  N);
    TESTING_HOSTALLOC(h_R, double, n2);
    TESTING_HOSTALLOC(h_S, double, n2);

    magma_int_t nb = magma_get_dsytrd_nb(N);
    magma_int_t lwork  = 1 + 6*N*nb + 2* N*N;
    magma_int_t liwork = 3 + 5*N;

    TESTING_HOSTALLOC(h_work, double,  lwork);
    TESTING_MALLOC(    iwork,     magma_int_t, liwork);
    
    printf("\n\n");
    printf("  N     CPU Time(s)    GPU Time(s) \n");
    printf("===================================\n");
    for(i=0; i<4; i++){
        if (!flagN){
            N = size[i];
            n2 = N*N;
        }

        /* Initialize the matrix */
        lapackf77_dlarnv( &ione, ISEED, &n2, h_A );
        lapackf77_dlarnv( &ione, ISEED, &n2, h_B );
        /* increase the diagonal */
        {
          magma_int_t i, j;
          for(i=0; i<N; i++) {
            MAGMA_D_SET2REAL( h_B[i*N+i], ( MAGMA_D_REAL(h_B[i*N+i]) + 1.*N ) );
          }
        }
        lapackf77_dlacpy( MagmaUpperLowerStr, &N, &N, h_A, &N, h_R, &N );
        lapackf77_dlacpy( MagmaUpperLowerStr, &N, &N, h_B, &N, h_S, &N );

        magma_dsygvd(itype, jobz[0], uplo[0],
                     N, h_R, N, h_S, N, w1,
                     h_work, lwork, 
                     iwork, liwork, 
                     &info);
        
        lapackf77_dlacpy( MagmaUpperLowerStr, &N, &N, h_A, &N, h_R, &N );
        lapackf77_dlacpy( MagmaUpperLowerStr, &N, &N, h_B, &N, h_S, &N );


        /* ====================================================================
           Performs operation using MAGMA
           =================================================================== */
        start = get_current_time();
        magma_dsygvd(itype, jobz[0], uplo[0],
                     N, h_R, N, h_S, N, w1,
                     h_work, lwork,
                     iwork, liwork,
                     &info);
        end = get_current_time();

        gpu_time = GetTimerValue(start,end)/1000.;

        if ( checkres ) {
          /* =====================================================================
             Check the results following the LAPACK's [zc]hegvd routine.
             A x = lambda B x is solved
             and the following 3 tests computed:
             (1)    | A Z - B Z D | / ( |A||Z| N )  (itype = 1)
                    | A B Z - Z D | / ( |A||Z| N )  (itype = 2)
                    | B A Z - Z D | / ( |A||Z| N )  (itype = 3)
             (2)    | I - V V' B | / ( N )           (itype = 1,2)
                    | B - V V' | / ( |B| N )         (itype = 3)
             (3)    | S(with V) - S(w/o V) | / | S |
             =================================================================== */
          double temp1, temp2;
          double *tau;

          if (itype == 1 || itype == 2){
            lapackf77_dlaset( "A", &N, &N, &d_zero, &c_one, h_S, &N);
            blasf77_dgemm("N", "C", &N, &N, &N, &c_one, h_R, &N, h_R, &N, &d_zero, h_work, &N);
            blasf77_dsymm("R", uplo, &N, &N, &c_neg_one, h_B, &N, h_work, &N, &c_one, h_S, &N);
            result[1]= lapackf77_dlange("1", &N, &N, h_S, &N, h_work) / N;
          }
          else if (itype == 3){
            lapackf77_dlacpy( MagmaUpperLowerStr, &N, &N, h_B, &N, h_S, &N);
            blasf77_dsyrk(uplo, "N", &N, &N, &d_neg_one, h_R, &N, &d_one, h_S, &N); 
            result[1]= lapackf77_dlansy("1",uplo, &N, h_S, &N, h_work) / N / 
              lapackf77_dlansy("1",uplo, &N, h_B, &N, h_work);
          }

          result[0] = 1.;
          result[0] /= lapackf77_dlansy("1",uplo, &N, h_A, &N, h_work);
          result[0] /= lapackf77_dlange("1",&N , &N, h_R, &N, h_work);

          if (itype == 1){
            blasf77_dsymm("L", uplo, &N, &N, &c_one, h_A, &N, h_R, &N, &d_zero, h_work, &N);
            for(int i=0; i<N; ++i)
              blasf77_ddscal(&N, &w1[i], &h_R[i*N], &ione);
            blasf77_dsymm("L", uplo, &N, &N, &c_neg_one, h_B, &N, h_R, &N, &c_one, h_work, &N);
            result[0] *= lapackf77_dlange("1", &N, &N, h_work, &N, &temp1)/N;
          }
          else if (itype == 2){
            blasf77_dsymm("L", uplo, &N, &N, &c_one, h_B, &N, h_R, &N, &d_zero, h_work, &N);
            for(int i=0; i<N; ++i)
              blasf77_ddscal(&N, &w1[i], &h_R[i*N], &ione);
            blasf77_dsymm("L", uplo, &N, &N, &c_one, h_A, &N, h_work, &N, &c_neg_one, h_R, &N);
            result[0] *= lapackf77_dlange("1", &N, &N, h_R, &N, &temp1)/N;
          }
          else if (itype == 3){
            blasf77_dsymm("L", uplo, &N, &N, &c_one, h_A, &N, h_R, &N, &d_zero, h_work, &N);
            for(int i=0; i<N; ++i)
              blasf77_ddscal(&N, &w1[i], &h_R[i*N], &ione);
            blasf77_dsymm("L", uplo, &N, &N, &c_one, h_B, &N, h_work, &N, &c_neg_one, h_R, &N);
            result[0] *= lapackf77_dlange("1", &N, &N, h_R, &N, &temp1)/N;
          }

/*          lapackf77_dsyt21(&ione, uplo, &N, &izero,
                           h_A, &N,
                           w1, w1,
                           h_R, &N,
                           h_R, &N,
                           tau, h_work, rwork, &result[0]);
*/          
          lapackf77_dlacpy( MagmaUpperLowerStr, &N, &N, h_A, &N, h_R, &N );
          lapackf77_dlacpy( MagmaUpperLowerStr, &N, &N, h_B, &N, h_S, &N );
 
          magma_dsygvd(itype, 'N', uplo[0],
                       N, h_R, N, h_S, N, w2,
                       h_work, lwork,
                       iwork, liwork,
                       &info);

          temp1 = temp2 = 0;
          for(int j=0; j<N; j++){
            temp1 = max(temp1, absv(w1[j]));
            temp1 = max(temp1, absv(w2[j]));
            temp2 = max(temp2, absv(w1[j]-w2[j]));
          }
          result[2] = temp2 / temp1;
        }

        /* =====================================================================
           Performs operation using LAPACK
           =================================================================== */
        start = get_current_time();
        lapackf77_dsygvd(&itype, jobz, uplo,
                         &N, h_A, &N, h_B, &N, w2,
                         h_work, &lwork,
                         iwork, &liwork,
                         &info);
        end = get_current_time();
        if (info < 0)
          printf("Argument %d of dsygvd had an illegal value.\n", -info);

        cpu_time = GetTimerValue(start,end)/1000.;


        /* =====================================================================
           Print execution time
           =================================================================== */
        printf("%5d     %6.2f         %6.2f\n",
               N, cpu_time, gpu_time);
        if ( checkres ){
          printf("Testing the eigenvalues and eigenvectors for correctness:\n");
          if(itype==1)
             printf("(1)    | A Z - B Z D | / (|A| |Z| N) = %e\n", result[0]);
          else if(itype==2)
             printf("(1)    | A B Z - Z D | / (|A| |Z| N) = %e\n", result[0]);
          else if(itype==3)
             printf("(1)    | B A Z - Z D | / (|A| |Z| N) = %e\n", result[0]);
          if(itype==1 || itype ==2)
             printf("(2)    | I -   Z Z' B | /  N      = %e\n", result[1]);
          else
             printf("(2)    | B -  Z Z' | / (|B| N)      = %e\n", result[1]);
          printf("(3)    | D(w/ Z)-D(w/o Z)|/ |D| = %e\n\n", result[2]);
        }

        if (flagN)
            break;
    }
 
    /* Memory clean up */
    TESTING_FREE(       h_A);
    TESTING_FREE(       h_B);
    TESTING_FREE(        w1);
    TESTING_FREE(        w2);
    TESTING_FREE(     iwork);
    TESTING_HOSTFREE(h_work);
    TESTING_HOSTFREE(   h_R);
    TESTING_HOSTFREE(   h_S);

    /* Shutdown */
    TESTING_CUDA_FINALIZE();
}
