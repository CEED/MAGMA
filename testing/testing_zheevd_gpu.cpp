/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    @author Raffaele Solca
    @author Stan Tomov 

    @precisions normal z -> c

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
   -- Testing zheevd_gpu
*/
int main( int argc, char** argv) 
{
    TESTING_CUDA_INIT();

    cuDoubleComplex *h_A, *h_R, *d_R, *h_work;
    double *rwork, *w1, *w2;
    magma_int_t *iwork;
    double gpu_time, cpu_time;

    magma_timestr_t start, end;

    /* Matrix size */
    magma_int_t N=0, n2;
    magma_int_t size[8] = {1024,2048,3072,4032,5184,6016,7040,8064};

    magma_int_t i, info;
    magma_int_t ione     = 1, izero = 0;
    magma_int_t ISEED[4] = {0,0,0,1};

    const char *uplo = MagmaLowerStr;
    const char *jobz = MagmaVectorsStr;

    magma_int_t checkres;
    double result[3], eps = lapackf77_dlamch( "E" );

    if (argc != 1){
        for(i = 1; i<argc; i++){
            if (strcmp("-N", argv[i])==0) {
                N = atoi(argv[++i]);
            }
            else if ( strcmp("-JV", argv[i]) == 0 ) {
                jobz = MagmaVectorsStr;
            }
            else if ( strcmp("-JN", argv[i]) == 0 ) {
                jobz = MagmaNoVectorsStr;
            }
        }
        if (N>0)
            printf("  testing_zheevd_gpu -N %d [-JV] [-JN]\n\n", (int) N);
        else {
            printf("\nUsage: \n");
            printf("  testing_zheevd_gpu -N %d [-JV] [-JN]\n\n", (int) N);
            exit(1);
        }
    }
    else {
        printf("\nUsage: \n");
        printf("  testing_zheevd_gpu -N %d [-JV] [-JN]\n\n", 1024);
        N = size[7];
    }

    checkres = getenv("MAGMA_TESTINGS_CHECK") != NULL;
    if ( checkres and jobz[0] == MagmaNoVectors ) {
        printf( "Cannot check results when vectors are not computed (jobz='N')\n" );
        checkres = false;
    }

    /* Query for workspace sizes */
    cuDoubleComplex aux_work[1];
    double          aux_rwork[1];
    magma_int_t     aux_iwork[1];
    magma_zheevd_gpu( jobz[0], uplo[0],
                      N, d_R, N, w1,
                      h_R, N,
                      aux_work,  -1,
                      aux_rwork, -1,
                      aux_iwork, -1,
                      &info );
    magma_int_t lwork, lrwork, liwork;
    lwork  = (magma_int_t) MAGMA_Z_REAL( aux_work[0] );
    lrwork = (magma_int_t) aux_rwork[0];
    liwork = aux_iwork[0];

    /* Allocate host memory for the matrix */
    magma_int_t ldda = ((N + 31)/32)*32;
    TESTING_MALLOC(    h_A, cuDoubleComplex, N*N );
    TESTING_MALLOC(    w1,  double,          N   );
    TESTING_MALLOC(    w2,  double,          N   );
    TESTING_HOSTALLOC( h_R, cuDoubleComplex, N*N );
    TESTING_DEVALLOC(  d_R, cuDoubleComplex, N*ldda );
    TESTING_HOSTALLOC( h_work, cuDoubleComplex, lwork  );
    TESTING_MALLOC(    rwork,  double,          lrwork );
    TESTING_MALLOC(    iwork,  magma_int_t,     liwork );
    
    printf("  N     CPU Time(s)    GPU Time(s) \n");
    printf("===================================\n");
    for(i=0; i<8; i++){
        if (argc==1){
            N = size[i];
        }
        n2 = N*N;

        /* Initialize the matrix */
        lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
        for(int k=0;k<N;k++)
        {
            MAGMA_Z_SET2REAL(h_A[k*N+k],  MAGMA_Z_REAL(h_A[k*N+k]));
        }
        magma_zsetmatrix( N, N, h_A, N, d_R, ldda );

        /* warm up run */
        magma_zheevd_gpu( jobz[0], uplo[0],
                          N, d_R, ldda, w1,
                          h_R, N,
                          h_work, lwork, 
                          rwork, lrwork, 
                          iwork, liwork, 
                          &info );
        
        magma_zsetmatrix( N, N, h_A, N, d_R, ldda );

        /* query for optimal workspace sizes */
        magma_zheevd_gpu( jobz[0], uplo[0],
                          N, d_R, ldda, w1,
                          h_R, N,
                          h_work, -1,
                          rwork,  -1,
                          iwork,  -1,
                          &info );
        int lwork_save  = lwork;
        int lrwork_save = lrwork;
        int liwork_save = liwork;
        lwork  = min( lwork,  (magma_int_t) MAGMA_Z_REAL( h_work[0] ));
        lrwork = min( lrwork, (magma_int_t) rwork[0] );
        liwork = min( liwork, iwork[0] );
        //printf( "lwork %d, query %d, used %d; liwork %d, query %d, used %d\n",
        //        lwork_save,  (magma_int_t) h_work[0], lwork,
        //        liwork_save, iwork[0], liwork );

        /* ====================================================================
           Performs operation using MAGMA
           =================================================================== */
        start = get_current_time();
        magma_zheevd_gpu( jobz[0], uplo[0],
                          N, d_R, ldda, w1,
                          h_R, N,
                          h_work, lwork,
                          rwork, lrwork,
                          iwork, liwork,
                          &info );
        end = get_current_time();

        gpu_time = GetTimerValue(start,end)/1000.;

        lwork  = lwork_save;
        lrwork = lrwork_save;
        liwork = liwork_save;
        
        if ( checkres ) {
          /* =====================================================================
             Check the results following the LAPACK's [zcds]drvst routine.
             A is factored as A = U S U' and the following 3 tests computed:
             (1)    | A - U S U' | / ( |A| N )
             (2)    | I - U'U | / ( N )
             (3)    | S(with U) - S(w/o U) | / | S |
             =================================================================== */
          double temp1, temp2;
          cuDoubleComplex *tau;

          magma_zgetmatrix( N, N, d_R, ldda, h_R, N );
          lapackf77_zhet21(&ione, uplo, &N, &izero,
                           h_A, &N,
                           w1, w1,
                           h_R, &N,
                           h_R, &N,
                           tau, h_work, rwork, &result[0]);
          
          magma_zsetmatrix( N, N, h_A, N, d_R, ldda );
          magma_zheevd_gpu( 'N', uplo[0],
                            N, d_R, ldda, w2,
                            h_R, N,
                            h_work, lwork,
                            rwork, lrwork,
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
        lapackf77_zheevd(jobz, uplo,
                         &N, h_A, &N, w2,
                         h_work, &lwork,
                         rwork, &lrwork,
                         iwork, &liwork,
                         &info);
        end = get_current_time();
        if (info < 0)
          printf("Argument %d of zheevd had an illegal value.\n", (int) -info);

        cpu_time = GetTimerValue(start,end)/1000.;

        /* =====================================================================
           Print execution time
           =================================================================== */
        printf("%5d     %6.2f         %6.2f\n",
               (int) N, cpu_time, gpu_time);
        if ( checkres ){
          printf("Testing the factorization A = U S U' for correctness:\n");
          printf("(1)    | A - U S U' | / (|A| N) = %e\n", result[0]*eps);
          printf("(2)    | I -   U'U  | /  N      = %e\n", result[1]*eps);
          printf("(3)    | S(w/ U)-S(w/o U)|/ |S| = %e\n\n", result[2]);
        }

        if (argc != 1)
            break;
    }
 
    /* Memory clean up */
    TESTING_FREE(     h_A    );
    TESTING_FREE(     w1     );
    TESTING_FREE(     w2     );
    TESTING_FREE(     rwork  );
    TESTING_FREE(     iwork  );
    TESTING_HOSTFREE( h_work );
    TESTING_HOSTFREE( h_R    );
    TESTING_DEVFREE(  d_R    );
  
    /* Shutdown */
    TESTING_CUDA_FINALIZE();
}
