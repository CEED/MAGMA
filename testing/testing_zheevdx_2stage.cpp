/*
    -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Raffaele Solca
       @author Azzam Haidar
       @author Mark Gates

       @precisions normal z -> c d s

*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"
#include "magma_threadsetting.h"

#define COMPLEX

static magma_int_t check_orthogonality(magma_int_t M, magma_int_t N, magmaDoubleComplex *Q, magma_int_t LDQ, double eps);
static magma_int_t check_reduction(magma_uplo_t uplo, magma_int_t N, magma_int_t bw, magmaDoubleComplex *A, double *D, magma_int_t LDA, magmaDoubleComplex *Q, double eps );
static magma_int_t check_solution(magma_int_t N, double *E1, double *E2, double eps);

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zhegvdx
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t gpu_time;

    magmaDoubleComplex *h_A, *h_R, *h_work;

    #ifdef COMPLEX
    double *rwork;
    magma_int_t lrwork;
    #endif

    /* Matrix size */
    double *w1, *w2;
    magma_int_t *iwork;
    magma_int_t N, n2, info, lwork, liwork;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t info_ortho     = 0;
    magma_int_t info_solution  = 0;
    magma_int_t info_reduction = 0;
    magma_int_t status = 0;

    magma_opts opts;
    opts.parse_opts( argc, argv );

    magma_range_t range = MagmaRangeAll;
    if (opts.fraction != 1)
        range = MagmaRangeI;

    // pass ngpu = -1 to test multi-GPU code using 1 gpu
    magma_int_t abs_ngpu = abs( opts.ngpu );
    
    printf("%% jobz = %s, range = %s, uplo = %s, fraction = %6.4f, ngpu %d\n",
           lapack_vec_const(opts.jobz), lapack_range_const(range), lapack_uplo_const(opts.uplo),
           opts.fraction, int(abs_ngpu) );

    printf("%%   N     M  GPU Time (sec)   ||I-Q^H Q||/N   ||A-QDQ^H||/(||A||N)   |D-D_magma|/(|D| * N)\n");
    printf("%%=========================================================================================\n");
    magma_int_t threads = magma_get_parallel_numthreads();
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            n2     = N*N;

            magma_zheevdx_getworksize(N, threads, (opts.jobz == MagmaVec), 
                                     &lwork, 
                                     #ifdef COMPLEX
                                     &lrwork, 
                                     #endif
                                     &liwork);
            
            /* Allocate host memory for the matrix */
            TESTING_MALLOC_CPU( h_A,   magmaDoubleComplex, n2 );
            TESTING_MALLOC_CPU( w1,    double, N );
            TESTING_MALLOC_CPU( w2,    double, N );
            TESTING_MALLOC_CPU( iwork, magma_int_t, liwork );
            
            TESTING_MALLOC_PIN( h_R,    magmaDoubleComplex, n2    );
            TESTING_MALLOC_PIN( h_work, magmaDoubleComplex, lwork );
            #ifdef COMPLEX
            TESTING_MALLOC_PIN( rwork, double, lrwork );
            #endif

            /* Initialize the matrix */
            lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
            magma_zmake_hermitian( N, h_A, N );

            magma_int_t m1 = 0;
            double vl = 0;
            double vu = 0;
            magma_int_t il = 0;
            magma_int_t iu = 0;
            if (opts.fraction == 0) {
                il = max( 1, magma_int_t(0.1*N) );
                iu = max( 1, magma_int_t(0.3*N) );
            }
            else {
                il = 1;
                iu = max( 1, magma_int_t(opts.fraction*N) );
            }

            if (opts.warmup) {
                // ==================================================================
                // Warmup using MAGMA
                // ==================================================================
                lapackf77_zlacpy( MagmaFullStr, &N, &N, h_A, &N, h_R, &N );
                if (opts.ngpu == 1) {
                    //printf("calling zheevdx_2stage 1 GPU\n");
                    magma_zheevdx_2stage( opts.jobz, range, opts.uplo, N, 
                                          h_R, N, 
                                          vl, vu, il, iu, 
                                          &m1, w1, 
                                          h_work, lwork, 
                                          #ifdef COMPLEX
                                          rwork, lrwork, 
                                          #endif
                                          iwork, liwork, 
                                          &info );
                } else {
                    //printf("calling zheevdx_2stage_m %d GPU\n", (int) opts.ngpu);
                    magma_zheevdx_2stage_m( abs_ngpu, opts.jobz, range, opts.uplo, N, 
                                            h_R, N, 
                                            vl, vu, il, iu, 
                                            &m1, w1, 
                                            h_work, lwork, 
                                            #ifdef COMPLEX
                                            rwork, lrwork, 
                                            #endif
                                            iwork, liwork, 
                                            &info );
                }
            }

            // ===================================================================
            // Performs operation using MAGMA
            // ===================================================================
            lapackf77_zlacpy( MagmaFullStr, &N, &N, h_A, &N, h_R, &N );
            gpu_time = magma_wtime();
            if (opts.ngpu == 1) {
                //printf("calling zheevdx_2stage 1 GPU\n");
                magma_zheevdx_2stage( opts.jobz, range, opts.uplo, N, 
                                      h_R, N, 
                                      vl, vu, il, iu, 
                                      &m1, w1, 
                                      h_work, lwork, 
                                      #ifdef COMPLEX
                                      rwork, lrwork, 
                                      #endif
                                      iwork, liwork, 
                                      &info );
            } else {
                //printf("calling zheevdx_2stage_m %d GPU\n", (int) opts.ngpu);
                magma_zheevdx_2stage_m( abs_ngpu, opts.jobz, range, opts.uplo, N, 
                                        h_R, N, 
                                        vl, vu, il, iu, 
                                        &m1, w1, 
                                        h_work, lwork, 
                                        #ifdef COMPLEX
                                        rwork, lrwork, 
                                        #endif
                                        iwork, liwork, 
                                        &info );
            }
            gpu_time = magma_wtime() - gpu_time;
            
            printf("%5d %5d  %7.2f      ",
                   (int) N, (int) m1, gpu_time );

            if ( opts.check ) {
                info_solution  = 0;
                info_ortho     = 0;
                info_reduction = 0;
                //double eps   = lapackf77_dlamch("E")*lapackf77_dlamch("B");
                double eps   = lapackf77_dlamch("E");
              
                /* Check the orthogonality, reduction and the eigen solutions */
                if (opts.jobz == MagmaVec) {
                    info_ortho = check_orthogonality(N, N, h_R, N, eps);
                    info_reduction = check_reduction(opts.uplo, N, 1, h_A, w1, N, h_R, eps);
                } else {
                    printf("         ---                ---  ");
                }
                lapackf77_zheevd("N", "L", &N, 
                                h_A, &N, w2, 
                                h_work, &lwork, 
                                #ifdef COMPLEX
                                rwork, &lrwork, 
                                #endif
                                iwork, &liwork, 
                                &info);
                info_solution = check_solution(N, w2, w1, eps);
                
                bool okay = (info_solution == 0) && (info_ortho == 0) && (info_reduction == 0);
                status += ! okay;
                printf("  %s", (okay ? "ok" : "failed"));
            }
            printf("\n");

            TESTING_FREE_CPU( h_A   );
            TESTING_FREE_CPU( w1    );
            TESTING_FREE_CPU( w2    );
            TESTING_FREE_CPU( iwork );
            
            TESTING_FREE_PIN( h_R    );
            TESTING_FREE_PIN( h_work );
            #ifdef COMPLEX
            TESTING_FREE_PIN( rwork  );
            #endif
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    opts.cleanup();
    TESTING_FINALIZE();
    return status;
}



/*-------------------------------------------------------------------
 * Check the orthogonality of Q
 */
static magma_int_t check_orthogonality(magma_int_t M, magma_int_t N, magmaDoubleComplex *Q, magma_int_t LDQ, double eps)
{
    double d_one     =  1.0;
    double d_neg_one = -1.0;
    magmaDoubleComplex c_zero    = MAGMA_Z_ZERO;
    magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    double  normQ, result;
    magma_int_t     info_ortho;
    magma_int_t     minMN = min(M, N);
    double *work;
    TESTING_MALLOC_CPU( work, double, minMN );

    /* Build the idendity matrix */
    magmaDoubleComplex *Id;
    TESTING_MALLOC_CPU( Id, magmaDoubleComplex, minMN*minMN );
    lapackf77_zlaset("A", &minMN, &minMN, &c_zero, &c_one, Id, &minMN);

    /* Perform Id - Q^H Q */
    if (M >= N)
        blasf77_zherk("U", "C", &N, &M, &d_one, Q, &LDQ, &d_neg_one, Id, &N);
    else
        blasf77_zherk("U", "N", &M, &N, &d_one, Q, &LDQ, &d_neg_one, Id, &M);

    normQ = safe_lapackf77_zlanhe("I", "U", &minMN, Id, &minMN, work);

    result = normQ / (minMN * eps);
    printf( "      %8.2e", normQ / minMN );

    // TODO: use opts.tolerance instead of hard coding 60
    if ( isnan(result) || isinf(result) || (result > 60.0) ) {
        info_ortho = 1;
    }
    else {
        info_ortho = 0;
    }
    TESTING_FREE_CPU( work );
    TESTING_FREE_CPU( Id   );
    
    return info_ortho;
}


/*------------------------------------------------------------
 *  Check the reduction 
 */
static magma_int_t check_reduction(magma_uplo_t uplo, magma_int_t N, magma_int_t bw, magmaDoubleComplex *A, double *D, magma_int_t LDA, magmaDoubleComplex *Q, double eps )
{
    magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex *TEMP, *Residual;
    double *work;
    double Anorm, Rnorm, result;
    magma_int_t info_reduction;
    magma_int_t i;
    magma_int_t ione=1;

    TESTING_MALLOC_CPU( TEMP,     magmaDoubleComplex, N*N );
    TESTING_MALLOC_CPU( Residual, magmaDoubleComplex, N*N );
    TESTING_MALLOC_CPU( work,     double, N );
    
    /* Compute TEMP =  Q * LAMBDA */
    lapackf77_zlacpy("A", &N, &N, Q, &LDA, TEMP, &N);        
    for (i = 0; i < N; i++) {
        blasf77_zdscal(&N, &D[i], &(TEMP[i*N]), &ione);
    }
    /* Compute Residual = A - Q * LAMBDA * Q^H */
    /* A is Hermitian but both upper and lower 
     * are assumed valable here for checking 
     * otherwise it need to be symetrized before 
     * checking.
     */ 
    lapackf77_zlacpy("A", &N, &N, A, &LDA, Residual, &N);        
    blasf77_zgemm("N", "C", &N, &N, &N, &c_neg_one, TEMP, &N, Q, &LDA, &c_one, Residual,     &N);

    // since A has been generated by larnv and we did not symmetrize, 
    // so only the uplo portion of A should be equal to Q*LAMBDA*Q^H 
    // for that Rnorm use zlanhe instead of zlange
    Rnorm = safe_lapackf77_zlanhe("1", lapack_uplo_const(uplo), &N, Residual, &N, work);
    Anorm = safe_lapackf77_zlanhe("1", lapack_uplo_const(uplo), &N, A,        &LDA, work);

    result = Rnorm / ( Anorm * N * eps);
    printf("           %8.2e",  Rnorm / ( Anorm * N));

    // TODO: use opts.tolerance instead of hard coding 60
    if ( isnan(result) || isinf(result) || (result > 60.0) ) {
        info_reduction = 1;
    }
    else {
        info_reduction = 0;
    }

    TESTING_FREE_CPU( TEMP     );
    TESTING_FREE_CPU( Residual );
    TESTING_FREE_CPU( work     );

    return info_reduction;
}


/*------------------------------------------------------------
 *  Check the eigenvalues 
 */
static magma_int_t check_solution(magma_int_t N, double *E1, double *E2, double eps)
{
    magma_int_t   info_solution, i;
    double unfl   = lapackf77_dlamch("Safe minimum");
    double resid;
    double maxtmp;
    double maxdif = fabs( fabs(E1[0]) - fabs(E2[0]) );
    double maxeig = max( fabs(E1[0]), fabs(E2[0]) );
    for (i = 1; i < N; i++) {
        resid   = fabs(fabs(E1[i])-fabs(E2[i]));
        maxtmp  = max(fabs(E1[i]), fabs(E2[i]));

        /* Update */
        maxeig = max(maxtmp, maxeig);
        maxdif  = max(resid,  maxdif );
    }
    maxtmp = maxdif / max(unfl, eps*max(maxeig, maxdif));

    printf("              %8.2e", maxdif / (max(maxeig, maxdif)) );

    // TODO: use opts.tolerance instead of hard coding 100
    if ( isnan(maxtmp) || isinf(maxtmp) || (maxtmp > 100) ) {
        info_solution = 1;
    }
    else {
        info_solution = 0;
    }
    return info_solution;
}
