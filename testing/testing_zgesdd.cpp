/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
       @author Mark Gates

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

#define COMPLEX

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgesdd (SVD with Divide & Conquer)
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t   gpu_time, cpu_time;
    magmaDoubleComplex *h_A, *h_R, *U, *VT, *h_work;
    double *S1, *S2;
    #ifdef COMPLEX
    double *rwork;
    #endif
    magma_int_t *iwork;
    magmaDoubleComplex dummy[1];
    
    magma_int_t M, N, lda, ldu, ldv, n2, min_mn, info, lwork;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_vec_t jobz;
    magma_int_t status = 0;
    
    magma_opts opts;
    parse_opts( argc, argv, &opts );
    
    double tol = opts.tolerance * lapackf77_dlamch("E");
    
    jobz = opts.jobu;
    
    magma_vec_t jobs[] = { MagmaNoVec, MagmaSomeVec, MagmaOverwriteVec, MagmaAllVec };
    
    if ( opts.check && ! opts.all && (jobz == MagmaNoVec)) {
        printf( "NOTE: some checks require that singular vectors are computed;\n"
                "      set jobz (option -U[NASO]) to be S, O, or A.\n\n" );
    }
    printf("jobz     M     N  CPU time (sec)  GPU time (sec)  |S1-S2|/.  |A-USV'|/. |I-UU'|/M  |I-VV'|/N  S sorted\n");
    printf("======================================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int ijobz = 0; ijobz < 4; ++ijobz ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            if ( opts.all ) {
                jobz = jobs[ ijobz ];
            }
            else if ( ijobz > 0 ) {
                // if not testing all, run only once, when ijobz = 0,
                // but jobz come from opts (above loops).
                continue;
            }
            
            M = opts.msize[itest];
            N = opts.nsize[itest];
            lda = M;
            ldu = M;
            ldv = N;
            n2 = lda*N;
            min_mn = min(M, N);
            //nb = magma_get_zgesvd_nb(N);
            //switch( opts.svd_work ) {
            //    default:
            //    #ifdef COMPLEX
            //    case 1: lwork = (M+N)*nb + 2*min_mn;                   break;  // minimum
            //    case 2: lwork = (M+N)*nb + 2*min_mn +   min_mn*min_mn; break;  // optimal for some paths
            //    case 3: lwork = (M+N)*nb + 2*min_mn + 2*min_mn*min_mn; break;  // optimal for all paths
            //    #else
            //    case 1: lwork = (M+N)*nb + 3*min_mn;                   break;  // minimum
            //    case 2: lwork = (M+N)*nb + 3*min_mn +   min_mn*min_mn; break;  // optimal for some paths
            //    case 3: lwork = (M+N)*nb + 3*min_mn + 2*min_mn*min_mn; break;  // optimal for all paths
            //    #endif
            //}
            
            TESTING_MALLOC_CPU( h_A, magmaDoubleComplex, lda*N );
            TESTING_MALLOC_CPU( VT,  magmaDoubleComplex, ldv*N );
            TESTING_MALLOC_CPU( U,   magmaDoubleComplex, ldu*M );
            TESTING_MALLOC_CPU( S1,  double, min_mn );
            TESTING_MALLOC_CPU( S2,  double, min_mn );
            TESTING_MALLOC_CPU( iwork, magma_int_t, 8*min_mn );
            #ifdef COMPLEX
            TESTING_MALLOC_CPU( rwork, double, 5*min_mn );
            #endif
            TESTING_MALLOC_PIN( h_R,    magmaDoubleComplex, lda*N );
            
            // query for workspace size
            lwork = -1;
            magma_zgesdd( jobz, M, N,
                          h_R, lda, S1, U, ldu, VT, ldv, dummy, lwork,
                          #ifdef COMPLEX
                          rwork,
                          #endif
                          iwork, &info );
            lwork = (magma_int_t) MAGMA_Z_REAL( dummy[0] );
            //printf( "lwork %d\n", lwork );
            
            TESTING_MALLOC_PIN( h_work, magmaDoubleComplex, lwork );
            
            /* Initialize the matrix */
            lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
            lapackf77_zlacpy( MagmaUpperLowerStr, &M, &N, h_A, &lda, h_R, &lda );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            magma_zgesdd( jobz, M, N,
                          h_R, lda, S1, U, ldu, VT, ldv, h_work, lwork,
                          #ifdef COMPLEX
                          rwork,
                          #endif
                          iwork, &info );
            gpu_time = magma_wtime() - gpu_time;
            if (info != 0)
                printf("magma_zgesdd returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            double eps = lapackf77_dlamch( "E" );
            double result[5] = { -1/eps, -1/eps, -1/eps, -1/eps, -1/eps };
            if ( opts.check ) {
                /* =====================================================================
                   Check the results following the LAPACK's [zcds]drvbd routine.
                   A is factored as A = U diag(S) VT and the following 4 tests computed:
                   (1)    | A - U diag(S) VT | / ( |A| max(M,N) )
                   (2)    | I - U'U | / ( M )
                   (3)    | I - VT VT' | / ( N )
                   (4)    S contains MNMIN nonnegative values in decreasing order.
                          (Return 0 if true, 1/ULP if false.)
                   =================================================================== */
                magma_int_t izero = 0;
                magmaDoubleComplex *h_work_err;
                magma_int_t lwork_err = max(5*min_mn, (3*min_mn + max(M,N)))*128;
                TESTING_MALLOC_CPU( h_work_err, magmaDoubleComplex, lwork_err );
                
                // get size and location of U and V^T depending on jobz
                // U2=NULL and VT2=NULL if they were not computed (e.g., jobz=N)
                magma_int_t M2 = (jobz == MagmaAllVec ? M : min_mn);
                magma_int_t N2 = (jobz == MagmaAllVec ? N : min_mn);
                magmaDoubleComplex *U2  = NULL;
                magmaDoubleComplex *VT2 = NULL;
                if ( jobz == MagmaSomeVec || jobz == MagmaAllVec ) {
                    U2  = U;
                    VT2 = VT;
                }
                else if ( jobz == MagmaOverwriteVec ) {
                    if ( M >= N ) {
                        U2  = h_R;
                        ldu = lda;
                        VT2 = VT;
                    }
                    else {
                        U2  = U;
                        VT2 = h_R;
                        ldv = lda;
                    }
                }
                
                // since KD=0 (3rd arg), E is not referenced so pass NULL (9th arg)
                #ifdef COMPLEX
                if ( U2 != NULL && VT2 != NULL ) {
                    lapackf77_zbdt01(&M, &N, &izero, h_A, &lda,
                                     U2, &ldu, S1, NULL, VT2, &ldv,
                                     h_work_err, rwork, &result[0]);
                }
                if ( U2 != NULL ) {
                    lapackf77_zunt01("Columns", &M, &M2, U2,  &ldu, h_work_err, &lwork_err, rwork, &result[1]);
                }
                if ( VT2 != NULL ) {
                    lapackf77_zunt01(   "Rows", &N2, &N, VT2, &ldv, h_work_err, &lwork_err, rwork, &result[2]);
                }
                #else
                if ( U2 != NULL && VT2 != NULL ) {
                    lapackf77_zbdt01(&M, &N, &izero, h_A, &lda,
                                      U2, &ldu, S1, NULL, VT2, &ldv,
                                      h_work_err, &result[0]);
                }
                if ( U2 != NULL ) {
                    lapackf77_zunt01("Columns", &M, &M2, U2,  &ldu, h_work_err, &lwork_err, &result[1]);
                }
                if ( VT2 != NULL ) {
                    // this step may be really slow for large N
                    lapackf77_zunt01(   "Rows", &N2, &N, VT2, &ldv, h_work_err, &lwork_err, &result[2]);
                }
                #endif
                
                result[3] = 0.;
                for(int j=0; j < min_mn-1; j++){
                    if ( S1[j] < S1[j+1] )
                        result[3] = 1.;
                    if ( S1[j] < 0. )
                        result[3] = 1.;
                }
                if (min_mn > 1 && S1[min_mn-1] < 0.)
                    result[3] = 1.;
                
                result[0] *= eps;
                result[1] *= eps;
                result[2] *= eps;
                
                TESTING_FREE_CPU( h_work_err );
            }
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                lapackf77_zgesdd( lapack_vec_const(jobz), &M, &N,
                                  h_A, &lda, S2, U, &ldu, VT, &ldv, h_work, &lwork,
                                  #ifdef COMPLEX
                                  rwork,
                                  #endif
                                  iwork, &info);
                cpu_time = magma_wtime() - cpu_time;
                if (info != 0)
                    printf("lapackf77_zgesdd returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                
                /* =====================================================================
                   Check the result compared to LAPACK
                   =================================================================== */
                double work[1], c_neg_one = -1;
                magma_int_t one = 1;
                
                blasf77_daxpy(&min_mn, &c_neg_one, S1, &one, S2, &one);
                result[4]  = lapackf77_dlange("f", &min_mn, &one, S2, &min_mn, work);
                result[4] /= lapackf77_dlange("f", &min_mn, &one, S1, &min_mn, work);
                
                printf("    %c %5d %5d  %7.2f         %7.2f         %8.2e",
                       lapack_vec_const(jobz)[0],
                       (int) M, (int) N, cpu_time, gpu_time, result[4] );
            }
            else {
                printf("    %c %5d %5d    ---           %7.2f           ---   ",
                       lapack_vec_const(jobz)[0],
                       (int) M, (int) N, gpu_time );
            }
            if ( opts.check ) {
                if ( result[0] < 0. ) { printf("     ---   "); } else { printf("  %#9.3g", result[0]); }
                if ( result[1] < 0. ) { printf("     ---   "); } else { printf("  %#9.3g", result[1]); }
                if ( result[2] < 0. ) { printf("     ---   "); } else { printf("  %#9.3g", result[2]); }
                int success = (result[0] < tol) && (result[1] < tol) && (result[2] < tol) && (result[3] == 0.) && (result[4] < tol);
                printf("   %3s  %s\n", (result[3] == 0. ? "yes" : "no"), (success ? "ok" : "failed"));
                status |= ! success;
            }
            else {
                printf("\n");
            }
            
            TESTING_FREE_CPU( h_A );
            TESTING_FREE_CPU( VT  );
            TESTING_FREE_CPU( U   );
            TESTING_FREE_CPU( S1  );
            TESTING_FREE_CPU( S2  );
            #ifdef COMPLEX
            TESTING_FREE_CPU( rwork );
            #endif
            TESTING_FREE_PIN( h_R    );
            TESTING_FREE_PIN( h_work );
        }}
        if ( opts.all || opts.niter > 1 ) {
            printf("\n");
        }
    }

    TESTING_FINALIZE();
    return status;
}
