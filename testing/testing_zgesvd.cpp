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
#define PRECISION_z

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgesvd
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t   gpu_time, cpu_time;
    magmaDoubleComplex *h_A, *h_R, *U, *VT, *h_work;
    double *S1, *S2;
#if defined(PRECISION_z) || defined(PRECISION_c)
    double *rwork;
#endif
    magma_int_t M, N, n2, min_mn, info, nb, lwork;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_vec_t jobu, jobvt;
    magma_int_t status = 0;
    
    magma_opts opts;
    parse_opts( argc, argv, &opts );
    
    double tol = opts.tolerance * lapackf77_dlamch("E");
    
    jobu  = opts.jobu;
    jobvt = opts.jobvt;

    magma_vec_t jobs[] = { MagmaNoVec, MagmaSomeVec, MagmaOverwriteVec, MagmaAllVec };
    
    if ( opts.check && (jobu == MagmaNoVec || jobvt == MagmaNoVec)) {
        printf( "NOTE: some checks require that singular vectors are computed;\n"
                "      set both jobu (option -U[NASO]) and jobvt (option -V[NASO]) to be S, O, or A.\n\n" );
    }
    printf("jobu jobv     M     N  CPU time (sec)  GPU time (sec)  |S1-S2|/.  |A-USV'|/. |I-UU'|/M  |I-VV'|/N  S sorted\n");
    printf("===========================================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int ijobu = 0; ijobu < 4; ++ijobu ) {
        for( int ijobv = 0; ijobv < 4; ++ijobv ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            if ( opts.all ) {
                jobu  = jobs[ ijobu ];
                jobvt = jobs[ ijobv ];
            }
            else if ( ijobu > 0 || ijobv > 0 ) {
                // if not testing all, run only once, with ijobu = ijobv = 0
                continue;
            }
            if ( jobu == 'O' && jobvt == 'O' ) {
                // illegal combination; skip
                continue;
            }
            
            M = opts.msize[itest];
            N = opts.nsize[itest];
            n2 = M*N;
            min_mn = min(M, N);
            nb = magma_get_zgesvd_nb(N);
            switch( opts.svd_work ) {
                default:
                #if defined(PRECISION_z) || defined(PRECISION_c)
                case 1: lwork = (M+N)*nb + 2*min_mn;                   break;  // minimum
                case 2: lwork = (M+N)*nb + 2*min_mn +   min_mn*min_mn; break;  // optimal for some paths
                case 3: lwork = (M+N)*nb + 2*min_mn + 2*min_mn*min_mn; break;  // optimal for all paths
                #else
                case 1: lwork = (M+N)*nb + 3*min_mn;                   break;  // minimum
                case 2: lwork = (M+N)*nb + 3*min_mn +   min_mn*min_mn; break;  // optimal for some paths
                case 3: lwork = (M+N)*nb + 3*min_mn + 2*min_mn*min_mn; break;  // optimal for all paths
                #endif
            }
            
            TESTING_MALLOC_CPU( h_A, magmaDoubleComplex, n2  );
            TESTING_MALLOC_CPU( VT,  magmaDoubleComplex, N*N );
            TESTING_MALLOC_CPU( U,   magmaDoubleComplex, M*M );
            TESTING_MALLOC_CPU( S1,  double, min_mn );
            TESTING_MALLOC_CPU( S2,  double, min_mn );
            #if defined(PRECISION_z) || defined(PRECISION_c)
            TESTING_MALLOC_CPU( rwork, double, 5*min_mn );
            #endif
            TESTING_MALLOC_PIN( h_R,    magmaDoubleComplex, n2    );
            TESTING_MALLOC_PIN( h_work, magmaDoubleComplex, lwork );
            
            /* Initialize the matrix */
            lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
            lapackf77_zlacpy( MagmaUpperLowerStr, &M, &N, h_A, &M, h_R, &M );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            #if defined(PRECISION_z) || defined(PRECISION_c)
            magma_zgesvd( jobu, jobvt, M, N,
                          h_R, M, S1, U, M,
                          VT, N, h_work, lwork, rwork, &info );
            #else
            magma_zgesvd( jobu, jobvt, M, N,
                          h_R, M, S1, U, M,
                          VT, N, h_work, lwork, &info );
            #endif
            gpu_time = magma_wtime() - gpu_time;
            if (info != 0)
                printf("magma_zgesvd returned error %d: %s.\n",
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
                
                // get size and location of U and V^T depending on jobu and jobvt
                // U2=NULL and VT2=NULL if they were not computed (e.g., jobu=N)
                magma_int_t M2  = (jobu  == 'A' ? M : min_mn);
                magma_int_t N2  = (jobvt == 'A' ? N : min_mn);
                magma_int_t ldu = M;
                magma_int_t ldv = (jobvt == 'O' ? M : N);
                magmaDoubleComplex *U2  = NULL;
                magmaDoubleComplex *VT2 = NULL;
                if ( jobu == 'S' || jobu == 'A' ) {
                    U2 = U;
                } else if ( jobu == 'O' ) {
                    U2 = h_R;
                }
                if ( jobvt == 'S' || jobvt == 'A' ) {
                    VT2 = VT;
                } else if ( jobvt == 'O' ) {
                    VT2 = h_R;
                }
                
                // since KD=0 (3rd arg), E is not referenced so pass NULL (9th arg)
                #if defined(PRECISION_z) || defined(PRECISION_c)
                if ( U2 != NULL && VT2 != NULL ) {
                    lapackf77_zbdt01(&M, &N, &izero, h_A, &M,
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
                    lapackf77_zbdt01(&M, &N, &izero, h_A, &M,
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
                #if defined(PRECISION_z) || defined(PRECISION_c)
                lapackf77_zgesvd( lapack_vec_const(jobu), lapack_vec_const(jobvt), &M, &N,
                                  h_A, &M, S2, U, &M,
                                  VT, &N, h_work, &lwork, rwork, &info);
                #else
                lapackf77_zgesvd( lapack_vec_const(jobu), lapack_vec_const(jobvt), &M, &N,
                                  h_A, &M, S2, U, &M,
                                  VT, &N, h_work, &lwork, &info);
                #endif
                cpu_time = magma_wtime() - cpu_time;
                if (info != 0)
                    printf("lapackf77_zgesvd returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                
                /* =====================================================================
                   Check the result compared to LAPACK
                   =================================================================== */
                double work[1], c_neg_one = -1;
                magma_int_t one = 1;
                
                blasf77_daxpy(&min_mn, &c_neg_one, S1, &one, S2, &one);
                result[4]  = lapackf77_dlange("f", &min_mn, &one, S2, &min_mn, work);
                result[4] /= lapackf77_dlange("f", &min_mn, &one, S1, &min_mn, work);
                
                printf("   %s    %s %5d %5d  %7.2f         %7.2f         %8.2e",
                       lapack_vec_const(jobu), lapack_vec_const(jobvt),
                       (int) M, (int) N, cpu_time, gpu_time, result[4] );
            }
            else {
                printf("   %s    %s %5d %5d    ---           %7.2f           ---   ",
                       lapack_vec_const(jobu), lapack_vec_const(jobvt),
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
            #if defined(PRECISION_z) || defined(PRECISION_c)
            TESTING_FREE_CPU( rwork );
            #endif
            TESTING_FREE_PIN( h_R    );
            TESTING_FREE_PIN( h_work );
        }}}
        if ( opts.all || opts.niter > 1 ) {
            printf("\n");
        }
    }

    TESTING_FINALIZE();
    return status;
}
