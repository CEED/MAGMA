/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c

*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <cblas.h>

#include <algorithm>  // for sorting

// includes, project
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

#define PRECISION_z
#define COMPLEX


// comparison operator for sorting
bool compare( magmaDoubleComplex a, magmaDoubleComplex b )
{
    return (MAGMA_Z_REAL(a) < MAGMA_Z_REAL(b)) ||
        (MAGMA_Z_REAL(a) == MAGMA_Z_REAL(b) && MAGMA_Z_IMAG(a) < MAGMA_Z_IMAG(b));
}

// DLAPY2 returns sqrt(x**2+y**2), taking care not to cause unnecessary overflow.
// TODO: put into auxiliary file. It's useful elsewhere.
extern "C"
double magma_dlapy2(double x, double y)
{
    double ret_val, d;
    double w, z, xabs, yabs;
    
    xabs = MAGMA_D_ABS(x);
    yabs = MAGMA_D_ABS(y);
    w    = max(xabs, yabs);
    z    = min(xabs, yabs);
    if (z == 0.) {
        ret_val = w;
    } else {
        d = z / w;
        ret_val = w * sqrt(d * d + 1.);
    }
    return ret_val;
}

extern "C"
double magma_dzlapy2(magmaDoubleComplex x)
{
    return magma_dlapy2( MAGMA_Z_REAL(x), MAGMA_Z_IMAG(x) );
}


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgeev
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t   gpu_time, cpu_time;
    magmaDoubleComplex *h_A, *h_R, *VL, *VR, *h_work, *w1, *w2;
    magmaDoubleComplex *w1copy, *w2copy;
    magmaDoubleComplex  c_neg_one = MAGMA_Z_NEG_ONE;
    double *rwork;
    double tnrm, result[9];
    magma_int_t N, n2, lda, nb, lwork, info;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    double ulp, ulpinv, error;
    magma_int_t status = 0;
    
    ulp = lapackf77_dlamch("P");
    ulpinv = 1./ulp;
    
    magma_opts opts;
    parse_opts( argc, argv, &opts );
    
    // need slightly looser bound (60*eps instead of 30*eps) for some tests
    opts.tolerance = max( 60., opts.tolerance );
    double tol    = opts.tolerance * lapackf77_dlamch("E");
    double tolulp = opts.tolerance * lapackf77_dlamch("P");
    
    // enable at least some minimal checks, if requested
    if ( opts.check && !opts.lapack && opts.jobvl == MagmaNoVec && opts.jobvr == MagmaNoVec ) {
        fprintf( stderr, "NOTE: Some checks require vectors to be computed;\n"
                "      set jobvl=V (option -LV), or jobvr=V (option -RV), or both.\n"
                "      Some checks require running lapack (-l); setting lapack.\n\n");
        opts.lapack = true;
    }
    
    printf("    N   CPU Time (sec)   GPU Time (sec)   |W_magma - W_lapack| / |W_lapack|\n");
    printf("===========================================================================\n");
    for( int i = 0; i < opts.ntest; ++i ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[i];
            lda   = N;
            n2    = lda*N;
            nb    = magma_get_zgehrd_nb(N);
            lwork = N*(1 + nb);
            // generous workspace - required by zget22
            lwork = max( lwork, N*(5 + 2*N) );
            
            TESTING_MALLOC_CPU( w1copy, magmaDoubleComplex, N );
            TESTING_MALLOC_CPU( w2copy, magmaDoubleComplex, N );
            TESTING_MALLOC_CPU( w1,     magmaDoubleComplex, N );
            TESTING_MALLOC_CPU( w2,     magmaDoubleComplex, N );
            TESTING_MALLOC_CPU( rwork,  double, 2*N );
            TESTING_MALLOC_CPU( h_A,    magmaDoubleComplex, n2 );
            
            TESTING_MALLOC_PIN( h_R,    magmaDoubleComplex, n2 );
            TESTING_MALLOC_PIN( VL,     magmaDoubleComplex, n2 );
            TESTING_MALLOC_PIN( VR,     magmaDoubleComplex, n2 );
            TESTING_MALLOC_PIN( h_work, magmaDoubleComplex, lwork );
            
            /* Initialize the matrix */
            lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
            lapackf77_zlacpy( MagmaUpperLowerStr, &N, &N, h_A, &lda, h_R, &lda );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            magma_zgeev( opts.jobvl, opts.jobvr,
                         N, h_R, lda, w1,
                         VL, lda, VR, lda,
                         h_work, lwork, rwork, &info );
            gpu_time = magma_wtime() - gpu_time;
            if (info != 0)
                printf("magma_zgeev returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            /* =====================================================================
               Check the result
               =================================================================== */
            if ( opts.check ) {
                /* ===================================================================
                 * Check the result following LAPACK's [zcds]drvev routine.
                 * The following tests are performed:
                 * (1)   | A * VR - VR * W | / ( n |A| )
                 *
                 *       Here VR is the matrix of unit right eigenvectors.
                 *       W is a diagonal matrix with diagonal entries W(j).
                 *
                 * (2)   | |VR(i)| - 1 |   and whether largest component real
                 *
                 *       VR(i) denotes the i-th column of VR.
                 *
                 * (3)   | A**H * VL - VL * W**H | / ( n |A| )
                 *
                 *       Here VL is the matrix of unit left eigenvectors, A**H is the
                 *       conjugate-transpose of A, and W is as above.
                 *
                 * (4)   | |VL(i)| - 1 |   and whether largest component real
                 *
                 *       VL(i) denotes the i-th column of VL.
                 *
                 * (5)   W(full) = W(partial, W only) -- currently skipped
                 * (6)   W(full) = W(partial, W and VR)
                 * (7)   W(full) = W(partial, W and VL)
                 *
                 *       W(full) denotes the eigenvalues computed when both VR and VL
                 *       are also computed, and W(partial) denotes the eigenvalues
                 *       computed when only W, only W and VR, or only W and VL are
                 *       computed.
                 *
                 * (8)   VR(full) = VR(partial, W and VR)
                 *
                 *       VR(full) denotes the right eigenvectors computed when both VR
                 *       and VL are computed, and VR(partial) denotes the result
                 *       when only VR is computed.
                 *
                 * (9)   VL(full) = VL(partial, W and VL)
                 *
                 *       VL(full) denotes the left eigenvectors computed when both VR
                 *       and VL are also computed, and VL(partial) denotes the result
                 *       when only VL is computed.
                 *
                 * (1, 2) only if jobvr = V
                 * (3, 4) only if jobvl = V
                 * (5-9)  only if check = 2 (option -c2)
                 ================================================================= */
                double vmx, vrmx, vtst;
                
                // Initialize result. -1 indicates test was not run.
                for( int j = 0; j < 9; ++j )
                    result[j] = -1.;
                
                if ( opts.jobvr == MagmaVec ) {
                    // Do test 1: | A * VR - VR * W | / ( n |A| )
                    // Note this writes result[1] also
                    lapackf77_zget22( MagmaNoTransStr, MagmaNoTransStr, MagmaNoTransStr,
                                      &N, h_A, &lda, VR, &lda, w1,
                                      h_work, rwork, &result[0] );
                    result[0] *= ulp;
                    
                    // Do test 2: | |VR(i)| - 1 |   and whether largest component real
                    result[1] = -1.;
                    for( int j = 0; j < N; ++j ) {
                        tnrm = cblas_dznrm2(N, &VR[j*lda], ione);
                        result[1] = max( result[1], min( ulpinv, fabs(tnrm-1.)/ulp ));
                        
                        vmx = vrmx = 0.;
                        for( int jj = 0; jj <N; ++jj ) {
                            vtst = magma_dzlapy2(VR[jj + j*lda]);
                            if (vtst > vmx)
                                vmx = vtst;
                            
                            if (MAGMA_Z_IMAG(VR[jj + j*lda])==0. &&
                                fabs( MAGMA_Z_REAL(VR[jj+j*lda]) ) > vrmx)
                            {
                                vrmx = fabs( MAGMA_Z_REAL( VR[jj+j*lda] ) );
                            }
                        }
                        if (vrmx / vmx < 1. - ulp*2.)
                            result[1] = ulpinv;
                    }
                    result[1] *= ulp;
                }
                
                if ( opts.jobvl == MagmaVec ) {
                    // Do test 3: | A**H * VL - VL * W**H | / ( n |A| )
                    // Note this writes result[3] also
                    lapackf77_zget22( MagmaConjTransStr, MagmaNoTransStr, MagmaConjTransStr,
                                      &N, h_A, &lda, VL, &lda, w1,
                                      h_work, rwork, &result[2] );
                    result[2] *= ulp;
                
                    // Do test 4: | |VL(i)| - 1 |   and whether largest component real
                    result[3] = -1.;
                    for( int j = 0; j < N; ++j ) {
                        tnrm = cblas_dznrm2(N, &VL[j*lda], ione);
                        result[3] = max( result[3], min( ulpinv, fabs(tnrm - 1.)/ ulp ));
                        
                        vmx = vrmx = 0.;
                        for( int jj = 0; jj < N; ++jj ) {
                            vtst = magma_dzlapy2(VL[jj + j*lda]);
                            if (vtst > vmx)
                                vmx = vtst;
                            
                            if (MAGMA_Z_IMAG(VL[jj + j*lda])==0. &&
                                fabs( MAGMA_Z_REAL( VL[jj + j*lda] ) ) > vrmx)
                            {
                                vrmx = fabs( MAGMA_Z_REAL( VL[jj+j*lda]) );
                            }
                        }
                        if (vrmx / vmx < 1. - ulp*2.)
                            result[3] = ulpinv;
                    }
                    result[3] *= ulp;
                }
            }
            if ( opts.check == 2 ) {
                // more extensive tests
                // this is really slow because it calls magma_zgeev multiple times
                magmaDoubleComplex *LRE, DUM;
                TESTING_MALLOC_PIN( LRE, magmaDoubleComplex, n2 );
                
                lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
                lapackf77_zlacpy( MagmaUpperLowerStr, &N, &N, h_A, &lda, h_R, &lda );
                
                // ----------
                // Compute eigenvalues, left and right eigenvectors
                magma_zgeev( MagmaVec, MagmaVec,
                             N, h_R, lda, w1,
                             VL, lda, VR, lda,
                             h_work, lwork, rwork, &info );
                if (info != 0)
                    printf("magma_zgeev (case V, V) returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                
                // ----------
                // Compute eigenvalues only
                // These are not exactly equal, and not in the same order, so skip for now.
                // lapackf77_zlacpy( MagmaUpperLowerStr, &N, &N, h_A, &lda, h_R, &lda );
                // magma_zgeev( MagmaNoVec, MagmaNoVec,
                //              N, h_R, lda, w2,
                //              &DUM, 1, &DUM, 1,
                //              h_work, lwork, rwork, &info );
                // if (info != 0)
                //     printf("magma_zgeev (case N, N) returned error %d: %s.\n",
                //            (int) info, magma_strerror( info ));
                // 
                // // Do test 5: W(full) = W(partial, W only)
                // result[4] = 1;
                // for( int j = 0; j < N; ++j )
                //     if ( ! MAGMA_Z_EQUAL( w1[j], w2[j] ))
                //         result[4] = 0;
                
                // ----------
                // Compute eigenvalues and right eigenvectors
                lapackf77_zlacpy( MagmaUpperLowerStr, &N, &N, h_A, &lda, h_R, &lda );
                magma_zgeev( MagmaNoVec, MagmaVec,
                             N, h_R, lda, w2,
                             &DUM, 1, LRE, lda,
                             h_work, lwork, rwork, &info );
                if (info != 0)
                    printf("magma_zgeev (case N, V) returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                
                // Do test 6: W(full) = W(partial, W and VR)
                result[5] = 1;
                for( int j = 0; j < N; ++j )
                    if ( ! MAGMA_Z_EQUAL( w1[j], w2[j] ))
                        result[5] = 0;
                
                // Do test 8: VR(full) = VR(partial, W and VR)
                result[7] = 1;
                for( int j = 0; j < N; ++j )
                    for( int jj = 0; jj < N; ++jj )
                        if ( ! MAGMA_Z_EQUAL( VR[j+jj*lda], LRE[j+jj*lda] ))
                            result[7] = 0;
                
                // ----------
                // Compute eigenvalues and left eigenvectors
                lapackf77_zlacpy( MagmaUpperLowerStr, &N, &N, h_A, &lda, h_R, &lda );
                magma_zgeev( MagmaVec, MagmaNoVec,
                             N, h_R, lda, w2,
                             LRE, lda, &DUM, 1,
                             h_work, lwork, rwork, &info );
                if (info != 0)
                    printf("magma_zgeev (case V, N) returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                
                // Do test 7: W(full) = W(partial, W and VL)
                result[6] = 1;
                for( int j = 0; j < N; ++j )
                    if ( ! MAGMA_Z_EQUAL( w1[j], w2[j] ))
                        result[6] = 0;
                
                // Do test 9: VL(full) = VL(partial, W and VL)
                result[8] = 1;
                for( int j = 0; j < N; ++j )
                    for( int jj = 0; jj < N; ++jj )
                        if ( ! MAGMA_Z_EQUAL( VL[j+jj*lda], LRE[j+jj*lda] ))
                            result[8] = 0;
                
                TESTING_FREE_PIN( LRE );
            }
            
            /* =====================================================================
               Performs operation using LAPACK
               Do this after checks, because it overwrites VL and VR.
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                lapackf77_zgeev( lapack_vec_const(opts.jobvl), lapack_vec_const(opts.jobvr),
                                 &N, h_A, &lda, w2,
                                 VL, &lda, VR, &lda,
                                 h_work, &lwork, rwork, &info );
                cpu_time = magma_wtime() - cpu_time;
                if (info != 0)
                    printf("lapackf77_zgeev returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                
                // check | W_magma - W_lapack | / | W |
                // need to sort eigenvalues first
                blasf77_zcopy( &N, w1, &ione, w1copy, &ione );
                blasf77_zcopy( &N, w2, &ione, w2copy, &ione );
                std::sort( w1copy, &w1copy[N], compare );
                std::sort( w2copy, &w2copy[N], compare );
                
                // adjust sorting to deal with numerical inaccuracy
                // search down w2 for eigenvalue that matches w1's eigenvalue
                for( int j=0; j < N; ++j ) {
                    for( int j2=j; j2 < N; ++j2 ) {
                        magmaDoubleComplex diff = MAGMA_Z_SUB( w1copy[j], w2copy[j2] );
                        double diff2 = magma_dzlapy2( diff ) / max( magma_dzlapy2( w1copy[j] ), tol );
                        if ( diff2 < 100*tol ) {
                            if ( j != j2 ) {
                                std::swap( w2copy[j], w2copy[j2] );
                            }
                            break;
                        }
                    }
                }
                
                blasf77_zaxpy( &N, &c_neg_one, w2copy, &ione, w1copy, &ione );
                error  = cblas_dznrm2( N, w1copy, 1 );
                error /= cblas_dznrm2( N, w2copy, 1 );
                
                printf("%5d   %7.2f          %7.2f          %.2e  %s\n",
                       (int) N, cpu_time, gpu_time,
                       error, (error < tolulp ? "ok" : "failed"));
                status |= ! (error < tolulp);
            }
            else {
                printf("%5d     ---            %7.2f\n",
                       (int) N, gpu_time);
            }
            if ( opts.check ) {
                // -1 indicates test was not run
                if ( result[0] != -1 ) { printf("        | A * VR - VR * W | / ( n |A| ) = %8.2e  %s\n", result[0], (result[0] < tol ? "ok" : "failed")); }
                if ( result[1] != -1 ) { printf("        |  |VR(i)| - 1    |             = %8.2e  %s\n", result[1], (result[1] < tol ? "ok" : "failed")); }
                if ( result[2] != -1 ) { printf("        | A'* VL - VL * W'| / ( n |A| ) = %8.2e  %s\n", result[2], (result[2] < tol ? "ok" : "failed")); }
                if ( result[3] != -1 ) { printf("        |  |VL(i)| - 1    |             = %8.2e  %s\n", result[3], (result[3] < tol ? "ok" : "failed")); }
                if ( result[4] != -1 ) { printf("        W  (full) == W  (partial, W only)           %s\n",         (result[4] == 1. ? "ok" : "failed")); }
                if ( result[5] != -1 ) { printf("        W  (full) == W  (partial, W and VR)         %s\n",         (result[5] == 1. ? "ok" : "failed")); }
                if ( result[6] != -1 ) { printf("        W  (full) == W  (partial, W and VL)         %s\n",         (result[6] == 1. ? "ok" : "failed")); }
                if ( result[7] != -1 ) { printf("        VR (full) == VR (partial, W and VR)         %s\n",         (result[7] == 1. ? "ok" : "failed")); }
                if ( result[8] != -1 ) { printf("        VL (full) == VL (partial, W and VL)         %s\n",         (result[8] == 1. ? "ok" : "failed")); }
                
                int newline = 0;
                if ( result[0] != -1 ) { status |= ! (result[0] < tol);  newline = 1; }
                if ( result[1] != -1 ) { status |= ! (result[1] < tol);  newline = 1; }
                if ( result[2] != -1 ) { status |= ! (result[2] < tol);  newline = 1; }
                if ( result[3] != -1 ) { status |= ! (result[3] < tol);  newline = 1; }
                if ( result[4] != -1 ) { status |= ! (result[4] == 1.);  newline = 1; }
                if ( result[5] != -1 ) { status |= ! (result[5] == 1.);  newline = 1; }
                if ( result[6] != -1 ) { status |= ! (result[6] == 1.);  newline = 1; }
                if ( result[7] != -1 ) { status |= ! (result[7] == 1.);  newline = 1; }
                if ( result[8] != -1 ) { status |= ! (result[8] == 1.);  newline = 1; }
                if ( newline ) {
                    printf( "\n" );
                }
            }
            
            TESTING_FREE_CPU( w1copy );
            TESTING_FREE_CPU( w2copy );
            TESTING_FREE_CPU( w1     );
            TESTING_FREE_CPU( w2     );
            TESTING_FREE_CPU( rwork  );
            TESTING_FREE_CPU( h_A    );
            
            TESTING_FREE_PIN( h_R    );
            TESTING_FREE_PIN( VL     );
            TESTING_FREE_PIN( VR     );
            TESTING_FREE_PIN( h_work );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    TESTING_FINALIZE();
    return status;
}
