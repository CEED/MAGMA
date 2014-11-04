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
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

/* ================================================================================================== */
extern "C" magma_int_t
magma_zhetrf_cpu(magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda,
                 magma_int_t *ipiv, magmaDoubleComplex *work, magma_int_t lwork, magma_int_t *info);
extern "C" magma_int_t
magma_zhetrf_hybrid(magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda,
             magma_int_t *ipiv, magmaDoubleComplex *work, magma_int_t lwork, magma_int_t *info);
extern "C" magma_int_t
magma_zhetrf(magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda,
                 magma_int_t *ipiv, magma_int_t *info);
extern "C" magma_int_t
magma_zhetrf_gpu_row(magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda,
                     magma_int_t *ipiv, magmaDoubleComplex *work, magma_int_t lwork, magma_int_t *info);
extern "C" magma_int_t
magma_zhetrf_nopiv(magma_uplo_t uplo, magma_int_t n,
                   cuDoubleComplex *a, magma_int_t lda, magma_int_t *info);

void
zhetrf_(char*, int*, magmaDoubleComplex*, int*, int*, magmaDoubleComplex*, int*, int*);
void 
zhetrs_(char*, int*, int*, magmaDoubleComplex*, int*, int*, magmaDoubleComplex*, int*, int*);
#ifdef __cplusplus
extern "C" {
#endif
magma_int_t 
magma_get_zhetrf_nb( magma_int_t m );
#ifdef __cplusplus
}
#endif
/* ================================================================================================== */

// Initialize matrix to random.
// Having this in separate function ensures the same ISEED is always used,
// so we can re-generate the identical matrix.
void init_matrix( int nopiv, int m, int n, magmaDoubleComplex *h_A, magma_int_t lda )
{
    magma_int_t ione = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t n2 = lda*n;
    //double *A = (double*)malloc(n2*sizeof(double));
    //lapackf77_dlarnv( &ione, ISEED, &n2, A );
    //for (int i=0; i<n; i++) for (int j=0; j<=i; j++) h_A[j+i*lda] = MAGMA_Z_MAKE(A[j+i*lda],0.0);
    //free(A);
    //
    lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
    // symmetrize
    for (int i=0; i<n; i++) for (int j=0; j<i; j++) h_A[i+j*lda] = MAGMA_Z_CNJG(h_A[j+i*lda]);
    if (nopiv) for (int i=0; i<n; i++) h_A[i+i*lda] = MAGMA_Z_MAKE(MAGMA_Z_REAL(h_A[i+i*lda]) + n, 0.0);
    else       for (int i=0; i<n; i++) h_A[i+i*lda] = MAGMA_Z_MAKE(MAGMA_Z_REAL(h_A[i+i*lda]), 0.0);
}


// On input, A and ipiv is LU factorization of A. On output, A is overwritten.
// Requires m == n.
// Uses init_matrix() to re-generate original A as needed.
// Generates random RHS b and solves Ax=b.
// Returns residual, |Ax - b| / (n |A| |x|).
double get_residual(
    int nopiv, magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    magma_int_t *ipiv )
{
    const magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    const magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    const magma_int_t ione = 1;
    magma_int_t upper = (uplo == MagmaUpper);
    
    // this seed should be DIFFERENT than used in init_matrix
    // (else x is column of A, so residual can be exactly zero)
    magma_int_t ISEED[4] = {0,0,0,2};
    magma_int_t info = 0;
    magmaDoubleComplex *x, *b;
    
    // initialize RHS
    TESTING_MALLOC_CPU( x, magmaDoubleComplex, n );
    TESTING_MALLOC_CPU( b, magmaDoubleComplex, n );
    lapackf77_zlarnv( &ione, ISEED, &n, b );
    blasf77_zcopy( &n, b, &ione, x, &ione );
    
    // solve Ax = b
    if (nopiv) {
        if (upper) {
            blasf77_ztrsm( "L", "U", "C", "U", &n, &ione, &c_one,
                           A, &lda, x, &n );
            for (int i=0; i<n; i++) x[i] = MAGMA_Z_DIV( x[i], A[i+i*lda] );
            blasf77_ztrsm( "L", "U", "N", "U", &n, &ione, &c_one,
                           A, &lda, x, &n );
        } else {
            blasf77_ztrsm( "L", "L", "N", "U", &n, &ione, &c_one,
                           A, &lda, x, &n );
            for (int i=0; i<n; i++) x[i] = MAGMA_Z_DIV( x[i], A[i+i*lda] );
            blasf77_ztrsm( "L", "L", "C", "U", &n, &ione, &c_one,
                           A, &lda, x, &n );
        }
    }else {
        zhetrs_( (upper ? MagmaUpperStr: MagmaLowerStr), &n, &ione, A, &lda, ipiv, x, &n, &info );
    }
    if (info != 0)
        printf("lapackf77_zhetrs returned error %d: %s.\n",
               (int) info, magma_strerror( info ));
    // reset to original A
    init_matrix( nopiv, n, n, A, lda );
    
    // compute r = Ax - b, saved in b
    blasf77_zgemv( "Notrans", &n, &n, &c_one, A, &lda, x, &ione, &c_neg_one, b, &ione );
    
    // compute residual |Ax - b| / (n*|A|*|x|)
    double norm_x, norm_A, norm_r, work[1];
    norm_A = lapackf77_zlange( "F", &n, &n, A, &lda, work );
    norm_r = lapackf77_zlange( "F", &n, &ione, b, &n, work );
    norm_x = lapackf77_zlange( "F", &n, &ione, x, &n, work );
    
    //printf( "r=\n" ); magma_zprint( 1, n, b, 1 );
    
    TESTING_FREE_CPU( x );
    TESTING_FREE_CPU( b );
    
    //printf( "r=%.2e, A=%.2e, x=%.2e, n=%d\n", norm_r, norm_A, norm_x, n );
    return norm_r / (n * norm_A * norm_x);
}


// On input, LU and ipiv is LU factorization of A. On output, LU is overwritten.
// Works for any m, n.
// Uses init_matrix() to re-generate original A as needed.
// Returns error in factorization, |PA - LU| / (n |A|)
// This allocates 3 more matrices to store A, L, and U.
double get_LU_error(magma_int_t M, magma_int_t N,
                    magmaDoubleComplex *LU, magma_int_t lda,
                    magma_int_t *ipiv)
{
    magma_int_t min_mn = min(M,N);
    magma_int_t ione   = 1;
    magma_int_t i, j;
    magmaDoubleComplex alpha = MAGMA_Z_ONE;
    magmaDoubleComplex beta  = MAGMA_Z_ZERO;
    magmaDoubleComplex *A, *L, *U;
    double work[1], matnorm, residual;
    
    TESTING_MALLOC_CPU( A, magmaDoubleComplex, lda*N    );
    TESTING_MALLOC_CPU( L, magmaDoubleComplex, M*min_mn );
    TESTING_MALLOC_CPU( U, magmaDoubleComplex, min_mn*N );
    memset( L, 0, M*min_mn*sizeof(magmaDoubleComplex) );
    memset( U, 0, min_mn*N*sizeof(magmaDoubleComplex) );

    // set to original A
    init_matrix( 0, M, N, A, lda );
    lapackf77_zlaswp( &N, A, &lda, &ione, &min_mn, ipiv, &ione);
    
    // copy LU to L and U, and set diagonal to 1
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

    TESTING_FREE_CPU( A );
    TESTING_FREE_CPU( L );
    TESTING_FREE_CPU( U );

    return residual / (matnorm * N);
}


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgetrf
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t   gflops, gpu_perf, gpu_time, cpu_perf=0, cpu_time=0;
    double          error, error_lapack = 0.0;
    magmaDoubleComplex *h_A, *work, temp;
    magma_int_t     *ipiv;
    magma_int_t     N, n2, lda, lwork, info;
    magma_int_t     status = 0;
    magma_int_t     cpu = 0, gpu = 0, nopiv = 0, row = 0;
    
    magma_opts opts;
    parse_opts( argc, argv, &opts );
    switch (opts.version) {
        case 1:
            gpu = 1;
            break;
        case 2:
            nopiv = 1;
            break;
        //case 3:
        //    cpu = 1;
        //    break;
        //case 4:
        //    row = 1;
        //    break;
        default:
            printf( " version = %d not supported\n\n",opts.version);
            return 0;
    }
    magma_uplo_t uplo = opts.uplo;

    if (nopiv)
        printf( "\n No-piv: Hybrid-version (A is SPD)" );
    else if (cpu)
        printf( "\n Bunch-Kauffman: CPU-only version" );
    else if (gpu)
        printf( "\n Bunch-Kauffman: GPU-only version" );
    else if (row)
        printf( "\n Bunch-Kauffman: GPU-only version (row-major)" );
    else
        printf( " hybrid CPU-GPU version" );
    printf( " (%s)\n",(uplo == MagmaUpper ? "upper" : "lower") );
    printf( " (--version: 1 = Bunch-Kauffman (GPU), 2 = No-piv (hybrid))\n\n" );
    
    magma_int_t upper = (uplo == MagmaUpper);
    double tol = opts.tolerance * lapackf77_dlamch("E");

    printf("ngpu %d\n", (int) opts.ngpu );
    if ( opts.check == 2 ) {
        printf("    M     N   CPU GFlop/s (sec)   GPU GFlop/s (sec)   |Ax-b|/(N*|A|*|x|)\n");
    }
    else {
        printf("    M     N   CPU GFlop/s (sec)   GPU GFlop/s (sec)   |PA-LU|/(N*|A|)\n");
    }
    printf("=========================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            lda    = N;
            n2     = lda*N;
            gflops = FLOPS_ZGETRF( N, N ) / 2e9;
            
            TESTING_MALLOC_CPU( ipiv, magma_int_t, N );
            TESTING_MALLOC_PIN( h_A,  magmaDoubleComplex, n2 );
            
            lwork = -1;
            lapackf77_zhetrf((upper ? MagmaUpperStr: MagmaLowerStr), &N, h_A, &lda, ipiv, &temp, &lwork, &info);
            lwork = max(N*(1+magma_get_zhetrf_nb(N)), (int)MAGMA_Z_REAL(temp));
            TESTING_MALLOC_PIN( work, magmaDoubleComplex, lwork );

            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                init_matrix( nopiv, N, N, h_A, lda );
                cpu_time = magma_wtime();
                lapackf77_zhetrf((upper ? MagmaUpperStr: MagmaLowerStr), &N, h_A, &lda, ipiv, work, &lwork, &info);
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0)
                    printf("lapackf77_zgetrf returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                error_lapack = get_residual( nopiv, uplo, N, h_A, lda, ipiv );
            }
           
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            init_matrix( nopiv, N, N, h_A, lda );

            gpu_time = magma_wtime();
            if (nopiv) {
                magma_setdevice(0);
                magma_zhetrf_nopiv( uplo, N, h_A, lda, &info);
            } else if (cpu) {
                //magma_zhetrf_cpu( uplo, N, h_A, lda, ipiv, work, lwork, &info);
            } else if (gpu) {
                magma_setdevice(0);
                magma_zhetrf( uplo, N, h_A, lda, ipiv, &info);
            } else if (row) {
                magma_setdevice(0);
                //magma_zhetrf_gpu_row( uplo, N, h_A, lda, ipiv, work, lwork, &info);
            } else {
                magma_setdevice(0);
                //magma_zhetrf_hybrid( uplo, N, h_A, lda, ipiv, work, lwork, &info);
            }
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0)
                printf("magma_zhetrf returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            /* =====================================================================
               Check the factorization
               =================================================================== */
            if ( opts.lapack ) {
                printf("%5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)",
                       (int) N, (int) N, cpu_perf, cpu_time, gpu_perf, gpu_time );
            }
            else {
                printf("%5d %5d     ---   (  ---  )   %7.2f (%7.2f)",
                       (int) N, (int) N, gpu_perf, gpu_time );
            }
            if ( opts.check == 2 ) {
                error = get_residual( nopiv, uplo, N, h_A, lda, ipiv );
                printf("   %8.2e   %s", error, (error < tol ? "ok" : "failed"));
                if (opts.lapack)
                    printf(" (lapack rel.res. = %8.2e)", error_lapack);
                printf("\n");
                status += ! (error < tol);
            }
            else if ( opts.check ) {
                printf( " not yet..\n" ); exit(0);
                //error = get_LU_error( M, N, h_A, lda, ipiv );
                //printf("   %8.2e   %s\n", error, (error < tol ? "ok" : "failed"));
                //status += ! (error < tol);
            }
            else {
                printf("     ---   \n");
            }
            
            TESTING_FREE_CPU( ipiv );
            TESTING_FREE_PIN( work );
            TESTING_FREE_PIN( h_A  );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    TESTING_FINALIZE();
    return status;
}
