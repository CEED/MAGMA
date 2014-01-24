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


// Initialize matrix to random.
// Having this in separate function ensures the same ISEED is always used,
// so we can re-generate the identical matrix.
void init_matrix( int m, int n, magmaDoubleComplex *h_A, magma_int_t lda )
{
    magma_int_t ione = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t n2 = lda*n;
    lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
}


// On input, A and ipiv is LU factorization of A. On output, A is overwritten.
// Requires m == n.
// Uses init_matrix() to re-generate original A as needed.
// Generates random RHS b and solves Ax=b.
// Returns residual, |Ax - b| / (n |A| |x|).
double get_residual(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    magma_int_t *ipiv )
{
    if ( m != n ) {
        printf( "\nERROR: residual check defined only for square matrices\n" );
        return -1;
    }
    
    const magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    const magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    const magma_int_t ione = 1;
    
    // this seed should be DIFFERENT than used in init_matrix
    // (else x is column of A, so residual can be exactly zero)
    magma_int_t ISEED[4] = {0,0,0,2};
    magma_int_t info = 0;
    magmaDoubleComplex *x, *b;
    
    // initialize RHS
    TESTING_MALLOC( x, magmaDoubleComplex, n );
    TESTING_MALLOC( b, magmaDoubleComplex, n );
    lapackf77_zlarnv( &ione, ISEED, &n, b );
    blasf77_zcopy( &n, b, &ione, x, &ione );
    
    // solve Ax = b
    lapackf77_zgetrs( "Notrans", &n, &ione, A, &lda, ipiv, x, &n, &info );
    if (info != 0)
        printf("lapackf77_zgetrs returned error %d: %s.\n",
               (int) info, magma_strerror( info ));
    
    // reset to original A
    init_matrix( m, n, A, lda );
    
    // compute r = Ax - b, saved in b
    blasf77_zgemv( "Notrans", &m, &n, &c_one, A, &lda, x, &ione, &c_neg_one, b, &ione );
    
    // compute residual |Ax - b| / (n*|A|*|x|)
    double norm_x, norm_A, norm_r, work[1];
    norm_A = lapackf77_zlange( "F", &m, &n, A, &lda, work );
    norm_r = lapackf77_zlange( "F", &n, &ione, b, &n, work );
    norm_x = lapackf77_zlange( "F", &n, &ione, x, &n, work );
    
    //printf( "r=\n" ); magma_zprint( 1, n, b, 1 );
    
    TESTING_FREE( x );
    TESTING_FREE( b );
    
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
    
    TESTING_MALLOC( A, magmaDoubleComplex, lda*N    );
    TESTING_MALLOC( L, magmaDoubleComplex, M*min_mn );
    TESTING_MALLOC( U, magmaDoubleComplex, min_mn*N );
    memset( L, 0, M*min_mn*sizeof(magmaDoubleComplex) );
    memset( U, 0, min_mn*N*sizeof(magmaDoubleComplex) );

    // set to original A
    init_matrix( M, N, A, lda );
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

    TESTING_FREE(A);
    TESTING_FREE(L);
    TESTING_FREE(U);

    return residual / (matnorm * N);
}


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgetrf
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t   gflops, gpu_perf, gpu_time, cpu_perf=0, cpu_time=0, sparse_perf=0, sparse_time=0;
    double          error;
    magmaDoubleComplex *h_A;
    magmaDoubleComplex *d_A;
    magma_int_t     *ipiv;
    magma_int_t M, N, n2, lda, ldda, info, min_mn;
    magma_int_t status   = 0;

    magma_opts opts;
    parse_opts( argc, argv, &opts );

    //double tol = opts.tolerance * lapackf77_dlamch("E");
    double tol = 0.000000001;
    /*
    if ( opts.check == 2 ) {
        printf("    M     N   CPU GFlop/s (sec)   MAGMA GFlop/s (sec)   sparse GFlop/s (sec)   |Ax-b|/(N*|A|*|x|)\n");
    }
    else {
        printf("    M     N   CPU GFlop/s (sec)   MAGMA GFlop/s (sec)   sparse GFlop/s (sec)   |Ax-b|/(N*|A|*|x|)\n");
    }
*/
            for( int z=16; (z<=1024); z=z*20000 ){
    printf("========================================================================================================================================\n");

        printf("#    size     blocksize  |  blockinfo alloc val col cpucopy getrf  rup  lup  trsm gemm  |  MAGMA GFlop/s (sec)   sparse GFlop/s (sec)\n");
        for( int iter = 0; iter < opts.niter; ++iter ) {
    for( int i = 1024*17; i < 1024*20; i+=1024 ) {


            int z=i;
            printf("%5d %5d  |", (int) i, (int) z );
            M = i;//opts.msize[i];
            N = i;//opts.nsize[i];
            min_mn = min(M, N);
            lda    = M;
            n2     = lda*N;
            ldda   = ((M+31)/32)*32;
            gflops = FLOPS_ZGETRF( M, N ) / 1e9;
            
            TESTING_MALLOC(    ipiv, magma_int_t,     min_mn );
            TESTING_MALLOC(    h_A,  magmaDoubleComplex, n2     );
            TESTING_DEVALLOC(  d_A,  magmaDoubleComplex, ldda*N );
            
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            init_matrix( M, N, h_A, lda );
            magma_zsetmatrix( M, N, h_A, lda, d_A, ldda );
            
            gpu_time = magma_wtime();
            magma_zgetrf_gpu( M, N, d_A, ldda, ipiv, &info);
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0)
                printf("magma_zgetrf_gpu returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));


            /* ====================================================================
               Performs operation using MAGMA-sparse using BCSR format
               =================================================================== */

            init_matrix( M, N, h_A, lda );
            // sparse framework
            magma_z_sparse_matrix A, B, C;
            A.num_rows = M;
            A.num_cols = N;
            A.nnz = M*N;
            A.val = h_A;
            A.storage_type = Magma_DENSE;
            A.memory_location = Magma_CPU;
            magma_z_mconvert( A, &B, Magma_DENSE, Magma_CSR);
            magma_z_mfree( &A );
            for(int defaultsize=512; defaultsize>=511; defaultsize--){
                if( A.num_rows%defaultsize == 0 )
                    A.blocksize = defaultsize;
            }
            A.blocksize = z;//M;//64;//M/17;
            magma_z_mconvert( B, &A, Magma_CSR, Magma_BCSR);
            magma_z_mfree( &B );
            magma_z_mtransfer( A, &B, Magma_CPU, Magma_DEV);
            magma_z_mfree( &A );
            sparse_time = magma_wtime();
            magma_zbcsrlu( B, &C, ipiv);
            sparse_time = magma_wtime() - sparse_time;
            sparse_perf = gflops / sparse_time;
            if (info != 0)
                printf("magmasparse_zgetrf_gpu returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            magma_z_mfree( &B );
            magma_z_mfree( &C );
            
            /* =====================================================================
               Print performance
               =================================================================== */

            printf("  %7.2f (%7.2f)   %7.2f (%7.2f)\n",
                   gpu_perf, gpu_time, sparse_perf, sparse_time );

            TESTING_FREE( ipiv );
            TESTING_DEVFREE( d_A );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
        }//z
        printf("\n");
    }

    TESTING_FINALIZE();
    return status;
}
