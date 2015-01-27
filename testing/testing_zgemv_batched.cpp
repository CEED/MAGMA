/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
       @author Mark Gates
       @author Azzam Haidar
       @author Tingxing Dong
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "testings.h"  // before magma.h, to include cublas_v2
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgemm_batched
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t   gflops, magma_perf, magma_time, cpu_perf, cpu_time;
    double          magma_error, magma_err, Ynorm, work[1];
    magma_int_t M, N, Xm, Ym, lda, ldda;
    magma_int_t sizeA, sizeX, sizeY;
    magma_int_t incx = 1;
    magma_int_t incy = 1;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t status = 0;
    magma_int_t batchCount;

    magmaDoubleComplex *h_A, *h_X, *h_Y, *h_Ymagma;
    magmaDoubleComplex *d_A, *d_X, *d_Y;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex alpha = MAGMA_Z_MAKE(  0.29, -0.86 );
    magmaDoubleComplex beta  = MAGMA_Z_MAKE( -0.48,  0.38 );
    magmaDoubleComplex **A_array = NULL;
    magmaDoubleComplex **X_array = NULL;
    magmaDoubleComplex **Y_array = NULL;


    magma_opts opts;
    parse_opts( argc, argv, &opts );
    batchCount = opts.batchcount;
    opts.lapack |= opts.check; 

    //double tol = opts.tolerance * lapackf77_dlamch("E");

    printf("trans = %s\n", lapack_trans_const(opts.transA) );

    printf("BatchCount    M     N     MAGMA Gflop/s (ms)  CPU Gflop/s (ms)  MAGMA error\n");

    printf("===================================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            lda    = ((M+31)/32)*32;
            gflops = FLOPS_ZGEMV( M, N ) / 1e9 * batchCount;

            if ( opts.transA == MagmaNoTrans ) {
                Xm = N;
                Ym = M;
            } else {
                Xm = M;
                Ym = N;
            }

            sizeA = lda*N*batchCount;
            sizeX = incx*Xm*batchCount;
            sizeY = incy*Ym*batchCount;

            ldda = ((lda+31)/32)*32;

            TESTING_MALLOC_CPU( h_A,  magmaDoubleComplex, sizeA );
            TESTING_MALLOC_CPU( h_X,  magmaDoubleComplex, sizeX );
            TESTING_MALLOC_CPU( h_Y,  magmaDoubleComplex, sizeY  );
            TESTING_MALLOC_CPU( h_Ymagma,  magmaDoubleComplex, sizeY  );


            TESTING_MALLOC_DEV( d_A, magmaDoubleComplex, ldda*N*batchCount );
            TESTING_MALLOC_DEV( d_X, magmaDoubleComplex, sizeX );
            TESTING_MALLOC_DEV( d_Y, magmaDoubleComplex, sizeY );

            magma_malloc((void**)&A_array, batchCount * sizeof(*A_array));
            magma_malloc((void**)&X_array, batchCount * sizeof(*X_array));
            magma_malloc((void**)&Y_array, batchCount * sizeof(*Y_array));

            /* Initialize the matrices */
            lapackf77_zlarnv( &ione, ISEED, &sizeA, h_A );
            lapackf77_zlarnv( &ione, ISEED, &sizeX, h_X );
            lapackf77_zlarnv( &ione, ISEED, &sizeY, h_Y );
            
            /* =====================================================================
               Performs operation using MAGMABLAS
               =================================================================== */
            magma_zsetmatrix( M, N*batchCount, h_A, lda, d_A, ldda );
            magma_zsetvector( Xm*batchCount, h_X, incx, d_X, incx );
            magma_zsetvector( Ym*batchCount, h_Y, incy, d_Y, incy );
            
            zset_pointer(A_array, d_A, ldda, 0, 0, ldda*N, batchCount, magma_stream);
            zset_pointer(X_array, d_X, 1, 0, 0, incx*Xm, batchCount, magma_stream);
            zset_pointer(Y_array, d_Y, 1, 0, 0, incy*Ym, batchCount, magma_stream);

            magma_time = magma_sync_wtime( NULL );
            magmablas_zgemv_batched(opts.transA, M, N,
                             alpha, A_array, ldda,
                                    X_array, incx,
                             beta,  Y_array, incy, batchCount, magma_stream);
            magma_time = magma_sync_wtime( NULL ) - magma_time;
            magma_perf = gflops / magma_time;            
            magma_zgetvector( Ym*batchCount, d_Y, incy, h_Ymagma, incy );
                      
            /* =====================================================================
               Performs operation using CPU BLAS
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                for(int i=0; i<batchCount; i++)
                {
                   blasf77_zgemv(
                               lapack_trans_const(opts.transA), 
                               &M, &N,
                               &alpha, h_A + i*lda*N, &lda,
                                       h_X + i*Xm, &incx,
                               &beta,  h_Y + i*Ym, &incy );
                }
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
            }
            
            /* =====================================================================
               Check the result
               =================================================================== */
            if ( opts.lapack ) {
                // compute relative error for both magma  relative to lapack,
                // |C_magma - C_lapack| / |C_lapack|
                magma_error = 0.0;

                for(int s=0; s<batchCount; s++)
                {

                    Ynorm = lapackf77_zlange( "M", &M, &ione, h_Y + s*Ym, &incy, work );

                    blasf77_zaxpy( &Ym, &c_neg_one, h_Y + s*Ym, &ione, h_Ymagma + s*Ym, &ione );
                    magma_err = lapackf77_zlange( "M", &M, &ione, h_Ymagma + s*Ym, &incy, work ) / Ynorm; 

                    if ( isnan(magma_err) || isinf(magma_err) ) {
                      magma_error = magma_err;
                      break;
                    }
                    magma_error = max(fabs(magma_err), magma_error); 

                }

                    printf("%10d %5d %5d  %7.2f (%7.2f)    %7.2f (%7.2f)   %8.2e  \n",
                       (int) batchCount, (int) M, (int) N, 
                       magma_perf,  1000.*magma_time,
                       cpu_perf,    1000.*cpu_time,
                       magma_error);
            }
            else {

                    printf("%10d %5d %5d  %7.2f (%7.2f)    ---   (  ---  )    ---\n",
                       (int) batchCount, (int) M, (int) N, 
                       magma_perf,  1000.*magma_time);
            }
            
            TESTING_FREE_CPU( h_A  );
            TESTING_FREE_CPU( h_X  );
            TESTING_FREE_CPU( h_Y  );
            TESTING_FREE_CPU( h_Ymagma  );


            TESTING_FREE_DEV( d_A );
            TESTING_FREE_DEV( d_X );
            TESTING_FREE_DEV( d_Y );
            TESTING_FREE_DEV( A_array );
            TESTING_FREE_DEV( X_array );
            TESTING_FREE_DEV( Y_array );

            
            fflush( stdout);

        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    TESTING_FINALIZE();
    return status;
}
