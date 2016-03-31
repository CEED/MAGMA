/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       
       @author Mark Gates
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// includes, project
#include "flops.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"

// define ICHI to test with Ichi's version, too
#undef ICHI


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing magma_zher2k_mgpu
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex alpha = MAGMA_Z_MAKE( 1.2345, 4.321 );
    double beta = 3.14159;
    
    real_Double_t    gflops, gpu_perf, cpu_perf, gpu_time, cpu_time;
    double           Anorm, error, work[1];
    magmaDoubleComplex *hA, *hR, *hR2, *hV, *hW;
    magmaDoubleComplex_ptr dV[MagmaMaxGPUs], dW[MagmaMaxGPUs], dA[MagmaMaxGPUs];
    magma_int_t n, k, size, lda, ldda, nb, ngpu, nqueue;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};

    magma_queue_t queues[MagmaMaxGPUs][20], queues0[MagmaMaxGPUs];
    magma_int_t status = 0;
    
    magma_opts opts;
    opts.parse_opts( argc, argv );
    opts.ngpu = abs( opts.ngpu );  // always uses multi-GPU code

    double tol = opts.tolerance * lapackf77_dlamch("E");
    
    ngpu    = opts.ngpu;
    nb      = (opts.nb      > 0 ? opts.nb      : 64);
    nqueue  = (opts.nqueue  > 0 ? opts.nqueue  :  2);
    
    printf( "%% version 1: magmablas_zher2k_mgpu2     %s\n", (opts.version == 1 ? "(enabled)" : ""));
    //printf( "%% version 2: magmablas_zher2k_mgpu_spec %s\n", (opts.version == 2 ? "(enabled)" : ""));
#ifdef ICHI
    printf( "%% version 3: magma_zher2k_mgpu (Ichi)   %s\n", (opts.version == 3 ? "(enabled)" : ""));
#endif
    printf( "\n" );
    
    printf("%% nb %d, ngpu %d, nqueue %d\n", (int) nb, (int) ngpu, (int) nqueue );
    printf("%%   n     k    nb offset  CPU Gflop/s (sec)   GPU Gflop/s (sec)   |R|/(|V|*|W|+|A|)\n");
    printf("%%==================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        n = opts.nsize[itest];
        k = opts.ksize[itest];
        
        for( int offset = 0; offset < n; offset += min(k,nb) ) {
            for( int iter = 0; iter < opts.niter; ++iter ) {
                lda    = n;
                ldda   = magma_roundup( n, opts.align );  // multiple of 32 by default
                gflops = FLOPS_ZHER2K( k, n-offset ) / 1e9;
                
                TESTING_CHECK( magma_zmalloc_cpu( &hA,  lda*n   ));
                TESTING_CHECK( magma_zmalloc_cpu( &hR,  lda*n   ));
                TESTING_CHECK( magma_zmalloc_cpu( &hR2, lda*n   ));
                TESTING_CHECK( magma_zmalloc_cpu( &hV,  lda*k*2 ));
                //TESTING_CHECK( magma_zmalloc_cpu( &hW,  lda*k   ));
                for( int dev = 0; dev < ngpu; ++dev ) {
                    magma_int_t nlocal = ((n / nb) / ngpu + 1) * nb;
                    magma_setdevice( dev );
                    TESTING_CHECK( magma_zmalloc( &dA[dev], ldda*nlocal ));
                    TESTING_CHECK( magma_zmalloc( &dV[dev], ldda*k*2    ));
                    //TESTING_CHECK( magma_zmalloc( &dW[dev], ldda*k      ));
                    for( int i = 0; i < nqueue; ++i ) {
                        magma_queue_create( dev, &queues[dev][i] );
                    }
                    queues0[dev] = queues[dev][0];
                }
                
                size = lda*n;
                lapackf77_zlarnv( &ione, ISEED, &size, hA );
                size = lda*k*2;
                lapackf77_zlarnv( &ione, ISEED, &size, hV );
                hW = hV + lda*k;
                //lapackf77_zlarnv( &ione, ISEED, &size, hW );
                
                /* ====================================================================
                   Performs operation using MAGMA
                   =================================================================== */
                magma_zsetmatrix_1D_col_bcyclic( n, n, hA, lda, dA, ldda, ngpu, nb, queues0 );
                for( int dev = 0; dev < ngpu; ++dev ) {
                    magma_setdevice( dev );
                    dW[dev] = dV[dev] + ldda*k;
                    magma_zsetmatrix( n, k, hV, lda, dV[dev], ldda, opts.queue );
                    magma_zsetmatrix( n, k, hW, lda, dW[dev], ldda, opts.queue );
                }
                
                gpu_time = magma_sync_wtime(0);
                
                if ( opts.version == 1 ) {
                    magmablas_zher2k_mgpu2(
                        MagmaLower, MagmaNoTrans, n-offset, k,
                        alpha, dV, ldda, 0,
                               dW, ldda, 0,
                        beta,  dA, ldda, offset,
                        ngpu, nb, queues, nqueue );
                }
                else if ( opts.version == 2 ) {
                    // see src/obsolete and magmablas/obsolete
                    printf( "magmablas_zher2k_mgpu_spec not compiled\n" );
                    //magmablas_zher2k_mgpu_spec(
                    //    MagmaLower, MagmaNoTrans, n-offset, k,
                    //    alpha, dV, ldda, 0,
                    //           dW, ldda, 0,
                    //    beta,  dA, ldda, offset,
                    //    ngpu, nb, queues, nqueue );
                }
                else {
#ifdef ICHI
                    magma_zher2k_mgpu(
                        ngpu, MagmaLower, MagmaNoTrans, nb, n-offset, k,
                        alpha, dV, ldda,
                               //dW, ldda,
                        beta,  dA, ldda, offset,
                        nqueue, queues );
#endif
                }
                
                gpu_time = magma_sync_wtime(0) - gpu_time;
                gpu_perf = gflops / gpu_time;
                
                // Get dA back to the CPU to compare with the CPU result.
                magma_zgetmatrix_1D_col_bcyclic( n, n, dA, ldda, hR, lda, ngpu, nb, queues0 );
                
                /* =====================================================================
                   Performs operation using LAPACK
                   =================================================================== */
                if ( opts.lapack || opts.check ) {
                    // store ||V||*||W|| + ||A||
                    magma_int_t n_offset = n - offset;
                    Anorm  = lapackf77_zlange("f", &n_offset, &k, hV, &lda, work );
                    Anorm *= lapackf77_zlange("f", &n_offset, &k, hW, &lda, work );
                    Anorm += lapackf77_zlange("f", &n_offset, &n_offset, &hA[offset + offset*lda], &lda, work );
                    
                    cpu_time = magma_wtime();
                    blasf77_zher2k( "Lower", "NoTrans", &n_offset, &k,
                                    &alpha, hV, &lda,
                                            hW, &lda,
                                    &beta,  &hA[offset + offset*lda], &lda );
                    cpu_time = magma_wtime() - cpu_time;
                    cpu_perf = gflops / cpu_time;
                    
                    // compute relative error ||R||/||A||, where R := A_magma - A_lapack = R - A
                    size = lda*n;
                    blasf77_zaxpy( &size, &c_neg_one, hA, &ione, hR, &ione );
                    error = safe_lapackf77_zlanhe("fro", "Lower", &n_offset, &hR[offset + offset*lda], &lda, work)
                          / Anorm;
                    
                    printf( "%5d %5d %5d %5d   %7.1f (%7.4f)   %7.1f (%7.4f)   %8.2e   %s\n",
                            (int) n, (int) k, (int) nb, (int) offset,
                            cpu_perf, cpu_time, gpu_perf, gpu_time,
                            error, (error < tol ? "ok" : "failed"));
                            //, gpu_perf2, gpu_time2, error, error2 );
                    status += ! (error < tol);
                }
                else {
                    printf( "%5d %5d %5d %5d     ---   (  ---  )   %7.1f (%7.4f)     ---\n",
                            (int) n, (int) k, (int) nb, (int) offset,
                            gpu_perf, gpu_time );
                }
                
                magma_free_cpu( hA  );
                magma_free_cpu( hR  );
                magma_free_cpu( hR2 );
                magma_free_cpu( hV  );
                //magma_free_cpu( hW );
                for( int dev = 0; dev < ngpu; ++dev ) {
                    magma_setdevice( dev );
                    magma_free( dA[dev] );
                    magma_free( dV[dev] );
                    //magma_free( dW[dev] );
                    for( int i = 0; i < nqueue; ++i ) {
                        magma_queue_destroy( queues[dev][i] );
                    }
                }
                fflush( stdout );
            }
            if ( opts.niter > 1 ) {
                printf( "\n" );
            }
        } // offset
        printf( "\n" );
    }
    
    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
