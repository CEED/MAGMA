/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> s d c
       
       @author Mark Gates
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <assert.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

//#include "trace.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing magma_zhemm_mgpu
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    cuDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    cuDoubleComplex calpha    = MAGMA_Z_MAKE( 3.456, 5.678 );
    cuDoubleComplex cbeta     = MAGMA_Z_MAKE( 1.234, 2.456 );
    
    real_Double_t    gflops, gpu_perf=0., cpu_perf=0., gpu_time=0., cpu_time=0.;
    real_Double_t    gpu_perf2=0., gpu_time2=0.;
    double           error=0., errorbis=0., work[1];
    cuDoubleComplex *hA, *hX, *hB, *hR;
    cuDoubleComplex *dA[MagmaMaxGPUs], *dX[MagmaMaxGPUs], *dB[MagmaMaxGPUs], *dwork[MagmaMaxGPUs], *hwork[MagmaMaxGPUs+1];
    cuDoubleComplex *dA2;
    
    /* Matrix size */
    magma_int_t m, size, lda, ldda;
    const int MAXTESTS = 10;
    // sizes are 1024*N - 32
    magma_int_t msize[MAXTESTS] = { 1024, 2048, 3072, 4032, 5184, 6016, 7040, 8064, 9088, 10112 };
    magma_int_t n       = 64;
    magma_int_t nb      = 64;
    int nstream = 3;
    int count   = 3;
    int ngpu    = magma_num_gpus();
    int ver =225;
    magma_int_t ione     = 1;
    magma_int_t iseed[4] = {0,0,0,1};
        
    printf( "Usage: %s -M m -N n -nb nb -nstream nstream -ngpu ngpu -count count -c\n"
            "    -M can be repeated %d times.\n"
            "    -ngpu or setting $MAGMA_NUM_GPUS sets number of GPUs to use.\n"
            "    -c or setting $MAGMA_TESTINGS_CHECK runs LAPACK and checks result.\n\n",
            argv[0], MAXTESTS );

    int checkres = (getenv("MAGMA_TESTINGS_CHECK") != NULL);

    int ntest = 0;
    int mmax = 0;
    for( int i = 1; i < argc; i++ ) {
        if ( strcmp("-M", argv[i]) == 0 && i+1 < argc ) {
            magma_assert( ntest < MAXTESTS, "error: -M repeated more than maximum %d tests\n", MAXTESTS );
            msize[ntest] = atoi( argv[++i] );
            magma_assert( msize[ntest] > 0, "error: -M %s is invalid; must be > 0.\n", argv[i] );
            mmax = max( mmax, msize[ntest] );
            ntest++;
        }
        else if ( strcmp("-N", argv[i]) == 0 && i+1 < argc ) {
            n = atoi( argv[++i] );
            magma_assert( n > 0, "error: -N %s is invalid; must be > 0.\n", argv[i] );
        }
        else if ( strcmp("-NB", argv[i]) == 0 && i+1 < argc ) {
            nb = atoi( argv[++i] );
            magma_assert( nb > 0, "error: -nb %s is invalid; must be > 0.\n", argv[i] );
        }
        else if ( strcmp("-v", argv[i]) == 0 && i+1 < argc ) {
            ver = atoi( argv[++i] );
            magma_assert( nb > 0, "error: -nb %s is invalid; must be > 0.\n", argv[i] );
        }
        else if ( strcmp("-count", argv[i]) == 0 && i+1 < argc ) {
            count = atoi( argv[++i] );
            magma_assert( count > 0, "error: -count %s is invalid; must be > 0.\n", argv[i] );
        }
        else if ( strcmp("-nstream", argv[i]) == 0 && i+1 < argc ) {
            nstream = atoi( argv[++i] );
            magma_assert( nstream > 0 && nstream <= 20,
                    "error: -nstream %s is invalid; must be > 0 and <= 20.\n", argv[i] );
        }
        else if ( strcmp("-ngpu", argv[i]) == 0 && i+1 < argc ) {
            ngpu = atoi( argv[++i] );
            magma_assert( ngpu > 0, "error: -ngpu %s is invalid; must be > 0.\n", argv[i] );
        }
        else if ( strcmp("-c", argv[i]) == 0 ) {
            checkres = true;
        }
        else {
            printf( "invalid argument: %s\n", argv[i] );
            //exit(1);
        }
    }
    if ( ntest == 0 ) {
        ntest = MAXTESTS;
        mmax = msize[ntest-1];
    }
    m = mmax;
    assert( m > 0 && n > 0 );
    
    // allocate memory for largest problem
    lda  = m;
    ldda = ((m + 31)/32)*32;

    
    magma_int_t gnode[MagmaMaxGPUs][MagmaMaxGPUs+2];
    magma_int_t nbcmplx=0;
    magma_buildconnection_mgpu(gnode, &nbcmplx,  ngpu);
    printf(" Initializin communication pattern.... GPU-ncmplx %d\n\n" , (int) nbcmplx);

    for (int i=0;i<nbcmplx;++i)
    {
        int myngpu =gnode[i][MagmaMaxGPUs];
        printf("cmplx %d has %d gpu ", i, myngpu);
        for(int j=0;j<myngpu;++j)
            printf("  %d", (int) gnode[i][j]);
        printf("\n");
    }

    TESTING_MALLOC( hA, cuDoubleComplex, lda*m );
    TESTING_MALLOC( hX, cuDoubleComplex, lda*n );
    TESTING_MALLOC( hB, cuDoubleComplex, lda*n );
    TESTING_HOSTALLOC( hR, cuDoubleComplex, lda*n );

    magma_int_t  nbevents =2;
    cudaStream_t streams[MagmaMaxGPUs][20];
    cudaEvent_t  redevents[MagmaMaxGPUs][20];
    cudaEvent_t  redevents2[MagmaMaxGPUs][MagmaMaxGPUs*MagmaMaxGPUs+10];
    for( int d = 0; d < ngpu; ++d ) {
        magma_int_t mlocal = ((m / nb) / ngpu + 1) * nb;
        cudaSetDevice( d );
        TESTING_DEVALLOC( dA[d], cuDoubleComplex, ldda*mlocal );
        TESTING_DEVALLOC( dX[d], cuDoubleComplex, ldda*n      );
        TESTING_DEVALLOC( dB[d], cuDoubleComplex, ldda*n      );
        TESTING_DEVALLOC( dwork[d], cuDoubleComplex, ldda*n*3  );
        TESTING_HOSTALLOC( hwork[d], cuDoubleComplex, lda*n );
        for( magma_int_t i = 0; i < nstream; ++i ) {
            cudaStreamCreate( &streams[d][i] );
        }
        for( magma_int_t i = 0; i < nbevents; ++i ) {
            cudaEventCreateWithFlags(&redevents[d][i], cudaEventDisableTiming);
            cudaEventCreateWithFlags(&redevents2[d][i], cudaEventDisableTiming);
        }
    }
    TESTING_HOSTALLOC( hwork[ngpu], cuDoubleComplex, lda*n );



    if ( checkres ) {
    cudaSetDevice( 0 );
    TESTING_DEVALLOC( dA2, cuDoubleComplex, ldda*m );
    }
    
    printf( "nb %d, ngpu %d, nstream %d version %d \n", (int) nb, ngpu, nstream, ver );
    printf("    m     n    nb offset  CPU GFlop/s (sec)   GPU GFlop/s (sec)   CUBLAS hemm (sec)   ||R|| / ||A||*||X||\n");
    printf("=========================================================================================================\n");

//    for( int nb = 64; nb < 256; nb+=64 ) {
//            if(nb==192) nb=256;
//            printf("\n\n\n\n\n");

    magma_int_t nbtime=0;
    for( int i = 0; i < ntest; ++i ) {
    for( int offst = 0; offst < 1; offst += min(n,nb) ) {
    for( int j = 0; j < count; ++j ) {
        m = msize[i];
        assert( m > 0 && n > 0 );
        magma_int_t msiz = m-offst;

        lda  = m;
        ldda = ((m + 31)/32)*32;
        gflops = FLOPS_ZHEMM( MagmaLeft, (double)msiz, (double)n ) / 1e9;

        size = lda*m;
        lapackf77_zlarnv( &ione, iseed, &size, hA );
        // make diagonal real
        for( int i = 0; i < m; ++i ) {
            hA[i + i*lda] = MAGMA_Z_MAKE( MAGMA_Z_REAL( hA[i+i*lda] ), 0. );
        }
        size = lda*n;
        lapackf77_zlarnv( &ione, iseed, &size, hX );
        lapackf77_zlarnv( &ione, iseed, &size, hB );
        lapackf77_zlacpy( "Full", &m, &n, hB, &lda, hR, &lda );
        
        /* ====================================================================
           Performs operation using MAGMA
           =================================================================== */
        magma_zsetmatrix_1D_col_bcyclic( m, m, hA, lda, dA, ldda, ngpu, nb );
        for( int d = 0; d < ngpu; ++d ) {
            cudaSetDevice( d );
            //magmablasSetKernelStream( streams[ d ][  0 ] );
            magma_zsetmatrix( m, n, hX, lda, dX[d], ldda );
            //if(d==0) magma_zsetmatrix( m, n, hB, lda, dB[d], ldda );// this is wrong coz when offset !=0 the gpu who do the beta*C may be not 0 so this should be related to stdev(starting device who own i=0 first col)
            magma_zsetmatrix( m, n, hB, lda, dB[d], ldda );
        }
    



        //memset(hR, 0, lda*n*sizeof(cuDoubleComplex));

        //trace_init( 1, ngpu, nstream, (cudaStream_t*) streams );

        //magma_int_t offst =0;//nb;

        //cudaDeviceSynchronize();
        gpu_time = magma_wtime();
        // 1GPU version light
        /*
        magmablas_zhemm_1gpu(
            MagmaLeft, MagmaLower, msiz, n,
            calpha,    dA, ldda, offst,
                       dX, ldda,
            cbeta,     dB, ldda, hR, lda,
            ngpu, nb, streams, nstream );
        */
        // multi gpu version
        
        if (ver==21) {
            // TODO: not available?
            //magmablas_zhemm_mgpu(
            //    MagmaLeft, MagmaLower, msiz, n,
            //    calpha,    dA, ldda, offst,
            //               dX, ldda,
            //    cbeta,     dB, ldda, dwork, ldda, hR, lda, hwork, lda,
            //    ngpu, nb, streams, nstream, redevents, nbevents );
        }
        else {
            magmablas_zhemm_mgpu_com(
                MagmaLeft, MagmaLower, msiz, n,
                calpha,    dA, ldda, offst,
                           dX, ldda,
                cbeta,     dB, ldda, dwork, ldda, hR, lda, hwork, lda,
                ngpu, nb, streams, nstream, redevents2, nbevents, gnode, nbcmplx);
        }
       
        cudaDeviceSynchronize();
        gpu_time = magma_wtime() - gpu_time;
        gpu_perf = gflops / gpu_time;
            
        #ifdef TRACING
        char buf[80];
        snprintf( buf, sizeof(buf), "zhemm-m%d-n%d-nb%d-stream%d-ngpu%d-run%d.svg",
                  (int) m, (int) n, (int) nb, (int) nstream, (int) ngpu, (int) j );
        trace_finalize( buf, "trace.css" );
        #endif
        
        /* ====================================================================
           Performs operation using CUBLAS
           =================================================================== */
        if (( checkres )&&(nbtime==0)) {
            nbtime =1;
            magma_setdevice( 0 );
            magmablasSetKernelStream(  0  );
            magma_zsetmatrix( m, m, hA, lda, dA2, ldda );
            magma_zsetmatrix( m, n, hX, lda, dX[0], ldda );
            magma_zsetmatrix( m, n, hB, lda, dwork[0], ldda );
            
            cudaDeviceSynchronize();
            gpu_time2 = magma_wtime();
            magma_zhemm(
                MagmaLeft, MagmaLower, msiz, n,
                calpha,    dA2+offst*ldda+offst,   ldda,
                           dX[0], ldda,
                cbeta,     dwork[0], ldda );
            cudaDeviceSynchronize();
            gpu_time2 = magma_wtime() - gpu_time2;
            gpu_perf2 = gflops / gpu_time2;
        }
        
        /* =====================================================================
           Performs operation using LAPACK
           =================================================================== */

        if ( checkres ) {
            // store ||A||*||X||
            errorbis  = lapackf77_zlange("fro", &msiz, &msiz, hA+offst*lda+offst, &lda, work );
            errorbis *= lapackf77_zlange("fro", &msiz, &n, hX, &lda, work );
            
            //printf( "A =" ); magma_zprint( m, m, hA, lda );
            //printf( "X =" ); magma_zprint( m, n, hX, lda );
            //printf( "B =" ); magma_zprint( m, n, hB, lda );
            
            cpu_time = magma_wtime();
            blasf77_zhemm( "Left", "Lower", &msiz, &n,
                            &calpha,    hA+offst*lda+offst, &lda,
                                        hX, &lda,
                            &cbeta,     hB, &lda );
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gflops / cpu_time;
            /*
              trace_file = fopen("AJETE/C", "w");
              for (int j = 0; j < n ; j++)
                    for (int i = 0; i < siz ; i++)
                                   fprintf(trace_file, "%10d%10d%40.30e\n", i+1, j+1, hB[j*lda+i]);
              fclose(trace_file);
            */
            magma_int_t firstprint=0;
            for(magma_int_t dev=0; dev<ngpu; ++dev){
            
                magma_setdevice( dev );
                magma_zgetmatrix( m, n,  dB[dev], ldda, hR, lda );

                // compute relative error ||R||/||A||*||X||, where R := B_magma - B_lapack = R - B
                size = lda*n;
                blasf77_zaxpy( &size, &c_neg_one, hB, &ione, hR, &ione );
                error = lapackf77_zlange("fro", &msiz, &n, hR, &lda, work) / errorbis;
                
                //printf( "R ="  ); magma_zprint( m, n, hR, lda );
                if(firstprint==0)
                   printf( "%5d %5d %5d %5d   %7.1f (%7.4f)   %7.1f (%7.4f)   %7.1f (%7.4f)   %8.2e\n",
                        (int) m, (int) n, (int) nb, (int) offst,
                        cpu_perf, cpu_time,
                        gpu_perf, gpu_time,
                        gpu_perf2, gpu_time2, error );
                else
                   printf( "%89s  %8.2e\n", " ", error );
                firstprint =1;
             }
        } else {
            printf( "%5d %5d %5d %5d     ---   (  ---  )   %7.1f (%7.4f)     ---   (  ---  )   ---\n",
                    (int) m, (int) n, (int) nb, (int) offst,
                    /*cpu_perf, cpu_time,*/
                    gpu_perf, gpu_time
                    /*, gpu_perf2, gpu_time2, error*/ );
        }

    }}}//}
    
    /* Memory clean up */
    for( int d = 0; d < ngpu; ++d ) {
        cudaSetDevice( d );
        magmablasSetKernelStream(  0  );
        TESTING_DEVFREE( dA[d] );
        TESTING_DEVFREE( dX[d] );
        //TESTING_DEVFREE( dB[d] );
    }
    
    TESTING_FREE( hA );
    TESTING_FREE( hX );
    TESTING_FREE( hB );
    TESTING_HOSTFREE( hR );
    
    /* Shutdown */
    TESTING_FINALIZE();
    return 0;
}
