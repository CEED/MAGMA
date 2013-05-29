/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> s d c
       @author Mark Gates
       
       This still has poor performance. Work in progress.
*/
#include "common_magma.h"
#include "trace.h"
#include <assert.h>

extern "C" void
magmablas_zsymmetrize( char uplo, magma_int_t m, magmaDoubleComplex *dA, magma_int_t ldda );

extern "C"
void magmablas_zhemm_1gpu_old(
    char side, char uplo, magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha, magmaDoubleComplex *dA[], magma_int_t ldda,  magma_int_t offset,
                           magmaDoubleComplex *dB[], magma_int_t lddb,
    magmaDoubleComplex beta,  magmaDoubleComplex *dC[], magma_int_t lddc,
                           magmaDoubleComplex *C,    magma_int_t ldc,
    magma_int_t ngpu, magma_int_t nb, cudaStream_t streams[][20], magma_int_t nstream )
{
    #define dA(dev, i, j) (dA[dev] + (i) + (j)*ldda)
    #define dB(dev, i, j) (dB[dev] + (i) + (j)*lddb)
    #define dC(dev, i, j) (dC[dev] + (i) + (j)*lddc)
    #define C(i, j) (C + (i) + (j)*ldc)
    
    assert( ldda >= m );
    assert( lddb >= m );
    assert( lddc >= m );
    
    magmaDoubleComplex c_one  = MAGMA_Z_ONE;
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO;
    magma_int_t ione = 1;
    
    // put init/finalize into testing_zhemm_mgpu,
    // so Gflop/s doesn't include writing file.
    //trace_init( 1, ngpu, nstream, (cudaStream_t*) streams );
        
    magma_device_t cdev;
    magma_getdevice( &cdev );
    
    // loop over all blocks
    // Faster to have several loops:
    // first  symmetrizes A[i,i]
    // second does C[i]      = A[i:m,  i]'*B[i:m]
    // third  does C[i+1:m] += A[i+1:m,i] *B[i]
    
    // tracing
    for( int dev = 0; dev < ngpu; ++dev ) {
        magma_setdevice( dev );
        for( int s = 0; s < nstream; ++s ) {
            trace_gpu_start( dev, s, "symmetrize", "symmetrize" );
        }
    }
    // 1. symmetrize
    for( magma_int_t i = 0; i < m; i += nb ) {
        magma_int_t ib     = min( nb, m-i );      // block size
        magma_int_t ioff   = i + offset;          // start global index in parent matrix
        magma_int_t iblock = (ioff / nb) / ngpu;  // local block id
        magma_int_t dev    = (ioff / nb) % ngpu;
        magma_int_t di     = iblock*nb;           // local index in parent matrix
        
        magma_setdevice( dev );
        magma_int_t s = iblock % nstream;
        magmablasSetKernelStream( streams[ dev ][ s ] );
        
        // make diagonal block symmetric
        magmablas_zsymmetrize( MagmaLower, ib, dA(dev,ioff,di), ldda );
    }
    // tracing
    for( int dev = 0; dev < ngpu; ++dev ) {
        magma_setdevice( dev );
        for( int s = 0; s < nstream; ++s ) {
            trace_gpu_end( dev, s );
            trace_gpu_start( dev, s, "gemm", "C[i] = A[i:m,i]'*B[i:m]" );
        }
    }
    // 2. row gemms
    for( magma_int_t i = 0; i < m; i += nb ) {
        magma_int_t ib     = min( nb, m-i );      // block size
        magma_int_t ioff   = i + offset;          // start global index in parent matrix
        magma_int_t iblock = (ioff / nb) / ngpu;  // local block id
        magma_int_t dev    = (ioff / nb) % ngpu;
        magma_int_t di     = iblock*nb;           // local index in parent matrix
        
        magma_setdevice( dev );
        magma_int_t s = iblock % nstream;
        magmablasSetKernelStream( streams[ dev ][ s ] );

        // C[i] = A[i:m,i]' * B[i:m0]
        //printf( "gemm1: A[%4d,%4d]*B[%4d] -> C[%4d] ib     %4d, n %4d, m-i %4d\n",
        //        ioff, di, i, i, ib, n, m-i );
        magma_zgemm( MagmaConjTrans, MagmaNoTrans, ib, n, m-i,
                     alpha,  dA(dev,ioff,di), ldda,
                             dB(dev,i,0),     lddb,
                     c_zero, dC(dev,i,0),     lddc );
    }
    // tracing
    for( int dev = 0; dev < ngpu; ++dev ) {
        magma_setdevice( dev );
        for( int s = 0; s < nstream; ++s ) {
            trace_gpu_end( dev, s );
            trace_gpu_start( dev, s, "sync", "sync C[i]" );
        }
    }
    // 2b. sync
    for( magma_int_t dev = 0; dev < ngpu; ++dev ) {
        for( magma_int_t s = 0; s < nstream; ++s ) {
            magma_setdevice( dev );
            magma_queue_sync( streams[ dev ][ s ] );
        }
    }
    // tracing
    for( int dev = 0; dev < ngpu; ++dev ) {
        magma_setdevice( dev );
        for( int s = 0; s < nstream; ++s ) {
            trace_gpu_end( dev, s );
        }
    }
    // 3. column gemms
    for( magma_int_t i = 0; i < m; i += nb ) {
        magma_int_t ib     = min( nb, m-i );      // block size
        magma_int_t ioff   = i + offset;          // start global index in parent matrix
        magma_int_t iblock = (ioff / nb) / ngpu;  // local block id
        magma_int_t dev    = (ioff / nb) % ngpu;
        magma_int_t di     = iblock*nb;           // local index in parent matrix
        
        // these have to be on same stream, since they write into same block,
        // unless we used separate C workspace for each stream
        magma_setdevice( dev );
        magmablasSetKernelStream( streams[dev][0] );
        
        // C[i+1:n] += A[i+1:n,i] * B[i]
        //printf( "gemm2: A[%4d,%4d]*B[%4d] -> C[%4d] m-i-ib %4d, n %4d, ib  %4d\n",
        //        ioff+ib, di, i, i+ib, m-i-ib, n, ib );
        trace_gpu_start( dev, 0, "gemm2", "C[i+1:m] += A[i+1:m,i]*B[i]" );
        magma_zgemm( MagmaNoTrans, MagmaNoTrans, m-i-ib, n, ib,
                     alpha, dA(dev,ioff+ib,di), ldda,
                            dB(dev,i,0),        lddb,
                     c_one, dC(dev,i+ib,0),     lddc );
        trace_gpu_end( dev, 0 );
    }
    
    // meanwhile on CPU, scale C := beta*C
    trace_cpu_start( 0, "scal", "C = beta*C" );
    for( magma_int_t j = 0; j < n; ++j ) {
        blasf77_zscal( &m, &beta, C(0,j), &ione );
    }
    trace_cpu_end( 0 );
    
    // wait and reduce results
    magma_int_t size = ldc*n;
    magmaDoubleComplex *Ctmp = C(0,n);
    for( magma_int_t dev = 0; dev < ngpu; ++dev ) {
        magma_setdevice( dev );
        trace_gpu_start( dev, 0, "get", "get C_dev" );
        magma_zgetmatrix( m, n, dC[dev], lddc, Ctmp, ldc );
        trace_gpu_end( dev, 0 );
        
        trace_cpu_start( 0, "axpy", "C += C_dev" );
        blasf77_zaxpy( &size, &c_one, Ctmp, &ione, C, &ione );
        trace_cpu_end( 0 );
    }
    
    magma_setdevice( cdev );
    
    //trace_finalize( "zhemm.svg", "trace.css" );
}
