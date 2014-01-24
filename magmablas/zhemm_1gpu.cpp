/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Mark Gates
       
       This still has poor performance. Work in progress.
*/
#include "common_magma.h"
#include "trace.h"
#include <assert.h>

extern "C"
void magmablas_zhemm_1gpu(
    magma_side_t side, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha, magmaDoubleComplex *dA[], magma_int_t ldda,  magma_int_t offset,
                           magmaDoubleComplex *dB[], magma_int_t lddb,
    magmaDoubleComplex beta,  magmaDoubleComplex *dC[], magma_int_t lddc,
                           magmaDoubleComplex *C,    magma_int_t ldc,
    magma_int_t ngpu, magma_int_t nb, magma_queue_t streams[][20], magma_int_t nstream )
{
    #define dA(dev, i, j) (dA[dev] + (i) + (j)*ldda)
    #define dB(dev, i, j) (dB[dev] + (i) + (j)*lddb)
    #define dC(dev, i, j) (dC[dev] + (i) + (j)*lddc)
    #define dwork(dev, i, j) (dwork[dev] + (i) + (j)*lddwork)
    #define C(i, j) (C + (i) + (j)*ldc)
        
    assert( ngpu == 1 );
    assert( ldda >= m );
    assert( lddb >= m );
    assert( lddc >= m );
    
    magmaDoubleComplex c_one  = MAGMA_Z_ONE;
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO;
    magma_int_t ione = 1;
    
    // put init/finalize into testing_zhemm_mgpu,
    // so Gflop/s doesn't include writing file.
    //trace_init( 1, ngpu, nstream, (magma_queue_t*) streams );
        
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
            trace_gpu_start( dev, 0, "gemm", "symmetrize+gemm" );
    }
    
    
    for( magma_int_t dev = 0; dev < ngpu; ++dev ) {
        magma_setdevice( dev );
        //magmablas_zlaset( MagmaUpperLower, m, n, dC(dev,0,0), lddc );
        cudaMemset(dC(dev,0,0), 0, (lddc)*(n)*sizeof(magmaDoubleComplex) );
    }
    // 1. symmetrize
    magma_int_t dev=0;    
    magma_int_t ntile = m / nb;
    magmablas_zsymmetrize_tiles(  MagmaLower,  nb,  dA(dev, offset, 0),  ldda,  ntile,  nb,  nb  );
    if(m%nb>0) magmablas_zsymmetrize(  MagmaLower,  m % nb,  dA(dev,offset+ntile*nb, ntile*nb),  ldda );  // last partial tile

    magma_int_t gemmstream=1;
    // 2. col gemms
    //    for( magma_int_t i = nb; i < m; i += nb ) {
    for( magma_int_t i = 0; i < m; i += nb ) {
        magma_int_t ib     = min( nb, m-i );      // block size
        magma_int_t ioff   = i + offset;          // start global index in parent matrix
        magma_int_t iblock = (ioff / nb) / ngpu;  // local block id
        magma_int_t dev    = (ioff / nb) % ngpu;
        magma_int_t di     = iblock*nb;           // local index in parent matrix
        
        magma_setdevice( dev );
        magma_int_t s = iblock % gemmstream;
        magmablasSetKernelStream( streams[ dev ][ 0 ] );
        /*
        if(i==0)
              magma_zgemm( MagmaNoTrans, MagmaNoTrans, m-i, n, ib,
                         alpha, dA(dev,ioff,di), ldda,
                            dB(dev,i,0),        lddb,
                         c_zero, dC(dev,i,0),     lddc );
        else
        */
              magma_zgemm( MagmaNoTrans, MagmaNoTrans, m-i, n, ib,
                         alpha, dA(dev,ioff,di), ldda,
                            dB(dev,i,0),        lddb,
                         c_one, dC(dev,i,0),     lddc );

        magma_zgemm( MagmaConjTrans, MagmaNoTrans, i, n, ib,
                     alpha,  dA(dev,ioff,0), ldda,
                             dB(dev,i,0),     lddb,
                     c_one, dC(dev,0,0),     lddc );

    }
    // 2b. sync
    //
    if(gemmstream>1){
        for( magma_int_t dev = 0; dev < ngpu; ++dev ) {
            magma_setdevice( dev );
            for( magma_int_t s = 0; s < gemmstream; ++s ) {
                magma_queue_sync( streams[ dev ][ s ] );
            }
        }
    }

    // tracing
    for( int dev = 0; dev < ngpu; ++dev ) {
        magma_setdevice( dev );
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
