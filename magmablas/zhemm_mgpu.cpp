/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> s d c
       @author Mark Gates
       @author Azzam Haidar
       
       This still has poor performance. Work in progress.
*/
#include "common_magma.h"
#include "magma_bulge.h"
//#include "trace.h"
#include <assert.h>

extern "C"
void magmablas_zhemm_mgpu(
    char side, char uplo, magma_int_t m, magma_int_t n,
    cuDoubleComplex alpha, cuDoubleComplex *dA[], magma_int_t ldda,  magma_int_t offset,
                           cuDoubleComplex *dB[], magma_int_t lddb,
    cuDoubleComplex beta,  cuDoubleComplex *dC[], magma_int_t lddc,
                           cuDoubleComplex *dwork[],    magma_int_t lddwork,
                           cuDoubleComplex *C,    magma_int_t ldc,
    magma_int_t ngpu, magma_int_t nb, cudaStream_t streams[][20], magma_int_t nstream )
{
    #define dA(dev, i, j) (dA[dev] + (i) + (j)*ldda)
    #define dB(dev, i, j) (dB[dev] + (i) + (j)*lddb)
    #define dC(dev, i, j) (dC[dev] + (i) + (j)*lddc)
    #define dwork(dev, i, j) (dwork[dev] + (i) + (j)*lddwork)
    #define C(i, j) (C + (i) + (j)*ldc)
    
    assert( ldda >= m );
    assert( lddb >= m );
    assert( lddc >= m );
    assert( lddwork >= m );
    assert( nstream > 1 );
    
    cuDoubleComplex c_one  = MAGMA_Z_ONE;
    cuDoubleComplex c_zero = MAGMA_Z_ZERO;
    magma_int_t ione = 1;
    
        
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_stream_t cstream;
    magmablasGetKernelStream(&cstream);

/*    
    // tracing
    for( int dev = 0; dev < ngpu; ++dev ) {
        magma_setdevice( dev );
        trace_gpu_start( dev, 0, "init", "initialize" );
    }
*/    

    magma_int_t stdev      = (offset/nb)%ngpu;  
    for( magma_int_t dev = 0; dev < ngpu; ++dev ) {
        magma_setdevice( dev );
        cudaMemset(dwork(dev,0,0), 0, (lddwork)*(n)*sizeof(cuDoubleComplex) );
        // put all dC on all dev to 0 except the one which
        // hold i==0 because this one has to multiply by beta.
        if(dev!=stdev){
           cudaMemset(dC(dev,0,0), 0, (lddc)*(n)*sizeof(cuDoubleComplex) );
        }
    }

    
    /*
            magma_zhemm(
                MagmaLeft, MagmaLower, m, n,
                alpha, dA[0]+offset*ldda+offset,   ldda,
                           dB[0], ldda,
                beta,     dC[0], ldda );
        magma_zgetmatrix( m, n, dC[0], ldda, C, ldc );
    return ;
*/
    
    /*
    // tracing
    for( int dev = 0; dev < ngpu; ++dev ) {
        magma_setdevice( dev );
        trace_gpu_end( dev, 0 );
        trace_gpu_start( dev, 0, "symmetrize", "symmetrize" );
    }
*/

    // 1. symmetrize
    for( magma_int_t dev = 0; dev < ngpu; ++dev ) {
        magma_setdevice( dev );
        magma_int_t nbblk = m/nb; // number of block of size nb. if m%nb>0 then a last block exist and is of size ib=m%nb
        magma_int_t myblk = (nbblk/ngpu) + (nbblk%ngpu > ((dev-stdev+ngpu)%ngpu) ?  1:0 );
        magma_int_t devperm   = (dev-stdev+ngpu)%ngpu;
        magma_int_t nbblkoffst = offset/nb;
        magma_int_t myblkoffst = (nbblkoffst/ngpu)+(nbblkoffst%ngpu > dev?1:0);
        //printf("dev %d  devperm %d   rowoff %d    coloff %d    myblk %d  \n",dev,devperm, offset+devperm*nb,myblkoffst*nb,myblk);
        magmablas_zsymmetrize_tiles(  MagmaLower,  nb,  dA(dev, offset+devperm*nb, myblkoffst*nb),  ldda,  myblk,  ngpu*nb,  nb  );
        if(m%nb>0){
            magma_int_t nblstblks = (nbblk+1)%ngpu;
            magma_int_t devlstblk = (nblstblks-1+ngpu)%ngpu;
            if(devperm==devlstblk)
                magmablas_zsymmetrize(  MagmaLower,  m % nb,  dA(dev,offset+nbblk*nb,myblkoffst*nb+ myblk*nb),  ldda );  // last partial tile
        }
    }
 
/*

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
    
    for( magma_int_t dev = 0; dev < ngpu; ++dev ) {
        magma_setdevice( dev );
        for( magma_int_t s = 0; s < nstream; ++s ) {
            magma_queue_sync( streams[ dev ][ s ] );
        }
    }
*/





    /*
    magma_int_t siz = m+offset;
    cuDoubleComplex *R=(cuDoubleComplex *) malloc(siz*siz*sizeof(cuDoubleComplex));
    magma_zgetmatrix( siz, siz, dA[0], ldda, R,siz );
    FILE *trace_file;
    trace_file = fopen("AJETE/Aafter", "w");
    for (int j = 0; j < siz ; j++) 
          for (int i = 0; i < siz ; i++) 
                         fprintf(trace_file,"%10d%10d%40.30e\n",i+1,j+1,R[j*siz+i]);
    fclose(trace_file);
return;
*/
    /*    
            magma_zhemm(
                MagmaLeft, MagmaLower, m, n,
                alpha, dA[0]+offset*ldda+offset,   ldda,
                           dB[0], ldda,
                beta,     dC[0], ldda );
        magma_zgetmatrix( m, n, dC[0], ldda, C, ldc );
    return ;
*/


    

/*
    // tracing
    for( int dev = 0; dev < ngpu; ++dev ) {
        magma_setdevice( dev );
        trace_gpu_end( dev, 0 );
        trace_gpu_start( dev, 1, "gemm", "ROW gemm" );
    }
*/

    // ROW GEMM transpose a row and make a gemm with a block
    // if only 1 GPU used the ROW GEMM is integrated with the 
    // COL GEMM (better accuracy observed) and better perf
    if(ngpu>1){
        for( magma_int_t i = nb; i < m; i += nb ) {
            magma_int_t ib     = min( nb, m-i );      // block size
            magma_int_t ioff   = i + offset;          // start global index in parent matrix
            magma_int_t iblock = (ioff / nb) / ngpu;  // local block id
            magma_int_t dev    = (ioff / nb) % ngpu;
            magma_int_t di     = iblock*nb;           // local index in parent matrix
            magma_int_t nbblkoffst = offset/nb;

            magma_int_t stdev      = (offset/nb)%ngpu; 
            for( magma_int_t dev = 0; dev < ngpu; ++dev ) {

                magma_int_t nbblk    = i/nb;
                magma_int_t myblk = (nbblk/ngpu) + (nbblk%ngpu > ((dev-stdev+ngpu)%ngpu) ?  1:0 );
                magma_int_t myblkoffst = (nbblkoffst/ngpu)+(nbblkoffst%ngpu > dev?1:0);
                //printf("voici i %d    ioff %d   nbblkoffst %d stdev %d  dev %d myblk %d  myblkoffset %d \n",i,ioff,nbblkoffst,stdev,dev,myblk,myblkoffst);

                magma_int_t myrowsize = myblk * nb;
                if(myrowsize>0){
                    magma_setdevice( dev );
                    magmablasSetKernelStream( streams[ dev ][ 1 ] );    
                    magma_zgemm( MagmaConjTrans, MagmaNoTrans, myrowsize, n, ib,
                                 alpha, dA(dev,ioff,myblkoffst*nb), ldda,
                                        dB(dev,i,0),    lddb,
                                 c_one, dwork(dev,0,0), lddwork );
                }
            }
        }
    }
    
    for( int dev = 0; dev < ngpu; ++dev ) {
               magma_queue_sync( streams[ dev ][ 1 ] );
    } 
    
    
 /*    
    // tracing
    for( int dev = 0; dev < ngpu; ++dev ) {
        magma_setdevice( dev );
        trace_gpu_end( dev, 1 );
        trace_gpu_start( dev, 0, "gemm", "COL gemm" );
        //if(ngpu==1) trace_gpu_start( dev, 1, "gemm", "ROW gemm" );        
    }
*/
    // COL GEMM
    for( magma_int_t i = 0; i < m; i += nb ) {
        magma_int_t ib     = min( nb, m-i );      // block size
        magma_int_t ioff   = i + offset;          // start global index in parent matrix
        magma_int_t iblock = (ioff / nb) / ngpu;  // local block id
        magma_int_t dev    = (ioff / nb) % ngpu;
        magma_int_t di     = iblock*nb;           // local index in parent matrix
        
        
        magma_setdevice( dev );
        magmablasSetKernelStream( streams[ dev ][ 0 ] );
        if(i==0){
           magma_zgemm( MagmaNoTrans, MagmaNoTrans, m-i, n, ib,
                        alpha, dA(dev,ioff,di), ldda,
                               dB(dev,i,0),     lddb,
                        beta,  dC(dev,i,0),     lddc );
        }else{
           magma_zgemm( MagmaNoTrans, MagmaNoTrans, m-i, n, ib,
                        alpha, dA(dev,ioff,di), ldda,
                               dB(dev,i,0),        lddb,
                        c_one, dC(dev,i,0),     lddc );
        }
        // if only 1 GPU is used, do the ROW GEMM
        if(ngpu==1){
            // NOTE THAT because the COL gemm write dC below the diagonal (i) 
            // and the ROW GEMM write dC from 0 to diag-1, so they could 
            // run in parallel on diferent stream.        
            // 
            // NO NO NO because
            // it might happen that col finished i and strated i+1 while row still at i    
            // magmablasSetKernelStream( streams[ dev ][ 1 ] );
            magma_zgemm( MagmaConjTrans, MagmaNoTrans, i, n, ib,
                         alpha, dA(dev,ioff,offset), ldda,
                                dB(dev,i,0),    lddb,
                         c_one, dC(dev,0,0),    lddc );
        }
    }
    

/*
    magma_int_t siz = m+offset;
    cuDoubleComplex *R=(cuDoubleComplex *) malloc(siz*siz*sizeof(cuDoubleComplex));    
    magma_zgetmatrix( siz, n, dC[0], lddc, R, siz );
    FILE *trace_file;
    trace_file = fopen("AJETE/DC", "w");
    for (int j = 0; j < n ; j++) 
          for (int i = 0; i < siz ; i++) 
                         fprintf(trace_file,"%10d%10d%40.30e\n",i+1,j+1,R[j*siz+i]);
    fclose(trace_file);
*/


    // wait and reduce results
    memset(C,0,ldc*n*sizeof(cuDoubleComplex));
    cuDoubleComplex *Ctmp = C(0,n);
//    memset(C,0,n*nb*sizeof(cuDoubleComplex));
    // receive and put on its placment the row block
    if(ngpu>1){
        for( magma_int_t dev = 0; dev < ngpu; ++dev ) {
            magma_setdevice( dev );
            magma_int_t nbblk = magma_ceildiv((m-nb),nb);
            magma_int_t stdev      = (offset/nb)%ngpu;
            magma_int_t devperm  = (dev-stdev+ngpu)%ngpu;
            magma_int_t myblk = (nbblk/ngpu) + (nbblk%ngpu > devperm ?  1:0 );
            magma_int_t myrowsize = myblk * nb;
            if(myrowsize>0){
                //trace_gpu_start( dev, 1, "get", "get C_row" );
                     /*
                     magma_zgetmatrix_async( myrowsize, n,
                                             dwork[dev], lddwork,
                                             Ctmp, ldc, streams[dev][1] );
                     *//*
                     magma_zcopymatrix_async( myrowsize, n,
                                             dwork[dev], lddwork,
                                             Ctmp, ldc, streams[dev][1] );
                     */                                 
                cudaMemcpy2DAsync(Ctmp, ldc*sizeof(cuDoubleComplex),
                                  dwork[dev], lddwork*sizeof(cuDoubleComplex),
                                  myrowsize*sizeof(cuDoubleComplex), n,
                                  cudaMemcpyDeviceToHost, streams[ dev ][ 1 ]);
                
                magma_queue_sync( streams[ dev ][ 1 ] );
                //trace_gpu_end( dev, 1 );
                // for each dev put the received block each on its placment
                //trace_cpu_start( 0, "axpy", "C += C_row" );
                for( magma_int_t blki = 0; blki < myblk; ++blki){
                    magma_int_t gbblki = (blki*ngpu + devperm)*nb;
                    magma_int_t ib     = nb;// min(nb,m-gbblki);
                    //printf("stdev %d  dev %d myblk %d  blki  %d  gbblki %d\n",stdev,dev,myblk,blki,gbblki);
                          lapackf77_zlacpy("A", &ib, &n, &Ctmp[blki*nb], &ldc, &C[gbblki], &ldc);
         
                }
                //trace_cpu_end( 0 );
            }        
        }
    }


    

    magma_int_t size = ldc*n;
    for( magma_int_t dev = 0; dev < ngpu; ++dev ) {
        magma_setdevice( dev );
        //trace_gpu_start( dev, 0, "get", "get C_dev" );
        //if(ngpu==1) magma_queue_sync( streams[ dev ][ 1 ] );
        magma_queue_sync( streams[ dev ][ 0 ] );
        magma_zgetmatrix( m, n, dC[dev], lddc, Ctmp, ldc );
        //trace_gpu_end( dev, 0 );
        
        //trace_cpu_start( 0, "axpy", "C += C_dev" );
        blasf77_zaxpy( &size, &c_one, Ctmp, &ione, C, &ione );
        //trace_cpu_end( 0 );
    }
    
    
        
/*
    for( magma_int_t dev = 0; dev < ngpu; ++dev ) {
        magma_setdevice( dev );
        magmablasSetKernelStream( cstream );
    }
*/    
    magma_setdevice( cdev );
    magmablasSetKernelStream( cstream );

}
