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
                           cuDoubleComplex *work[], magma_int_t ldwork,
                           magma_int_t ngpu, magma_int_t nb, 
                           cudaStream_t streams[][20], magma_int_t nstream, 
                           cudaEvent_t redevents[][20],magma_int_t nbevents )
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
    assert( nbevents > 1 );
    
    cuDoubleComplex c_one  = MAGMA_Z_ONE;
    cuDoubleComplex c_zero = MAGMA_Z_ZERO;
    magma_int_t ione = 1;
    
        
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_stream_t cstream;
    magmablasGetKernelStream(&cstream);

   

    magma_int_t stdev      = (offset/nb)%ngpu;  
    for( magma_int_t dev = 0; dev < ngpu; ++dev ) {
        cudaSetDevice( dev );
        magmablasSetKernelStream( streams[ dev ][ 0 ] );
        cudaMemset(dwork(dev,0,0), 0, (lddwork)*(n)*sizeof(cuDoubleComplex) );
        // put all dC on all dev to 0 except the one which
        // hold i==0 because this one has to multiply by beta.
        if(dev!=stdev){
           cudaMemset(dC(dev,0,0), 0, (lddc)*(n)*sizeof(cuDoubleComplex) );
        }
    }

    magma_int_t blockoffset = offset % nb;
    magma_int_t remm      = m;
    magma_int_t fstblksiz = 0;
    magma_int_t neworig = 0;
    magma_int_t newoffset  = offset;

    // 1. symmetrize
    if(blockoffset>0){
        fstblksiz  = min(m, (nb - blockoffset));
        remm       = m - fstblksiz;
        newoffset  = offset+fstblksiz; // newoffset is adjusted over nb
        magma_int_t nbblkoffst = offset/nb;
        magma_int_t myblkoffst = (nbblkoffst/ngpu)+(nbblkoffst%ngpu > stdev?1:0);
        //printf("STDEV %d  voici offset %d remm %d   myblockoffset %d    siz %d \n",stdev,offset,remm,myblkoffst, fstblksiz);
        cudaSetDevice( stdev );
        magmablasSetKernelStream( streams[ stdev ][ 0 ] );
        magmablas_zsymmetrize_tiles(  MagmaLower,  fstblksiz,  dA(stdev, offset, myblkoffst*nb+blockoffset),  ldda,  1,  ngpu*nb,  nb  );         }else{
        remm = m;
    }

    for( magma_int_t dev = 0; dev < ngpu; ++dev ) {
        magma_int_t newstdev      = (newoffset/nb)%ngpu;
        magma_int_t nbblk = remm/nb; // number of block of size nb. if m%nb>0 then a last block exist and is of size ib=m%nb
        magma_int_t devperm   = (dev-newstdev+ngpu)%ngpu;
        magma_int_t myblk = (nbblk/ngpu) + (nbblk%ngpu > devperm ?  1:0 );
        magma_int_t nbblkoffst = newoffset/nb;
        magma_int_t myblkoffst = (nbblkoffst/ngpu)+(nbblkoffst%ngpu > dev?1:0);
        //printf("dev %d  devperm %d   newoffset %d  rowoff %d    coloff %d    myblk %d  \n",dev,devperm,newoffset,newoffset+devperm*nb,myblkoffst*nb,myblk);
        cudaSetDevice( dev );
        magmablasSetKernelStream( streams[ dev ][ 0 ] );
        magmablas_zsymmetrize_tiles(  MagmaLower,  nb,  dA(dev, newoffset+devperm*nb, myblkoffst*nb),  ldda,  myblk,  ngpu*nb,  nb  );
        if(remm%nb>0){
            magma_int_t nblstblks = (nbblk+1)%ngpu;
            magma_int_t devlstblk = (nblstblks-1+ngpu)%ngpu;
            //printf("==> siz %d devperm %d,    devlstblk %d,    newoffset+nbblk*nb %d,   myblkoffst*nb+ myblk*nb %d\n",remm % nb,devperm,devlstblk,newoffset+nbblk*nb,myblkoffst*nb+ myblk*nb);
            if(devperm==devlstblk)
                magmablas_zsymmetrize(  MagmaLower,  remm % nb,  dA(dev,newoffset+nbblk*nb,myblkoffst*nb+ myblk*nb),  ldda );  // last partial tile
        }
    }


    

/*
    magma_int_t siz = m+offset;
    cuDoubleComplex *R=(cuDoubleComplex *) malloc(siz*siz*sizeof(cuDoubleComplex));
    // collecte back A
    magmablas_zgetmatrix_1D_bcyclic( siz, siz, dA, ldda, R, siz, ngpu, nb );
    cudaSetDevice( 0 );
    magmablasSetKernelStream( streams[ dev ][ 0 ] );
    //magma_zgetmatrix( siz, siz, dA[0], ldda, R, siz );
    FILE *trace_file;
    trace_file = fopen("AJETE/Aafter", "w");
    for (int j = 0; j < siz ; j++) 
          for (int i = 0; i < siz ; i++) 
                         fprintf(trace_file,"%10d%10d%40.30e\n",i+1,j+1,R[j*siz+i]);
    fclose(trace_file);
return;
*/
    

    // ROW GEMM transpose a row and make a gemm with a block
    // if only 1 GPU used the ROW GEMM is integrated with the 
    // COL GEMM (better accuracy observed) and better perf
    magma_int_t storingpt = 0;
    if(ngpu>1){
        for( magma_int_t i = fstblksiz; i < m; i += nb ) {
            magma_int_t ib     = min( nb, m-i );      // block size
            magma_int_t ioff   = i + offset;          // start global index in parent matrix
            magma_int_t dev    = (ioff / nb) % ngpu;
            magma_int_t nbblkoffst = offset/nb;
            magma_int_t nbblk      = magma_ceildiv(i,nb);
            magma_int_t stdev      = (offset/nb)%ngpu; 
            for( magma_int_t dev = 0; dev < ngpu; ++dev ) {
                magma_int_t myblk = (nbblk/ngpu) + (nbblk%ngpu > ((dev-stdev+ngpu)%ngpu) ?  1:0 );
                magma_int_t myblkoffst = (nbblkoffst/ngpu)+(nbblkoffst%ngpu > dev?1:0);
                magma_int_t myrowsize = myblk * nb;
                magma_int_t coloffset = myblkoffst*nb;
                if(dev==stdev) {
                    myrowsize = myrowsize -blockoffset;
                    coloffset = myblkoffst*nb+blockoffset;
                }
                //printf("ROW GEMM: voici i %d   ib %d    ioff %d   nbblkoffst %d stdev %d  dev %d myblk %d  myblkoffset %d  coloffset %d  rowsize %d\n",i,ib,ioff,nbblkoffst,stdev,dev,myblk,myblkoffst,coloffset,myrowsize);
                if(myrowsize>0){
                    cudaSetDevice( dev );
                    magmablasSetKernelStream( streams[ dev ][ 1 ] );    
                    magma_zgemm( MagmaConjTrans, MagmaNoTrans, myrowsize, n, ib,
                                 alpha, dA(dev,ioff,coloffset), ldda,
                                        dB(dev,i,0),    lddb,
                                 c_one, dwork(dev,0,0), lddwork );
                }
            }
        }
        // start async receiving and for all GPU put all their dwork, on work[ngpu] 
        // note that because each GPU own a myblk so all GPU own nbblk so they could 
        // fit in work[ngpu] so I store them one after the other. 
        storingpt = 0;
        for( magma_int_t dev = 0; dev < ngpu; ++dev ) {
            magma_int_t nbblk    = magma_ceildiv((m+blockoffset),nb);
            magma_int_t nbblkrow = nbblk-1; 
            magma_int_t stdev    = (offset/nb)%ngpu;
            magma_int_t devperm  = (dev-stdev+ngpu)%ngpu;
            magma_int_t myblk = (nbblkrow/ngpu) + (nbblkrow%ngpu > devperm ?  1:0 );
            magma_int_t myrowsize = myblk * nb;
             if(dev==stdev) {
                myrowsize = myrowsize - blockoffset;
            }
            //printf("blockoffset %d nbblkrow %d devperm %d  DEV %d RECEIVING myblk %d  myrowsize %d\n",blockoffset,nbblkrow,devperm,dev,myblk,myrowsize);
            if(myrowsize>0){
                cudaSetDevice( dev );
                magmablasSetKernelStream( streams[ dev ][ 1 ] );
                cudaMemcpy2DAsync(&work[ngpu][storingpt], ldwork*sizeof(cuDoubleComplex),
                                  dwork[dev], lddwork*sizeof(cuDoubleComplex),
                                  myrowsize*sizeof(cuDoubleComplex), n,
                                  cudaMemcpyDeviceToHost, streams[ dev ][ 1 ]);
                storingpt =  storingpt + myrowsize;
            }
        }
    }


    // COL GEMM
    // blockoffset is offset within first block; for subsequent blocks it is 0
    if(blockoffset>0){
        magma_int_t ib     = min( nb-blockoffset, m );  // block size
        magma_int_t iblock = (offset / nb) / ngpu;          // local block id
        magma_int_t di     = iblock*nb+blockoffset;       // local index in parent matrix
        magma_int_t stdev  = (offset/nb)%ngpu; 
        cudaSetDevice( stdev );
        magmablasSetKernelStream( streams[ stdev ][ 0 ] );        
        //printf("DEV %d COL GEMM first   ioff %d  di %d   m %d   n %d   ib %d \n",stdev,offset,di,m,n,ib);
        magma_zgemm( MagmaNoTrans, MagmaNoTrans, m, n, ib,
                        alpha, dA(stdev,offset,di), ldda,
                               dB(stdev,0,0),     lddb,
                        beta,  dC(stdev,0,0),     lddc );
    }
   


    // COL GEMM
    for( magma_int_t i = fstblksiz; i < m; i += nb ) {
        magma_int_t ib     = min( nb, m-i );      // block size
        magma_int_t ioff   = i + offset;          // start global index in parent matrix
        magma_int_t iblock = (ioff / nb) / ngpu;  // local block id
        magma_int_t dev    = (ioff / nb) % ngpu;
        magma_int_t di     = iblock*nb;           // local index in parent matrix
        
        //printf("DEV %d COL GEMM i %d      ioff %d  di %d m-i %d    n %d   ib %d \n",dev,i,ioff,di,m-i,n,ib);
        
        cudaSetDevice( dev );
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
            // magmablasSetKernelStream( streams[ dev ][ 0 ] );
            magma_zgemm( MagmaConjTrans, MagmaNoTrans, i, n, ib,
                         alpha, dA(dev,ioff,offset), ldda,
                                dB(dev,i,0),    lddb,
                         c_one, dC(dev,0,0),    lddc );
        }
    }
    

    // wait and reduce results
    memset(C,0,ldc*n*sizeof(cuDoubleComplex));
    //    memset(C,0,n*nb*sizeof(cuDoubleComplex));
    // receive and put on its placment the row block
    
    if(ngpu>1){
        storingpt = 0;
        for( magma_int_t dev = 0; dev < ngpu; ++dev ) {
            magma_int_t nbblk    = magma_ceildiv((m+blockoffset),nb);
            magma_int_t nbblkrow = nbblk-1; 
            magma_int_t stdev      = (offset/nb)%ngpu;
            magma_int_t devperm  = (dev-stdev+ngpu)%ngpu;
            magma_int_t myblk = (nbblkrow/ngpu) + (nbblkrow%ngpu > devperm ?  1:0 );
            magma_int_t myrowsize = myblk * nb;
             if(dev==stdev) {
                myrowsize = myrowsize - blockoffset;
            }
      
            //printf("blockoffset %d nbblkrow %d devperm %d  DEV %d RECEIVING myblk %d  myrowsize %d\n",blockoffset,nbblkrow,devperm,dev,myblk,myrowsize);
            if(myrowsize>0){
                cudaSetDevice( dev );
                magma_queue_sync( streams[ dev ][ 1 ] );
                // for each dev put the received block each on its placment
                for(magma_int_t blki = 0; blki < myblk; ++blki){
                    magma_int_t gbblki = (blki*ngpu + devperm)*nb - blockoffset;
                    magma_int_t lcblki = blki*nb;
                    magma_int_t ib     = nb;// min(nb,m-gbblki);
                    if(dev==stdev){
                        lcblki = blki*nb-blockoffset;
                        if(blki==0){
                            gbblki = 0;
                            lcblki = 0;
                            ib     = nb-blockoffset;
                        }
                    }
                    //printf("blockoffset %d nbblkrow %d devperm %d  DEV %d RECEIVING myblk %d  myrowsize %d copying blki %d of size %d from work[%d] to C[%d]\n",blockoffset,nbblkrow,devperm,dev,myblk,myrowsize,blki,ib,lcblki,gbblki);
                    //printf("stdev %d  dev %d myblk %d  blki  %d  gbblki %d\n",stdev,dev,myblk,blki,gbblki);
                    lcblki = lcblki + storingpt;
                    lapackf77_zlacpy("A", &ib, &n, &work[ngpu][lcblki], &ldwork, &C[gbblki], &ldc);
         
                }
                storingpt =  storingpt + myrowsize;
            }
        }
    }

    

    magma_int_t size   = ldc*n;
    magma_int_t fstcpy = 1;   
    magma_int_t nxtdev,nxtdevperm,nxtmyblk;
    for( magma_int_t dev = 0; dev < ngpu; ++dev ) {
        magma_int_t nbblk   = magma_ceildiv(m+blockoffset,nb);
        magma_int_t stdev   = (offset/nb)%ngpu;
        magma_int_t devperm = (dev-stdev+ngpu)%ngpu;
        magma_int_t myblk   = (nbblk/ngpu) + (nbblk%ngpu > devperm ?  1:0 );
        //printf("DEV %d COL RECEIVING stdev %d devperm %d myblk %d  m %d   n %d  \n",dev,stdev,devperm,myblk,m,n);
        if(myblk>0){
           cudaSetDevice( dev );       
           magmablasSetKernelStream( streams[ dev ][ 0 ] );
           magma_queue_sync( streams[ dev ][ 0 ] );
           if(ngpu==1){
               magma_zgetmatrix( m, n, dC[dev], lddc, C, ldc );
           }else{
               if(fstcpy)magma_zgetmatrix( m, n, dC[dev], lddc, work[dev], ldwork );
               fstcpy = 0;
               // preparing next receive from the next available GPU
               for (nxtdev=dev+1; nxtdev<ngpu; ++nxtdev)
               {
                   nxtdevperm = (nxtdev-stdev+ngpu)%ngpu;
                   nxtmyblk   = (nbblk/ngpu) + (nbblk%ngpu > nxtdevperm ?  1:0 );
                   if(nxtmyblk>0) {
                       cudaSetDevice( nxtdev ); 
                       magma_zgetmatrix_async( m, n, dC[nxtdev], lddc, work[nxtdev], ldwork, streams[ nxtdev ][ 0 ] );
                       break;
                   }
               }
               blasf77_zaxpy( &size, &c_one, work[dev], &ione, C, &ione );
           }
        }
    }
    
 
/*
    for( magma_int_t dev = 0; dev < ngpu; ++dev ) {
        cudaSetDevice( dev );
        magmablasSetKernelStream( streams[ dev ][ 0 ] );
        for( magma_int_t s = 0; s < nstream; ++s ) {
            magma_queue_sync( streams[ dev ][ s ] );
        }
    }
*/
    // Synchronous send X=AVT from CPU to all active GPUs 
    // because async is the same by cutting the bandwidth
    for( magma_int_t dev = 0; dev < ngpu; ++dev ) {
        magma_int_t nbblk   = magma_ceildiv(m+blockoffset,nb);
        magma_int_t stdev   = (offset/nb)%ngpu;
        magma_int_t devperm = (dev-stdev+ngpu)%ngpu;
        magma_int_t myblk   = (nbblk/ngpu) + (nbblk%ngpu > devperm ?  1:0 );
        if(myblk>0){
           cudaSetDevice( dev );  
           magma_zsetmatrix( m, n,
                 C, ldc,
                 dC[dev],  lddc );
        }
    }

    
    cudaSetDevice( cdev );
    magmablasSetKernelStream( cstream );

}
