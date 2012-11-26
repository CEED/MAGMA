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
void magmablas_zhemm_mgpu_com(
    char side, char uplo, magma_int_t m, magma_int_t n,
    cuDoubleComplex alpha, cuDoubleComplex *dA[], magma_int_t ldda,  magma_int_t offset,
                           cuDoubleComplex *dB[], magma_int_t lddb,
    cuDoubleComplex beta,  cuDoubleComplex *dC[], magma_int_t lddc,
                           cuDoubleComplex *dwork[],    magma_int_t lddwork,
                           cuDoubleComplex *C,    magma_int_t ldc,
                           cuDoubleComplex *work[], magma_int_t ldwork,
                           magma_int_t ngpu, magma_int_t nb, 
                           cudaStream_t streams[][20], magma_int_t nstream, 
                           cudaEvent_t redevents[][20],magma_int_t nbevents, 
                           magma_int_t gnode[MagmaMaxGPUs][MagmaMaxGPUs+2], magma_int_t nbcmplx )
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
    

    cuDoubleComplex *dwork1[MagmaMaxGPUs];
    cuDoubleComplex *dwork2[MagmaMaxGPUs];
    cuDoubleComplex *dwork3[MagmaMaxGPUs];



    for( magma_int_t dev = 0; dev < ngpu; ++dev ) {
        dwork1[dev] = dwork[dev];
        dwork2[dev] = dwork[dev]+n*lddwork;
        dwork3[dev] = dwork[dev]+2*n*lddwork;
    }




        
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
        magma_int_t myblk = (nbblk/ngpu) + (nbblk%ngpu > ((dev-newstdev+ngpu)%ngpu) ?  1:0 );
        magma_int_t devperm   = (dev-newstdev+ngpu)%ngpu;
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
        for( magma_int_t dev = 0; dev < ngpu; ++dev ) {
            cudaSetDevice( dev );
            cudaEventRecord(redevents[dev][1], streams[dev][1]);
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


    
    if(ngpu>1){
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
                magmablasSetKernelStream( streams[ dev ][ 0 ] );
                cudaStreamWaitEvent(streams[ dev ][ 0 ], redevents[dev][1], 0);
                //magma_queue_sync( streams[ dev ][ 1 ] );
                // for each dev add the computed ROW block each on its placment with dC
                for( magma_int_t blki = 0; blki < myblk; ++blki){
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
                    magmablas_zgeadd(ib, n, c_one, 
                                    &dwork[dev][lcblki], lddwork, 
                                    &dC[dev][gbblki]   , lddc   );
                }
                cudaEventRecord(redevents[dev][0], streams[dev][0]);                
            }
        }
    }


    // ===========================================================
    //             COMMUNICATION ALL_REDUCE_SUM 
    // ===========================================================
    magma_int_t CPUREDUCE = 1;
    if(ngpu==1){
        magma_zgetmatrix( m, n, dC[0], lddc, C, ldwork );
    }else{
        // wait and reduce results
        memset(C,0,ldc*n*sizeof(cuDoubleComplex));
        magma_int_t size   = ldc*n;
        magma_int_t flip, blkreceived, fstdevcpy, myngpu;
        magma_int_t nxtdev,nxtdevperm,nxtmyblk;

        // put the fstdevcpy stored in gnode in MagmaMaxGPUs+1 to -1;
        // compute the nb of active cmplx
        magma_int_t firsttime=0;
        magma_int_t nbcmplxactive =0;
        magma_int_t cmplxisactive =0;
        for( magma_int_t cmplxid = 0; cmplxid < nbcmplx; ++cmplxid ) {
            gnode[cmplxid][MagmaMaxGPUs+1] = -1;
            cmplxisactive = 0;
            myngpu = gnode[cmplxid][MagmaMaxGPUs];
            for( magma_int_t idev = 0; idev < myngpu; ++idev ) {
                magma_int_t dev     = gnode[cmplxid][idev];
                magma_int_t nbblk   = magma_ceildiv(m+blockoffset,nb);
                magma_int_t stdev   = (offset/nb)%ngpu;
                magma_int_t devperm = (dev-stdev+ngpu)%ngpu;
                magma_int_t myblk   = (nbblk/ngpu) + (nbblk%ngpu > devperm ?  1:0 );
                if(myblk>0)cmplxisactive = 1;
            }
            if(cmplxisactive)nbcmplxactive = nbcmplxactive +1;
        }
        // printf("nbcmplxactive %d\n",nbcmplxactive);
        //*******************************
        //  each Master GPU is collecting
        //  from other on the same board
        //  and doing the addition, then
        //  either sending to CPU if the 
        //  CPU has to REDUCE over the board
        //  or to the other board if GPU 
        //  will do the full reduction
        //*******************************
        for( magma_int_t cmplxid = 0; cmplxid < nbcmplx; ++cmplxid ) {
            flip        = -1;  
            blkreceived =  0;  
            fstdevcpy   = -1;
            gnode[cmplxid][MagmaMaxGPUs+1] = -1;
            myngpu = gnode[cmplxid][MagmaMaxGPUs];
            for( magma_int_t idev = 0; idev < myngpu; ++idev ) {
                magma_int_t dev     = gnode[cmplxid][idev];
                magma_int_t nbblk   = magma_ceildiv(m+blockoffset,nb);
                magma_int_t stdev   = (offset/nb)%ngpu;
                magma_int_t devperm = (dev-stdev+ngpu)%ngpu;
                magma_int_t myblk   = (nbblk/ngpu) + (nbblk%ngpu > devperm ?  1:0 );
                //printf("DEV %d COL RECEIVING stdev %d devperm %d myblk %d  m %d   n %d  \n",dev,stdev,devperm,myblk,m,n);
                if(myblk>0){
                    cudaSetDevice( dev );        
                    magmablasSetKernelStream( streams[ dev ][ 0 ] );
                    //cudaStreamWaitEvent(0, redevents[dev][0], 0);
                    //cudaStreamWaitEvent(streams[ dev ][ 0 ], redevents[dev][0], 0);
                    blkreceived = blkreceived +1;
                    if(blkreceived==1){
                        fstdevcpy = dev;
                        gnode[cmplxid][MagmaMaxGPUs+1] = fstdevcpy;
                    }else{
                        flip = (flip+1)%2;                                            
                        if(blkreceived==2){
                            // flip =0 put it on dwork2
                            cudaMemcpy2DAsync(dwork2[fstdevcpy], lddwork*sizeof(cuDoubleComplex),
                                               dC[dev], lddc*sizeof(cuDoubleComplex),
                                               m*sizeof(cuDoubleComplex), n,
                                               cudaMemcpyDeviceToDevice, streams[dev][0]);
                            cudaEventRecord(redevents[dev][0], streams[dev][0]);
                            //printf("device %d send to device %d flip %d\n",dev,fstdevcpy,flip);
                        }
                        // preparing next receive from the next available GPU
                        if(blkreceived>=2){
                            for (magma_int_t k=idev+1; k<myngpu; ++k)
                            {
                                nxtdev = gnode[cmplxid][k];
                                nxtdevperm = (nxtdev-stdev+ngpu)%ngpu;
                                nxtmyblk   = (nbblk/ngpu) + (nbblk%ngpu > nxtdevperm ?  1:0 );
                                if(nxtmyblk>0) {
                                    cudaSetDevice( nxtdev );
                                    // sync with the previous copy because I don't want to have parallel copy within the same complex
                                    cudaStreamWaitEvent(streams[ nxtdev ][ 0 ], redevents[dev][0], 0);
                                    //  ATTENTION        
                                    // this is the first time I may write dwork of fstdevcpy
                                    // so i should be sure that his ROW+COL GEMM-ADDITION which read dwork has been done.
                                    // On the second hand I want to be sure that fstdevcpy has been 
                                    // finishing the adition on the previous dev.
                                    cudaStreamWaitEvent(streams[ nxtdev ][ 0 ], redevents[fstdevcpy][0], 0); 
                           
                                    if(flip==0)
                                        cudaMemcpy2DAsync(dwork[fstdevcpy], lddwork*sizeof(cuDoubleComplex),
                                                    dC[nxtdev], lddc*sizeof(cuDoubleComplex),
                                                    m*sizeof(cuDoubleComplex), n,
                                                    cudaMemcpyDeviceToDevice, streams[nxtdev][0]);
                                    else
                                        cudaMemcpy2DAsync(dwork2[fstdevcpy], lddwork*sizeof(cuDoubleComplex),
                                                    dC[nxtdev], lddc*sizeof(cuDoubleComplex),
                                                    m*sizeof(cuDoubleComplex), n,
                                                    cudaMemcpyDeviceToDevice, streams[nxtdev][0]);
                                   cudaEventRecord(redevents[nxtdev][0], streams[nxtdev][0]);
                                   //printf("device %d send to device %d flip %d\n",nxtdev,fstdevcpy,flip);
                                   break;
                                }
                            }
                            cudaSetDevice( fstdevcpy );
                            magmablasSetKernelStream( streams[ fstdevcpy ][ 0 ] );
                            cudaStreamWaitEvent(streams[ fstdevcpy ][ 0 ], redevents[dev][0], 0);
                            //printf("device %d adding from device %d flip %d\n",fstdevcpy,dev,flip);
                           
                            if(flip==0)
                                magmablas_zgeadd(m, n, c_one, 
                                                      dwork2[fstdevcpy], lddwork, 
                                                      dC[fstdevcpy]   , lddc   );
                            else
                                magmablas_zgeadd(m, n, c_one, 
                                                      dwork[fstdevcpy], lddwork, 
                                                      dC[fstdevcpy]   , lddc   );
                            
                        }
                    }

                } // ifmyblk>0
            } // end for idev 1:myngpu
            if(fstdevcpy !=-1){
                cudaSetDevice( fstdevcpy );
                cudaEventRecord(redevents[fstdevcpy][0], streams[fstdevcpy][0]);
            }
            if(nbcmplxactive>1){
                fstdevcpy = gnode[cmplxid][MagmaMaxGPUs+1];
                if(fstdevcpy !=-1){
                    //printf(" cmplx %d device %d is sending to host cmplxid %d \n", cmplxid+1,fstdevcpy,cmplxid);
                    cudaSetDevice( fstdevcpy );
                    cudaStreamWaitEvent(streams[ fstdevcpy ][ 0 ], redevents[fstdevcpy][0], 0);
                    if(CPUREDUCE==1){
                        if(firsttime==0){
                            firsttime = 1;
                            cudaMemcpy2DAsync(C, ldc*sizeof(cuDoubleComplex),
                                              dC[fstdevcpy], lddc*sizeof(cuDoubleComplex),
                                              m*sizeof(cuDoubleComplex), n,
                                              cudaMemcpyDeviceToHost, streams[ fstdevcpy ][ 0 ]);
                        }else{
                            cudaMemcpy2DAsync(work[fstdevcpy], ldwork*sizeof(cuDoubleComplex),
                                              dC[fstdevcpy], lddc*sizeof(cuDoubleComplex),
                                              m*sizeof(cuDoubleComplex), n,
                                              cudaMemcpyDeviceToHost, streams[ fstdevcpy ][ 0 ]);
                        }
                    }else{
                        // REDUCTION IS DONE DIRECTLY ON GPU where each complex send its
                        // results to the other complex.                   
                    }
                }
            }
        } // for cmplxid

        //*******************************
        //  CPU IS DOING THE REDUCE SUM
        //*******************************
        firsttime = 0;
        if(CPUREDUCE==1){
            // CPU sync and add, only in case where more than 1 cmplx is active
            if(nbcmplxactive>1){
                // make the addition of the dC received from each complex.    
                for( magma_int_t cmplxid = 0; cmplxid < nbcmplx; ++cmplxid ) {
                    fstdevcpy = gnode[cmplxid][MagmaMaxGPUs+1];        
                    if(fstdevcpy !=-1){
                        firsttime = firsttime+1;
                        magma_queue_sync( streams[ fstdevcpy ][ 0 ] );
                        if(firsttime>1){
                            blasf77_zaxpy( &size, &c_one, work[fstdevcpy], &ione, C, &ione );
                        }
                    }
                }
                // addition done, broadcast the result to the master of each complex
                for( magma_int_t cmplxid = 0; cmplxid < nbcmplx; ++cmplxid ) {
                    fstdevcpy = gnode[cmplxid][MagmaMaxGPUs+1];
                    if(fstdevcpy !=-1){
                        cudaSetDevice( fstdevcpy );
                        cudaStreamWaitEvent(streams[ fstdevcpy ][ 0 ], redevents[fstdevcpy][0], 0);
                        cudaMemcpy2DAsync(dC[fstdevcpy], lddc*sizeof(cuDoubleComplex),
                                          C, ldc*sizeof(cuDoubleComplex),
                                          m*sizeof(cuDoubleComplex), n,
                                          cudaMemcpyHostToDevice, streams[ fstdevcpy ][ 0 ]);
                        cudaEventRecord(redevents[fstdevcpy][0], streams[fstdevcpy][0]);
                    }
                }
            }
        }else{
            // REDUCTION IS DONE DIRECTLY ON GPU where each complex send its
            // results to the other complex.
            // here each master must add the received portion in dwork3 to its dC
        }

        //*******************************
        //  each Master GPU has the final
        //  result either by receiving 
        //  from CPU of by making the add
        //  by himself, so now it is time 
        //  to broadcast over the GPUs of 
        //  its board.
        //*******************************
        for( magma_int_t cmplxid = 0; cmplxid < nbcmplx; ++cmplxid ) {
            myngpu    = gnode[cmplxid][MagmaMaxGPUs];
            fstdevcpy = gnode[cmplxid][MagmaMaxGPUs+1];
            //printf(" cmplx %d fstdevcpy %d broadcasting over %d gpu\n",cmplxid,fstdevcpy,myngpu);
            for( magma_int_t idev = 0; idev < myngpu; ++idev ) {
                magma_int_t dev     = gnode[cmplxid][idev];
                magma_int_t nbblk   = magma_ceildiv(m+blockoffset,nb);
                magma_int_t stdev   = (offset/nb)%ngpu;
                magma_int_t devperm = (dev-stdev+ngpu)%ngpu;
                magma_int_t myblk   = (nbblk/ngpu) + (nbblk%ngpu > devperm ?  1:0 );
                //printf("DEV %d RECEIVING from fstdevcpy %d myblk %d  m %d   n %d  \n",dev,fstdevcpy,myblk,m,n);
                if((myblk>0)&&(dev!=fstdevcpy)){
                    // parallel broadcast inside each board
                    cudaSetDevice( dev );        
                    
                    magmablasSetKernelStream( streams[ dev ][ 0 ] );
                    cudaStreamWaitEvent(streams[ dev ][ 0 ], redevents[fstdevcpy][0], 0);
                    cudaMemcpy2DAsync(dC[dev], lddc*sizeof(cuDoubleComplex),
                                      dC[fstdevcpy], lddc*sizeof(cuDoubleComplex),
                                      m*sizeof(cuDoubleComplex), n,
                                      cudaMemcpyDeviceToDevice, streams[dev][0]);
                    cudaEventRecord(redevents[dev][0], streams[dev][0]);
                    /*
                    // sequential broadcast
                    cudaSetDevice( fstdevcpy );        
                    magmablasSetKernelStream( streams[ fstdevcpy ][ 0 ] );
                    cudaStreamWaitEvent(streams[ fstdevcpy ][ 0 ], redevents[fstdevcpy][0], 0);
                    cudaMemcpy2DAsync(dC[dev], lddc*sizeof(cuDoubleComplex),
                                      dC[fstdevcpy], lddc*sizeof(cuDoubleComplex),
                                      m*sizeof(cuDoubleComplex), n,
                                      cudaMemcpyDeviceToDevice, streams[fstdevcpy][0]);
                    cudaEventRecord(redevents[fstdevcpy][0], streams[fstdevcpy][0]);
                    */

                } // ifmyblk>0
            } // for idev 1:myngpu
        } // for cmplxid


        // ONLY FOR NOW IN CASE OF 1 COMPLEX_GPU, we still coying tothe CPU (temporary)
        if((nbcmplx==1)||(nbcmplxactive==1)){
            for( magma_int_t cmplxid = 0; cmplxid < nbcmplx; ++cmplxid ) {  
                magma_int_t fstdevcpy = gnode[cmplxid][MagmaMaxGPUs+1];
                if(fstdevcpy != -1){
                    cudaSetDevice( fstdevcpy );
                    magmablasSetKernelStream( streams[ fstdevcpy ][ 0 ] );
                    magma_queue_sync( streams[ fstdevcpy ][ 0 ] );
                    //printf("%d sending to CPU\n",fstdevcpy);
                    /*
                    cudaMemcpy2DAsync(C, ldc*sizeof(cuDoubleComplex),
                                              dC[fstdevcpy], lddc*sizeof(cuDoubleComplex),
                                              m*sizeof(cuDoubleComplex), n,
                                              cudaMemcpyDeviceToHost, streams[ fstdevcpy ][ 0 ]);
                    magma_queue_sync( streams[ fstdevcpy ][ 0 ] );
                    */
                }
            }
        } // the loop of FOR NOW

   } // if ngpu>1
 
    



 
        for( magma_int_t cmplxid = 0; cmplxid < nbcmplx; ++cmplxid ) {
            magma_int_t myngpu    = gnode[cmplxid][MagmaMaxGPUs];
            magma_int_t fstdevcpy = gnode[cmplxid][MagmaMaxGPUs+1];
            for( magma_int_t idev = 0; idev < myngpu; ++idev ) {
                magma_int_t dev     = gnode[cmplxid][idev];
                cudaSetDevice( dev );
                // parallel broadcast
                cudaStreamWaitEvent(streams[ dev ][ 0 ], redevents[dev][0], 0);
                magma_queue_sync( streams[ dev ][ 0 ] );
                
                // sequential broadcast
                //cudaStreamWaitEvent(streams[ dev ][ 0 ], redevents[fstdevcpy][0], 0);
            }
        }



/*
    // send X=AVT stored in dW to all GPUs 
    for( magma_int_t dev = 0; dev < ngpu; ++dev ) {
        cudaSetDevice( dev );
        magma_zsetmatrix_async( m, n,
                 work, ldwork,
                 dC[dev],  lddc, streams[dev][0] );

    }
*/


    cudaSetDevice( cdev );
    magmablasSetKernelStream( cstream );

}
