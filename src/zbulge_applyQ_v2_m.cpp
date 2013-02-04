/*
 * Copyright (c) 2011      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 *
 *     @author Azzam Haidar
 *     @author Stan Tomov
 *     @author Raffaele Solca
 *
 *     @precisions normal z -> s d c
 *
 */

#include "common_magma.h"
#include "magma_bulge.h"
#include "magma_zbulgeinc.h"
#include <cblas.h>
// === Define what BLAS to use ============================================
#define PRECISION_z

// === End defining what BLAS to use ======================================

////////////////////////////////////////////////////////////////////////////////////////////////////

#define dE(dev,i,j)  (dE[dev]+(i) + ldde*(j))
#define V(j)     (V+(j))
#define T(j)     (T+(j))
/***************************************************************************
 *  Parallel apply Q2 from bulgechasing symetric matrices - static scheduling
 *  Lower case is treated
 **/
    /*
     * side == magmaLeft:
     *     meaning apply E = Q*E = (q_1*q_2*.....*q_n) * E ==> so
     *     traverse Vs in reverse order (forward) from q_n to q_1 Also
     *     E is splitten by block of col over cores because each apply
     *     consist in a block of row (horizontal block)
     */
    /*
     * side == magmaRight:
     *     meaning apply E = E*Q = E * (q_1*q_2*.....*q_n) ==> so
     *     traverse Vs in normal order (forward) from q_1 to q_n Also
     *     E is splitten by block of row over core because each apply 
     *     consist in a block of col (vertical block)
     */
/***************************************************************************/
extern "C" magma_int_t magma_zbulge_applyQ_v2_m(magma_int_t ngpu, char side, magma_int_t NE, magma_int_t N, magma_int_t NB, magma_int_t Vblksiz, cuDoubleComplex *E, magma_int_t lde, cuDoubleComplex *V, magma_int_t ldv, cuDoubleComplex *T, magma_int_t ldt, magma_int_t *info)
{
    //%===========================
    //%   local variables
    //%===========================
    magma_int_t Vm, Vn, mt, nt;
    magma_int_t myrow, mycol, blkj, blki;
    magma_int_t blkid,vpos,tpos;
    magma_int_t firstrow, nbcolinvolvd;
    magma_int_t versionL  = 113;
    magma_int_t versionR  = 92;
    magma_int_t Vchunksiz = 10;
    *info=0;

    /* Quick return */
    if ( NE == 0 ) {
        return MAGMA_SUCCESS;
    }
    if ( N == 0 ) {
        return MAGMA_SUCCESS;
    }
    if ( NB == 0 ) {
        return MAGMA_SUCCESS;
    }
    /* ==========================================
     * some infos for developer 
     * Initialisation and checking nb of cores
     * ==========================================*/
    /* we have 2 algo for left (113 114) and 2 algo for right (91 92)
     * which correspond to versionL versionR.
     * They are very similar (detail explained in tech report and matlab code)
     * however version 114 and 92 improve locality.
     * while version 113 is used in case WNATZ=1 (construct Q2) which allow 
     * the construction to be done in an optimized way taking into 
     * consideration that the matrix is Identity so making less flops.
     * 
    */

    // Initialize streaming and events
    cudaDeviceSynchronize();
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_stream_t cstream;
    magmablasGetKernelStream(&cstream);

    magma_int_t  nbevents =2, nstream=2;
    cudaStream_t streams[MagmaMaxGPUs][20];    
    cudaEvent_t  myevent[MagmaMaxGPUs][20]; 
    for( magma_int_t dev = 0; dev < ngpu; ++dev ) {
        cudaSetDevice( dev );
        for( magma_int_t i = 0; i < nstream; ++i ) {
            cudaStreamCreate( &streams[dev][i] );
        }
        for( magma_int_t i = 0; i < nbevents; ++i ) {
            cudaEventCreateWithFlags(&myevent[dev][i],cudaEventDisableTiming);
        }
    }

    // Azzam 21/11/2012
    // NOTE THAT dwork was of size 2*lddwork*Vblksiz+...
    // but I am thinking why not modifing it to lddwork*Vblksiz+...
    cuDoubleComplex *dE[MagmaMaxGPUs], *dwork[MagmaMaxGPUs], *dwvt[MagmaMaxGPUs];
    cuDoubleComplex *dT0[MagmaMaxGPUs], *dV0[MagmaMaxGPUs], *dT1[MagmaMaxGPUs], *dV1[MagmaMaxGPUs];
    magma_int_t dev;
    magma_int_t ldde = N;
    magma_int_t lddv = ldv;
    magma_int_t lddt = ldt;
    magma_int_t lddwvt = ldv; 
    magma_int_t lddwork = NE;
    magma_int_t ne_loc = magma_ceildiv(NE, ngpu);

    if (ne_loc<256)
       ne_loc=256;

    ngpu = min(ngpu, magma_ceildiv(NE,ne_loc)); // Don't use GPU that will not have data.

    // copy dE to GPUs 
    for (dev=0; dev < ngpu; ++dev){
        cudaSetDevice( dev );
        lddwork = ne_loc;
        if(MAGMA_SUCCESS != magma_zmalloc( &dE[dev], ldde * ne_loc)) {
            printf ("!!!!  magma_zbulge_applyQ magma_alloc failed for: dE\n" );
            exit(-1);
        }
        if(MAGMA_SUCCESS != magma_zmalloc( &dwork[dev], 2*lddwork*Vblksiz + 2*Vchunksiz* (Vblksiz* (lddv+lddt)) + lddwvt*Vblksiz )) {
            printf ("!!!!  magma_zbulge_applyQ magma_alloc failed for: dwork\n" );
            exit(-1);
        }
        dV0[dev] = dwork[dev] + 2*lddwork*Vblksiz;
        dT0[dev] = dV0[dev] + Vchunksiz*Vblksiz*lddv;
        dV1[dev] = dT0[dev] + Vchunksiz*Vblksiz*lddt;
        dT1[dev] = dV1[dev] + Vchunksiz*Vblksiz*lddv;
        dwvt[dev]= dT1[dev] + Vchunksiz*Vblksiz*lddt;

        magma_int_t ie_loc = min(ne_loc, NE - ne_loc*dev);
        magma_zsetmatrix_async( N, ie_loc, E+lde*ne_loc*dev, lde, dE(dev, 0, 0), ldde, streams[dev][1] );
    }


    // make overlapped copy
    magma_int_t ncpy = 0;
    magma_int_t copyed=0, copyst=0;
    magma_int_t blkcnt,nothing, mysiz, flip, vld,tld, locpos;
    findVTsiz(N, NB, Vblksiz, &blkcnt, &nothing);


    /* SIDE LEFT  meaning apply E = Q*E = (q_1*q_2*.....*q_n) * E ==> so traverse Vs in reverse order (forward) from q_n to q_1
     *            Also E is splitten by row meaning each apply consist in a block of row (horizontal block) */
    /* SIDE RIGHT meaning apply E = E*Q = E * (q_1*q_2*.....*q_n) ==> so tarverse Vs in normal  order (forward) from q_1 to q_n
     *            Also E is splitten by col meaning each apply consist in a block of col (vertical block) */


    printf("  APPLY Q_v22_m GPU with NGPU %d  N %d, NE %d,  NB %d, Vblksiz %d, versionL %d versionR %d  SIDE %c \n",
          ngpu, N, NE, NB, Vblksiz, versionL, versionR, side);


    /*
     * MagmamaLeft
     */
    if(side=='L'){
        /*
         * Version 113:
         * loop over the block_col (nt) and for each find the 
         * number of tiles (mt) in this block_col. then loop over mt, find 
         * the size of the V's(Vm,Vn) and apply it to the corresponding
         * portion of E. 
         */
        if( versionL == 113 ) {
            nt  = magma_ceildiv((N-1),Vblksiz);
            for (blkj=nt-1; blkj>=0; blkj--) {
                /* the index of the first row on the top of block (blkj) */ 
                firstrow = blkj * Vblksiz + 1;
                /*find the number of tile for this block */
                if( blkj == nt-1 )
                    mt = magma_ceildiv( N -  firstrow,    NB);
                else
                    mt = magma_ceildiv( N - (firstrow+1), NB);
                /*loop over the tiles find the size of the Vs and apply it */
                for (blki=mt; blki>0; blki--) {
                    /*calculate the size of each losange of Vs= (Vm,Vn)*/
                    myrow     = firstrow + (mt-blki)*NB;
                    mycol     = blkj*Vblksiz;
                    Vm = min( NB+Vblksiz-1, N-myrow);
                    if( ( blkj == nt-1 ) && ( blki == mt ) ){
                        Vn = min (Vblksiz, Vm);
                    } else {
                        Vn = min (Vblksiz, Vm-1);
                    }
                    /*calculate the pointer to the Vs and the Ts.
                     * Note that Vs and Ts have special storage done
                     * by the bulgechasing function*/
                    //printf("voici blkj %d blki %d  Vm %d  Vn %d mycol %d vpos %d \n",blkj,blki,Vm, Vn,mycol,vpos);
                    magma_bulge_findpos113(N, NB, Vblksiz, mycol, myrow, &blkid);
               
                    // COPY Vchunksiz Vs and Vchunksiz Ts to GPU and store it in dV0/dV1 and dT0/dT1
                    if(ncpy==0){
                        // flip = 1 for this.
                        copyst = 0;                             // meaning that copy will start copying from blkid =copyst
                        copyed = min(copyst+Vchunksiz, blkcnt); // meaning that copy will end copying at blkid =copyed-1==> next copy had to start at copyed
                        mysiz  = copyed-copyst;                 // the size of the chunk to be copied
                        if(mysiz>0){        
                            ncpy = 1;
                            flip = 1;
                            vpos = copyst*Vblksiz*ldv;
                            tpos = copyst*Vblksiz*ldt;
                            vld  = mysiz * ldv;
                            tld  = mysiz * ldt;
                            for( dev = 0; dev < ngpu; ++dev ) {
                                cudaSetDevice( dev );
                                magmablasSetKernelStream( streams[ dev ][ 1 ] );
                                magma_zsetmatrix_async(vld, Vblksiz, V(vpos), vld, dV1[dev], vld, streams[dev][1]);
                                magma_zsetmatrix_async(tld, Vblksiz, T(tpos), tld, dT1[dev], tld, streams[dev][1]);
                            }
                            //printf("doing the first copy   of mysiz %2d copyst %2d copyed %2d vpos %8d tpos %8d into dV1 dT1\n",mysiz,copyst,copyed,vpos,tpos);
                        }
                    }
                   
                    if(blkid == copyst){
                        flip   = ncpy % 2;
                        copyst = copyed;                             // meaning that copy will start copying from blkid =copyst
                        copyed = min(copyst+Vchunksiz, blkcnt); // meaning that copy will end copying at blkid =copyed-1==> next copy had to start at copyed
                        mysiz  = copyed-copyst;                 // the size of the chunk to be copied
                        //printf(" get to copy blkid %d blkid+(2*Vchunksiz) %d copyst %d copyed %d\n",blkid,blkid+(Vchunksiz),copyst,copyed);
                        if(mysiz>0){              
                            ncpy = ncpy + 1;        
                            vpos = copyst*Vblksiz*ldv;
                            tpos = copyst*Vblksiz*ldt;
                            vld  = mysiz * ldv;
                            tld  = mysiz * ldt;
                            if(flip==0){ // now I am working on dV0 so copy the next and put it on dV1
                                //printf("doing overlapping copy of mysiz %2d copyst %2d copyed %2d vpos %8d tpos %8d into dV1 dT1\n",mysiz,copyst,copyed,vpos,tpos);  
                                for( dev = 0; dev < ngpu; ++dev ) {
                                    cudaSetDevice( dev );
                                    magmablasSetKernelStream( streams[ dev ][ 1 ] );
                                    magma_zsetmatrix_async(vld, Vblksiz, V(vpos), vld, dV1[dev], vld, streams[dev][1]);
                                    magma_zsetmatrix_async(tld, Vblksiz, T(tpos), tld, dT1[dev], tld, streams[dev][1]);
                                }
                            }else{ // now I am working on dV1 so copy the next and put it on dV0
                                //printf("doing overlapping copy of mysiz %2d copyst %2d copyed %2d vpos %8d tpos %8d into dV0 dT0\n",mysiz,copyst,copyed,vpos,tpos);   
                                for( dev = 0; dev < ngpu; ++dev ) {
                                    cudaSetDevice( dev );
                                    magmablasSetKernelStream( streams[ dev ][ 0 ] );
                                    magma_zsetmatrix_async(vld, Vblksiz, V(vpos), vld, dV0[dev], vld, streams[dev][0]);
                                    magma_zsetmatrix_async(tld, Vblksiz, T(tpos), tld, dT0[dev], tld, streams[dev][0]);
                                }
                            }
                        }
                    }

                    if((Vm>0)&&(Vn>0)){
                        locpos = blkid%Vchunksiz;
                        magma_int_t lcvpos   = locpos*Vblksiz*lddv;
                        magma_int_t lctpos   = locpos*Vblksiz*lddt;
                        //printf("voici blkj %d blki %d  Vm %d  Vn %d mycol %d locvpos %5d loctpos %5d  blkid %2d  using data in dV%1d dT%1d \n",blkj,blki,Vm, Vn,mycol,lcvpos,lctpos, blkid,flip,flip);
                        if(flip==0){
                            for( dev = 0; dev < ngpu; ++dev ) {
                                magma_int_t ie_loc   = min(ne_loc, NE - ne_loc*dev);
                                magma_int_t nr_bl = magma_ceildiv(ie_loc,10000);       //nr of blocks
                                magma_int_t sz_bl = magma_ceildiv(ie_loc,nr_bl*64)*64; //maximum size of blocks (to have blocks of around the same size and multiple of 64)
                                magma_int_t ib;                                        //size of current block
                                cudaSetDevice( dev );
                                magmablasSetKernelStream(streams[dev][0]);
                                cudaStreamWaitEvent(streams[dev][0], myevent[dev][1], 0);
                                for(magma_int_t i=0; i<ie_loc; i+= sz_bl){
                                    ib = min(sz_bl, ie_loc-i);
                                    //magma_zlarfb_gpu( MagmaLeft, MagmaNoTrans, MagmaForward, MagmaColumnwise, Vm, ib, Vn, dV0[dev]+lcvpos, lddv, dT0[dev]+lctpos, lddt, dE(dev,myrow,i), ldde, dwork[dev], lddwork);
                                    magma_zlarfb_gpu_gemm( MagmaLeft, MagmaNoTrans, MagmaForward, MagmaColumnwise, Vm, ib, Vn, dV0[dev]+lcvpos, lddv, dT0[dev]+lctpos, lddt, dE(dev,myrow,i), ldde, dwork[dev], ib, dwvt[dev], Vm);
                                }
                                cudaEventRecord(myevent[dev][0], streams[dev][0]);
                            }
                        }else{
                            for( dev = 0; dev < ngpu; ++dev ) {
                                magma_int_t ie_loc   = min(ne_loc, NE - ne_loc*dev);
                                magma_int_t nr_bl = magma_ceildiv(ie_loc,10000);       //nr of blocks
                                magma_int_t sz_bl = magma_ceildiv(ie_loc,nr_bl*64)*64; //maximum size of blocks (to have blocks of around the same size and multiple of 64)
                                magma_int_t ib;                                        //size of current block
                                cudaSetDevice( dev );
                                magmablasSetKernelStream(streams[dev][1]);
                                cudaStreamWaitEvent(streams[dev][1], myevent[dev][0], 0);
                                for(magma_int_t i=0; i<ie_loc; i+= sz_bl){
                                    ib = min(sz_bl, ie_loc-i);
                                    //magma_zlarfb_gpu( MagmaLeft, MagmaNoTrans, MagmaForward, MagmaColumnwise, Vm, ib, Vn, dV1[dev]+lcvpos, lddv, dT1[dev]+lctpos, lddt, dE(dev,myrow,i), ldde, dwork[dev], ib);
                                    magma_zlarfb_gpu_gemm( MagmaLeft, MagmaNoTrans, MagmaForward, MagmaColumnwise, Vm, ib, Vn, dV1[dev]+lcvpos, lddv, dT1[dev]+lctpos, lddt, dE(dev,myrow,i), ldde, dwork[dev], ib, dwvt[dev], Vm);
                                }
                                cudaEventRecord(myevent[dev][1], streams[dev][1]);
                            }
                        }
                    }  // end for (Vm &Vn) > 0   
                } // end for blki
            } // end for blkj
        } // end if version=113
        /*
         * Version 114:
         * loop over the block_row (mt) and for each find diagonally the 
         * number of tiles (nt) in this block_row. then loop over nt, find 
         * the size of the V's(Vm,Vn) and apply it to the corresponding
         * portion of E. 
         */
        else {
            printf("versionL 114 not implemented in zbulge_applyQ_v2_m\n");
            exit(-1);
            mt    = magma_ceildiv((N-1),NB);
            for (blki = mt; blki>0; blki--) {
                /* nbcolinvolvd = number of column corresponding to this block_row (blki) */
                nbcolinvolvd = min(N-1, blki*NB);
                /*find the number of tile for this block (diagonal row of tiles) */
                nt = magma_ceildiv(nbcolinvolvd,Vblksiz);
                /*loop over the tiles find the size of the Vs and apply it */
                for (blkj = nt-1; blkj>=0; blkj--) {
                    /* the index of the first row of the first col meaning 
                     * the block on the top left (blki) */ 
                    firstrow = (mt-blki)*NB+1;                
                    /*calculate the size of each losange of Vs= (Vm,Vn)*/
                    myrow    = firstrow + blkj*Vblksiz;
                    mycol    = blkj*Vblksiz;
                    Vm = min( NB+Vblksiz-1, N-myrow);
                    if( ( blkj == nt-1 ) && ( blki == mt ) ){
                        Vn = min (Vblksiz, Vm);
                    }else{
                        Vn = min (Vblksiz, Vm-1);
                    }

                    if((Vm>0)&&(Vn>0)){
                        /*calculate the pointer to the Vs and the Ts.
                         * Note that Vs and Ts have special storage done
                         * by the bulgechasing function*/
                        /*
                        magma_bulge_findVTpos(N, NB, Vblksiz, mycol, myrow, ldv, ldt, &vpos, &tpos);
                        magma_zsetmatrix_async(Vm, Vn, V(vpos), ldv, dV0, lddv, NULL);
                        magma_zsetmatrix_async(Vn,  Vn, T(tpos), ldt, dT0, lddt, NULL);
                        //printf("voici blki %d  rownbm %d mycol %d  coled %d  blkid %d vpos %d  tpos %d\n", blki, rownbm, mycol, coled, blkid, vpos, tpos);
                        for(magma_int_t i=0; i<NE; i+= sz_bl){
                            ib = min(sz_bl, NE-i);
                            magma_zlarfb_gpu( MagmaLeft, MagmaNoTrans, MagmaForward, MagmaColumnwise, Vm, ib, Vn, dV0, lddv, dT0, lddt, dE(myrow,i), ldde, dwork, NE);
                        }
                        */
                    } // end for (Vm &Vn) > 0
                } // end for blkj
            } // end for blki
        } // end version 114
    } // end LEFT
    /*
     * MagmaRight
     */
    else {
        printf("Side 'R' not implemented in zbulge_applyQ_v2_m\n");
        exit(-1);
        /*
         * Version 91:
         */
        if( versionR == 91 ) {
            nt  = magma_ceildiv((N-1),Vblksiz);
            for (blkj=0; blkj<nt; blkj++) {
                /* the index of the first myrow on the top of block (blkj) */ 
                firstrow = blkj * Vblksiz + 1;
                /*find the number of tile for this block */
                if( blkj == nt-1 )
                    mt = magma_ceildiv( N -  firstrow,    NB);
                else
                    mt = magma_ceildiv( N - (firstrow+1), NB);
                /*loop over the tiles find the size of the Vs and apply it */
                for (blki=1; blki<=mt; blki++) {
                    /*calculate the size of each losange of Vs= (Vm,Vn)*/
                    myrow  = firstrow + (mt-blki)*NB;
                    Vm = min( NB+Vblksiz-1, N-myrow);
                    if( ( blkj == nt-1 ) && ( blki == mt ) )
                    {
                        Vn = min (Vblksiz, Vm);
                    }else{
                        Vn = min (Vblksiz, Vm-1);
                    }
                    mycol     = blkj*Vblksiz;
                    if((Vm>0)&&(Vn>0)){
                        /*calculate the pointer to the Vs and the Ts.
                         * Note that Vs and Ts have special storage done
                         * by the bulgechasing function*/
                        /*
                        magma_bulge_findVTpos(N, NB, Vblksiz ,mycol, myrow, ldv, ldt, &vpos, &tpos);
                        magma_zsetmatrix_async(Vm, Vn, V(vpos), ldv, dV0, lddv, NULL);
                        magma_zsetmatrix_async(Vn,  Vn, T(tpos), ldt, dT0, lddt, NULL);
                        magma_zlarfb_gpu( MagmaRight, MagmaNoTrans, MagmaForward, MagmaColumnwise, NE, Vm, Vn, dV0, lddv, dT0, lddt, dE(0, myrow), ldde, dwork, NE);
                        */
                    } // end for (Vm &Vn) > 0
                } // end for blki
            } // end fo blkj
        } // end of version 91
        /*
         * Version 92:
         */
        else {
            mt    = magma_ceildiv((N-1),NB);
            for (blki = 1; blki<=mt; blki++) {
                /* nbcolinvolvd = number of column corresponding to this block_row (blki) */
                nbcolinvolvd = min(N-1, blki*NB);
                /*find the number of tile for this block (diagonal row of tiles) */
                nt = magma_ceildiv(nbcolinvolvd,Vblksiz);
                /*loop over the tiles find the size of the Vs and apply it */
                for (blkj = 0; blkj<nt; blkj++) {
                    /* the index of the first row of the first col meaning 
                     * the block on the top left (blki) */ 
                    firstrow = (mt-blki)*NB+1;                
                    /*calculate the size of each losange of Vs= (Vm,Vn)*/
                    myrow    = firstrow + blkj*Vblksiz;
                    mycol    = blkj*Vblksiz;
                    Vm = min( NB+Vblksiz-1, N-myrow);
                    if( ( blkj == nt-1 ) && ( blki == mt ) ){
                        Vn = min (Vblksiz, Vm);
                    }else{
                        Vn = min (Vblksiz, Vm-1);
                    }
                    if((Vm>0)&&(Vn>0)){
                        /*calculate the pointer to the Vs and the Ts.
                         * Note that Vs and Ts have special storage done
                         * by the bulgechasing function*/
                        /*
                        magma_bulge_findVTpos(N, NB, Vblksiz ,mycol, myrow, ldv, ldt, &vpos, &tpos);
                        magma_zsetmatrix_async(Vm, Vn, V(vpos), ldv, dV0, lddv, NULL);
                        magma_zsetmatrix_async(Vn,  Vn, T(tpos), ldt, dT0, lddt, NULL);
                        magma_zlarfb_gpu( MagmaRight, MagmaNoTrans, MagmaForward, MagmaColumnwise, NE, Vm, Vn, dV0, lddv, dT0, lddt, dE(0, myrow), ldde, dwork, NE);
                        */
                    } // end for (Vm &Vn) > 0
                } //end for blkj
            } // end for blki
        } //end of version 92
    } // end RIGHT

    // copy back the dE form each GPU 
    for( dev = 0; dev < ngpu; ++dev ) {
        cudaSetDevice( dev );
        magmablasSetKernelStream(streams[dev][0]);
        cudaStreamWaitEvent(streams[dev][0], myevent[dev][1], 0);
        cudaStreamWaitEvent(streams[dev][0], myevent[dev][0], 0);
        magma_int_t ie_loc   = min(ne_loc, NE - ne_loc*dev);
        magma_zgetmatrix_async( N, ie_loc, dE(dev, 0, 0), ldde, E+lde*ne_loc*dev, lde, streams[dev][0] );
        cudaEventRecord(myevent[dev][0], streams[dev][0]);
    }


    for( magma_int_t dev = 0; dev < ngpu; ++dev ) {
        cudaSetDevice( dev );
        magmablasSetKernelStream(streams[dev][0]);
        cudaStreamWaitEvent(streams[dev][0], myevent[dev][0], 0);
        cudaDeviceSynchronize(); // no need for synchronize
        magma_free(dwork[dev]);
        magma_free(dE[dev]);
        for( magma_int_t i = 0; i < nbevents; ++i ) {
            cudaEventDestroy(myevent[dev][i]);           
        }
        for( magma_int_t i = 0; i < nstream; ++i ) {
            magma_queue_destroy( streams[dev][i] );
        }
    }

    cudaSetDevice( cdev );
    magmablasSetKernelStream( cstream );
    return MAGMA_SUCCESS;

}
#undef V
#undef T
#undef dE
////////////////////////////////////////////////////////////////////////////////////////////////////



