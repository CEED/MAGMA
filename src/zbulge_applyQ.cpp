/*
 * Copyright (c) 2011      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 *
 *     @author Azzam Haidar
 *     @author Stan Tomov
 *
 *     @precisions normal z -> s d c
 *
 */

#include "common_magma.h"
#include "magma_zbulgeinc.h"
#include <cblas.h>
// === Define what BLAS to use ============================================
#define PRECISION_z

// === End defining what BLAS to use ======================================
 

 
#ifdef __cplusplus
extern "C" {
#endif
void  magmablas_zlaset_identity(magma_int_t m, magma_int_t n,
                          cuDoubleComplex *A, magma_int_t lda);

#ifdef __cplusplus
}
#endif


////////////////////////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////////////////////////
#define dE(m,n)  (dE+(m) + LDE*(n))
#define dV(m)    (dV+(m))
#define dT(m)    (dT+(m))
#define E(m,n)   &(E[(m) + LDE*(n)])
#define V(m)     &(V[(m)])
#define TAU(m)   &(TAU[(m)])
#define T(m)     &(T[(m)])
extern "C" void magma_zbulge_applyQ(magma_int_t WANTZ, char SIDE, magma_int_t NE, magma_int_t N, magma_int_t NB, magma_int_t Vblksiz, cuDoubleComplex *E, magma_int_t LDE, cuDoubleComplex *V, cuDoubleComplex *TAU, cuDoubleComplex *T, magma_int_t *INFO, cuDoubleComplex *dV, cuDoubleComplex *dT, cuDoubleComplex *dE, magma_int_t copytype )
{

    //%===========================
    //%   local variables
    //%===========================
    magma_int_t LDT,LDV,blklen,firstcolj;
    magma_int_t bg, nbGblk,rownbm, k, m, n;
    magma_int_t st,ed,fst,vlen,vnb,colj,len;
    magma_int_t blkid, vpos,taupos,tpos;
    cuDoubleComplex *WORK;
    magma_int_t LWORK;
    magma_int_t  cur_blksiz,avai_blksiz, ncolinvolvd;
    magma_int_t  nbgr, colst, coled, version;
    magma_int_t blkcnt=-1;

    *INFO=0;
    version = 113;
    LDT     = Vblksiz;
    LDV     = NB+Vblksiz-1;
    blklen  = LDV*Vblksiz;
    nbGblk  = plasma_ceildiv((N-1),Vblksiz);
    //WORK    = (cuDoubleComplex *) malloc (LWORK*sizeof(cuDoubleComplex));

#if defined(USEMAGMA)
    /* find the size of the matrix T V*/
    findVTsiz(N, NB, Vblksiz, &blkcnt, &LDV);
    /* Copy E & V & T to the GPU in dE and dV and dT 
     * depending on copytype:
     * 1: mean copy only V
     * 2: mean copy V and T
     * 3: mean copy V, T and E
     * */
    if(copytype>0)cublasSetMatrix( LDV, blkcnt*Vblksiz, sizeof(cuDoubleComplex), V, LDV, dV, LDV);
    if(copytype>1)cublasSetMatrix( LDT, blkcnt*Vblksiz, sizeof(cuDoubleComplex), T, LDT, dT, LDT);
    if(copytype>2)cublasSetMatrix( N, NE, sizeof(cuDoubleComplex), E, N, dE, N);
    cuDoubleComplex *dwork;
    magma_int_t ldwork;
    ldwork  = NE;
    LWORK   = 2*N*max(Vblksiz,64);
    if( CUBLAS_STATUS_SUCCESS != cublasAlloc( LWORK, sizeof(cuDoubleComplex), (void**)&dwork) ) { 
       printf ("!!!!  magma_zbulge_applyQ cublasAlloc failed for: dwork\n" );       
       exit(-1);                                                           
    }
#else
    LWORK   = 2*N*max(Vblksiz,64);
    WORK    = (cuDoubleComplex *) malloc (LWORK*sizeof(cuDoubleComplex));
#endif

    /* SIDE LEFT  meaning apply E = Q*E = (q_1*q_2*.....*q_n) * E ==> so traverse Vs in reverse order (forward) from q_n to q_1
     *            Also E is splitten by row meaning each apply consist in a block of row (horizontal block) */
    /* SIDE RIGHT meaning apply E = E*Q = E * (q_1*q_2*.....*q_n) ==> so tarverse Vs in normal  order (forward) from q_1 to q_n 
     *            Also E is splitten by col meaning each apply consist in a block of col (vertical block) */

     /* WANTZ = 1 meaning E is IDENTITY so form Q using optimized update. 
      *         So we use the reverse order from small q to large one, 
      *         so from q_n to q_1 so Left update to Identity.
      *         Use version 113 because in 114 we need to update the whole matrix and not in icreasing order.
      * WANTZ = 2 meaning E is a full matrix and need to be updated from Left or Right so use normal update
      * */
    if(WANTZ==1) 
    {
        version=113;
        SIDE='L';
        //set the matrix to Identity here to avoid copying it from the CPU
        magmablas_zlaset_identity(N, N, dE, N);        
    }
    


    printf("  APPLY Q_v115 GPU with  N %d   NB %d   Vblksiz %d SIDE %c version %d WANTZ %d \n",N,NB,Vblksiz,SIDE,version,WANTZ);


    magma_int_t N2=N/2;
    magma_int_t N1=N-N2;   
#if defined(USESTREAM)
    //static cudaStream_t stream[2];
    //cudaStreamCreate(&stream[0]);
    //cudaStreamCreate(&stream[1]);
#endif
    

    if(SIDE=='L'){
    if(version==113){            
        for (bg = nbGblk; bg>0; bg--)
        {
           firstcolj = (bg-1)*Vblksiz + 1;
           rownbm    = plasma_ceildiv((N-(firstcolj+1)),NB);
           if(bg==nbGblk) rownbm    = plasma_ceildiv((N-(firstcolj)),NB);  // last blk has size=1 used for complex to handle A(N,N-1)
           for (m = rownbm; m>0; m--)
           {
               vlen = 0;
               vnb  = 0;
               colj      = (bg-1)*Vblksiz; // for k=0;I compute the fst and then can remove it from the loop
               fst       = (rownbm -m)*NB+colj +1;
               for (k=0; k<Vblksiz; k++)
               {
                   colj     = (bg-1)*Vblksiz + k;
                   st       = (rownbm -m)*NB+colj +1;
                   ed       = min(st+NB-1,N-1);
                   if(st>ed)break;
                   if((st==ed)&&(colj!=N-2))break;
                   vlen=ed-fst+1;
                   vnb=k+1;
               }        
               colst     = (bg-1)*Vblksiz;
               findVTpos(N,NB,Vblksiz,colst,fst, &vpos, &taupos, &tpos, &blkid);
               //printf("voici bg %d m %d  vlen %d  vnb %d fcolj %d vpos %d taupos %d \n",bg,m,vlen, vnb,colst+1,vpos+1,taupos+1);
               if((vlen>0)&&(vnb>0)){
#if defined(USEMAGMA)
                       if(WANTZ==1){
                          len =  N-colst;    
                          magma_zlarfb_gpu( 'L', 'N', 'F', 'C', vlen, len, vnb, dV(vpos), LDV, dT(tpos), LDT, dE(fst,colst), LDE, dwork, len);
                       }else{
                          magma_zlarfb_gpu( 'L', 'N', 'F', 'C', vlen, NE, vnb, dV(vpos), LDV, dT(tpos), LDT, dE(fst,0), LDE, dwork, NE);
                       }
                          // magma_zormqr2_gpu('L', 'N', vlen, N, vnb, dV(vpos), LDV, TAU(taupos), dE(fst,0), LDE, V(vpos), LDV, INFO );
#else
                       if(WANTZ==1){
                          len =  N-colst;    
                          lapackf77_zlarfb( "L", "N", "F", "C", &vlen, &len, &vnb, V(vpos), &LDV, T(tpos), &LDT, E(fst,colst), &LDE,  WORK, &len); 
                       }else{
                          lapackf77_zlarfb( "L", "N", "F", "C", &vlen, &NE, &vnb, V(vpos), &LDV, T(tpos), &LDT, E(fst,0), &LDE,  WORK, &NE); 
                       }
                           //DORMQR( "L", "N", &vlen, &N, &vnb, V(vpos), &LDV, TAU(taupos), E(fst,0), &LDE,  WORK, &LWORK, INFO );
                       //DORMQR_BLG( "L", "N", &vlen, &N, &vnb, &NB, V(vpos), &LDV, TAU(taupos), E(fst,0), &LDE,  WORK, &LWORK, INFO );
#endif           
               }           
               if(*INFO!=0) 
                       printf("ERROR DORMQR INFO %d \n",*INFO);
           }
        }
    }else if(version==114){
        rownbm    = plasma_ceildiv((N-1),NB);
        for (m = rownbm; m>0; m--)
        {
           ncolinvolvd = min(N-1, m*NB);
           avai_blksiz=min(Vblksiz,ncolinvolvd);
           nbgr = plasma_ceildiv(ncolinvolvd,avai_blksiz);
           for (n = nbgr; n>0; n--)
           {
               vlen = 0;
               vnb  = 0;
               cur_blksiz = min(ncolinvolvd-(n-1)*avai_blksiz, avai_blksiz);
               colst = (n-1)*avai_blksiz;
               coled = colst + cur_blksiz -1;
               fst   = (rownbm -m)*NB+colst +1;
               for (colj=colst; colj<=coled; colj++)
               {
                   st       = (rownbm -m)*NB+colj +1;
                   ed       = min(st+NB-1,N-1);
                   if(st>ed)break;
                   if((st==ed)&&(colj!=N-2))break;
                   vlen=ed-fst+1;
                   vnb=vnb+1;
               }        
               findVTpos(N,NB,Vblksiz,colst,fst, &vpos, &taupos, &tpos, &blkid);
               //printf("voici bg %d m %d  vlen %d  vnb %d fcolj %d vpos %d taupos %d \n",bg,m,vlen, vnb,colst+1,vpos+1,taupos+1);
               if((vlen>0)&&(vnb>0))
                   //DORMQR( "L", "N", &vlen, &N, &vnb, V(vpos), &LDV, TAU(taupos), E(fst,0), &LDE,  WORK, &LWORK, INFO );
                   lapackf77_zlarfb( "L", "N", "F", "C", &vlen, &NE, &vnb, V(vpos), &LDV, T(tpos), &LDT, E(fst,0), &LDE,  WORK, &NE);       
               if(*INFO!=0) 
                       printf("ERROR DORMQR INFO %d \n",*INFO);
       
           }
        }
    }
    }else if (SIDE=='R'){
        for (bg =1; bg<=nbGblk; bg++)
        {
           firstcolj = (bg-1)*Vblksiz + 1;
           rownbm    = plasma_ceildiv((N-(firstcolj+1)),NB);
           if(bg==nbGblk) rownbm    = plasma_ceildiv((N-(firstcolj)),NB);  // last blk has size=1 used for complex to handle A(N,N-1)
           for (m = 1; m<=rownbm; m++)
           {
               vlen = 0;
               vnb  = 0;
               // for k=0;I compute the fst and then can remove it from the loop
               colj     = (bg-1)*Vblksiz; 
               fst      = (rownbm -m)*NB+colj +1;
               for (k=0; k<Vblksiz; k++)
               {
                   colj     = (bg-1)*Vblksiz + k;
                   st       = (rownbm -m)*NB+colj +1;
                   ed       = min(st+NB-1,N-1);
                   if(st>ed)break;
                   if((st==ed)&&(colj!=N-2))break;
                   vlen=ed-fst+1;
                   vnb=k+1;
               }        
               colj     = (bg-1)*Vblksiz;
               findVTpos(N,NB,Vblksiz,colj,fst, &vpos, &taupos, &tpos, &blkid);
               //printf("voici bg %d m %d  vlen %d  vnb %d fcolj %d vpos %d taupos %d \n",bg,m,vlen, vnb,colj,vpos,taupos);
               if((vlen>0)&&(vnb>0)){
#if defined(USEMAGMA)
                #if defined(USESTREAM)
                   magmablasSetKernelStream(stream[0]);                       
                   magma_zlarfb_gpu( 'R', 'N', 'F', 'C', N1, vlen, vnb, dV(vpos), LDV, dT(tpos), LDT, dE(0, fst), LDE, dwork, N1);
                   magmablasSetKernelStream(stream[1]);        
                   magma_zlarfb_gpu( 'R', 'N', 'F', 'C', N2, vlen, vnb, dV(vpos), LDV, dT(tpos), LDT, dE(N1, fst), LDE, &dwork[N1*Vblksiz], N2);
                #else
                   magma_zlarfb_gpu( 'R', 'N', 'F', 'C', NE, vlen, vnb, dV(vpos), LDV, dT(tpos), LDT, dE(0, fst), LDE, dwork, NE);
                #endif
                   //magma_zormqr2_gpu('R', 'N',N, vlen, vnb, dV(vpos), LDV, TAU(tpos), dE(0, fst), LDE, V(vpos), LDV, INFO );
#else                       
                   //DORMQR( "R", "N", &N, &vlen, &vnb, V(vpos), &LDV, TAU(taupos), E(0,fst), &LDE,  WORK, &LWORK, INFO );
                   lapackf77_zlarfb( "R", "N", "F", "C", &NE, &vlen, &vnb, V(vpos), &LDV, T(tpos), &LDT, E(0, fst), &LDE,  WORK, &NE);       
#endif
               }
               if(*INFO!=0) 
                   printf("Right ERROR DORMQR INFO %d \n",*INFO);
       
           }
        }
    }else{
            printf("ERROR SIDE %d \n",SIDE);
    }

#if defined(USESTREAM)
    magmablasSetKernelStream(NULL);        
    cudaStreamDestroy( stream[0] );
    cudaStreamDestroy( stream[1] );
#endif

#if defined(USEMAGMA)
        //printf("difference  =  %e\n", cpu_gpu_ddiff(N, N, E, N, dE, N));
        // no need to send it if WANTZ=3 or 4 because I will use the GPU to make the GEMM with Q1 or the apply with V1.
        //if(WANTZ==5) cublasGetMatrix( N, N, sizeof(cuDoubleComplex), dE, N, E, N);
#endif        
}
#undef E
#undef V
#undef TAU
#undef T
#undef dE
#undef dV
#undef dT
////////////////////////////////////////////////////////////////////////////////////////////////////



