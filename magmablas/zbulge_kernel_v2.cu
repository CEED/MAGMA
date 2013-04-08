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
#include "magma_bulge.h"
#include <cblas.h>

//#include "magma_zbulgeinc.h"
// === Define what BLAS to use ============================================
#define PRECISION_z

// === End defining what BLAS to use ======================================

extern "C" {

    void magma_ztrdtype1cbHLsym_withQ_v2(magma_int_t n, magma_int_t nb, cuDoubleComplex *A, magma_int_t lda, cuDoubleComplex *V, magma_int_t ldv, cuDoubleComplex *TAU,
                                         magma_int_t st, magma_int_t ed, magma_int_t sweep, magma_int_t Vblksiz, cuDoubleComplex *work);

    void magma_ztrdtype2cbHLsym_withQ_v2(magma_int_t n, magma_int_t nb, cuDoubleComplex *A, magma_int_t lda, cuDoubleComplex *V, magma_int_t ldv, cuDoubleComplex *TAU,
                                         magma_int_t st, magma_int_t ed, magma_int_t sweep, magma_int_t Vblksiz, cuDoubleComplex *work);

    void magma_ztrdtype3cbHLsym_withQ_v2(magma_int_t n, magma_int_t nb, cuDoubleComplex *A, magma_int_t lda, cuDoubleComplex *V, magma_int_t ldv, cuDoubleComplex *TAU,
                                         magma_int_t st, magma_int_t ed, magma_int_t sweep, magma_int_t Vblksiz, cuDoubleComplex *work);

    void magma_zlarfxsym_v2(magma_int_t n, cuDoubleComplex *A, magma_int_t lda, cuDoubleComplex *V, cuDoubleComplex *TAU, cuDoubleComplex *work);

}

__device__ void zsum_reduce( int n, int i, cuDoubleComplex* x )
{
    __syncthreads();
    if ( n >  128 ) { if ( i <  128 && i +  128 < n ) { x[i] += x[i+ 128]; }  __syncthreads(); }
    if ( n >   64 ) { if ( i <   64 && i +   64 < n ) { x[i] += x[i+  64]; }  __syncthreads(); }
    if ( n >   32 ) { if ( i <   32 && i +   32 < n ) { x[i] += x[i+  32]; }  __syncthreads(); }
    // probably don't need __syncthreads for < 16 threads
    // because of implicit warp level synchronization.
    if ( n >   16 ) { if ( i <   16 && i +   16 < n ) { x[i] += x[i+  16]; }  __syncthreads(); }
    if ( n >    8 ) { if ( i <    8 && i +    8 < n ) { x[i] += x[i+   8]; }  __syncthreads(); }
    if ( n >    4 ) { if ( i <    4 && i +    4 < n ) { x[i] += x[i+   4]; }  __syncthreads(); }
    if ( n >    2 ) { if ( i <    2 && i +    2 < n ) { x[i] += x[i+   2]; }  __syncthreads(); }
    if ( n >    1 ) { if ( i <    1 && i +    1 < n ) { x[i] += x[i+   1]; }  __syncthreads(); }
}


 __device__ void sum_reduce(int n, int i, double* x )
{
    __syncthreads();
    if ( n >  128 ) { if ( i <  128 && i +  128 < n ) { x[i] += x[i+ 128]; }  __syncthreads(); }
    if ( n >   64 ) { if ( i <   64 && i +   64 < n ) { x[i] += x[i+  64]; }  __syncthreads(); }
    if ( n >   32 ) { if ( i <   32 && i +   32 < n ) { x[i] += x[i+  32]; }  __syncthreads(); }
    // probably don't need __syncthreads for < 16 threads
    // because of implicit warp level synchronization.
    if ( n >   16 ) { if ( i <   16 && i +   16 < n ) { x[i] += x[i+  16]; }  __syncthreads(); }
    if ( n >    8 ) { if ( i <    8 && i +    8 < n ) { x[i] += x[i+   8]; }  __syncthreads(); }
    if ( n >    4 ) { if ( i <    4 && i +    4 < n ) { x[i] += x[i+   4]; }  __syncthreads(); }
    if ( n >    2 ) { if ( i <    2 && i +    2 < n ) { x[i] += x[i+   2]; }  __syncthreads(); }
    if ( n >    1 ) { if ( i <    1 && i +    1 < n ) { x[i] += x[i+   1]; }  __syncthreads(); }
}


///////////////////////////////////////////////////////////
//// add -1 because of C
#define dA(i,j)   &(dA[((i)-(j)) + ldda*((j)-1)])
#define dAC(i,j)   &(dA[((i)-(j)) + ldda*((j))])

#define dV(i)     &(dV[(i)])
#define dTAU(i)   &(dTAU[(i)])

// nb is assumed to be < BLOCK_SIZE; if not, increase BLOCK_SIZE
#define BLOCK_SIZE 128

__device__ void zlarfxsym_v2(magma_int_t n, 
                             cuDoubleComplex *dA, magma_int_t ldda, 
                             cuDoubleComplex *dV, cuDoubleComplex *dTAU, 
                             cuDoubleComplex *dwork) 
{
/*
    WORK (workspace) double complex array, dimension N
*/

    magma_int_t j, i = threadIdx.x;
    cuDoubleComplex dtmp     = MAGMA_Z_ZERO;
    cuDoubleComplex c_half   =  MAGMA_Z_HALF;

    __shared__ cuDoubleComplex sum[ BLOCK_SIZE ];

    /* 
        X = tau A V 
        blasf77_zhemv("L", &n, TAU, A, &lda, V, &ione, &c_zero, work, &ione);
    */
    for(j = 0; j< i; j++)
         dtmp += *dAC(i, j) * dV[j];
    for(j = i; j< n; j++)
         dtmp += MAGMA_Z_CNJG(*dAC(j, i)) * dV[j];

    dwork[i] = dTAU[0] * dtmp;

    /* compute dtmp= X'*V */
    sum[i] = MAGMA_Z_CNJG( dwork[i]) * dV[i];
    zsum_reduce(n, i, sum);

    /* compute 1/2 X'*V*t = 1/2*dtmp*tau  */
    dtmp = sum[0] * c_half * dTAU[0];

    /*
       compute W=X-1/2VX'Vt = X - dtmp*V 
       blasf77_zaxpy(&n, &dtmp, V, &ione, work, &ione); 
    */
    dwork[i] -= dtmp * dV[i]; 

    /* 
       performs the symmetric rank 2 operation A := alpha*x*y' + alpha*y*x' + A 
       blasf77_zher2("L", &n, &c_neg_one, work, &ione, V, &ione, A, &lda);
    */
    __syncthreads();
    for(j=0; j<=i; j++)
       *dAC(i, j) -= dwork[i]*MAGMA_Z_CNJG( dV[j] ) + dV[i]*MAGMA_Z_CNJG( dwork[j] ); 

    *dAC(i, i) = MAGMA_Z_MAKE( MAGMA_Z_REAL(*dAC(i, i)), 0.);
    // synch the routine
    __syncthreads();

}

///////////////////////////////////////////////////////////
//                  TYPE 1-BAND Householder
///////////////////////////////////////////////////////////
__device__ void zlarfg(int n, cuDoubleComplex *dA, cuDoubleComplex *dx,
                       cuDoubleComplex *dtau)
{
    const int i = threadIdx.x;
    __shared__ cuDoubleComplex scale;
    __shared__ double sum[ BLOCK_SIZE ], beta;
    cuDoubleComplex alpha;

#if (defined(PRECISION_s) || defined(PRECISION_d))
#else
    double alphar;
    __shared__ double alphai;
#endif


#if (defined(PRECISION_s) || defined(PRECISION_d))
    if( n <= 1 ) {
#else
    if( n <= 0 ) {
#endif
        *dtau = MAGMA_Z_ZERO;
        return;
    }

/*
    sum[i] = MAGMA_D_ZERO
*/    
/*
#if (defined(PRECISION_c) || defined(PRECISION_z))
    if (( n == 1 ) &&( i==0 ) ){
       sum[0] = MAGMA_D_ZERO
    }
#endif
*/

    /* Compute the norm of dx
      XNORM = DZNRM2( N-1, X, INCX )
    */
    if (i<n-1){
#if (defined(PRECISION_s) || defined(PRECISION_d))
         {
         double re = dx[i];
         sum[i] = re*re;
         }
#else
         {
         double re = MAGMA_Z_REAL(dx[i]), im = MAGMA_Z_IMAG(dx[i]);
         sum[i] = re*re + im*im;
         }
#endif
        sum_reduce( n-1, i, sum );
    }


    if ( i == 0 ) {
    alpha = *dA;
#if (defined(PRECISION_s) || defined(PRECISION_d))
    beta = sqrt(sum[0]);
#else
    alphar = MAGMA_Z_REAL(alpha);
    alphai = MAGMA_Z_IMAG(alpha);
    if ( n == 1 )
        beta = MAGMA_D_ZERO;
    else
        beta = sqrt(sum[0]);
#endif 
    }
    __syncthreads();



#if (defined(PRECISION_s) || defined(PRECISION_d))
    if( beta == 0) {
#else
    if( beta == 0 && alphai == 0) {
#endif
        *dtau = MAGMA_Z_ZERO;
        return;
    }

    if ( i == 0 ) {
#if (defined(PRECISION_s) || defined(PRECISION_d))
            beta  = beta*beta + alpha*alpha;
            beta  = sqrt(beta);
            beta  = -copysign( beta, alpha );

            // todo: deal with badly scaled vectors (see lapack's larfg)
            *dtau = (beta - alpha) / beta;
            *dA = beta;

            scale = 1. / (alpha - beta);
#else
            beta  = beta*beta + alphar*alphar + alphai*alphai;
            beta  = sqrt(beta);
            beta  = -copysign( beta, alphar );

            // todo: deal with badly scaled vectors (see lapack's larfg)
            *dtau = MAGMA_Z_MAKE((beta - alphar)/beta, -alphai/beta);

            *dA = MAGMA_Z_MAKE(beta, 0.);

            alpha = MAGMA_Z_MAKE( MAGMA_Z_REAL(alpha) - beta, MAGMA_Z_IMAG(alpha));
            scale = MAGMA_Z_DIV( MAGMA_Z_ONE, alpha);
#endif
    }

    // scale x
    __syncthreads();
    if ( i < n-1)
        dx[i] = MAGMA_Z_MUL(dx[i], scale);



    // synch the routine
    __syncthreads();
    
}


__global__
void magma_ztrdtype1cbHLsym_withQ_v2_gpu_kernel(cuDoubleComplex *dA, int ldda,
                                                cuDoubleComplex *dV, cuDoubleComplex *dTAU,
                                                int st, int len, cuDoubleComplex *dwork)
{
       const int i = threadIdx.x;

       if (i < len) {
          /*
             V(0)  = c_one;
             cblas_zcopy(len-1, A(st+1, st-1), ione, V(1), ione);
             memset(A(st+1, st-1), 0, (len-1)*sizeof(cuDoubleComplex));
          */
          if (i==0){
             dV[0] = MAGMA_Z_ONE;
          } else {
             dV[i] = *dA(st+i, st-1);
             *dA(st+i, st-1) = MAGMA_Z_ZERO;
          }
       
          /*
             Eliminate the col  at st-1
             lapackf77_zlarfg( &len, A(st, st-1), V(1), &ione, TAU );
          */
          __syncthreads();
          zlarfg(len, dA(st,st-1), dV(1), dTAU);

          /*
             apply left and right on A(st:ed,st:ed)
             magma_zlarfxsym_v2(len, A(st,st), lda-1, V, TAU, work);
          */
          zlarfxsym_v2(len, dA(st,st), ldda, dV, dTAU, dwork);
       }
}

extern "C" void
magma_ztrdtype1cbHLsym_withQ_v2_gpu(magma_int_t n, magma_int_t nb, 
                                    cuDoubleComplex *dA, magma_int_t ldda, 
                                    cuDoubleComplex *dV, magma_int_t lddv, 
                                    cuDoubleComplex *dTAU,
                                    magma_int_t st, magma_int_t ed, magma_int_t sweep, 
                                    magma_int_t Vblksiz, cuDoubleComplex *dwork) 
{
/*
    WORK (workspace) double complex array, dimension N
*/
    magma_int_t vpos, taupos, len;

    if (nb > BLOCK_SIZE)
       printf("magma_ztrdtype1cbHLsym_withQ_v2_gpu: BLOCK_SIZE should be > %d\n", nb);
 
    magma_bulge_findVTAUpos(n, nb, Vblksiz, sweep-1, st-1, lddv, &vpos, &taupos);
    //printf("voici vpos %d taupos %d  tpos %d  blkid %d \n", vpos, taupos, tpos, blkid);

    len     = ed-st+1;

    /* === Compute the following using one multiprocessor with BLOCK_SIZE threads ===
       *V(vpos)  = c_one;
       cblas_zcopy(len-1, A(st+1, st-1), ione, V(vpos+1), ione);
       memset(A(st+1, st-1), 0, (len-1)*sizeof(cuDoubleComplex));

       // Eliminate the col  at st-1 
       lapackf77_zlarfg( &len, A(st, st-1), V(vpos+1), &ione, TAU(taupos) );

       // apply left and right on A(st:ed,st:ed)
       magma_zlarfxsym_v2(len, A(st,st), lda-1, V(vpos), TAU(taupos), work);
    */ 
    magma_ztrdtype1cbHLsym_withQ_v2_gpu_kernel<<<1, BLOCK_SIZE>>>(dA, ldda, dV+vpos,
                                                                  dTAU+taupos,
                                                                  st, len, dwork);
}
#undef dA
#undef dV
#undef dTAU

///////////////////////////////////////////////////////////
//                  TYPE 1-LPK Householder
///////////////////////////////////////////////////////////
//// add -1 because of C
#define A(i,j)   &(A[((i)-(j)) + lda*((j)-1)])
#define V(i)     &(V[(i)])
#define TAU(i)   &(TAU[(i)])
extern "C" void
magma_ztrdtype2cbHLsym_withQ_v2_fake(magma_int_t n, magma_int_t nb, cuDoubleComplex *A, magma_int_t lda, cuDoubleComplex *V, magma_int_t ldv, cuDoubleComplex *TAU,
                                magma_int_t st, magma_int_t ed, magma_int_t sweep, magma_int_t Vblksiz, cuDoubleComplex *work) {

    /*
     WORK (workspace) double complex array, dimension NB
    */

    magma_int_t ione = 1;
    magma_int_t vpos, taupos;

    cuDoubleComplex conjtmp;

    cuDoubleComplex c_one = MAGMA_Z_ONE;

    magma_int_t ldx = lda-1;
    magma_int_t len = ed - st + 1;
    magma_int_t lem = min(ed+nb, n) - ed;

    if(lem>0){
        magma_bulge_findVTAUpos(n, nb, Vblksiz, sweep-1, st-1, ldv, &vpos, &taupos);
        /* apply remaining right coming from the top block */
        lapackf77_zlarfx("R", &lem, &len, V(vpos), TAU(taupos), A(ed+1, st), &ldx, work);
    }
    if(lem>1){
        magma_bulge_findVTAUpos(n, nb, Vblksiz, sweep-1, ed, ldv, &vpos, &taupos);

        /* remove the first column of the created bulge */
        *V(vpos)  = c_one;
        //memcpy(V(vpos+1), A(ed+2, st), (lem-1)*sizeof(cuDoubleComplex));
        cblas_zcopy(lem-1, A(ed+2, st), ione, V(vpos+1), ione);
        memset(A(ed+2, st),0,(lem-1)*sizeof(cuDoubleComplex));

        /* Eliminate the col at st */
        lapackf77_zlarfg( &lem, A(ed+1, st), V(vpos+1), &ione, TAU(taupos) );

        /* apply left on A(J1:J2,st+1:ed) */
        len = len-1; /* because we start at col st+1 instead of st. col st is the col that has been removed;*/
        conjtmp = MAGMA_Z_CNJG(*TAU(taupos));
        lapackf77_zlarfx("L", &lem, &len, V(vpos),  &conjtmp, A(ed+1, st+1), &ldx, work);
    }

}
#undef A
#undef V
#undef TAU

///////////////////////////////////////////////////////////
//                  TYPE 1-LPK Householder
///////////////////////////////////////////////////////////
//// add -1 because of C
#define A(i,j)   &(A[((i)-(j)) + lda*((j)-1)])
#define V(i)     &(V[(i)])
#define TAU(i)   &(TAU[(i)])
extern "C" void
magma_ztrdtype3cbHLsym_withQ_v2_fake(magma_int_t n, magma_int_t nb, cuDoubleComplex *A, magma_int_t lda, cuDoubleComplex *V, magma_int_t ldv, cuDoubleComplex *TAU,
                                magma_int_t st, magma_int_t ed, magma_int_t sweep, magma_int_t Vblksiz, cuDoubleComplex *work) {

    /*
     WORK (workspace) double complex array, dimension N
     */

    magma_int_t vpos, taupos;

    magma_bulge_findVTAUpos(n, nb, Vblksiz, sweep-1, st-1, ldv, &vpos, &taupos);

    magma_int_t len = ed-st+1;

    /* apply left and right on A(st:ed,st:ed)*/
    magma_zlarfxsym_v2(len, A(st,st), lda-1, V(vpos), TAU(taupos), work);

}
#undef A
#undef V
#undef TAU
///////////////////////////////////////////////////////////





