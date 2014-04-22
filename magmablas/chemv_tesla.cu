/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       Note: [ds] precisions generated from csymv_tesla.cu
       
*/
#include "common_magma.h"
#define PRECISION_c

/* The version for fermi can be found in chemv_fermi.cu */

#define hemv_bs          64
#define thread_x         64
#define thread_y          4
#define bank_shift       33
#define quarter_thread_x 16
#define half_thread_x    32

/*******************************************************************************
 *    Lower case, where n is multiple of block size (hemv_bs)
 */

__global__ void
chemv_kernel_tesla_L_special(
    int n, magmaFloatComplex alpha,
    const magmaFloatComplex * __restrict__ A, int lda,
    const magmaFloatComplex * __restrict__ x, int incx,
    magmaFloatComplex  beta,
    magmaFloatComplex * __restrict__ y, int incy,
    magmaFloatComplex * __restrict__ WC)
{
    int tx   = threadIdx.x;
    int ty   = threadIdx.y;
    int blkc = blockIdx.x;

    magmaFloatComplex res  = MAGMA_C_ZERO;
    magmaFloatComplex res_ = MAGMA_C_ZERO;
    magmaFloatComplex res1 = MAGMA_C_ZERO;

    // la must be at least half_thread_x*bank_shift = 32x33 = 1056;
    // quarter_thread_x*(thread_x+2) = 16*(64+2) = 1056
    __shared__ magmaFloatComplex la   [quarter_thread_x][thread_x+3]; /* Why +3? */
    __shared__ magmaFloatComplex buff [thread_x];
    __shared__ magmaFloatComplex buff2[thread_x];

    magmaFloatComplex tr[4];
    magmaFloatComplex b[4];

    int break_d   =  thread_x * blkc;
    const int td  = (thread_x * ty) + tx;
    int       tx_ = td % half_thread_x;
    int       ty_ = td / half_thread_x;

    WC +=  break_d + tx;
    x  += (break_d + tx)*incx;
    A  +=  break_d * (lda+1);
    A  += ty_*lda + tx_;

    // load x[block] into buff
    if ( ty == 0 ) {
        buff[tx] = x[0];
    } // obtain the vector x store in buff;

    tx = tx_; ty = ty_;

    #pragma unroll
    for(int j=0; j < half_thread_x; j += 8)
        la[0][ bank_shift * (ty_+j) + tx_] = A[ j * lda];
    __syncthreads();

    #pragma unroll
    for(int i=ty_*4; i<(ty_ * 4 + 4); i++) {
        if ( i < tx_ ) {
            la[0][bank_shift * tx_ + i] = cuConjf( la[0][ i * bank_shift + tx_] );
        }
        else
            la[0][bank_shift * tx_ + i] = la[0][ bank_shift * tx_ + i];
    }
    __syncthreads();

    #pragma unroll
    for(int j=0; j < 4; j++)
        res += cuConjf( la[0][bank_shift * tx_ + j + ty_ * 4] ) * buff[j + ty_ * 4];
    __syncthreads();

    la[0][bank_shift*tx_+ty_] = res;
    __syncthreads();

    if ( ty_== 0 ) {
        res1 = la[0][tx_*bank_shift+0]+la[0][tx_*bank_shift+1]
             + la[0][tx_*bank_shift+2]+la[0][tx_*bank_shift+3]
             + la[0][tx_*bank_shift+4]+la[0][tx_*bank_shift+5]
             + la[0][tx_*bank_shift+6]+la[0][tx_*bank_shift+7];
    }
    else {
        res1 = MAGMA_C_ZERO;
    }
    __syncthreads();

    res = MAGMA_C_ZERO;

    A += half_thread_x + half_thread_x*lda;

    #pragma unroll
    for(int j=0; j < half_thread_x; j += 8)
        la[0][bank_shift*(ty_+j)+tx_] = A[ j * lda];
    __syncthreads();

    #pragma unroll
    for(int i=ty_*4; i<(4+ty_*4); i++) {
        if ( i < tx_ ) {
            la[0][bank_shift*tx_+i] = cuConjf( la[0][bank_shift*i+tx_] );
        }
        else
            la[0][bank_shift*tx_+i] = la[0][bank_shift*tx_+i];
    }
    __syncthreads();

    #pragma unroll
    for(int j=0; j < 4; j++)
        res += cuConjf( la[0][bank_shift*tx_+j+ty_*4] ) * buff[half_thread_x + j + 4 * ty_];
    __syncthreads();
    la[0][bank_shift*tx_+ty_] = res;
    __syncthreads();

    magmaFloatComplex res2;
    res2 = MAGMA_C_ZERO;
    if ( ty_== 1 ) {
        res2 = la[0][tx_*bank_shift+0]+la[0][tx_*bank_shift+1]
             + la[0][tx_*bank_shift+2]+la[0][tx_*bank_shift+3]
             + la[0][tx_*bank_shift+4]+la[0][tx_*bank_shift+5]
             + la[0][tx_*bank_shift+6]+la[0][tx_*bank_shift+7];
    }
    else {
        res2 = MAGMA_C_ZERO;
    }
    __syncthreads();

    res = MAGMA_C_ZERO;

    A -= half_thread_x*lda;

    res_ = MAGMA_C_ZERO;

    #pragma unroll
    for(int j=0; j < half_thread_x; j += 8)
        tr[j/8] = A[ j * lda];

    #pragma unroll
    for(int j=0; j < 4; j++) {
        res += tr[j] * buff[ j*8 + ty_];
        la[0][bank_shift*(ty_+j*8)+tx_] = tr[j];
    }
    __syncthreads();

    #pragma unroll
    for(int j=0; j < 4; j++)
        res_ += cuConjf(la[0][bank_shift*tx_+j+ty_*4]) * buff[half_thread_x +j+ty_*4];
    __syncthreads();

    la[0][bank_shift*tx_+ty_] = res;
    __syncthreads();
    if ( ty_ == 1 ) {
        res2 = res2
             + la[0][tx_*bank_shift+0]+la[0][tx_*bank_shift+1]
             + la[0][tx_*bank_shift+2]+la[0][tx_*bank_shift+3]
             + la[0][tx_*bank_shift+4]+la[0][tx_*bank_shift+5]
             + la[0][tx_*bank_shift+6]+la[0][tx_*bank_shift+7];
    }
    else {
        res2 = MAGMA_C_ZERO;
    }
    __syncthreads();

    la[0][bank_shift*tx_+ty_] = res_;
    __syncthreads();
    if ( ty_ == 0 ) {
        res1 = res1
             + la[0][tx_*bank_shift+0]+la[0][tx_*bank_shift+1]
             + la[0][tx_*bank_shift+2]+la[0][tx_*bank_shift+3]
             + la[0][tx_*bank_shift+4]+la[0][tx_*bank_shift+5]
             + la[0][tx_*bank_shift+6]+la[0][tx_*bank_shift+7];
    }
    else {
        res1 = MAGMA_C_ZERO;
    }
    A -= half_thread_x;

    __syncthreads();
    tx = threadIdx.x;
    ty = threadIdx.y;

    if ( ty_ == 0  && ty == 0  )
        res = res1;
    else if ( ty_ == 1  && ty == 0  )
        res = res2;
    else {
            res = MAGMA_C_ZERO;
        }

    A -= ty_* lda;
    A -= tx_;

    A = A - lda * blkc * thread_x;
    x = x - blkc * thread_x * incx;

    A += 4 * ty* lda;
    A += tx;

    int wc_c = 0;
    int count = 0;

    tx_ = td % quarter_thread_x;
    ty_ = td / quarter_thread_x;

    WC -= tx;
    WC += tx_;

    if ( blkc * thread_x >= thread_x ) {
        #pragma unroll
        for( int i=0; i < thread_x; i += thread_x ) {
            res_ = MAGMA_C_ZERO;
            count++;
            if ( ty == 0 )
                buff2[tx] = x[i*incx];
            __syncthreads();

            #pragma unroll
            for( int k=0; k < 4; k++ ) {
                #pragma unroll
                for(int j=0; j < 4; j++)
                    tr[j] = A[j*lda];

                #pragma unroll
                for(int j=0; j < 4; j++) {
                    res += tr[j] * buff2[ quarter_thread_x * k + ty*4 + j];
                    la[j + ty*4][tx] = cuConjf(tr[j]) * buff[tx];
                }
                __syncthreads();

                res_ = MAGMA_C_ZERO;

                #pragma unroll
                for(int j=0; j < 4; j++) {
                    res_ += la[tx_][ty_*4+j];
                }
                b[k] = res_;
                __syncthreads();

                A += lda * quarter_thread_x;
            }

            #pragma unroll
            for(int k=0; k < 4; k++) {
                la[tx_][ty_+quarter_thread_x*k] = b[k];
            }
            __syncthreads();
            if ( ty_ < 4 ) {
                int k = ty_*quarter_thread_x;
                res_ = la[tx_][0+k] + la[tx_][1+k]
                     + la[tx_][2+k] + la[tx_][3+k]
                     + la[tx_][4+k] + la[tx_][5+k]
                     + la[tx_][6+k] + la[tx_][7+k]
                     + la[tx_][8+k] + la[tx_][9+k]
                     + la[tx_][10+k]+ la[tx_][11+k]
                     + la[tx_][12+k]+ la[tx_][13+k]
                     + la[tx_][14+k]+ la[tx_][15+k];
                WC[k + wc_c*lda ] =   res_;
            }

            wc_c++;
            __syncthreads();
        }
    }
    
    for(int i=thread_x; i < (blkc * thread_x); i += thread_x ) {
        res_ = MAGMA_C_ZERO;
        count++;
        if ( ty == 0 )
            buff2[tx] = x[i*incx];
        __syncthreads();

        #pragma unroll
        for( int k=0; k < 4; k++ ) {
            #pragma unroll
            for(int j=0; j < 4; j++)
                tr[j] = A[j*lda];

            #pragma unroll
            for(int j=0; j < 4; j++) {
                res += tr[j] * buff2[ quarter_thread_x*k + ty*4 + j];
                la[j + ty*4][tx] = cuConjf( tr[j] ) * buff[tx];
            }
            __syncthreads();

            res_ = MAGMA_C_ZERO;

            #pragma unroll
            for(int j=0; j < 4; j++)
                res_ += la[tx_][ty_*4+j];

            b[k] = res_;
            __syncthreads();

            A += lda * quarter_thread_x;
        }

        #pragma unroll
        for(int k=0; k < 4; k++) {
            la[tx_][ty_+quarter_thread_x*k] = b[k];
        }
        __syncthreads();
        if ( ty_ < 4 ) {
            int k = ty_*quarter_thread_x;
            res_ = la[tx_][0+k] + la[tx_][1+k]
                 + la[tx_][2+k] + la[tx_][3+k]
                 + la[tx_][4+k] + la[tx_][5+k]
                 + la[tx_][6+k] + la[tx_][7+k]
                 + la[tx_][8+k] + la[tx_][9+k]
                 + la[tx_][10+k]+ la[tx_][11+k]
                 + la[tx_][12+k]+ la[tx_][13+k]
                 + la[tx_][14+k]+ la[tx_][15+k];
            WC[k + wc_c*lda ] =   res_;
        }

        wc_c++;
        __syncthreads();
    }

    WC += tx;
    WC -= tx_;

    la[ty][tx] = res;
    __syncthreads();
    if ( ty == 0 ) {
        res = la[0][tx]+ la[1][tx]
            + la[2][tx]+ la[3][tx];
        WC[0+lda*(blkc)  ] =  res;
    }
}

/**************************************************************
 *    Lower case for generic sizes
 */
__global__ void
chemv_kernel_tesla_L_generic(
    int n, magmaFloatComplex alpha,
    const magmaFloatComplex * __restrict__ A, int lda,
    const magmaFloatComplex * __restrict__ x, int incx,
    magmaFloatComplex beta,
    magmaFloatComplex * __restrict__ y, int incy,
    magmaFloatComplex * __restrict__ WC,
    int m_mod_thread_x)
{
    int tx   = threadIdx.x;
    int ty   = threadIdx.y;
    int blkc = blockIdx.x;

    magmaFloatComplex res  = MAGMA_C_ZERO;
    magmaFloatComplex res_ = MAGMA_C_ZERO;
    magmaFloatComplex res1 = MAGMA_C_ZERO;

    __shared__ magmaFloatComplex la   [quarter_thread_x][thread_x+3];
    __shared__ magmaFloatComplex buff [thread_x];
    __shared__ magmaFloatComplex buff2[thread_x];

    magmaFloatComplex tr[4];
    magmaFloatComplex b[8];

    int break_d   =  thread_x * blkc;
    const int td  = (thread_x * ty) + tx;
    int       tx_ = td % half_thread_x;
    int       ty_ = td / half_thread_x;

    WC +=  break_d + tx;
    x += (break_d + tx) * incx;
    A +=  break_d * (lda+1);
    A += lda * ty_;

    int trackA;
    if ( blkc == ( gridDim.x - 1 ) ) {
        if ( ty == 0 ) {
            if ( tx > m_mod_thread_x ) {
                buff[tx] = MAGMA_C_ZERO;
            }
            else
                buff[tx]  = x[0];
        }
        if ( tx_ > m_mod_thread_x )
            trackA=m_mod_thread_x;
        else
            trackA=tx_;
        A += trackA;
    }
    else {
        if ( ty == 0 ) {
            buff[tx]  = x[0];
        }
        trackA = tx_;
        A += trackA;
    }

    // Somehow merging these two if - else creates problem
    // It could be a potential bug -- from synchronization or from cuda or compiler
    if ( blkc == ( gridDim.x - 1 ) ) {
        #pragma unroll
        for(int j=0; j < half_thread_x; j += 8) {
            if ( ( ty_ + j ) > m_mod_thread_x ) {
                la[0][bank_shift*(ty_+j)+tx_] = MAGMA_C_MAKE( 9999, 0 );
            }
            else
                la[0][bank_shift*(ty_+j)+tx_] = A[ j * lda];
        }
        A -= trackA;
    }
    else {
        #pragma unroll
        for(int j=0; j < half_thread_x; j += 8) {
            la[0][bank_shift*(ty_+j)+tx_] = A[ j * lda];
        }
    }
    tx = tx_;
    ty = ty_;
    __syncthreads();

    #pragma unroll
    for(int i=ty_*4; i<(ty_*4+4); i++) {
        if ( i < tx_ ) {
            la[0][bank_shift*tx_+i] = cuConjf(la[0][i*bank_shift+tx_]);
        }
        else
            la[0][bank_shift*tx_+i] = la[0][bank_shift*tx_+i];
    }
    __syncthreads();

    #pragma unroll
    for(int j=0; j < 4; j++)
        res += cuConjf(la[0][bank_shift*tx_+j+ty_*4]) * buff[j+ty_*4];
    __syncthreads();

    la[0][bank_shift*tx_+ty_] = res;
    __syncthreads();
    if ( ty_== 0 ) {
        res1 = la[0][tx_*bank_shift+0]
             + la[0][tx_*bank_shift+1]
             + la[0][tx_*bank_shift+2]
             + la[0][tx_*bank_shift+3]
             + la[0][tx_*bank_shift+4]
             + la[0][tx_*bank_shift+5]
             + la[0][tx_*bank_shift+6]
             + la[0][tx_*bank_shift+7];
    }
    else {
        res1 = MAGMA_C_ZERO;
    }
    __syncthreads();

    res = MAGMA_C_ZERO;

    if ( blkc == ( gridDim.x - 1 ) ) {
        if ( (tx_+half_thread_x) > m_mod_thread_x )
            trackA = m_mod_thread_x;
        else
            trackA = tx_ + half_thread_x;
        A += trackA+half_thread_x*lda;

        #pragma unroll
        for(int j=0; j < half_thread_x; j += 8) {
            if ( ( ty_ + j+half_thread_x ) > m_mod_thread_x ) {
                la[0][bank_shift*(ty_+j)+tx_] = MAGMA_C_MAKE( 99999, 0 );
            }
            else
                la[0][bank_shift*(ty_+j)+tx_] = A[ j * lda];
        }

        A -= trackA+half_thread_x*lda;
        A += tx_;
        A += half_thread_x + half_thread_x*lda;
    }
    else {
        A += half_thread_x + half_thread_x*lda;

        #pragma unroll
        for(int j=0; j < half_thread_x; j += 8) {
            la[0][bank_shift*(ty_+j)+tx_] = A[ j * lda];
        }
    }

    __syncthreads();
    #pragma unroll
    for(int i=ty_*4; i<(4+ty_*4); i++) {
        if ( i < tx_ ) {
            la[0][bank_shift*tx_+i] = cuConjf(la[0][bank_shift*i+tx_]);
        }
        else
            la[0][bank_shift*tx_+i] = la[0][bank_shift*tx_+i];
    }
    __syncthreads();

    #pragma unroll
    for(int j=0; j < 4; j++)
        res += cuConjf(la[0][bank_shift*tx_+j+ty_*4]) * buff[half_thread_x + j + 4 * ty_];
    __syncthreads();

    la[0][bank_shift*tx_+ty_] = res;
    __syncthreads();

    magmaFloatComplex res2;
    res2 = MAGMA_C_ZERO;
    if ( ty_== 1 ) {
        res2 = la[0][tx_*bank_shift+0]
             + la[0][tx_*bank_shift+1]
             + la[0][tx_*bank_shift+2]
             + la[0][tx_*bank_shift+3]
             + la[0][tx_*bank_shift+4]
             + la[0][tx_*bank_shift+5]
             + la[0][tx_*bank_shift+6]
             + la[0][tx_*bank_shift+7];
    }
    else {
        res2 = MAGMA_C_ZERO;
    }
    __syncthreads();

    res = MAGMA_C_ZERO;
    res_ = MAGMA_C_ZERO;

    A -= half_thread_x*lda;
    if ( blkc == ( gridDim.x - 1 ) ) {
        A -= tx_;
        if ( tx_ > m_mod_thread_x )
            trackA=m_mod_thread_x;
        else
            trackA=tx_;
        A += trackA;

        #pragma unroll
        for(int j=0; j < half_thread_x; j += 8)
            if ( ( ty_ + j ) > m_mod_thread_x ) {
                tr[j/8] = MAGMA_C_MAKE( 99999, 0 );
            }
            else
                tr[j/8] = A[ j * lda];
        A -= trackA;
        A += tx_;
    }
    else {
        #pragma unroll
        for(int j=0; j < half_thread_x; j += 8)
            tr[j/8] = A[ j * lda];
    }
    __syncthreads();

    #pragma unroll
    for(int j=0; j < 4; j++) {
        res += tr[j] * buff[ j*8 + ty_];
        la[0][bank_shift*(ty_+j*8)+tx_] = tr[j];
    }
    __syncthreads();

    #pragma unroll
    for(int j=0; j < 4; j++)
        res_ += cuConjf(la[0][bank_shift*tx_+j+ty_*4]) * buff[half_thread_x +j+ty_*4];
    __syncthreads();

    la[0][bank_shift*tx_+ty_] = res;
    __syncthreads();
    if ( ty_ == 1 ) {
        res2 = res2
             + la[0][tx_*bank_shift+0]
             + la[0][tx_*bank_shift+1]
             + la[0][tx_*bank_shift+2]
             + la[0][tx_*bank_shift+3]
             + la[0][tx_*bank_shift+4]
             + la[0][tx_*bank_shift+5]
             + la[0][tx_*bank_shift+6]
             + la[0][tx_*bank_shift+7];
    }
    else {
        res2 = MAGMA_C_ZERO;
    }
    __syncthreads();

    la[0][bank_shift*tx_+ty_] = res_;
    __syncthreads();

    if ( ty_ == 0 ) {
        res1 = res1
             + la[0][tx_*bank_shift+0]
             + la[0][tx_*bank_shift+1]
             + la[0][tx_*bank_shift+2]
             + la[0][tx_*bank_shift+3]
             + la[0][tx_*bank_shift+4]
             + la[0][tx_*bank_shift+5]
             + la[0][tx_*bank_shift+6]
             + la[0][tx_*bank_shift+7];
    }
    else {
        res1 = MAGMA_C_ZERO;
    }
    A -= half_thread_x;

    __syncthreads();
    tx = threadIdx.x;
    ty = threadIdx.y;

    if ( ty_ == 0  && ty == 0  )
        res = res1;
    else if ( ty_ == 1  && ty == 0  )
        res = res2;
    else {
        res = MAGMA_C_ZERO;
    }

    A -= ty_* lda;
    A -= tx_;

    A = A - lda*break_d;
    x = x - break_d*incx;

    A += 4 * ty* lda;

    if ( blkc  == ( gridDim.x - 1 ) ) {
        if ( tx <= m_mod_thread_x )
            A += tx;
        else
            A += m_mod_thread_x;
    }
    else{
        A += tx;
    }

    int wc_c = 0;
    int count = 0;

    tx_ = td % quarter_thread_x;
    ty_ = td / quarter_thread_x;

    WC -= tx;
    WC += tx_;

    #pragma unroll
    for(int j=0; j < 4; j++)
        b[j] =  buff[ty_*4+j];

    if ( break_d > 0 )
        #pragma unroll
        for( int i=0; i < thread_x; i += thread_x ) {
            res_ = MAGMA_C_ZERO;
            count++;
            if ( ty == 0 ) {
                buff2[tx]  = x[i*incx];
            }
            __syncthreads();

            #pragma unroll
            for( int k=0; k < 4; k++ ) {
                #pragma unroll
                for(int j=0; j < 4; j++)
                    tr[j] = A[j*lda];

                #pragma unroll
                for(int j=0; j < 4; j++) {
                    res += tr[j]*buff2[quarter_thread_x*k + ty*4+(j)];
                    la[( (j)+ty*4)][tx] = cuConjf(tr[j]);
                }
                __syncthreads();

                res_ = MAGMA_C_ZERO;

                #pragma unroll
                for(int j=0; j < 4; j++)
                    res_ += la[tx_][ty_*4+j]* b[j];
                b[4+k] = res_;
                __syncthreads();
                A += lda* quarter_thread_x;
            }

            #pragma unroll
            for(int k=0; k < 4; k++) {
                la[tx_][ty_+quarter_thread_x*k] = b[4+k];
            }
            __syncthreads();

            if ( ty_ < 4 ) {
                int k = ty_*quarter_thread_x;
                res_ = la[tx_][0+k] + la[tx_][1+k]
                     + la[tx_][2+k] + la[tx_][3+k]
                     + la[tx_][4+k] + la[tx_][5+k]
                     + la[tx_][6+k] + la[tx_][7+k]
                     + la[tx_][8+k] + la[tx_][9+k]
                     + la[tx_][10+k]+ la[tx_][11+k]
                     + la[tx_][12+k]+ la[tx_][13+k]
                     + la[tx_][14+k]+ la[tx_][15+k];
                WC[k + wc_c*lda ] =   res_;
            }
            wc_c++;
            __syncthreads();
        }

    for( int i=thread_x; i < break_d; i += thread_x ) {
        res_ = MAGMA_C_ZERO;
        count++;
        if ( ty == 0 )
            buff2[tx]  = x[i*incx];
        __syncthreads();

        #pragma unroll
        for( int k=0; k < 4; k++ ) {
            #pragma unroll
            for(int j=0; j < 4; j++)
                tr[j] = A[j*lda];
            #pragma unroll
            for(int j=0; j < 4; j++) {
                res += tr[j]*buff2[quarter_thread_x*k + ty*4+(j)];
                la[( (j)+ty*4)][tx] = cuConjf(tr[j]);
            }
            __syncthreads();

            res_ = MAGMA_C_ZERO;

            #pragma unroll
            for(int j=0; j < 4; j++)
                res_ += la[tx_][ty_*4+j]* b[j];
            b[4+k] = res_;
            __syncthreads();
            A += lda* quarter_thread_x;
        }

        #pragma unroll
        for(int k=0; k < 4; k++) {
            la[tx_][ty_+quarter_thread_x*k] = b[4+k];
        }
        __syncthreads();

        if ( ty_ < 4 ) {
            int k = ty_*quarter_thread_x;
            res_ = la[tx_][0+k] + la[tx_][1+k]
                 + la[tx_][2+k] + la[tx_][3+k]
                 + la[tx_][4+k] + la[tx_][5+k]
                 + la[tx_][6+k] + la[tx_][7+k]
                 + la[tx_][8+k] + la[tx_][9+k]
                 + la[tx_][10+k]+ la[tx_][11+k]
                 + la[tx_][12+k]+ la[tx_][13+k]
                 + la[tx_][14+k]+ la[tx_][15+k];
            WC[k + wc_c*lda ] =   res_;
        }
        wc_c++;
        __syncthreads();
    }

    WC += tx;
    WC -= tx_;
    la[ty][tx] = res;
    __syncthreads();

    if ( ty == 0 ) {
        res=la[0][tx]+ la[1][tx]+ la[2][tx]+ la[3][tx];
        WC[0+lda*(blkc)] = res;
    }
}

__global__ void
chemv_kernel_tesla_L_update(
    int n, magmaFloatComplex alpha,
    const magmaFloatComplex * __restrict__ A, int lda,
    const magmaFloatComplex * __restrict__ x, int incx,
    magmaFloatComplex beta,
    magmaFloatComplex * __restrict__ y, int incy,
    magmaFloatComplex * __restrict__ WC )
{
    int i;
    int tx  = threadIdx.x;
    int ind = blockIdx.x * thread_x + tx;
    magmaFloatComplex Ca;

    Ca = MAGMA_C_ZERO;
    WC += ind + lda * blockIdx.x;

    for(i = blockIdx.x*thread_x; i < n; i += thread_x) {
        Ca += WC[0];
        WC += thread_x;
    }
    if ( ind < n )
        y[ind * incy] = beta * y[ind * incy]  + alpha * Ca;
}


extern "C"
void magmablas_chemv_tesla_L(
    magma_int_t n, magmaFloatComplex alpha,
    const magmaFloatComplex *A, magma_int_t lda,
    const magmaFloatComplex *x, magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex *y, magma_int_t incy,
    magmaFloatComplex *dwork)
{
    magma_int_t blocks = (n - 1)/hemv_bs + 1;
    dim3 grid(blocks, 1, 1);
    dim3 threads(thread_x, thread_y, 1);

    /*
     * If matrix size is multiple of hemv_bs, we use a specific code.
     * otherwise, we call the generic case.
     */
    if ( n % hemv_bs == 0 ) {
        chemv_kernel_tesla_L_special<<< grid, threads, 0, magma_stream >>>
            (n, alpha, A, lda, x, incx, beta, y, incy, dwork);
    }
    else{
        magma_int_t m_mod_thread_x = (n % hemv_bs) - 1;
        chemv_kernel_tesla_L_generic<<< grid, threads, 0, magma_stream >>>
            (n, alpha, A, lda, x, incx, beta, y, incy, dwork, m_mod_thread_x);
    }

    dim3 threads_u(hemv_bs, 1, 1);
    chemv_kernel_tesla_L_update<<< grid, threads_u, 0, magma_stream >>>
        (n, alpha, A, lda, x, incx, beta, y, incy, dwork);
}

/**
    Purpose
    -------
    magmablas_chemv performs the matrix-vector operation:

        y := alpha*A*x + beta*y,

    where alpha and beta are scalars, x and y are n element vectors and
    A is an n by n hermitian matrix.

    Arguments
    ----------
    @param[in]
    uplo    CHARACTER*1.
            On entry, UPLO specifies whether the upper or lower
            triangular part of the array A is to be referenced as
            follows:
      -     = 'U':  Only the upper triangular part of A is to be referenced.
      -     = 'L':  Only the lower triangular part of A is to be referenced.


    @param[in]
    n       INTEGER.
            On entry, N specifies the order of the matrix A.
            N must be at least zero.

    @param[in]
    alpha   COMPLEX*16.
            On entry, ALPHA specifies the scalar alpha.

    @param[in]
    A       COMPLEX*16 array of DIMENSION ( LDA, n ).
            Before entry with UPLO = 'U' or 'u', the leading n by n
            upper triangular part of the array A must contain the upper
            triangular part of the hermitian matrix and the strictly
            lower triangular part of A is not referenced.
            Before entry with UPLO = 'L' or 'l', the leading n by n
            lower triangular part of the array A must contain the lower
            triangular part of the hermitian matrix and the strictly
            upper triangular part of A is not referenced.
            Note that the imaginary parts of the diagonal elements need
            not be set and are assumed to be zero.

    @param[in]
    lda     INTEGER.
            On entry, LDA specifies the first dimension of A as declared
            in the calling (sub) program. LDA must be at least
            max( 1, n ).
            It is recommended that lda is multiple of 16. Otherwise
            performance would be deteriorated as the memory accesses
            would not be fully coalescent.

    @param[in]
    x       COMPLEX*16 array of dimension at least
            ( 1 + ( n - 1 )*abs( INCX ) ).
            Before entry, the incremented array X must contain the n
            element vector x.

    @param[in]
    incx    INTEGER.
            On entry, INCX specifies the increment for the elements of
            X. INCX must not be zero.

    @param[in]
    beta    COMPLEX*16.
            On entry, BETA specifies the scalar beta. When BETA is
            supplied as zero then Y need not be set on input.

    @param[in,out]
    y       COMPLEX*16 array of dimension at least
            ( 1 + ( n - 1 )*abs( INCY ) ).
            Before entry, the incremented array Y must contain the n
            element vector y. On exit, Y is overwritten by the updated
            vector y.

    @param[in]
    incy    INTEGER.
            On entry, INCY specifies the increment for the elements of
            Y. INCY must not be zero.

    @ingroup magma_zblas2
    ********************************************************************/
extern "C"
magma_int_t
magmablas_chemv_tesla(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex alpha,
    const magmaFloatComplex *A, magma_int_t lda,
    const magmaFloatComplex *x, magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex *y, magma_int_t incy)
{
    int upper = (uplo == MagmaUpper);

    /*
     * Test the input parameters.
     */
    if ((! upper) && (uplo != MagmaLower)) {
        return -1;
    } else if ( n < 0 ) {
        return -2;
    } else if ( lda < max(1,n) ) {
        return -5;
    } else if ( incx == 0 ) {
        return -7;
    } else if ( incy == 0 ) {
        return -10;
    }

    /*
     * Quick return if possible.
     */
    if ( (n == 0) || ( MAGMA_C_EQUAL(alpha, MAGMA_C_ZERO) && MAGMA_C_EQUAL(beta, MAGMA_C_ONE) ) )
        return MAGMA_SUCCESS;

    /* TODO: Upper case is not implemented in MAGMA */
    if ( upper )
        cublasChemv( lapacke_uplo_const(uplo), n, alpha, A, lda, x, incx, beta, y, incy);
    else {
        magmaFloatComplex *dwork;
        magma_int_t blocks = (n - 1)/thread_x + 1;
        magma_int_t lwork  = lda * (blocks + 1);

        // TODO deal with error
        magma_cmalloc( &dwork, lwork );

        magmablas_chemv_tesla_L(n, alpha, A, lda, x, incx, beta, y, incy, dwork);

        magma_free( dwork );
    }
    return MAGMA_SUCCESS;
}
