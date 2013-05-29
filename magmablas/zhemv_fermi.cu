/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> c

*/
#include "common_magma.h"
#define PRECISION_z

/*The version for tesla can be found in zhemv_tesla.cu */
#if (GPUSHMEM >= 200)
#define magmablas_zhemv_200 magmablas_zhemv
#define magmablas_zhemv2_200 magmablas_zhemv2

#define NB_64
/*
     turning on NB_64, it will call routine blocksize = 64 
     otherwise it will can blocksize = 32 which is 10% faster in z,c precision
*/

#ifdef NB_64// using block size 64

#define zhemv_bs         64
#define thread_x         64
#define thread_y          4
#define bank_shift       33
#define quarter_thread_x 16
#define half_thread_x    32

#else // using block size 32

#define zhemv_bs         32
#define thread_x         32
#define thread_y          8
#define bank_shift       33
#define SWITCH  1400

#endif

/*******************************************************************************
 *     Functions for each specific cases - Lower case
 */

#ifdef NB_64

__global__ void
magmablas_zhemv_200_L_special( magma_int_t n, magmaDoubleComplex alpha,
                               const magmaDoubleComplex *A, magma_int_t lda,
                               const magmaDoubleComplex *x, magma_int_t incx,
                               magmaDoubleComplex  beta,
                               magmaDoubleComplex *y, magma_int_t incy,
                               magmaDoubleComplex *WC)
{
    magma_int_t tx   = threadIdx.x ;
    magma_int_t ty   = threadIdx.y ;
    magma_int_t blkc = blockIdx.x ;

    magmaDoubleComplex res  = MAGMA_Z_ZERO;
    magmaDoubleComplex res_ = MAGMA_Z_ZERO;
    magmaDoubleComplex res1 = MAGMA_Z_ZERO;

    __shared__ magmaDoubleComplex la   [quarter_thread_x][thread_x+2];
    __shared__ magmaDoubleComplex buff [thread_x];
    __shared__ magmaDoubleComplex buff2 [thread_x];

    magmaDoubleComplex tr[4];
    magmaDoubleComplex b[4];

    magma_int_t break_d   =  thread_x * blkc;
    const magma_int_t td  = (thread_x * ty ) + tx;
    magma_int_t       tx_ = td % half_thread_x;
    magma_int_t       ty_ = td / half_thread_x;

    WC +=  break_d + tx;
    x  += (break_d + tx ) * incx;
    A  +=  break_d * (lda+1);
    A  += ty_* lda + tx_ ;

    if( ty == 0 ){
        buff[tx] = x[0];
    } // obtain the vector x store in buff;

    tx = tx_ ; ty = ty_ ;

    #pragma unroll
    for(magma_int_t j =0; j<half_thread_x; j +=8)
        la[0][ bank_shift * (ty_+j) + tx_] =  A[ j * lda];
    __syncthreads();

    #pragma unroll
    for(magma_int_t  i=ty_*4; i<(ty_ * 4 + 4)  ; i++){
        if ( i < tx_ )   {
            la[0][bank_shift * tx_ + i] = cuConj( la[0][ i * bank_shift + tx_] ) ;
        }
        else
            la[0][bank_shift * tx_ + i] = la[0][ bank_shift * tx_ + i]  ;
    }
    __syncthreads();

    #pragma unroll
    for(magma_int_t j=0; j < 4 ; j++)
        res+= cuConj( la[0][bank_shift * tx_ + j + ty_ * 4] ) * buff[j + ty_ * 4];
    __syncthreads();

    la[0][bank_shift*tx_+ty_]= res ;
    __syncthreads();

    if( ty_== 0 )
      res1 = la[0][tx_*bank_shift+0]+la[0][tx_*bank_shift+1]
        +    la[0][tx_*bank_shift+2]+la[0][tx_*bank_shift+3]
        +    la[0][tx_*bank_shift+4]+la[0][tx_*bank_shift+5]
        +    la[0][tx_*bank_shift+6]+la[0][tx_*bank_shift+7];
    else
        {
            MAGMA_Z_SET2REAL(res1,0);
        }
    __syncthreads();


    MAGMA_Z_SET2REAL(res, 0) ;

    A+= half_thread_x + half_thread_x *lda ;

    #pragma unroll
    for(magma_int_t j =0; j<half_thread_x; j+=8)
        la[0][bank_shift*(ty_+j)+tx_] = A[ j * lda];
    __syncthreads();

    #pragma unroll
    for(magma_int_t  i=ty_*4; i<(4+ty_*4) ; i++){
        if ( i < tx_ )   {
            la[0][bank_shift*tx_+i] = cuConj( la[0][bank_shift*i+tx_] ) ;
        }
        else
            la[0][bank_shift*tx_+i] = la[0][bank_shift*tx_+i]  ;
    }
    __syncthreads();

    #pragma unroll
    for(magma_int_t j=0; j < 4 ; j++)
        res+= cuConj( la[0][bank_shift*tx_+j+ty_*4] ) * buff[half_thread_x + j + 4 * ty_];
    __syncthreads();
    la[0][bank_shift*tx_+ty_]= res ;
    __syncthreads();

    magmaDoubleComplex res2;
    MAGMA_Z_SET2REAL(res2,0);
    if( ty_== 1 )
        res2 = la[0][tx_*bank_shift+0]+la[0][tx_*bank_shift+1]
          +    la[0][tx_*bank_shift+2]+la[0][tx_*bank_shift+3]
          +    la[0][tx_*bank_shift+4]+la[0][tx_*bank_shift+5]
          +    la[0][tx_*bank_shift+6]+la[0][tx_*bank_shift+7];
    else
    {
        MAGMA_Z_SET2REAL(res2,0);
    }
    __syncthreads();

    MAGMA_Z_SET2REAL(res,0);

    A-=half_thread_x *lda ;

    MAGMA_Z_SET2REAL(res_,0);

    #pragma unroll
    for(magma_int_t j=0; j<half_thread_x; j+=8)
        tr[j/8] = A[ j * lda];

    #pragma unroll
    for(magma_int_t j=0; j < 4 ; j++){
        res += tr[j] * buff[ j*8 + ty_];
        la[0][bank_shift*(ty_+j*8)+tx_] = tr[j];
    }
    __syncthreads();

    #pragma unroll
    for(magma_int_t j=0; j < 4 ; j++)
        res_+= cuConj(la[0][bank_shift*tx_+j+ty_*4]) * buff[half_thread_x +j+ty_*4];
    __syncthreads();

    la[0][bank_shift*tx_+ty_]= res ;
    __syncthreads();
    if( ty_ == 1 )
        res2 = res2 
            +  la[0][tx_*bank_shift+0]+la[0][tx_*bank_shift+1]
            +  la[0][tx_*bank_shift+2]+la[0][tx_*bank_shift+3]
            +  la[0][tx_*bank_shift+4]+la[0][tx_*bank_shift+5]
            +  la[0][tx_*bank_shift+6]+la[0][tx_*bank_shift+7];
    else
        {
            MAGMA_Z_SET2REAL(res2,0);
        }
    __syncthreads();

    la[0][bank_shift*tx_+ty_]= res_ ;
    __syncthreads();
    if( ty_ == 0 ) {
        res1 = res1
            +  la[0][tx_*bank_shift+0]+la[0][tx_*bank_shift+1]
            +  la[0][tx_*bank_shift+2]+la[0][tx_*bank_shift+3]
            +  la[0][tx_*bank_shift+4]+la[0][tx_*bank_shift+5]
            +  la[0][tx_*bank_shift+6]+la[0][tx_*bank_shift+7];
    }
    else
        {
            MAGMA_Z_SET2REAL(res1,0);
        }
    A-=half_thread_x;

    __syncthreads();
    tx = threadIdx.x ;
    ty = threadIdx.y ;

    if( ty_ == 0  && ty == 0  )
        res = res1 ;
    else if( ty_ == 1  && ty == 0  )
        res = res2 ;
    else
        {
            MAGMA_Z_SET2REAL(res,0);
        }

    A-=ty_* lda  ;
    A-=tx_;

    A= A - lda * blkc * thread_x;
    x= x - blkc * thread_x  *incx  ;


    A+=4 * ty* lda  ;
    A+=tx;

    magma_int_t wc_c = 0 ;
    magma_int_t count = 0 ;

    tx_ = td % quarter_thread_x ;
    ty_ = td / quarter_thread_x ;

    WC-=tx ;
    WC+=tx_;

    if( blkc * thread_x >=thread_x)
        #pragma unroll
        for(magma_int_t i=0; i<thread_x; i += thread_x )
        {
            MAGMA_Z_SET2REAL(res_,0);
            count++;
            
            if( ty== 0 ) {
                buff2[tx]  = x[i*incx];
            }
            __syncthreads();

            #pragma unroll
            for( magma_int_t k=0;k<4;k++)
            {

                #pragma unroll
                for(magma_int_t j=0; j < 4 ; j++)
                    tr[j] = A[j*lda];

                #pragma unroll
                for(magma_int_t j=0; j < 4 ; j++)
                {
                    res += tr[j] * buff2[ quarter_thread_x * k + ty * 4 + j];
                    la[( j + ty * 4)][tx] = cuConj(tr[j]) * buff[tx];
                }
                __syncthreads();


                MAGMA_Z_SET2REAL(res_,0);

                #pragma unroll
                for(magma_int_t j=0; j < 4 ; j++)
                {
                    res_+=la[tx_][ty_*4+j] ;
                }
                b[k] = res_ ;
                __syncthreads();

                A += lda * quarter_thread_x ;
            }

            #pragma unroll
            for(magma_int_t k=0; k < 4 ; k++){
                la[tx_][ty_+quarter_thread_x*k]= b[k] ;
            }
            __syncthreads();
            if( ty_ < 4 ) {
                magma_int_t k = ty_*quarter_thread_x;
                res_ = la[tx_][0+k] + la[tx_][1+k]
                    +  la[tx_][2+k] + la[tx_][3+k]
                    +  la[tx_][4+k] + la[tx_][5+k]
                    +  la[tx_][6+k] + la[tx_][7+k]
                    +  la[tx_][8+k] + la[tx_][9+k]
                    +  la[tx_][10+k]+ la[tx_][11+k]
                    +  la[tx_][12+k]+ la[tx_][13+k]
                    +  la[tx_][14+k]+ la[tx_][15+k];
                WC[k + wc_c*lda ] =   res_;
            }

            wc_c++;
            __syncthreads();

        }

    for(magma_int_t  i=thread_x; i< (blkc * thread_x); i += thread_x )
    {
        MAGMA_Z_SET2REAL(res_,0);
        count++;
        if( ty== 0 ) {
            buff2[tx]  = x[i*incx];
        }
        __syncthreads();

        #pragma unroll
        for( magma_int_t k=0;k<4;k++)
        {
            #pragma unroll
            for(magma_int_t j=0; j < 4 ; j++)
                tr[j] = A[j*lda] ;
            

            #pragma unroll
            for(magma_int_t j=0; j < 4 ; j++)
            {
                res += tr[j] * buff2[quarter_thread_x*k + ty*4+(j)];
                la[( j + ty * 4)][tx] = cuConj( tr[j] )* buff[tx];
            }
            __syncthreads();

            MAGMA_Z_SET2REAL(res_,0);

            #pragma unroll
            for(magma_int_t j=0; j < 4 ; j++)
                res_+=la[tx_][ty_*4+j] ;

            b[k] = res_ ;
            __syncthreads();

            A += lda * quarter_thread_x ;
        }

        #pragma unroll
        for(magma_int_t k=0; k < 4 ; k++){
            la[tx_][ty_+quarter_thread_x*k]= b[k] ;
        }
        __syncthreads();
        if( ty_ < 4 ) {
            magma_int_t k = ty_*quarter_thread_x;
            res_ = la[tx_][0+k] + la[tx_][1+k]
                +  la[tx_][2+k] + la[tx_][3+k]
                +  la[tx_][4+k] + la[tx_][5+k]
                +  la[tx_][6+k] + la[tx_][7+k]
                +  la[tx_][8+k] + la[tx_][9+k]
                +  la[tx_][10+k]+ la[tx_][11+k]
                +  la[tx_][12+k]+ la[tx_][13+k]
                +  la[tx_][14+k]+ la[tx_][15+k];
            WC[k + wc_c*lda ] =   res_;
        }

        wc_c++;
        __syncthreads();
    }

    WC+=tx ;
    WC-=tx_;

    la[ty][tx]= res ;
    __syncthreads();
    if( ty == 0 ) {
        res = la[0][tx]+ la[1][tx]
            + la[2][tx]+ la[3][tx];
        WC[0+lda*(blkc)  ] =  res;
    }
}

/**************************************************************
 *    Lower case for generic sizes
 */
__global__ void
magmablas_zhemv_200_L_generic(magma_int_t n, magmaDoubleComplex alpha,
                              const magmaDoubleComplex *A, magma_int_t lda,
                              const magmaDoubleComplex *x, magma_int_t incx,
                              magmaDoubleComplex beta,
                              magmaDoubleComplex *y, magma_int_t incy,
                              magmaDoubleComplex *WC,
                              magma_int_t m_mod_thread_x)
{
    magma_int_t tx   = threadIdx.x ;
    magma_int_t ty   = threadIdx.y ;
    magma_int_t blkc = blockIdx.x ;

    magmaDoubleComplex res  = MAGMA_Z_ZERO;
    magmaDoubleComplex res_ = MAGMA_Z_ZERO;
    magmaDoubleComplex res1 = MAGMA_Z_ZERO;

    __shared__ magmaDoubleComplex la   [quarter_thread_x][thread_x+2];
    __shared__ magmaDoubleComplex buff [thread_x];
    __shared__ magmaDoubleComplex buff2[thread_x];

    magmaDoubleComplex tr[4];
    magmaDoubleComplex b[8];

    magma_int_t break_d   =  thread_x * blkc;
    const magma_int_t td  = (thread_x * ty ) + tx;
    magma_int_t       tx_ = td % half_thread_x;
    magma_int_t       ty_ = td / half_thread_x;

    WC+=  break_d + tx;
    x += (break_d + tx ) * incx;
    A +=  break_d * (lda+1);
    A += lda * ty_;

    magma_int_t trackA ;
    if( blkc == ( gridDim.x - 1 ) ) {
        if( ty == 0 ){
            if( tx > m_mod_thread_x )
            {
                MAGMA_Z_SET2REAL(buff[tx],0);
            }
            else
                buff[tx]  = x[0];
        }
        if ( tx_ > m_mod_thread_x )
            trackA=m_mod_thread_x;
        else
            trackA=tx_;
        A += trackA ;
    }
    else {
        if( ty == 0 ){
            buff[tx]  = x[0];
        }
        trackA = tx_;
        A += trackA ;
    }

    // Somehow merging these two if - else creates problem
    // It could be a potential bug -- from synchronization or from cuda or compiler
    if( blkc == ( gridDim.x - 1 ) ) {
        #pragma unroll
        for(magma_int_t j =0; j<half_thread_x; j+=8){
            if( ( ty_ + j ) > m_mod_thread_x )
            {
                MAGMA_Z_SET2REAL(la[0][bank_shift*(ty_+j)+tx_], 9999);
            }
            else
                la[0][bank_shift*(ty_+j)+tx_] =  A[ j * lda];
        }
        A-=trackA;
    }
    else {
        #pragma unroll
        for(magma_int_t j =0; j<half_thread_x; j+=8){
            la[0][bank_shift*(ty_+j)+tx_] = A[ j * lda];
        }
    }
    tx = tx_ ;
    ty = ty_ ;
    __syncthreads();

    #pragma unroll
    for(magma_int_t  i=ty_*4; i<(ty_*4+4)  ; i++){
        if ( i < tx_ )   {
            la[0][bank_shift*tx_+i] = cuConj(la[0][i*bank_shift+tx_]) ;
        }
        else
            la[0][bank_shift*tx_+i] = la[0][bank_shift*tx_+i]  ;
    }
    __syncthreads();

    #pragma unroll
    for(magma_int_t j=0; j < 4 ; j++)
        res += cuConj(la[0][bank_shift*tx_+j+ty_*4])* buff[j+ty_*4];
    __syncthreads();

    la[0][bank_shift*tx_+ty_]= res ;
    __syncthreads();
    if( ty_== 0 )
        res1 = la[0][tx_*bank_shift+0] 
            +  la[0][tx_*bank_shift+1]
            +  la[0][tx_*bank_shift+2]
            +  la[0][tx_*bank_shift+3]
            +  la[0][tx_*bank_shift+4]
            +  la[0][tx_*bank_shift+5]
            +  la[0][tx_*bank_shift+6]
            +  la[0][tx_*bank_shift+7];
    else
    {
        MAGMA_Z_SET2REAL(res1,0);
    }
    __syncthreads();


    MAGMA_Z_SET2REAL(res,0);

    if( blkc == ( gridDim.x - 1 ) ) {
        if ( (tx_+half_thread_x) > m_mod_thread_x )
            trackA = m_mod_thread_x;
        else
            trackA = tx_ + half_thread_x;
        A+= trackA+half_thread_x*lda ;

        #pragma unroll
        for(magma_int_t j =0; j<half_thread_x; j+=8){
            if( ( ty_ + j+half_thread_x ) > m_mod_thread_x )
            {
                MAGMA_Z_SET2REAL(la[0][bank_shift*(ty_+j)+tx_], 99999);
            }
            else
                la[0][bank_shift*(ty_+j)+tx_] =  A[ j * lda];
        }

        A-= trackA+half_thread_x*lda ;
        A+=tx_ ;
        A+= half_thread_x + half_thread_x *lda ;
    }
    else {
        A+= half_thread_x + half_thread_x *lda ;

        #pragma unroll
        for(magma_int_t j =0; j<half_thread_x; j+=8){
            la[0][bank_shift*(ty_+j)+tx_] = A[ j * lda];
        }
    }

    __syncthreads();
    #pragma unroll
    for(magma_int_t  i=ty_*4; i<(4+ty_*4) ; i++){
        if ( i < tx_ )   {
            la[0][bank_shift*tx_+i] = cuConj(la[0][bank_shift*i+tx_]) ;
        }
        else
            la[0][bank_shift*tx_+i] = la[0][bank_shift*tx_+i]  ;
    }
    __syncthreads();

    #pragma unroll
    for(magma_int_t j=0; j < 4 ; j++)
        res+= cuConj(la[0][bank_shift*tx_+j+ty_*4]) * buff[half_thread_x + j + 4 * ty_];
    __syncthreads();

    la[0][bank_shift*tx_+ty_]= res ;
    __syncthreads();

    magmaDoubleComplex res2;
    MAGMA_Z_SET2REAL(res2,0);
    if( ty_== 1 )
        res2 = la[0][tx_*bank_shift+0]
            +  la[0][tx_*bank_shift+1]
            +  la[0][tx_*bank_shift+2]
            +  la[0][tx_*bank_shift+3]
            +  la[0][tx_*bank_shift+4]
            +  la[0][tx_*bank_shift+5]
            +  la[0][tx_*bank_shift+6]
            +  la[0][tx_*bank_shift+7];
    else
    {
        MAGMA_Z_SET2REAL(res2,0);
    }
    __syncthreads();

    MAGMA_Z_SET2REAL(res,0);
    MAGMA_Z_SET2REAL(res_,0);

    A-=half_thread_x *lda ;
    if( blkc == ( gridDim.x - 1 ) ) {
        A-=tx_;
        if ( tx_ > m_mod_thread_x )
            trackA=m_mod_thread_x;
        else
            trackA=tx_;
        A+= trackA ;

        #pragma unroll
        for(magma_int_t j =0; j<half_thread_x; j+=8)
            if( ( ty_ + j ) > m_mod_thread_x )
            {
                MAGMA_Z_SET2REAL(tr[j/8], 99999);
            }
            else
                tr[j/8] = A[ j * lda];
        A-=trackA;
        A+=tx_;
    }
    else {
        #pragma unroll
        for(magma_int_t j =0; j<half_thread_x; j+=8)
            tr[j/8] = A[ j * lda];
    }
    __syncthreads();

    #pragma unroll
    for(magma_int_t j=0; j < 4 ; j++){
        res+= tr[j] * buff[ j*8 + ty_];
        la[0][bank_shift*(ty_+j*8)+tx_] = tr[j];
    }
    __syncthreads();

    #pragma unroll
    for(magma_int_t j=0; j < 4 ; j++)
        res_+= cuConj(la[0][bank_shift*tx_+j+ty_*4]) * buff[half_thread_x +j+ty_*4];
    __syncthreads();


    la[0][bank_shift*tx_+ty_]= res ;
    __syncthreads();
    if( ty_ == 1 )
        res2 = res2
            +  la[0][tx_*bank_shift+0]
            +  la[0][tx_*bank_shift+1]
            +  la[0][tx_*bank_shift+2]
            +  la[0][tx_*bank_shift+3]
            +  la[0][tx_*bank_shift+4]
            +  la[0][tx_*bank_shift+5]
            +  la[0][tx_*bank_shift+6]
            +  la[0][tx_*bank_shift+7];
    else
    {
        MAGMA_Z_SET2REAL(res2,0);
    }
    __syncthreads();

    la[0][bank_shift*tx_+ty_]= res_ ;
    __syncthreads();

    if( ty_ == 0 ) {
        res1 = res1
            +  la[0][tx_*bank_shift+0]
            +  la[0][tx_*bank_shift+1]
            +  la[0][tx_*bank_shift+2]
            +  la[0][tx_*bank_shift+3]
            +  la[0][tx_*bank_shift+4]
            +  la[0][tx_*bank_shift+5]
            +  la[0][tx_*bank_shift+6]
            +  la[0][tx_*bank_shift+7];
    }
    else
    {
        MAGMA_Z_SET2REAL(res1,0);
    }
    A-=half_thread_x;

    __syncthreads();
    tx = threadIdx.x ;
    ty = threadIdx.y ;

    if( ty_ == 0  && ty == 0  )
        res = res1 ;
    else if( ty_ == 1  && ty == 0  )
        res = res2 ;
    else
    {
        MAGMA_Z_SET2REAL(res,0);
    }

    A-=ty_* lda  ;
    A-=tx_;

    A= A - lda*break_d;
    x= x - break_d *incx  ;

    A+=4 * ty* lda  ;

    if( blkc  == ( gridDim.x - 1 ) ) {
        if(tx <= m_mod_thread_x )
            A+=tx;
        else
            A+=m_mod_thread_x;
    }
    else{
        A+=tx;
    }

    magma_int_t wc_c = 0 ;
    magma_int_t count = 0 ;

    tx_ = td % quarter_thread_x ;
    ty_ = td / quarter_thread_x ;

    WC-=tx ;
    WC+=tx_;

    #pragma unroll
    for(magma_int_t j=0; j < 4 ; j++)
        b[j] =  buff[ty_*4+j];

    if( break_d > 0)
        #pragma unroll
        for(magma_int_t  i=0; i< thread_x; i += thread_x ){
            MAGMA_Z_SET2REAL(res_,0);
            count++;
            if( ty== 0 ) {
                buff2[tx]  = x[i*incx];
            }
            __syncthreads();

            #pragma unroll
            for( magma_int_t k=0;k<4;k++){
                #pragma unroll
                for(magma_int_t j=0; j < 4 ; j++)
                    tr[j] = A[j*lda] ;

                #pragma unroll
                for(magma_int_t j=0; j < 4 ; j++){
                    res+=tr[j]*buff2[quarter_thread_x*k + ty*4+(j)];
                    la[( (j)+ty*4)][tx] = cuConj(tr[j]);
                }
                __syncthreads();

                MAGMA_Z_SET2REAL(res_, 0) ;

                #pragma unroll
                for(magma_int_t j=0; j < 4 ; j++)
                    res_+=la[tx_][ty_*4+j]* b[j] ;
                b[4+k] = res_ ;
                __syncthreads();
                A+=lda* quarter_thread_x ;
            }

            #pragma unroll
            for(magma_int_t k=0; k < 4 ; k++){
                la[tx_][ty_+quarter_thread_x*k]= b[4+k] ;
            }
            __syncthreads();

            if( ty_ < 4 ) {
                magma_int_t k = ty_*quarter_thread_x;
                res_ = la[tx_][0+k] + la[tx_][1+k] 
                    +  la[tx_][2+k] + la[tx_][3+k]
                    +  la[tx_][4+k] + la[tx_][5+k]
                    +  la[tx_][6+k] + la[tx_][7+k]
                    +  la[tx_][8+k] + la[tx_][9+k]
                    +  la[tx_][10+k]+ la[tx_][11+k]
                    +  la[tx_][12+k]+ la[tx_][13+k]
                    +  la[tx_][14+k]+ la[tx_][15+k];
                WC[k + wc_c*lda ] =   res_;
            }
            wc_c++;
            __syncthreads();
        }

    for(magma_int_t  i=thread_x; i<break_d; i += thread_x ){
        MAGMA_Z_SET2REAL(res_, 0) ;
        count++;
        if(ty == 0 )
            buff2[tx]  = x[i*incx];
        __syncthreads();

        #pragma unroll
        for( magma_int_t k=0;k<4;k++){
            #pragma unroll
            for(magma_int_t j=0; j < 4 ; j++)
                tr[j] = A[j*lda] ;
            #pragma unroll
            for(magma_int_t j=0; j < 4 ; j++){
                res+=tr[j]*buff2[quarter_thread_x*k + ty*4+(j)];
                la[( (j)+ty*4)][tx] = cuConj(tr[j]);
            }
            __syncthreads();

            MAGMA_Z_SET2REAL(res_, 0) ;

            #pragma unroll
            for(magma_int_t j=0; j < 4 ; j++)
                res_+=la[tx_][ty_*4+j]* b[j] ;
            b[4+k] = res_ ;
            __syncthreads();
            A+=lda* quarter_thread_x ;
        }

        #pragma unroll
        for(magma_int_t k=0; k < 4 ; k++){
            la[tx_][ty_+quarter_thread_x*k]= b[4+k] ;
        }
        __syncthreads();

        if( ty_ < 4 ) {
            magma_int_t k = ty_*quarter_thread_x;
            res_ = la[tx_][0+k] + la[tx_][1+k] 
                +  la[tx_][2+k] + la[tx_][3+k]
                +  la[tx_][4+k] + la[tx_][5+k]
                +  la[tx_][6+k] + la[tx_][7+k]
                +  la[tx_][8+k] + la[tx_][9+k]
                +  la[tx_][10+k]+ la[tx_][11+k]
                +  la[tx_][12+k]+ la[tx_][13+k]
                +  la[tx_][14+k]+ la[tx_][15+k];
            WC[k + wc_c*lda ] =   res_;
        }
        wc_c++;
        __syncthreads();
    }


    WC+=tx ;
    WC-=tx_;
    la[ty][tx]= res ;
    __syncthreads();

    if( ty == 0 ) {
        res=la[0][tx]+ la[1][tx]+ la[2][tx]+ la[3][tx] ;
        WC[0+lda*(blkc)] = res;
    }
}

__global__ void
magmablas_zhemv_200_L_update(magma_int_t n, magmaDoubleComplex alpha,
                         const magmaDoubleComplex* A, magma_int_t lda,
                         const magmaDoubleComplex *x, magma_int_t incx,
                         magmaDoubleComplex beta,
                         magmaDoubleComplex *y, magma_int_t incy,
                         magmaDoubleComplex *WC )
{
    magma_int_t i;
    magma_int_t tx  = threadIdx.x ;
    magma_int_t ind = blockIdx.x * thread_x + tx ;
    magmaDoubleComplex Ca;

    MAGMA_Z_SET2REAL(Ca, 0) ;
    WC+= ind + lda * blockIdx.x;

    for(i = blockIdx.x*thread_x; i<n; i+=thread_x){
        Ca += WC[0] ;
        WC += thread_x;
    }
    if( ind < n )
        y[ind * incy] = beta * y[ind * incy]  + alpha * Ca ;
}


extern "C"
void magmablas_zhemv_200_L(magma_int_t m, magmaDoubleComplex alpha,
                           const magmaDoubleComplex *A, magma_int_t lda,
                           const magmaDoubleComplex *X, magma_int_t incx,
                           magmaDoubleComplex beta,
                           magmaDoubleComplex *Y, magma_int_t incy,
                           magmaDoubleComplex *dC_work)
{
    magma_int_t blocks;

    if (m % zhemv_bs==0)
        blocks = m / zhemv_bs;
    else
        blocks = m / zhemv_bs + 1;

    dim3 grid(blocks, 1, 1);
    dim3 threads(thread_x, thread_y, 1);
    dim3 threads_u(zhemv_bs, 1, 1);

    /*
     * If matrix size is multiple of zhemv_bs, we use a specific code.
     * otherwise, we call the generic case.
     */
    if(m % zhemv_bs == 0 ) {
        magmablas_zhemv_200_L_special <<< grid, threads, 0, magma_stream >>>(
            m, alpha, A, lda, X, incx, beta, Y, incy, dC_work);
    }
    else{
        magma_int_t m_mod_thread_x = m%zhemv_bs - 1;
        magmablas_zhemv_200_L_generic <<< grid, threads, 0, magma_stream >>> (
            m, alpha, A, lda, X, incx ,beta, Y, incy, dC_work, m_mod_thread_x);
    }

    magmablas_zhemv_200_L_update<<< grid, threads_u, 0, magma_stream >>>(
        m, alpha, A, lda, X, incx, beta, Y, incy, dC_work);
}


#else


/*******************************************************************************
 *     Functions for each specific cases - Lower case nb = 32
 */


__global__ void
magmablas_zhemv_200_L_special_32_s( magma_int_t n, magmaDoubleComplex alpha,
                               magmaDoubleComplex *A, magma_int_t lda,
                               magmaDoubleComplex *x, magma_int_t incx,
                               magmaDoubleComplex  beta,
                               magmaDoubleComplex *y, magma_int_t incy,
                               magmaDoubleComplex *WC, 
                         magma_int_t nb)
{


    if(blockIdx.y > blockIdx.x) return;

    magma_int_t tx   = threadIdx.x ;
    magma_int_t ty   = threadIdx.y ;


    magmaDoubleComplex res  = MAGMA_Z_ZERO;// used in scan the row
    magmaDoubleComplex res_ = MAGMA_Z_ZERO;// used in scan the column

    __shared__ magmaDoubleComplex la   [1056];
    __shared__ magmaDoubleComplex buff [zhemv_bs];
    __shared__ magmaDoubleComplex buff2 [zhemv_bs];


    magma_int_t break_d   =  zhemv_bs * blockIdx.x;
                  
    A +=  break_d ;
    A +=  lda * ty + tx;
    A +=  lda * (blockIdx.y ) * zhemv_bs; // 

    x +=  tx;


    if ( blockIdx.x == blockIdx.y ) // diagonal 
    {    
    
        x  += (blockIdx.y * zhemv_bs) * incx;    
        if( ty == 0 )
        {
            buff[tx] = x[0];
        } // obtain the vector x store in buff;
       
        #pragma unroll
        for(magma_int_t j =0; j<zhemv_bs; j +=8)
        la[ bank_shift * (ty+j) + tx] =  A[ j * lda];
        __syncthreads();

        #pragma unroll
        for(magma_int_t  i=ty*4; i<(ty * 4 + 4)  ; i++)
        {
            if ( i < tx )   
            {
                la[bank_shift * tx + i] = cuConj(la[ i * bank_shift + tx])  ;
            }
        }
        __syncthreads();

        #pragma unroll
        for(magma_int_t j=0; j < 4 ; j++)
            res += cuConj(la[bank_shift * tx + j + ty * 4])  * buff[j + ty * 4];
    
        __syncthreads();
    
    }
    else // non diagonal
    {
        x  += (blockIdx.x * zhemv_bs) * incx;    
        if( ty == 0 )
        {
            buff[tx] = x[0];
        } // obtain the vector x and  store in buff; buff store its corresponding upper elements instead of buff2; 
            
        x  -= (blockIdx.x * zhemv_bs ) * incx;
        
        x  += (blockIdx.y * zhemv_bs ) * incx;    
            
        if( ty == 0 )
        {
            buff2[tx] = x[0]; 
        } // obtain the vector x store in buff2; 
    
        #pragma unroll
        for(magma_int_t j =0; j<zhemv_bs; j +=8)
        {
            la[ bank_shift * (ty+j) + tx] =  A[ j * lda];
        }
            
        __syncthreads();

            #pragma unroll
            for(magma_int_t j=0; j < 4 ; j++)
            {
                    res += (la[bank_shift * (ty + j * 8) + tx] )* buff2[ ty + j * 8];
                    res_ += cuConj(la[bank_shift * tx + j + ty * 4]) * buff[j + ty * 4]; // 
            }
            __syncthreads();

            
            la[bank_shift*tx+ty]= res_ ;
            __syncthreads();

            if( ty== 0 )
            {
                  res_ = la[tx*bank_shift+0]+la[tx*bank_shift+1]
                +    la[tx*bank_shift+2]+la[tx*bank_shift+3]
                +    la[tx*bank_shift+4]+la[tx*bank_shift+5]
                +    la[tx*bank_shift+6]+la[tx*bank_shift+7];
            
                WC[ tx + blockIdx.y * zhemv_bs + lda * blockIdx.x ] =   res_; // write to its corresponding upper side position
            }
            __syncthreads();
            
      } // end if else 

        la[bank_shift*tx+ty]= res ;
        __syncthreads();

            if( ty== 0 )
            {
                  res = la[tx*bank_shift+0]+la[tx*bank_shift+1]
                +    la[tx*bank_shift+2]+la[tx*bank_shift+3]
                +    la[tx*bank_shift+4]+la[tx*bank_shift+5]
                +    la[tx*bank_shift+6]+la[tx*bank_shift+7];
            
                 WC[ tx + blockIdx.x * zhemv_bs + lda * blockIdx.y] =  res;
            }
        
}


__global__ void
magmablas_zhemv_200_L_special_32( magma_int_t n, magmaDoubleComplex alpha,
                               magmaDoubleComplex *A, magma_int_t lda,
                               magmaDoubleComplex *x, magma_int_t incx,
                               magmaDoubleComplex  beta,
                               magmaDoubleComplex *y, magma_int_t incy,
                               magmaDoubleComplex *WC, 
                         magma_int_t nb)
{
    magma_int_t tx   = threadIdx.x ;
    magma_int_t ty   = threadIdx.y ;
    magma_int_t blkc = blockIdx.x ;


    magmaDoubleComplex res  = MAGMA_Z_ZERO;// used in scan the row
    magmaDoubleComplex res_ = MAGMA_Z_ZERO;// used in scan the column
    magmaDoubleComplex res1 = MAGMA_Z_ZERO;// tem for res
    magmaDoubleComplex res2 = MAGMA_Z_ZERO;// tem for res_

    __shared__ magmaDoubleComplex la   [16][64+2];
    __shared__ magmaDoubleComplex buff [zhemv_bs];
    __shared__ magmaDoubleComplex buff2 [zhemv_bs];


    magma_int_t break_d   =  zhemv_bs * blkc;

    x  += (break_d + tx ) * incx;
    A  +=  break_d ;
    A  +=  ty * lda + tx ;

    if( ty == 0 )
    {
        buff[tx] = x[0];
    } // obtain the vector x store in buff;


    
    {
        A += lda * (blkc) * zhemv_bs; // change

        #pragma unroll
        for(magma_int_t j =0; j<zhemv_bs; j +=8)
        la[0][ bank_shift * (ty+j) + tx] =  A[ j * lda];
        __syncthreads();

        #pragma unroll
        for(magma_int_t  i=ty*4; i<(ty * 4 + 4)  ; i++){
        if ( i < tx )   {
            la[0][bank_shift * tx + i] = cuConj( la[0][ i * bank_shift + tx] ) ;
        }
        }
        __syncthreads();

        #pragma unroll
        for(magma_int_t j=0; j < 4 ; j++)
            res += cuConj( la[0][bank_shift * tx + j + ty * 4] ) * buff[j + ty * 4];
    
            __syncthreads();

             A -= lda * (blkc) * zhemv_bs; 
    
        }


        x -= blkc * zhemv_bs  *incx  ;

        x= x- tx*incx;

        magma_int_t wc_c = 0 ;
        magma_int_t count = 0 ;

        WC +=  break_d + tx;


        if( blkc > 0)

        for(magma_int_t  s=0; s< (blkc * zhemv_bs); s += zhemv_bs )
        {
            MAGMA_Z_SET2REAL(res_,0);
            count++;

                     #pragma unroll
            for(magma_int_t j =0; j<zhemv_bs; j +=8)
            la[0][ bank_shift * (ty+j) + tx] =  A[ j * lda];

            if( ty == 0 )
            {
            buff2[tx] = x[tx];
            } // obtain the vector x store in buff;
            __syncthreads();

            #pragma unroll
            for(magma_int_t j=0; j < 4 ; j++)
            {
                        res += (la[0][bank_shift * (ty + j * 8) + tx] )* buff2[ ty + j * 8];
                        res_ += cuConj( la[0][bank_shift * tx + j + ty * 4] ) * buff[j + ty * 4]; //iterate colum
                }
                     __syncthreads();

                    la[0][bank_shift*tx+ty]= res_ ;
            __syncthreads();

            if( ty== 0 )
            {
              res2 = la[0][tx*bank_shift+0]+la[0][tx*bank_shift+1]
            +    la[0][tx*bank_shift+2]+la[0][tx*bank_shift+3]
            +    la[0][tx*bank_shift+4]+la[0][tx*bank_shift+5]
            +    la[0][tx*bank_shift+6]+la[0][tx*bank_shift+7];
                    
                    WC[wc_c*lda ] =   res2;
            }
 
            __syncthreads();


                    wc_c += 1;
                    x += zhemv_bs;
                    A += lda * zhemv_bs ;


               }


        la[0][bank_shift*tx+ty]= res ;
        __syncthreads();

            if( ty== 0 )
                    {
              res1 = la[0][tx*bank_shift+0]+la[0][tx*bank_shift+1]
            +    la[0][tx*bank_shift+2]+la[0][tx*bank_shift+3]
            +    la[0][tx*bank_shift+4]+la[0][tx*bank_shift+5]
            +    la[0][tx*bank_shift+6]+la[0][tx*bank_shift+7];
                 
                 WC[0+lda*(blkc)] =  res1;
                    }
}

/**************************************************************
 *    Lower case for generic sizes
 */

__global__ void
magmablas_zhemv_200_L_generic_32_s( magma_int_t n, magmaDoubleComplex alpha,
                               magmaDoubleComplex *A, magma_int_t lda,
                               magmaDoubleComplex *x, magma_int_t incx,
                               magmaDoubleComplex  beta,
                               magmaDoubleComplex *y, magma_int_t incy,
                               magmaDoubleComplex *WC, 
                               magma_int_t m_mod_thread_x,
                         magma_int_t nb)
{


    if(blockIdx.y > blockIdx.x) return;


    magma_int_t tx   = threadIdx.x ;
    magma_int_t ty   = threadIdx.y ;


    magmaDoubleComplex res  = MAGMA_Z_ZERO;// used in scan the row
    magmaDoubleComplex res_ = MAGMA_Z_ZERO;// used in scan the column

    __shared__ magmaDoubleComplex la   [1056];
    __shared__ magmaDoubleComplex buff [zhemv_bs];
    __shared__ magmaDoubleComplex buff2 [zhemv_bs];


    magma_int_t break_d   =  zhemv_bs * blockIdx.x;
                  
    A +=  break_d ;
    A +=  lda * ty;
    A +=  lda * (blockIdx.y ) * zhemv_bs; // 
    x +=  tx;
    x  += (blockIdx.x * zhemv_bs) * incx;    

    magma_int_t trackA ;
    if( blockIdx.x == ( gridDim.x - 1 ) ) {
        if( ty == 0 ){
            if( tx > m_mod_thread_x )
            {
                MAGMA_Z_SET2REAL(buff[tx],0);
            }
            else
                buff[tx]  = x[0];
        }
        if ( tx > m_mod_thread_x )
            trackA=m_mod_thread_x;
        else
            trackA=tx;
        A += trackA ;
    }
    else {
        if( ty == 0 ){
            buff[tx]  = x[0];
        }
        trackA = tx;
        A += trackA ;
    }

    __syncthreads();


    if ( blockIdx.x == blockIdx.y) // diagonal 
    {    

               if( blockIdx.x == ( gridDim.x - 1 ) ) {
        #pragma unroll
        for(magma_int_t j =0; j<zhemv_bs; j+=8){
            if( ( ty + j ) > m_mod_thread_x )
            {
                MAGMA_Z_SET2REAL(la[bank_shift*(ty+j)+tx], 9999);
            }
            else
                la[bank_shift*(ty+j)+tx] =  A[ j * lda];
        }
        }
        else {
        #pragma unroll
        for(magma_int_t j =0; j<zhemv_bs; j+=8){
            la[bank_shift*(ty+j)+tx] = A[ j * lda];
        }
        }
        __syncthreads();

        #pragma unroll
        for(magma_int_t  i=ty*4; i<(ty * 4 + 4)  ; i++)
        {
            if ( i < tx )   
            {
                la[bank_shift * tx + i] = cuConj(la[ i * bank_shift + tx])  ;
            }
        }
        __syncthreads();

        #pragma unroll
        for(magma_int_t j=0; j < 4 ; j++)
            res += cuConj(la[bank_shift * tx + j + ty * 4])  * buff[j + ty * 4];
    
        __syncthreads();
    
    }
    else // non diagonal
    {

        // obtain the vector x and  store in buff; buff store its corresponding upper elements instead of buff2; 
            
        x  -= (blockIdx.x * zhemv_bs ) * incx;
        
        x  += (blockIdx.y * zhemv_bs ) * incx;    
            
        if( ty == 0 )
        {
            buff2[tx] = x[0]; 
        } // obtain the vector x store in buff2; 
    
        #pragma unroll
        for(magma_int_t j =0; j<zhemv_bs; j +=8)
        {
            la[ bank_shift * (ty+j) + tx] =  A[ j * lda];
        }
            
        __syncthreads();

            #pragma unroll
            for(magma_int_t j=0; j < 4 ; j++)
            {
                    res += (la[bank_shift * (ty + j * 8) + tx] )* buff2[ ty + j * 8];
                    res_ += cuConj(la[bank_shift * tx + j + ty * 4]) * buff[j + ty * 4]; // 
            }
            __syncthreads();

            
            la[bank_shift*tx+ty]= res_ ;
            __syncthreads();

            if( ty== 0 )
            {
                  res_ = la[tx*bank_shift+0]+la[tx*bank_shift+1]
                +    la[tx*bank_shift+2]+la[tx*bank_shift+3]
                +    la[tx*bank_shift+4]+la[tx*bank_shift+5]
                +    la[tx*bank_shift+6]+la[tx*bank_shift+7];
            
                WC[ tx + blockIdx.y * zhemv_bs + lda * blockIdx.x ] =   res_; // write to its corresponding upper side position
            }
            __syncthreads();
            
      } // end if else 

        la[bank_shift*tx+ty]= res ;
        __syncthreads();

        if( ty== 0 )
            {
                  res = la[tx*bank_shift+0]+la[tx*bank_shift+1]
                +    la[tx*bank_shift+2]+la[tx*bank_shift+3]
                +    la[tx*bank_shift+4]+la[tx*bank_shift+5]
                +    la[tx*bank_shift+6]+la[tx*bank_shift+7];
            
                 WC[ tx + blockIdx.x * zhemv_bs + lda * blockIdx.y] =  res;
            }
        
}

__global__ void
magmablas_zhemv_200_L_generic_32(magma_int_t n, magmaDoubleComplex alpha,
                              magmaDoubleComplex *A, magma_int_t lda,
                              magmaDoubleComplex *x, magma_int_t incx,
                              magmaDoubleComplex beta,
                              magmaDoubleComplex *y, magma_int_t incy,
                              magmaDoubleComplex *WC,
                              magma_int_t m_mod_thread_x,
                         magma_int_t nb)
{
    magma_int_t tx   = threadIdx.x ;
    magma_int_t ty   = threadIdx.y ;
    magma_int_t blkc = blockIdx.x ;


    magmaDoubleComplex res  = MAGMA_Z_ZERO;
    magmaDoubleComplex res_ = MAGMA_Z_ZERO;
    magmaDoubleComplex res1 = MAGMA_Z_ZERO;
    magmaDoubleComplex res2 = MAGMA_Z_ZERO;

    __shared__ magmaDoubleComplex la   [16][64+2];
    __shared__ magmaDoubleComplex buff [zhemv_bs];
    __shared__ magmaDoubleComplex buff2 [zhemv_bs];


    magma_int_t break_d   =  zhemv_bs * blkc;

    x += (break_d + tx ) * incx;
    A +=  break_d ;
    A += lda * ty;

    magma_int_t trackA ;
    if( blkc == ( gridDim.x - 1 ) ) {
        if( ty == 0 ){
            if( tx > m_mod_thread_x )
            {
                MAGMA_Z_SET2REAL(buff[tx],0);
            }
            else
                buff[tx]  = x[0];
        }
        if ( tx > m_mod_thread_x )
            trackA=m_mod_thread_x;
        else
            trackA=tx;
        A += trackA ;
    }
    else {
        if( ty == 0 ){
            buff[tx]  = x[0];
        }
        trackA = tx;
        A += trackA ;
    }

    {
        A += lda * (blkc) * zhemv_bs; // change
        // Somehow merging these two if - else creates problem
        // It could be a potential bug -- from synchronization or from cuda or compiler
        if( blkc == ( gridDim.x - 1 ) ) {
        #pragma unroll
        for(magma_int_t j =0; j<zhemv_bs; j+=8){
            if( ( ty + j ) > m_mod_thread_x )
            {
                MAGMA_Z_SET2REAL(la[0][bank_shift*(ty+j)+tx], 9999);
            }
            else
                la[0][bank_shift*(ty+j)+tx] =  A[ j * lda];
        }
        }
        else {
        #pragma unroll
        for(magma_int_t j =0; j<zhemv_bs; j+=8){
            la[0][bank_shift*(ty+j)+tx] = A[ j * lda];
        }
        }
        __syncthreads();

        #pragma unroll
        for(magma_int_t  i=ty*4; i<(ty*4+4)  ; i++){
        if ( i < tx )   {
            la[0][bank_shift*tx+i] = cuConj(la[0][i*bank_shift+tx]) ;
        }
        else
            la[0][bank_shift*tx+i] = la[0][bank_shift*tx+i]  ;
        }
        __syncthreads();

        #pragma unroll
        for(magma_int_t j=0; j < 4 ; j++)
        res += cuConj(la[0][bank_shift*tx+j+ty*4])* buff[j+ty*4];
        __syncthreads();

       
          A -= lda * (blkc) * zhemv_bs; 
    
        
    }

    __syncthreads();


    x= x - break_d *incx  ;
    x= x - tx * incx ;


    magma_int_t wc_c = 0 ;
    magma_int_t count = 0 ;

    WC +=  break_d + tx;


        if( blkc > 0)

        for(magma_int_t  s=0; s< (blkc * zhemv_bs); s += zhemv_bs )
        {
            MAGMA_Z_SET2REAL(res_,0);
            count++;

                     #pragma unroll
            for(magma_int_t j =0; j<zhemv_bs; j +=8)
            la[0][ bank_shift * (ty+j) + tx] =  A[ j * lda];
            __syncthreads();

            if( ty == 0 )
            {
            buff2[tx] = x[tx];
            } // obtain the vector x store in buff2;
            __syncthreads();

            #pragma unroll
            for(magma_int_t j=0; j < 4 ; j++)
            {
            
                        res += (la[0][bank_shift * (ty + j * 8) + tx] )* buff2[ ty + j * 8];
                        res_ += cuConj( la[0][bank_shift * tx + j + ty * 4] ) * buff[j + ty * 4]; //iterate colum
                }
                     __syncthreads();

                    la[0][bank_shift*tx+ty]= res_ ;
            __syncthreads();

            if( ty== 0 )
            {
              res2 = la[0][tx*bank_shift+0]+la[0][tx*bank_shift+1]
            +    la[0][tx*bank_shift+2]+la[0][tx*bank_shift+3]
            +    la[0][tx*bank_shift+4]+la[0][tx*bank_shift+5]
            +    la[0][tx*bank_shift+6]+la[0][tx*bank_shift+7];
             WC[wc_c*lda ] =   res2;
            }
 
            __syncthreads();

                        

                    wc_c += 1;
                x +=  zhemv_bs;
                    A += lda * zhemv_bs ;

                 
               }


        la[0][bank_shift*tx+ty]= res ;
        __syncthreads();

            if( ty== 0 )
         {
              res1 = la[0][tx*bank_shift+0]+la[0][tx*bank_shift+1]
            +    la[0][tx*bank_shift+2]+la[0][tx*bank_shift+3]
            +    la[0][tx*bank_shift+4]+la[0][tx*bank_shift+5]
            +    la[0][tx*bank_shift+6]+la[0][tx*bank_shift+7];

        WC[0+lda*(blkc)] =  res1;
        }
}



__global__ void
magmablas_zhemv_200_L_update_32_s(magma_int_t n, magmaDoubleComplex alpha,
                         magmaDoubleComplex* A, magma_int_t lda,
                         magmaDoubleComplex *x, magma_int_t incx,
                         magmaDoubleComplex beta,
                         magmaDoubleComplex *y, magma_int_t incy,
                         magmaDoubleComplex *WC,
                         magma_int_t nb )
{
    magma_int_t i;
    magma_int_t tx  = threadIdx.x ;
    magma_int_t ind = blockIdx.x * zhemv_bs + tx ;
    magmaDoubleComplex Ca;

    MAGMA_Z_SET2REAL(Ca, 0) ;
    WC+= ind;

    for(i =0; i<n; i+=zhemv_bs){
        Ca += WC[i/zhemv_bs * lda] ;
    }
    if( ind < n )
        y[ind * incy] = beta * y[ind * incy]  + alpha * Ca ;
}


__global__ void
magmablas_zhemv_200_L_update_32(magma_int_t n, magmaDoubleComplex alpha,
                         magmaDoubleComplex* A, magma_int_t lda,
                         magmaDoubleComplex *x, magma_int_t incx,
                         magmaDoubleComplex beta,
                         magmaDoubleComplex *y, magma_int_t incy,
                         magmaDoubleComplex *WC,
                         magma_int_t nb )
{
    magma_int_t i;
    magma_int_t tx  = threadIdx.x ;
    magma_int_t ind = blockIdx.x * zhemv_bs + tx ;
    magmaDoubleComplex Ca;

    MAGMA_Z_SET2REAL(Ca, 0) ;
    WC+= ind + lda * blockIdx.x;

    for(i = blockIdx.x*zhemv_bs; i<n; i+=zhemv_bs){
        Ca += WC[0] ;
        WC += zhemv_bs;
    }
    if( ind < n )
        y[ind * incy] = beta * y[ind * incy]  + alpha * Ca ;
}


extern "C"
void magmablas_zhemv_200_L_32(magma_int_t m, magmaDoubleComplex alpha,
                           magmaDoubleComplex *A, magma_int_t lda,
                           magmaDoubleComplex *X, magma_int_t incx,
                           magmaDoubleComplex beta,
                           magmaDoubleComplex *Y, magma_int_t incy,
                           magmaDoubleComplex *dC_work,
                         magma_int_t nb)
{
    magma_int_t blocks;

    if (m % zhemv_bs==0)
        blocks = m / zhemv_bs;
    else
        blocks = m / zhemv_bs + 1;

    dim3 grid(blocks, 1, 1);
    dim3 grid_s(blocks, blocks, 1);

    dim3 threads(thread_x, thread_y, 1);
    dim3 threads_u(zhemv_bs, 1, 1);

    /*
     * If matrix size is multiple of zhemv_bs, we use a specific code.
     * otherwise, we call the generic case.
     */
    if(m % zhemv_bs == 0 ) {
    if(m  < SWITCH)
        magmablas_zhemv_200_L_special_32_s <<< grid_s, threads, 0, magma_stream >>>(
            m, alpha, A, lda, X, incx, beta, Y, incy, dC_work,  nb);
        else
        magmablas_zhemv_200_L_special_32 <<< grid, threads, 0, magma_stream >>>(
            m, alpha, A, lda, X, incx, beta, Y, incy, dC_work,  nb);

    }
    else{
        magma_int_t m_mod_thread_x = m%zhemv_bs - 1;
    if(m  < SWITCH)
        magmablas_zhemv_200_L_generic_32_s <<< grid_s, threads, 0, magma_stream >>> (
            m, alpha, A, lda, X, incx ,beta, Y, incy, dC_work, m_mod_thread_x,  nb);
    else
    magmablas_zhemv_200_L_generic_32 <<< grid, threads, 0, magma_stream >>> (
            m, alpha, A, lda, X, incx ,beta, Y, incy, dC_work, m_mod_thread_x,  nb);
    }
    if(m  < SWITCH)
    magmablas_zhemv_200_L_update_32_s<<< grid, threads_u, 0, magma_stream >>>(
        m, alpha, A, lda, X, incx, beta, Y, incy, dC_work,  nb);
    else
    magmablas_zhemv_200_L_update_32<<< grid, threads_u, 0, magma_stream >>>(
        m, alpha, A, lda, X, incx, beta, Y, incy, dC_work,  nb);
}



#endif


/*************************************************************************

    Purpose
    =======

    magmablas_zhemv  performs the matrix-vector operation on fermi:

       y := alpha*A*x + beta*y,

    where alpha and beta are scalars, x and y are n element vectors and
    A is an n by n hermitian matrix.

    Arguments
    ==========

    UPLO   - CHARACTER*1.
             On entry, UPLO specifies whether the upper or lower
             triangular part of the array A is to be referenced as
             follows:

                UPLO = 'U' or 'u'   Only the upper triangular part of A
                                    is to be referenced.

                UPLO = 'L' or 'l'   Only the lower triangular part of A
                                    is to be referenced.

             Unchanged on exit.

    N      - INTEGER.
             On entry, N specifies the order of the matrix A.
             N must be at least zero.
             Unchanged on exit.

    ALPHA  - COMPLEX*16      .
             On entry, ALPHA specifies the scalar alpha.
             Unchanged on exit.

    A      - COMPLEX*16       array of DIMENSION ( LDA, n ).
             Before entry with  UPLO = 'U' or 'u', the leading n by n
             upper triangular part of the array A must contain the upper
             triangular part of the hermitian matrix and the strictly
             lower triangular part of A is not referenced.
             Before entry with UPLO = 'L' or 'l', the leading n by n
             lower triangular part of the array A must contain the lower
             triangular part of the hermitian matrix and the strictly
             upper triangular part of A is not referenced.
             Note that the imaginary parts of the diagonal elements need
             not be set and are assumed to be zero.
             Unchanged on exit.

    LDA    - INTEGER.
             On entry, LDA specifies the first dimension of A as declared
             in the calling (sub) program. LDA must be at least
             max( 1, n ).
             Unchanged on exit.
             It is recommended that lda is multiple of 16. Otherwise
             performance would be deteriorated as the memory accesses
             would not be fully coalescent.

    X      - COMPLEX*16       array of dimension at least
             ( 1 + ( n - 1 )*abs( INCX ) ).
             Before entry, the incremented array X must contain the n
             element vector x.
             Unchanged on exit.

    INCX   - INTEGER.
             On entry, INCX specifies the increment for the elements of
             X. INCX must not be zero.
             Unchanged on exit.

    BETA   - COMPLEX*16      .
             On entry, BETA specifies the scalar beta. When BETA is
             supplied as zero then Y need not be set on input.
             Unchanged on exit.

    Y      - COMPLEX*16       array of dimension at least
             ( 1 + ( n - 1 )*abs( INCY ) ).
             Before entry, the incremented array Y must contain the n
             element vector y. On exit, Y is overwritten by the updated
             vector y.

    INCY   - INTEGER.
             On entry, INCY specifies the increment for the elements of
             Y. INCY must not be zero.
             Unchanged on exit.

*/

extern "C"
magma_int_t
magmablas_zhemv_200( char uplo, magma_int_t n,
                     magmaDoubleComplex alpha, 
                     const magmaDoubleComplex *A, magma_int_t lda,
                     const magmaDoubleComplex *X, magma_int_t incx,
                     magmaDoubleComplex beta,  
                     magmaDoubleComplex *Y, magma_int_t incy)
{
    char uplo_[2] = {uplo, 0};
    int  upper    = lapackf77_lsame(uplo_, "U");

    /*
     * Test the input parameters.
     */
    if ((! upper) && (! lapackf77_lsame(uplo_, "L"))) {
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
    if ( (n == 0) || ( MAGMA_Z_EQUAL(alpha, MAGMA_Z_ZERO) && MAGMA_Z_EQUAL(beta, MAGMA_Z_ONE) ) )
        return MAGMA_SUCCESS;

    /* TODO: Upper case is not implemented in MAGMA */
    if ( upper )
        cublasZhemv(uplo, n, alpha, A, lda, X, incx, beta, Y, incy);
    else
    {
        magmaDoubleComplex *dC_work;
        magma_int_t blocks    = n / zhemv_bs + (n % zhemv_bs != 0);
        magma_int_t workspace = lda * (blocks + 1);

        /* TODO: need to add a MAGMA context to handle workspaces */
        cublasAlloc( workspace, sizeof(magmaDoubleComplex), (void**)&dC_work ) ;
        cublasGetError( ) ;

#ifdef NB_64
        magmablas_zhemv_200_L(n, alpha, A, lda, X, incx, beta, Y, incy, dC_work);

#else     
        magmablas_zhemv_200_L_32(n, alpha, A, lda, X, incx, beta, Y, incy, dC_work, zhemv_bs);     
#endif



        cublasFree(dC_work);
        cublasGetError( ) ;
    }
    return MAGMA_SUCCESS;
}

extern "C"
magma_int_t
magmablas_zhemv2_200( char uplo, magma_int_t n,
                      magmaDoubleComplex alpha,
                      const magmaDoubleComplex *A, magma_int_t lda,
                      const magmaDoubleComplex *X, magma_int_t incx,
                      magmaDoubleComplex beta,
                      magmaDoubleComplex *Y, magma_int_t incy,
                      magmaDoubleComplex *work, int lwork)
{
    char uplo_[2] = {uplo, 0};
    int  upper    = lapackf77_lsame(uplo_, "U");

    /*
     * Test the input parameters.
     */
    if ((! upper) && (! lapackf77_lsame(uplo_, "L"))) {
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
    if ( (n == 0) || ( MAGMA_Z_EQUAL(alpha, MAGMA_Z_ZERO) && MAGMA_Z_EQUAL(beta, MAGMA_Z_ONE) ) )
        return MAGMA_SUCCESS;


    /* TODO: Upper case is not implemented in MAGMA */
    if ( upper )
        cublasZhemv(uplo, n, alpha, A, lda, X, incx, beta, Y, incy);
    else
    {
        magma_int_t blocks    = n / zhemv_bs + (n % zhemv_bs != 0);
        magma_int_t workspace = lda * (blocks + 1);

        if (lwork < workspace){
           printf("Not enough work space in magmablas_zhemv: passed %d, required %d\n",
                  lwork, workspace);
           exit(1);
        }
        //printf("You are using zhemv_bs=%d\n", zhemv_bs);

#ifdef NB_64
        if( n < 1622)
             cublasZhemv(uplo, n, alpha, A, lda, X, incx, beta, Y, incy);
        else
             magmablas_zhemv_200_L(n, alpha, A, lda, X, incx, beta, Y, incy, work);

#else     
        magmablas_zhemv_200_L_32(n, alpha, A, lda, X, incx, beta, Y, incy, work, zhemv_bs);     
#endif

    }
    return MAGMA_SUCCESS;
}

#endif /* (GPUSHMEM >= 200) */
