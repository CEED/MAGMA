/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> s d c

*/
#include "common_magma.h"
#define PRECISION_z


#if (GPUSHMEM >= 200)

#define magmablas_zhemv_200_mgpu_offset magmablas_zhemv_mgpu_offset
#define magmablas_zhemv2_200_mgpu_offset magmablas_zhemv2_mgpu_offset

#define zhemv_bs         32
#define bank_shift       33


/*******************************************************************************
 *     Functions for each specific cases - Lower case
 */



__global__ void
magmablas_zhemv_200_L_special_mgpu_offset( magma_int_t n, magmaDoubleComplex alpha,
                               magmaDoubleComplex *A, magma_int_t lda,
                               magmaDoubleComplex *x, magma_int_t incx,
                               magmaDoubleComplex  beta,
                               magmaDoubleComplex *y, magma_int_t incy,
                               magmaDoubleComplex *WC, 
                         magma_int_t my_gpu_id,
                         magma_int_t num_gpus, 
                         magma_int_t nb,
                         magma_int_t kstan)
{
    magma_int_t tx   = threadIdx.x ;
    magma_int_t ty   = threadIdx.y ;
    magma_int_t blkc = blockIdx.x ;

    if(blkc < my_gpu_id)
    {
    return;
    }

    magmaDoubleComplex res  = MAGMA_Z_ZERO;// used in scan the row
    magmaDoubleComplex res_ = MAGMA_Z_ZERO;// used in scan the column
    magmaDoubleComplex res1 = MAGMA_Z_ZERO;// tem for res
    magmaDoubleComplex res2 = MAGMA_Z_ZERO;// tem for res_

    __shared__ magmaDoubleComplex la   [zhemv_bs][bank_shift];
    __shared__ magmaDoubleComplex sdata   [zhemv_bs][9];
    __shared__ magmaDoubleComplex buff [zhemv_bs];
    __shared__ magmaDoubleComplex buff2 [zhemv_bs];


    magma_int_t break_d   =  zhemv_bs * blkc;

    x  += (break_d + tx ) * incx;
    A  +=  break_d ;
    A  +=  ty * lda + tx ;

    if( ty == 0 )
    {
        buff[tx] = x[0];
    if(blkc == 0 && my_gpu_id == 0 && tx < kstan)
    {
             MAGMA_Z_SET2REAL(buff[tx], 0.0);
        }
    } // obtain the vector x store in buff;
    

    magma_int_t flag = 0;
    
    if ( (blkc % num_gpus) == my_gpu_id) 
    {
        A += lda * (blkc/num_gpus) * zhemv_bs; // change

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
            

             A -= lda * (blkc/num_gpus) * zhemv_bs; 
    
              flag = 1;
        }

        

        x -= blkc * zhemv_bs  *incx  ;

        x= x- tx*incx;

        magma_int_t wc_c = my_gpu_id ;
        magma_int_t count = 0 ;

               WC +=  break_d + tx;
  
        magma_int_t num_blocks_iters = (blkc +1) /num_gpus - flag;
    
        if( my_gpu_id < ( (blkc+1) % num_gpus) )
        {
        num_blocks_iters += 1;
        }

        x += (my_gpu_id ) * zhemv_bs ;

        if( blkc > my_gpu_id)

        for(magma_int_t s=0; s<num_blocks_iters; s++)
        {
            MAGMA_Z_SET2REAL(res_,0);
            count++;

                     #pragma unroll
            for(magma_int_t j =0; j< zhemv_bs; j +=8)
            la[0][ bank_shift * (ty+j) + tx] =  A[ j * lda];

            if( ty == 0 )
            {
                buff2[tx] = x[tx];
                if(my_gpu_id == 0 && tx < kstan && count==1)
                {
                     MAGMA_Z_SET2REAL(buff2[tx], 0.0);
                }
            } // obtain the vector x store in buff2;
            __syncthreads();

            #pragma unroll
            for(magma_int_t j=0; j < 4 ; j++)
            {
                        res += (la[0][bank_shift * (ty + j * 8) + tx] )* buff2[ ty + j * 8];
                        res_ += cuConj( la[0][bank_shift * tx + j + ty * 4] ) * buff[j + ty * 4]; //iterate colum
                }

                    sdata[tx][ty]= res_ ;
            __syncthreads();

            if( ty== 1 )
            {
              res2 = sdata[tx][0]+sdata[tx][1]
              + sdata[tx][2]+sdata[tx][3]
              + sdata[tx][4]+sdata[tx][5]
              + sdata[tx][6]+sdata[tx][7];
                        
                WC[wc_c*lda ] =   res2;
            }


                    wc_c += num_gpus;
                x += num_gpus * zhemv_bs;
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
magmablas_zhemv_200_L_generic_mgpu_offset(magma_int_t n, magmaDoubleComplex alpha,
                              magmaDoubleComplex *A, magma_int_t lda,
                              magmaDoubleComplex *x, magma_int_t incx,
                              magmaDoubleComplex beta,
                              magmaDoubleComplex *y, magma_int_t incy,
                              magmaDoubleComplex *WC,
                              magma_int_t m_mod_nb,
                         magma_int_t my_gpu_id,
                         magma_int_t num_gpus, 
                         magma_int_t nb,
                         magma_int_t kstan)
{
    magma_int_t tx   = threadIdx.x ;
    magma_int_t ty   = threadIdx.y ;
    magma_int_t blkc = blockIdx.x ;

    if(blkc < my_gpu_id)
    {
    return;
    }

    magmaDoubleComplex res  = MAGMA_Z_ZERO;
    magmaDoubleComplex res_ = MAGMA_Z_ZERO;
    magmaDoubleComplex res1 = MAGMA_Z_ZERO;
    magmaDoubleComplex res2 = MAGMA_Z_ZERO;

    __shared__ magmaDoubleComplex la   [zhemv_bs][bank_shift];
    __shared__ magmaDoubleComplex sdata   [zhemv_bs][9];
    __shared__ magmaDoubleComplex buff [zhemv_bs];
    __shared__ magmaDoubleComplex buff2 [zhemv_bs];


    magma_int_t break_d   =  zhemv_bs * blkc;

    x += (break_d + tx ) * incx;
    A +=  break_d ;
    A += lda * ty;

    magma_int_t trackA ;
    if( blkc == ( gridDim.x - 1 ) ) {
        if( ty == 0 ){
            if( tx > m_mod_nb )
            {
                MAGMA_Z_SET2REAL(buff[tx],0);
            }
            else
                buff[tx]  = x[0];
        }
        if ( tx > m_mod_nb )
            trackA=m_mod_nb;
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

     if(ty == 0 )
     { 
           if(my_gpu_id == 0 && blkc ==0  && tx < kstan)//
       {
                 MAGMA_Z_SET2REAL(buff[tx], 0.0);
       }
     }

    magma_int_t flag = 0;
    
    if ( (blkc % num_gpus) == my_gpu_id) 
    {
        A += lda * (blkc/num_gpus) * zhemv_bs; // change
        // Somehow merging these two if - else creates problem
        // It could be a potential bug -- from synchronization or from cuda or compiler
        if( blkc == ( gridDim.x - 1 ) ) {
        #pragma unroll
        for(magma_int_t j =0; j< zhemv_bs; j+=8){
            if( ( ty + j ) > m_mod_nb )
            {
                MAGMA_Z_SET2REAL(la[0][bank_shift*(ty+j)+tx], 9999);
            }
            else
                la[0][bank_shift*(ty+j)+tx] =  A[ j * lda];
        }
        }
        else {
        #pragma unroll
        for(magma_int_t j =0; j< zhemv_bs; j+=8){
            la[0][bank_shift*(ty+j)+tx] = A[ j * lda];
        }
        }
        __syncthreads();

        #pragma unroll
        for(magma_int_t  i=ty*4; i<(ty*4+4)  ; i++){
        if ( i < tx )   {
            la[0][bank_shift*tx+i] = cuConj(la[0][i*bank_shift+tx]) ;
        }
        }
        __syncthreads();

        #pragma unroll
        for(magma_int_t j=0; j < 4 ; j++)
        res += cuConj(la[0][bank_shift*tx+j+ty*4])* buff[j+ty*4];
        __syncthreads();

       
          A -= lda * (blkc/num_gpus) * zhemv_bs; 
    
          flag = 1;
    }

    __syncthreads();


    x= x - break_d *incx  ;
    x= x - tx * incx ;


    magma_int_t wc_c = my_gpu_id ;
    magma_int_t count = 0 ;

    WC +=  break_d + tx;


    magma_int_t num_blocks_iters = (blkc +1) /num_gpus - flag;
    
    if( my_gpu_id < ( (blkc+1) % num_gpus) )
    {
    num_blocks_iters += 1;
    }

    x += (my_gpu_id ) * zhemv_bs ;

        if( blkc > my_gpu_id)

        for(magma_int_t s=0; s<num_blocks_iters; s++)
        {
            MAGMA_Z_SET2REAL(res_,0);
            count++;

                     #pragma unroll
            for(magma_int_t j =0; j< zhemv_bs; j +=8)
            la[0][ bank_shift * (ty+j) + tx] =  A[ j * lda];

            if( ty == 0 )
            {
                buff2[tx] = x[tx];
                if(my_gpu_id == 0 && tx < kstan && count==1)//
                {
                     MAGMA_Z_SET2REAL(buff2[tx], 0.0);
                }
            } // obtain the vector x store in buff2;
            __syncthreads();

            #pragma unroll
            for(magma_int_t j=0; j < 4 ; j++)
            {
            
                        res += (la[0][bank_shift * (ty + j * 8) + tx] )* buff2[ ty + j * 8];
                        res_ += cuConj( la[0][bank_shift * tx + j + ty * 4] ) * buff[j + ty * 4]; //iterate colum
                }

                   sdata[tx][ty]= res_ ;
            __syncthreads();


            if( ty== 1 )
            {
              res2 = sdata[tx][0]+sdata[tx][1]
              + sdata[tx][2]+sdata[tx][3]
              + sdata[tx][4]+sdata[tx][5]
              + sdata[tx][6]+sdata[tx][7];
                        
                WC[wc_c*lda ] =   res2;
            }


                    wc_c += num_gpus;
                x += num_gpus * zhemv_bs;
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
 *    
 */


__global__ void
magmablas_zhemv_200_L_update_mgpu_offset(magma_int_t n, magmaDoubleComplex alpha,
                         magmaDoubleComplex* A, magma_int_t lda,
                         magmaDoubleComplex *x, magma_int_t incx,
                         magmaDoubleComplex beta,
                         magmaDoubleComplex *y, magma_int_t incy,
                         magmaDoubleComplex *WC,
                         magma_int_t my_gpu_id,
                         magma_int_t num_gpus, 
                         magma_int_t nb,
                                                 magma_int_t kstan )
{
    magma_int_t i;
    magma_int_t tx  = threadIdx.x ;
    magma_int_t ind = blockIdx.x * zhemv_bs + tx ;
    magmaDoubleComplex Ca;

    MAGMA_Z_SET2REAL(Ca, 0) ;
    WC+= ind + lda * blockIdx.x;

    for(i = blockIdx.x* zhemv_bs; i<n; i+= zhemv_bs){
        Ca += WC[0] ;
        WC += zhemv_bs;
    }
    if( ind < n && ind >= kstan)
        y[ind * incy] = beta * y[ind * incy]  + alpha * Ca ;
}


extern "C"
void magmablas_zhemv_200_L_mgpu_offset(magma_int_t m, magmaDoubleComplex alpha,
                           magmaDoubleComplex *A, magma_int_t lda,
                           magmaDoubleComplex *X, magma_int_t incx,
                           magmaDoubleComplex beta,
                           magmaDoubleComplex *Y, magma_int_t incy,
                           magmaDoubleComplex *dC_work,
                         magma_int_t my_gpu_id,
                         magma_int_t num_gpus, 
                         magma_int_t nb,
                             magma_int_t offset,
                         magma_int_t num_blocks_skipped)
{

    magma_int_t the_chosen_block_id = offset / nb; 
   
    magma_int_t kstan = offset % nb;

    A += lda * num_blocks_skipped * nb + the_chosen_block_id * nb;
    X += the_chosen_block_id * nb;
    Y += the_chosen_block_id * nb;

    magma_int_t blocks;

    if (m % zhemv_bs==0)
        blocks = m / zhemv_bs;
    else
        blocks = m / zhemv_bs + 1;

    blocks -= the_chosen_block_id;

    dim3 grid(blocks, 1, 1);

    dim3 threads(nb, 8, 1);
    dim3 threads_u(nb, 1, 1);

        
    /*
         * If matrix size is multiple of zhemv_bs, we use a specific code.
         * otherwise, we call the generic case.
         */
      if(m % zhemv_bs == 0 ) 
      {

        magmablas_zhemv_200_L_special_mgpu_offset <<< grid, threads, 0, magma_stream >>>(
            m, alpha, A, lda, X, incx, beta, Y, incy, dC_work, my_gpu_id, num_gpus, nb, kstan);
        }
    else
        {
         magma_int_t m_mod_nb = m%zhemv_bs - 1;



        magmablas_zhemv_200_L_generic_mgpu_offset <<< grid, threads, 0, magma_stream >>> (
            m, alpha, A, lda, X, incx ,beta, Y, incy, dC_work, m_mod_nb, my_gpu_id, num_gpus, nb, kstan);
        }

        magmablas_zhemv_200_L_update_mgpu_offset<<< grid, threads_u, 0, magma_stream >>>(
        m, alpha, A, lda, X, incx, beta, Y, incy, dC_work, my_gpu_id, num_gpus, nb, kstan);
    
}



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
magmablas_zhemv_200_mgpu_offset( char uplo, magma_int_t n,
                      magmaDoubleComplex alpha,
                      magmaDoubleComplex **A, magma_int_t lda,
                      magmaDoubleComplex **X, magma_int_t incx,
                      magmaDoubleComplex beta,
                      magmaDoubleComplex **Y, magma_int_t incy,
                      magmaDoubleComplex **work, magma_int_t lwork,
              magma_int_t num_gpus, 
              magma_int_t nb,
                      magma_int_t offset,
                      cudaStream_t stream[][10])

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

        magma_int_t blocks    = n / zhemv_bs + (n % zhemv_bs != 0);
        magma_int_t workspace = lda * (blocks + 1);

        if (lwork < workspace){
           printf("Not enough work space in magmablas_zhemv: passed %d, required %d\n",
                  lwork, workspace);
           exit(1);
        }
        if(nb != 32)
        {
        printf("Error in magmablas_zsymv_200_mgpu: nb != 32, program will exit! please reallocate your matrix among GPUs\n");
        exit(0);
        }
        magma_int_t i = 0;
        for(i=0; i<num_gpus; i++)
        {
             magma_setdevice(i);
             magmablasSetKernelStream(stream[i][0]);

             magma_int_t the_chosen_block_id = offset / nb; 
         magma_int_t the_chosen_gpu_id = the_chosen_block_id % num_gpus; 

         magma_int_t  num_blocks_skipped = the_chosen_block_id / num_gpus;

         if(i < the_chosen_gpu_id)     
             {
         num_blocks_skipped += 1;
             }
              
             int new_gpu_id = ( i + num_gpus - the_chosen_gpu_id ) % num_gpus;
             


           magmablas_zhemv_200_L_mgpu_offset(n, alpha, A[i], lda, X[i], incx, beta, Y[i], incy, work[i], 
                                                        new_gpu_id, num_gpus, nb, offset, num_blocks_skipped);
             
        
      }



    return MAGMA_SUCCESS;
}



extern "C"
magma_int_t
magmablas_zhemv2_200_mgpu_offset( char uplo, magma_int_t n,
                      magmaDoubleComplex alpha,
                      magmaDoubleComplex **A, magma_int_t lda,
                      magmaDoubleComplex **X, magma_int_t incx,
                      magmaDoubleComplex beta,
                      magmaDoubleComplex **Y, magma_int_t incy,
                      magmaDoubleComplex **work, magma_int_t lwork,
              magma_int_t num_gpus, 
              magma_int_t nb,
                      magma_int_t offset)

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

        magma_int_t blocks    = n / zhemv_bs + (n % zhemv_bs != 0);
        magma_int_t workspace = lda * (blocks + 1);

        if (lwork < workspace){
           printf("Not enough work space in magmablas_zhemv: passed %d, required %d\n",
                  lwork, workspace);
           exit(1);
        }
        if(nb != 32)
        {
        printf("Error in magmablas_zsymv_200_mgpu: nb != 32, program will exit! please reallocate your matrix among GPUs\n");
        exit(0);
        }
        magma_int_t i = 0;
        for(i=0; i<num_gpus; i++)
        {
             magma_setdevice(i);
            // magmablasSetKernelStream(stream[i][0]);

             magma_int_t the_chosen_block_id = offset / nb; 
         magma_int_t the_chosen_gpu_id = the_chosen_block_id % num_gpus; 

         magma_int_t  num_blocks_skipped = the_chosen_block_id / num_gpus;

         if(i < the_chosen_gpu_id)     
             {
         num_blocks_skipped += 1;
             }
              
             int new_gpu_id = ( i + num_gpus - the_chosen_gpu_id ) % num_gpus;
             


         magmablas_zhemv_200_L_mgpu_offset(n, alpha, A[i], lda, X[i], incx, beta, Y[i], incy, work[i], 
                                                        new_gpu_id, num_gpus, nb, offset, num_blocks_skipped);
             
        
      }



    return MAGMA_SUCCESS;

}


/*
__global__ void 
kernel_fillZero(magmaDoubleComplex *A, magma_int_t size)
{
    magma_int_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < size)
    {
        MAGMA_Z_SET2REAL(A[id], 0.0); 
    }
}


void fillZero(magmaDoubleComplex *A, magma_int_t size)
{

    magma_int_t blocks = (size-1)/512 + 1;
    
    dim3 grid(blocks, 1, 1);
    dim3 threads(512, 1, 1);
    
    kernel_fillZero<<<grid, threads>>>(A, size);

}
*/


#endif /* (GPUSHMEM >= 200) */
