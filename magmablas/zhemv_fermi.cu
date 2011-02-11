/*
    -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

       @precisions normal z -> c

*/

#include <stdio.h>
#include <cuda.h>
#include <cublas.h>
#include "magma.h"
#include "operators.h"

#define magmablas_zhemv_fermi magmablas_zhemv 

#define dgemv_bs 64
#define thread_x 64
#define thread_y 4
#define bank_shift 33
#define quarter_thread_x 16
#define half_thread_x 32

__global__ void
magma_l_zhemv_special_v6_ts_fermi(magma_int_t n, cuDoubleComplex alpha,  cuDoubleComplex* A, magma_int_t lda, cuDoubleComplex *x, 
                           magma_int_t incx, cuDoubleComplex beta, cuDoubleComplex *y, magma_int_t iny, cuDoubleComplex *WC, 
                           magma_int_t kstan)
{
  magma_int_t tx = threadIdx.x ; 
  magma_int_t ty = threadIdx.y ; 

  magma_int_t blkc= blockIdx.x ;
  

  cuDoubleComplex res;
  cuDoubleComplex res_;
  cuDoubleComplex res1; 
  MAGMA_Z_SET2REAL(res, 0) ; 
  MAGMA_Z_SET2REAL(res_, 0) ; 
  MAGMA_Z_SET2REAL(res1, 0) ; 

  __shared__ cuDoubleComplex la   [quarter_thread_x][thread_x+2]; 
  __shared__ cuDoubleComplex buff [thread_x];

  cuDoubleComplex tr[4];
  cuDoubleComplex b[4];


  WC += tx + blkc * thread_x;
  x  += blkc * thread_x  * incx;
  A  += blkc * thread_x + lda * blkc * thread_x;

  const magma_int_t td = (thread_x * ty ) + tx  ; 
  magma_int_t tx_ = td % half_thread_x ; 
  magma_int_t ty_ = td /half_thread_x ; 

  A += ty_* lda + tx_ ;  
  x += tx * incx;

  if( ty == 0 ){
      if ( blkc ==0 && tx <= kstan )
      {
          MAGMA_Z_SET2REAL(buff[tx], 0);
      }
      else
      {
          buff[tx] = x[0];
      }
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
      res1 = la[0][tx_*bank_shift+0]+la[0][tx_*bank_shift+1]+
             la[0][tx_*bank_shift+2]+la[0][tx_*bank_shift+3]+
             la[0][tx_*bank_shift+4]+la[0][tx_*bank_shift+5]+
             la[0][tx_*bank_shift+6]+la[0][tx_*bank_shift+7];
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

   cuDoubleComplex res2;
   MAGMA_Z_SET2REAL(res2,0);
   if( ty_== 1 ) 
      res2 = la[0][tx_*bank_shift+0]+la[0][tx_*bank_shift+1]+
             la[0][tx_*bank_shift+2]+la[0][tx_*bank_shift+3]+
             la[0][tx_*bank_shift+4]+la[0][tx_*bank_shift+5]+
             la[0][tx_*bank_shift+6]+la[0][tx_*bank_shift+7];
    else
	{
	  MAGMA_Z_SET2REAL(res2,0);
    }
   __syncthreads();

   MAGMA_Z_SET2REAL(res,0);

   A-=half_thread_x *lda ;

   MAGMA_Z_SET2REAL(res_,0);

   #pragma unroll  
   for(magma_int_t j =0; j<half_thread_x; j+=8)
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
      res2 =res2+  la[0][tx_*bank_shift+0]+la[0][tx_*bank_shift+1]+
                   la[0][tx_*bank_shift+2]+la[0][tx_*bank_shift+3]+
                   la[0][tx_*bank_shift+4]+la[0][tx_*bank_shift+5]+
                   la[0][tx_*bank_shift+6]+la[0][tx_*bank_shift+7];
   else
   {
       MAGMA_Z_SET2REAL(res2,0);
   }
   __syncthreads();

   la[0][bank_shift*tx_+ty_]= res_ ;
   __syncthreads();
   if( ty_ == 0 ) {
      res1 =res1+  la[0][tx_*bank_shift+0]+la[0][tx_*bank_shift+1]+
                   la[0][tx_*bank_shift+2]+la[0][tx_*bank_shift+3]+
                   la[0][tx_*bank_shift+4]+la[0][tx_*bank_shift+5]+
                   la[0][tx_*bank_shift+6]+la[0][tx_*bank_shift+7];
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
   
   x= x- tx*incx; 

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
     for(magma_int_t  i=0; i<thread_x; i += thread_x )
	 {
       MAGMA_Z_SET2REAL(res_,0);
		count++;
       #pragma unroll  
       for( magma_int_t k=0;k<4;k++)
	   {
			#pragma unroll  
			for(magma_int_t j=0; j < 4 ; j++)
			tr[j] = A[j*lda];

	 	   #pragma unroll  
			for(magma_int_t j=0; j < 4 ; j++)
			{
				res += tr[j] * x[ quarter_thread_x * k + ty * 4 + j];
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
     	 res_ = la[tx_][0+k] + la[tx_][1+k]+ 
                la[tx_][2+k] + la[tx_][3+k]+ 
                la[tx_][4+k] + la[tx_][5+k]+ 
                la[tx_][6+k] + la[tx_][7+k]+ 
                la[tx_][8+k] + la[tx_][9+k]+ 
                la[tx_][10+k]+ la[tx_][11+k]+
                la[tx_][12+k]+ la[tx_][13+k]+
                la[tx_][14+k]+ la[tx_][15+k];
     	 WC[k + wc_c*lda ] =   res_; 
       }
	   
       wc_c++;
       __syncthreads();
	   
    }

  for(magma_int_t  i=thread_x; i< (blkc * thread_x); i += thread_x )
  {
       MAGMA_Z_SET2REAL(res_,0);
		count++;

		#pragma unroll  
		for( magma_int_t k=0;k<4;k++)
		{
			#pragma unroll  
			for(magma_int_t j=0; j < 4 ; j++)
					tr[j] = A[j*lda] ;
			 #pragma unroll  
			for(magma_int_t j=0; j < 4 ; j++)
			{
				 res += tr[j] * x[ i + quarter_thread_x*k + ty*4+(j)];
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
     	res_ = la[tx_][0+k] + la[tx_][1+k]+ 
               la[tx_][2+k] + la[tx_][3+k]+
               la[tx_][4+k] + la[tx_][5+k]+
               la[tx_][6+k] + la[tx_][7+k]+
               la[tx_][8+k] + la[tx_][9+k]+
               la[tx_][10+k]+ la[tx_][11+k]+
               la[tx_][12+k]+ la[tx_][13+k]+
               la[tx_][14+k]+ la[tx_][15+k];
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
     WC[0+lda*(blkc)  ] =  res;
  }
}

__global__ void
magma_l_zhemv_special_update_v6_ts_fermi(magma_int_t n, cuDoubleComplex alpha, cuDoubleComplex* A, magma_int_t lda, cuDoubleComplex *x, 
                                  magma_int_t inx, cuDoubleComplex beta, cuDoubleComplex *y, magma_int_t iny, 
                                  cuDoubleComplex *WC, magma_int_t kstan)
{
  magma_int_t tx = threadIdx.x ;
  magma_int_t ind = (blockIdx.x)* thread_x + tx ;
  cuDoubleComplex Ca ;
       MAGMA_Z_SET2REAL(Ca,0);
  WC+= tx+(blockIdx.x)*thread_x + lda*blockIdx.x  ;
  for(magma_int_t i=(blockIdx.x)*thread_x ;i<n;i+=thread_x){
          Ca+=WC[0] ;
          WC+=thread_x;
  }
  if( ind > kstan )
  y[ind * iny ] =beta * y[ind * iny  ]  + alpha * Ca ; 
}

__global__ void
magma_l_zhemv_generic_v6_ts_fermi(magma_int_t n, cuDoubleComplex alpha, cuDoubleComplex* A, magma_int_t lda, cuDoubleComplex *x, 
                           magma_int_t inx, cuDoubleComplex beta, cuDoubleComplex *y, magma_int_t iny, cuDoubleComplex *WC,
                           magma_int_t m_mod_thread_x, magma_int_t kstan)
{
  magma_int_t tx = threadIdx.x ; 
  magma_int_t ty = threadIdx.y ; 
  magma_int_t blkc= blockIdx.x ;
  
  cuDoubleComplex res;
       MAGMA_Z_SET2REAL(res,0);
  cuDoubleComplex res_;
       MAGMA_Z_SET2REAL(res_,0);
  __shared__ cuDoubleComplex la   [quarter_thread_x][thread_x+2];  
  __shared__ cuDoubleComplex buff [thread_x];
  __shared__ cuDoubleComplex buff2 [thread_x];
  cuDoubleComplex tr[4];
  cuDoubleComplex b[8];
  magma_int_t break_d  =   (blkc)* thread_x  ;
  const magma_int_t td = (thread_x * ty ) + tx  ; 
  magma_int_t tx_ = td % half_thread_x ; 
  magma_int_t ty_ = td /half_thread_x ; 
  cuDoubleComplex res1; 
       MAGMA_Z_SET2REAL(res1,0);
  WC+= tx+(blkc)*thread_x;
  A+= (blkc)* thread_x  ;
  A+=lda*break_d;
  A+=ty_* lda  ;  
  x+=break_d *inx  ;
  x+=tx*inx;

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
   A+=trackA ;
 }
 else{
  if( ty == 0 ){
      buff[tx]  = x[0];
  }
  trackA= tx_; 
  A+=trackA ;
 }
if( ty==0 && tx<=kstan  && blkc == 0 ) {
       MAGMA_Z_SET2REAL(buff[tx],0);
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
else{
  #pragma unroll  
  for(magma_int_t j =0; j<half_thread_x; j+=8){
         la[0][bank_shift*(ty_+j)+tx_] = A[ j * lda];
  }
}
  tx = tx_ ; ty = ty_ ; 
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
      res1 = la[0][tx_*bank_shift+0]+la[0][tx_*bank_shift+1]+la[0][tx_*bank_shift+2]+la[0][tx_*bank_shift+3]+la[0][tx_*bank_shift+4]+la[0][tx_*bank_shift+5]+la[0][tx_*bank_shift+6]+la[0][tx_*bank_shift+7];
  else
      {
	      MAGMA_Z_SET2REAL(res1,0);
	  }
  __syncthreads();


       MAGMA_Z_SET2REAL(res,0);

if( blkc == ( gridDim.x - 1 ) ) {
  if ( (tx_+half_thread_x) > m_mod_thread_x ) 
       trackA=m_mod_thread_x;
  else 
       trackA=tx_+half_thread_x;
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
else{
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
   cuDoubleComplex res2;  
       MAGMA_Z_SET2REAL(res2,0);
   if( ty_== 1 ) 
      res2 = la[0][tx_*bank_shift+0]+la[0][tx_*bank_shift+1]+la[0][tx_*bank_shift+2]+la[0][tx_*bank_shift+3]+la[0][tx_*bank_shift+4]+la[0][tx_*bank_shift+5]+la[0][tx_*bank_shift+6]+la[0][tx_*bank_shift+7];
    else 
      {
	     MAGMA_Z_SET2REAL(res2,0);
	  }

  __syncthreads();

       MAGMA_Z_SET2REAL(res,0);
       MAGMA_Z_SET2REAL(res_,0);


//  __syncthreads();


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
else{
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
      res2 =res2+  la[0][tx_*bank_shift+0]+la[0][tx_*bank_shift+1]+la[0][tx_*bank_shift+2]+la[0][tx_*bank_shift+3]+la[0][tx_*bank_shift+4]+la[0][tx_*bank_shift+5]+la[0][tx_*bank_shift+6]+la[0][tx_*bank_shift+7];
   else 
      {
	     MAGMA_Z_SET2REAL(res2,0);
	  }
   __syncthreads();
   la[0][bank_shift*tx_+ty_]= res_ ;
   __syncthreads();
   if( ty_ == 0 ) {
      res1 =res1+  la[0][tx_*bank_shift+0]+la[0][tx_*bank_shift+1]+la[0][tx_*bank_shift+2]+la[0][tx_*bank_shift+3]+la[0][tx_*bank_shift+4]+la[0][tx_*bank_shift+5]+la[0][tx_*bank_shift+6]+la[0][tx_*bank_shift+7];
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
  x= x - break_d *inx  ; 

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
    if(tx > kstan )
           buff2[tx]  = x[i*inx];
    else
	    {
           MAGMA_Z_SET2REAL(buff2[tx], 0) ; 
		}
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
     	res_ = la[tx_][0+k] + la[tx_][1+k]+ la[tx_][2+k]+la[tx_][3+k]+la[tx_][4+k]+la[tx_][5+k]+la[tx_][6+k]+la[tx_][7+k]+la[tx_][8+k]+la[tx_][9+k]+la[tx_][10+k]+la[tx_][11+k]+la[tx_][12+k]+la[tx_][13+k]+la[tx_][14+k]+la[tx_][15+k];
     	WC[k + wc_c*lda ] =   res_; 
    }
    wc_c++;
   __syncthreads();
  }
  for(magma_int_t  i=thread_x; i<break_d; i += thread_x ){
    MAGMA_Z_SET2REAL(res_, 0) ; 
    count++;
    if(ty == 0 )
           buff2[tx]  = x[i*inx];
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
     	res_ = la[tx_][0+k] + la[tx_][1+k]+ la[tx_][2+k]+la[tx_][3+k]+la[tx_][4+k]+la[tx_][5+k]+la[tx_][6+k]+la[tx_][7+k]+la[tx_][8+k]+la[tx_][9+k]+la[tx_][10+k]+la[tx_][11+k]+la[tx_][12+k]+la[tx_][13+k]+la[tx_][14+k]+la[tx_][15+k];
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
     WC[0+lda*(blkc)  ] =  res;
   }


}

__global__ void
magma_l_zhemv_generic_update_v6_ts_fermi (magma_int_t n, cuDoubleComplex alpha ,  cuDoubleComplex* A, magma_int_t lda, cuDoubleComplex *x, magma_int_t inx , cuDoubleComplex beta ,  cuDoubleComplex *y , magma_int_t iny , cuDoubleComplex *WC, magma_int_t kstan ){
  magma_int_t tx = threadIdx.x ;
  magma_int_t ind = (blockIdx.x)* thread_x + tx ;
  cuDoubleComplex Ca;
  MAGMA_Z_SET2REAL(Ca, 0) ; 
  WC+= tx+(blockIdx.x)*thread_x + lda*blockIdx.x  ;
  for(magma_int_t i=(blockIdx.x)*thread_x ;i<n;i+=thread_x){
          Ca+=WC[0] ;
          WC+=thread_x;
  }
  if( ind >kstan && ind < n ) 
     y[ind * iny ] =beta * y[ind * iny  ]  + alpha * Ca ; 
}




extern "C"
void magmablas_zhemv6_fermi(char uplo, magma_int_t m, cuDoubleComplex alpha, cuDoubleComplex *A, magma_int_t lda, 
                      cuDoubleComplex *X, magma_int_t incx, cuDoubleComplex beta, cuDoubleComplex *Y, magma_int_t incy, 
                      cuDoubleComplex *dC_work, magma_int_t kstan)


{
/*
   zhemv performs the matrix-vector operation

     y := alpha*A*x + beta*y,

   where alpha and beta are scalars, x and y are m element vectors and
   A is an m by m symmetric matrix. Matrix A is stored in the lower triangular
   part of A (always).

   This version is designed for the two-sided tridiagonal factorization.
   Parameter kstan can have the following values
   kstan == -1   y              := alpha*A*x + beta*y
   kstan != -1   y(kstan+1:m-1) := alpha*A(kstan+1:m-1,kstan+1:m-1)*x(kstan+1:m-1)+
                                   beta*y(kstan+1:m-1)

   It ia recommended that lda is multiple of 16. Otherwise performance would be 
   deteriorated as the memory accesses would not be fully coalescent.
*/

    magma_int_t blocks;
    if (m % dgemv_bs==0)
        blocks = m/ dgemv_bs;
    else
        blocks = m/ dgemv_bs + 1;

    dim3 grid(blocks, 1, 1);
    dim3 threads(thread_x, thread_y, 1);
    dim3 threads_u(dgemv_bs, 1, 1);
    if(m % dgemv_bs == 0 ) {
       if( uplo == 'L' || uplo == 'l'){
	  magma_l_zhemv_special_v6_ts_fermi <<<grid, threads>>>(m, alpha, 
                   A, lda, X, incx ,beta,  Y , incy, dC_work, kstan);
	  magma_l_zhemv_special_update_v6_ts_fermi<<<grid, threads_u>>>(m, alpha, 
                        A, lda, X, incx ,beta,  Y , incy, dC_work, kstan);
       }
       else{
          printf("Not Available Now.\n");
      } 
		
    } 
    else{	
      magma_int_t  m_mod_thread_x = m%dgemv_bs ; 
      if (uplo == 'L' || uplo == 'l'){
         magma_l_zhemv_generic_v6_ts_fermi <<<grid, threads>>> (m, alpha, A, lda, 
                    X, incx ,beta,  Y , incy, dC_work, m_mod_thread_x-1, kstan);
         magma_l_zhemv_generic_update_v6_ts_fermi<<<grid, threads_u>>>(m, alpha, 
                        A, lda, X, incx ,beta,  Y , incy, dC_work, kstan);
      }	
      else{
         printf("Not Available Now.\n");
      }	
    }
}

extern "C"
void  magmablas_zhemv_fermi( char uplo , magma_int_t m , cuDoubleComplex alpha,  cuDoubleComplex *A , magma_int_t lda , 
				cuDoubleComplex *X , magma_int_t incx, cuDoubleComplex beta, cuDoubleComplex *Y, magma_int_t incy)
{

	if (uplo == 'U' || uplo == 'u')
		cublasZhemv(uplo, m, alpha, A, lda, X, incx, beta, Y, incy);
	else
	{	
	
	cuDoubleComplex *dC_work;
	magma_int_t bsz = thread_x;
	magma_int_t blocks = m / bsz + (m %bsz != 0);
	magma_int_t workspace = lda * (blocks + 1);
	cublasAlloc( workspace, sizeof(cuDoubleComplex), (void**)&dC_work ) ;
			
	cublasGetError( ) ;

	magma_int_t kstan = -1;

	magmablas_zhemv6_fermi(uplo, m, alpha, A, lda, X, incx, beta, Y, incy, 
                      dC_work, kstan);

	cublasFree(dC_work);
	}

}

#undef thread_x 
#undef thread_y 
#undef dgemv_bs 
