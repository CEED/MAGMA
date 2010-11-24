#include <stdio.h>
#include "cuda.h"
#include "cublas.h"

#define dgemv_bs 64
#define thread_x 64
#define thread_y 4

#define magmablas_dsymv_tesla magmablas_dsymv

__global__ void
test_l_dsymv_special_v6_ts_tesla(int n, double alpha,  double* A, int lda, double *x, 
                           int incx, double beta, double *y, int iny, double *WC, 
                           int kstan)
{
  int tx = threadIdx.x ; 
  int ty = threadIdx.y ; 

  int blkc= blockIdx.x ;
  
/*  if ( gridDim.x %60 > 29 ) 
     blkc= (blockIdx.x + 1 ) % gridDim.x;
*/ 
  double res = 0.f;
  double res_ = 0.f;
  double res1 = 0.f ; 

  __shared__ double la   [16][66]; 
  __shared__ double buff [64];

  double tr[4];
  double b[4];


  WC += tx + blkc * 64;
  x  += blkc * 64  * incx;
  A  += blkc * 64 + lda * blkc * 64;

  const int td = (64 * ty ) + tx  ; 
  int tx_ = td % 32 ; 
  int ty_ = td /32 ; 

  A += ty_* lda + tx_ ;  
  x += tx * incx;

  if( ty == 0 ){
      if ( blkc ==0 && tx <= kstan )
          buff[tx] = 0.0 ; 
      else
          buff[tx]  = x[0];
   } // obtain the vector x store in buff;

   tx = tx_ ; ty = ty_ ; 

   #pragma unroll  
   for(int j =0; j<32; j +=8)
         la[0][ 33 * (ty_+j) + tx_] =  A[ j * lda];
   __syncthreads();

   #pragma unroll 
   for(int  i=ty_*4; i<(ty_ * 4 + 4)  ; i++){
         if ( i < tx_ )   {
	        la[0][33 * tx_ + i] = la[0][ i * 33 + tx_] ; 
         }
	 else 
	        la[0][33 * tx_ + i] = la[0][ 33 * tx_ + i]  ;
   }
   __syncthreads();
 
   #pragma unroll 
   for(int j=0; j < 4 ; j++)
      res+=la[0][33 * tx_ + j + ty_ * 4] * buff[j + ty_ * 4];
   __syncthreads();

   la[0][33*tx_+ty_]= res ;  
   __syncthreads();
   if( ty_== 0 ) 
      res1 = la[0][tx_*33+0]+la[0][tx_*33+1]+
             la[0][tx_*33+2]+la[0][tx_*33+3]+
             la[0][tx_*33+4]+la[0][tx_*33+5]+
             la[0][tx_*33+6]+la[0][tx_*33+7];
   else
      res1 = 0 ;
   __syncthreads();


   res = 0 ; 

   A+= 32 + 32 *lda ;
   #pragma unroll 
   for(int j =0; j<32; j+=8)
         la[0][33*(ty_+j)+tx_] = A[ j * lda];
   __syncthreads();
   #pragma unroll 
   for(int  i=ty_*4; i<(4+ty_*4) ; i++){
         if ( i < tx_ )   {
	        la[0][33*tx_+i] = la[0][33*i+tx_] ; 
         }
	 else 
	        la[0][33*tx_+i] = la[0][33*tx_+i]  ;
   }
   __syncthreads();

   #pragma unroll 
   for(int j=0; j < 4 ; j++)
      res+= la[0][33*tx_+j+ty_*4]*buff[32 + j + 4 * ty_];
   __syncthreads();
   la[0][33*tx_+ty_]= res ;  
   __syncthreads();

   double res2 = 0.f;  
   if( ty_== 1 ) 
      res2 = la[0][tx_*33+0]+la[0][tx_*33+1]+
             la[0][tx_*33+2]+la[0][tx_*33+3]+
             la[0][tx_*33+4]+la[0][tx_*33+5]+
             la[0][tx_*33+6]+la[0][tx_*33+7];
    else 
      res2 = 0 ; 

   __syncthreads();

   res  = 0.0f ; 

   A-=32 *lda ;

   res_=0.f;

   #pragma unroll  
   for(int j =0; j<32; j+=8)
       tr[j/8] = A[ j * lda];
   #pragma unroll 
   for(int j=0; j < 4 ; j++){
      res+=tr[j]*buff[ j*8 + ty_];
      la[0][33*(ty_+j*8)+tx_] = tr[j];	
   }
   __syncthreads();
   #pragma unroll  
   for(int j=0; j < 4 ; j++)
      res_+=la[0][33*tx_+j+ty_*4]*buff[32 +j+ty_*4];
   __syncthreads();

   la[0][33*tx_+ty_]= res ;
   __syncthreads();
   if( ty_ == 1 ) 
      res2 =res2+  la[0][tx_*33+0]+la[0][tx_*33+1]+
                   la[0][tx_*33+2]+la[0][tx_*33+3]+
                   la[0][tx_*33+4]+la[0][tx_*33+5]+
                   la[0][tx_*33+6]+la[0][tx_*33+7];
   else 
       res2 = 0 ; 
   __syncthreads();

   la[0][33*tx_+ty_]= res_ ;
   __syncthreads();
   if( ty_ == 0 ) {
      res1 =res1+  la[0][tx_*33+0]+la[0][tx_*33+1]+
                   la[0][tx_*33+2]+la[0][tx_*33+3]+
                   la[0][tx_*33+4]+la[0][tx_*33+5]+
                   la[0][tx_*33+6]+la[0][tx_*33+7];
   }
   else 
       res1 = 0 ; 
   A-=32;

   __syncthreads(); 
   tx = threadIdx.x ; 
   ty = threadIdx.y ; 
   
   if( ty_ == 0  && ty == 0  ) 
      res = res1 ; 
   else if( ty_ == 1  && ty == 0  ) 
      res = res2 ; 
   else res = 0.0f ; 

   A-=ty_* lda  ;  
   A-=tx_; 

   A= A - lda * blkc * 64;
   x= x - blkc * 64  *incx  ; 
   
   x= x- tx*incx; 

   A+=4 * ty* lda  ;  
   A+=tx; 

   int wc_c = 0 ; 
   int count = 0 ; 


   tx_ = td % 16 ; 
   ty_ = td / 16 ; 

     WC-=tx ;
     WC+=tx_; 

   if( blkc * 64 >=64) 
     #pragma unroll 
     for(int  i=0; i<64; i += 64 )
	 {
		res_=0.f;
		count++;
       #pragma unroll  
       for( int k=0;k<4;k++)
	   {
			#pragma unroll  
			for(int j=0; j < 4 ; j++)
			tr[j] = A[j*lda];

	 	   #pragma unroll  
			for(int j=0; j < 4 ; j++)
			{
				res += tr[j] * x[ 16 * k + ty * 4 + j];

            	la[( j + ty * 4)][tx] = tr[j] * buff[tx]; 
			}
	 		  __syncthreads();


		   res_= 0 ;

            #pragma unroll  
			for(int j=0; j < 4 ; j++)
			{
				res_+=la[tx_][ty_*4+j] ;
			}
			b[k] = res_ ;
			__syncthreads();

			A += lda * 16 ;
       }

       #pragma unroll  
       for(int k=0; k < 4 ; k++){
         la[tx_][ty_+16*k]= b[k] ;
       }
       __syncthreads();
       if( ty_ < 4 ) {	
		int k = ty_*16;
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

  for(int  i=64; i< (blkc * 64); i += 64 )
  {
	    res_=0.f;
		count++;

		#pragma unroll  
		for( int k=0;k<4;k++)
		{
			#pragma unroll  
			for(int j=0; j < 4 ; j++)
					tr[j] = A[j*lda] ;
			 #pragma unroll  
			for(int j=0; j < 4 ; j++)
			{
				 res += tr[j] * x[ i + 16*k + ty*4+(j)];
		         la[( j + ty * 4)][tx] = tr[j] * buff[tx]; 
			}
			__syncthreads();

			
		    res_= 0 ; 
		    #pragma unroll  
			for(int j=0; j < 4 ; j++)
                  res_+=la[tx_][ty_*4+j] ;
		
			b[k] = res_ ;
            __syncthreads();
	
			A += lda * 16 ;
		}

	
   #pragma unroll  
   for(int k=0; k < 4 ; k++){
       la[tx_][ty_+16*k]= b[k] ;
   }
   __syncthreads();
   if( ty_ < 4 ) {	
	int k = ty_*16;
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
test_l_dsymv_special_update_v6_ts_tesla(int n, double alpha, double* A, int lda, double *x, 
                                  int inx, double beta, double *y, int iny, 
                                  double *WC, int kstan)
{
  int tx = threadIdx.x ;
  int ind = (blockIdx.x)* 64 + tx ;
  double Ca = 0.f ;
  WC+= tx+(blockIdx.x)*64 + lda*blockIdx.x  ;
  for(int i=(blockIdx.x)*64 ;i<n;i+=64){
          Ca+=WC[0] ;
          WC+=64;
  }
  if( ind > kstan )
  y[ind * iny ] =beta * y[ind * iny  ]  + alpha * Ca ; 
}

__global__ void
test_l_dsymv_generic_v6_ts_tesla(int n, double alpha, double* A, int lda, double *x, 
                           int inx, double beta, double *y, int iny, double *WC,
                           int m_mod_64, int kstan)
{
  int tx = threadIdx.x ; 
  int ty = threadIdx.y ; 
  int blkc= blockIdx.x ;
  double res = 0.f;
  double res_ = 0.f;
  __shared__ double la   [16][66];  
  __shared__ double buff [64];
  __shared__ double buff2 [64];
  double tr[4];
  double b[8];
  int break_d  =   (blkc)* 64  ;
  const int td = (64 * ty ) + tx  ; 
  int tx_ = td % 32 ; 
  int ty_ = td /32 ; 
  double res1 = 0.f ; 
  WC+= tx+(blkc)*64;
  A+= (blkc)* 64  ;
  A+=lda*break_d;
  A+=ty_* lda  ;  
  x+=break_d *inx  ;
  x+=tx*inx;

 int trackA ; 
 if( blkc == ( gridDim.x - 1 ) ) {
   if( ty == 0 ){
      if( tx > m_mod_64 )
          buff[tx]=0;
      else
          buff[tx]  = x[0];
   } 
   if ( tx_ > m_mod_64 ) 
       trackA=m_mod_64;
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
   buff[tx]=0; 
}
// Somehow merging these two if - else creates problem 
// It could be a potential bug -- from synchronization or from cuda or compiler 
if( blkc == ( gridDim.x - 1 ) ) {
  #pragma unroll  
  for(int j =0; j<32; j+=8){
         if( ( ty_ + j ) > m_mod_64 ) 
             la[0][33*(ty_+j)+tx_] = 9999;
         else
             la[0][33*(ty_+j)+tx_] =  A[ j * lda];
  }
  A-=trackA; 
}
else{
  #pragma unroll  
  for(int j =0; j<32; j+=8){
         la[0][33*(ty_+j)+tx_] = A[ j * lda];
  }
}
  tx = tx_ ; ty = ty_ ; 
  __syncthreads();
  #pragma unroll 
  for(int  i=ty_*4; i<(ty_*4+4)  ; i++){
         if ( i < tx_ )   {
	        la[0][33*tx_+i] = la[0][i*33+tx_] ; 
         }
	 else 
	        la[0][33*tx_+i] = la[0][33*tx_+i]  ;
  }
  __syncthreads();
  #pragma unroll 
  for(int j=0; j < 4 ; j++)
      res+=la[0][33*tx_+j+ty_*4]* buff[j+ty_*4];
  __syncthreads();
  la[0][33*tx_+ty_]= res ;  
  __syncthreads();
  if( ty_== 0 ) 
      res1 = la[0][tx_*33+0]+la[0][tx_*33+1]+la[0][tx_*33+2]+la[0][tx_*33+3]+la[0][tx_*33+4]+la[0][tx_*33+5]+la[0][tx_*33+6]+la[0][tx_*33+7];
  else
      res1 = 0 ;
  __syncthreads();


  res = 0 ; 

if( blkc == ( gridDim.x - 1 ) ) {
  if ( (tx_+32) > m_mod_64 ) 
       trackA=m_mod_64;
  else 
       trackA=tx_+32;
  A+= trackA+32*lda ;
  #pragma unroll  
  for(int j =0; j<32; j+=8){
         if( ( ty_ + j+32 ) > m_mod_64 ) 
              la[0][33*(ty_+j)+tx_] = 99999;
         else
              la[0][33*(ty_+j)+tx_] =  A[ j * lda];
  }
  A-= trackA+32*lda ;
  A+=tx_ ; 
  A+= 32 + 32 *lda ;
}
else{
  A+= 32 + 32 *lda ;
  #pragma unroll  
  for(int j =0; j<32; j+=8){
         la[0][33*(ty_+j)+tx_] = A[ j * lda];
  }
}

  __syncthreads();
  #pragma unroll 
  for(int  i=ty_*4; i<(4+ty_*4) ; i++){
         if ( i < tx_ )   {
	        la[0][33*tx_+i] = la[0][33*i+tx_] ; 
         }
	 else 
	        la[0][33*tx_+i] = la[0][33*tx_+i]  ;
  }
  __syncthreads();
  #pragma unroll 
  for(int j=0; j < 4 ; j++)
      res+= la[0][33*tx_+j+ty_*4]*buff[32 + j + 4 * ty_];
  __syncthreads();
  la[0][33*tx_+ty_]= res ;  
  __syncthreads();
   double res2 = 0.f;  
   if( ty_== 1 ) 
      res2 = la[0][tx_*33+0]+la[0][tx_*33+1]+la[0][tx_*33+2]+la[0][tx_*33+3]+la[0][tx_*33+4]+la[0][tx_*33+5]+la[0][tx_*33+6]+la[0][tx_*33+7];
    else 
      res2 = 0 ; 

  __syncthreads();

res  = 0 ; 


  res_=0.f;
  __syncthreads();

  A-=32 *lda ;
if( blkc == ( gridDim.x - 1 ) ) {
  A-=tx_; 
  if ( tx_ > m_mod_64 ) 
       trackA=m_mod_64;
  else 
       trackA=tx_;
  A+= trackA ;
  #pragma unroll  
  for(int j =0; j<32; j+=8)
       if( ( ty_ + j ) > m_mod_64 ) 
       tr[j/8] = 99999;
       else
       tr[j/8] = A[ j * lda];
  A-=trackA; 
  A+=tx_; 
}
else{
  #pragma unroll  
  for(int j =0; j<32; j+=8)
       tr[j/8] = A[ j * lda];
}
   __syncthreads();
  #pragma unroll 
  for(int j=0; j < 4 ; j++){
      res+=tr[j]*buff[ j*8 + ty_];
      la[0][33*(ty_+j*8)+tx_] = tr[j];	
  }
  __syncthreads();
  #pragma unroll  
  for(int j=0; j < 4 ; j++)
      res_+=la[0][33*tx_+j+ty_*4]*buff[32 +j+ty_*4];
  __syncthreads();




   la[0][33*tx_+ty_]= res ;
   __syncthreads();
   if( ty_ == 1 ) 
      res2 =res2+  la[0][tx_*33+0]+la[0][tx_*33+1]+la[0][tx_*33+2]+la[0][tx_*33+3]+la[0][tx_*33+4]+la[0][tx_*33+5]+la[0][tx_*33+6]+la[0][tx_*33+7];
   else 
       res2 = 0 ; 
   __syncthreads();
   la[0][33*tx_+ty_]= res_ ;
   __syncthreads();
   if( ty_ == 0 ) {
      res1 =res1+  la[0][tx_*33+0]+la[0][tx_*33+1]+la[0][tx_*33+2]+la[0][tx_*33+3]+la[0][tx_*33+4]+la[0][tx_*33+5]+la[0][tx_*33+6]+la[0][tx_*33+7];
   }
   else 
       res1 = 0 ; 
  A-=32;

   __syncthreads();
   tx = threadIdx.x ; 
   ty = threadIdx.y ; 
   
   if( ty_ == 0  && ty == 0  ) 
      res = res1 ; 
    else if( ty_ == 1  && ty == 0  ) 
      res = res2 ; 
    else res = 0 ; 

  A-=ty_* lda  ;  
  A-=tx_; 


  A= A - lda*break_d;
  x= x - break_d *inx  ; 

  A+=4 * ty* lda  ;  

if( blkc  == ( gridDim.x - 1 ) ) {
  if(tx <= m_mod_64 )  
     A+=tx;
  else
    A+=m_mod_64; 
}
else{
  A+=tx; 
}

  int wc_c = 0 ; 
  int count = 0 ; 


  tx_ = td % 16 ; 
  ty_ = td / 16 ; 

WC-=tx ;
WC+=tx_; 


   #pragma unroll  
   for(int j=0; j < 4 ; j++)
        b[j] =  buff[ty_*4+j];

  if( break_d > 0) 
  #pragma unroll 
  for(int  i=0; i< 64; i += 64 ){
    res_=0.f;
    count++;
    if( ty== 0 ) {
    if(tx > kstan )
           buff2[tx]  = x[i*inx];
    else
           buff2[tx]  = 0;
    }
           
    __syncthreads();
    #pragma unroll  
    for( int k=0;k<4;k++){
  	   #pragma unroll  
	   for(int j=0; j < 4 ; j++)
	      tr[j] = A[j*lda] ;
   	   #pragma unroll  
 	   for(int j=0; j < 4 ; j++){
   	     res+=tr[j]*buff2[16*k + ty*4+(j)];
  	     la[( (j)+ty*4)][tx] = tr[j]; 
	    }
	    __syncthreads();
	    res_= 0 ; 
   	    #pragma unroll  
	    for(int j=0; j < 4 ; j++)
                  res_+=la[tx_][ty_*4+j]* b[j] ;
	    b[4+k] = res_ ; 	
   	    __syncthreads();
	    A+=lda* 16 ;
   }
   #pragma unroll  
   for(int k=0; k < 4 ; k++){
       la[tx_][ty_+16*k]= b[4+k] ;
   }
   __syncthreads();
   if( ty_ < 4 ) {	
	int k = ty_*16;
     	res_ = la[tx_][0+k] + la[tx_][1+k]+ la[tx_][2+k]+la[tx_][3+k]+la[tx_][4+k]+la[tx_][5+k]+la[tx_][6+k]+la[tx_][7+k]+la[tx_][8+k]+la[tx_][9+k]+la[tx_][10+k]+la[tx_][11+k]+la[tx_][12+k]+la[tx_][13+k]+la[tx_][14+k]+la[tx_][15+k];
     	WC[k + wc_c*lda ] =   res_; 
    }
    wc_c++;
   __syncthreads();
  }
  for(int  i=64; i<break_d; i += 64 ){
    res_=0.f;
    count++;
    if(ty == 0 )
           buff2[tx]  = x[i*inx];
    __syncthreads();
    #pragma unroll  
    for( int k=0;k<4;k++){
  	   #pragma unroll  
	   for(int j=0; j < 4 ; j++)
	      tr[j] = A[j*lda] ;
   	   #pragma unroll  
 	   for(int j=0; j < 4 ; j++){
   	     res+=tr[j]*buff2[16*k + ty*4+(j)];
  	     la[( (j)+ty*4)][tx] = tr[j]; 
	    }
	    __syncthreads();
	    res_= 0 ; 
   	    #pragma unroll  
	    for(int j=0; j < 4 ; j++)
                  res_+=la[tx_][ty_*4+j]* b[j] ;
	    b[4+k] = res_ ; 	
   	    __syncthreads();
	    A+=lda* 16 ;
   }
   #pragma unroll  
   for(int k=0; k < 4 ; k++){
       la[tx_][ty_+16*k]= b[4+k] ;
   }
   __syncthreads();
   if( ty_ < 4 ) {	
	int k = ty_*16;
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
test_l_dsymv_generic_update_v6_ts_tesla (int n, double alpha ,  double* A, int lda, double *x, int inx , double beta ,  double *y , int iny , double *WC, int kstan ){
  int tx = threadIdx.x ;
  int ind = (blockIdx.x)* 64 + tx ;
  double Ca = 0.f ;
  WC+= tx+(blockIdx.x)*64 + lda*blockIdx.x  ;
  for(int i=(blockIdx.x)*64 ;i<n;i+=64){
          Ca+=WC[0] ;
          WC+=64;
  }
  if( ind >kstan && ind < n ) 
     y[ind * iny ] =beta * y[ind * iny  ]  + alpha * Ca ; 
}



__global__ void
test_u_dsymv_generic_v6_ts_tesla (int n, double alpha ,  double* A, int lda, double *x, int inx , double beta ,  double *y , int iny , int m_full_block , int m_mod_32){
}
__global__ void
test_u_dsymv_special_v6_ts_tesla (int n, double alpha ,  double* A, int lda, double *x, int inx , double beta ,  double *y , int iny ){
}



extern "C"
void magmablas_dsymv6_tesla(char uplo, int m, double alpha, double *A, int lda, 
                      double *X, int incx, double beta, double *Y, int incy, 
                      double *dC_work, int kstan)


{
/*
   dsymv performs the matrix-vector operation

     y := alpha*A*x + beta*y,

   where alpha and beta are scalars, x and y are m element vectors and
   A is an m by m symmetric matrix. Matrix A is stored in the lower triangular
   part of A (always).

   This version is designed for the two-sided tridiagonal factorization.
   Parameter kstan can have the following values
   kstan == -1   y              := alpha*A*x + beta*y
   kstan != -1   y(kstan+1:m-1) := alpha*A(kstan+1:m-1,kstan+1:m-1)*x(kstan+1:m-1)+
                                   beta*y(kstan+1:m-1)

   This kernel is recommended for GTX280. It achieves up to 102 GFlop/s.
   It ia recommended that lda is multiple of 16. Otherwise performance would be 
   deteriorated as the memory accesses would not be fully coalescent.
*/

    int blocks;
    if (m % dgemv_bs==0)
        blocks = m/ dgemv_bs;
    else
        blocks = m/ dgemv_bs + 1;

    dim3 grid(blocks, 1, 1);
    dim3 threads(thread_x, thread_y, 1);
    dim3 threads_u(dgemv_bs, 1, 1);
    if(m % dgemv_bs == 0 ) {
       if( uplo == 'L' || uplo == 'l'){
	  test_l_dsymv_special_v6_ts_tesla <<<grid, threads>>>(m, alpha, 
                   A, lda, X, incx ,beta,  Y , incy, dC_work, kstan);
	  test_l_dsymv_special_update_v6_ts_tesla<<<grid, threads_u>>>(m, alpha, 
                        A, lda, X, incx ,beta,  Y , incy, dC_work, kstan);
       }
       else{
          printf("Not Available Now.\n");
      } 
		
    } 
    else{	
      int  m_mod_64 = m%dgemv_bs ; 
      if (uplo == 'L' || uplo == 'l'){
         test_l_dsymv_generic_v6_ts_tesla <<<grid, threads>>> (m, alpha, A, lda, 
                    X, incx ,beta,  Y , incy, dC_work, m_mod_64-1, kstan);
         test_l_dsymv_generic_update_v6_ts_tesla<<<grid, threads_u>>>(m, alpha, 
                        A, lda, X, incx ,beta,  Y , incy, dC_work, kstan);
      }	
      else{
         printf("Not Available Now.\n");
      }	
    }
}

extern "C"
void  magmablas_dsymv_tesla( char uplo , int m , double alpha,  double *A , int lda , 
	 		    double *X , int incx, double beta, double *Y, int incy)
{

	
	double *dC_work;
	int bsz = 64;
	int blocks = m / bsz + (m %bsz != 0);
	int workspace = lda * (blocks + 1);
	cublasAlloc( workspace, sizeof(double), (void**)&dC_work ) ;
			
	cublasGetError( ) ;


	int kstan = -1;

	magmablas_dsymv6_tesla(uplo, m, alpha, A, lda, X, incx, beta, Y, incy, 
                      dC_work, kstan);

	cublasFree(dC_work);

}

#undef thread_x 
#undef thread_y 
