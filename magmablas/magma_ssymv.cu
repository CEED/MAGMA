#include <stdio.h>
#include "cuda.h"
#define BLOCK_SIZE 32

//#define num_threads 64
#define dgemv_bs 32




__global__ void
l_ssymv_special (int n, float alpha ,  float* A, int lda, float *x, int inx , float beta ,  float *y , int iny ){
  int tx = threadIdx.x ; 
  int ty = threadIdx.y ; 
  int ind = blockIdx.x*  dgemv_bs + tx ;
  float res = 0.f;


  __shared__ float buff [dgemv_bs];
  __shared__ float la   [dgemv_bs][dgemv_bs+1];


  A += ind;
  x += tx  * inx ;
  A+= ty * lda  ;  
  int break_d  =   blockIdx.x* dgemv_bs ;

  for(int  i=0; i<break_d; i += dgemv_bs ){
    #pragma unroll 8 
    for(int j=0; j < dgemv_bs ; j+=4){
        la[tx][ty+j] = A[j*lda] ;
    }
    buff[tx]  = x[i*inx];
    __syncthreads();

    #pragma unroll 8 
    for(int j=0; j < 8 ; j++){
       res+=la[tx][j+ty*8]*buff[j+ty*8];
    }
    A+=lda* dgemv_bs ;
    __syncthreads(); 
  }

 

  #pragma unroll 8
  for(int j =0; j<dgemv_bs; j+=4)
         la[ty+j][tx] = A[ j * lda];


  A+= dgemv_bs ;
  x+= break_d *inx  ; 
  __syncthreads();
  //buff[tx]  = x[break_d*inx];
  buff[tx]  = x[0*inx];
  #pragma unroll 8
  for(int  i=ty*8; i<(1+ty)* dgemv_bs/4 ; i++){
         if ( i < tx )   {
	        la[tx][i] = la[i][tx] ; 
         }
	 else 
	        la[tx][i] = la[tx][i]  ;
  
  }
  __syncthreads();
    #pragma unroll 8
    for(int j=0; j < dgemv_bs/4 ; j++){
     res+=la[tx][j+ty*8]*buff[j+ty*8];
    }
  x-= break_d *inx ; 
  break_d  += dgemv_bs ; 
  __syncthreads();




  for(int i=break_d; i<n; i += dgemv_bs ){
    buff[tx]  = x[i*inx];
   #pragma unroll 8
    for(int j=0; j<dgemv_bs; j+=4)
       la[ty+j][tx] = A[ j * lda];
    A+= dgemv_bs ;
      __syncthreads();
    #pragma unroll 8
    for(int j=0; j < dgemv_bs/4;j++){
       res+=la[tx][j+ty*8]*buff[j+ty*8];
    }
      __syncthreads();
  }


   la[tx][ty]= res ;
   __syncthreads();
   if( ty == 0 ) {
     res = res + la[tx][1]+ la[tx][2]+ la[tx][3] ;
     y[ind * iny ] = beta * y[ind * iny  ]  + alpha * res;
   }

}

__global__ void
l_ssymv_generic (int n, float alpha ,  float* A, int lda, float *x, int inx , float beta ,  float *y , int iny , int m_full_block , int m_mod_32){

  
  int tx = threadIdx.x ; 
  int ty = threadIdx.y ; 

  int ind = blockIdx.x*  dgemv_bs + tx ;
  
  float res = 0.f;


  __shared__ float buff [dgemv_bs];
  __shared__ float la   [dgemv_bs][dgemv_bs+1];

  if( blockIdx.x == m_full_block ) {
  /************************************************************************
   -- Last block --
   -- We will do something unusual here 
   -- For sufficiently large matrix the overhead will be very low
  *************************************************************************/
       if  ( tx < m_mod_32 ){
		A+= ( blockIdx.x * dgemv_bs + tx ) ;
       } 	 	
       else{
		A+= ( blockIdx.x * dgemv_bs + m_mod_32 -1) ; 
		//A+= ( blockIdx.x * dgemv_bs + 0) ; 
       }
       x+=tx *inx;
       A+= ty * lda  ;  
       int break_d  =   blockIdx.x* dgemv_bs ;

	  /*----------------------------
		Go Right
	  -------------------------------*/

	  for(int  i=0; i<break_d; i += dgemv_bs ){
	    #pragma unroll 8 
	    for(int j=0; j < dgemv_bs ; j+=4){
	        la[tx][ty+j] = A[j*lda] ;
	    }
	    buff[tx]  = x[i * inx];
	    __syncthreads();

	    #pragma unroll 8 
	    for(int j=0; j < 8 ; j++){
	       res+=la[tx][j+ty*8]*buff[j+ty*8];
	    }
	    A+=lda* dgemv_bs ;
	    __syncthreads(); 
	  }
	  /*
           we don't need to make zero, as those computation will be discarded. 
          */
          if( ty==0  ) {
		x+= ( break_d -tx ) ; 
	        //buff[tx]  = x[i*inx];
		/*--------------------------------------------
			he will compute the triangular parts
			others will be waiting with values. 
                -----------------------------------------------*/
		int j ;
                int count = 1 ; 
		if( tx < m_mod_32 ) 
			count = tx ; 
		else
			count = m_mod_32 ;
		for(j =0;j<=count;j++){
			res+= A[j*lda] * x[j*inx];
                }
		A+=(tx)*lda;
		count = 1 ; 
		for(;j<m_mod_32;j++){
			res+= A[count] * x[j*inx];
			count++;
		}
          }
          else{
          }
	  __syncthreads(); 
   	 la[tx][ty]= res ;
          __syncthreads();
         /*--------------------------------------------------------
	 The leader accumulates all the results from his peer. 
         ----------------------------------------------------------*/
         if( ty == 0 ) {
             res = res + la[tx][1]+ la[tx][2]+ la[tx][3] ;
	     if( tx < m_mod_32)
                 y[ind * iny ] = beta * y[ind * iny ]  + alpha * res;
         }
	 
  }

  else{ 
  /***************************************
    -----------------------------------
  -- All the blocks but the last one --
  ****************************************
  -------------------------------------*/
  A += ind;
  x += tx*inx ;
  A+= ty * lda  ;  
  int break_d  =   blockIdx.x* dgemv_bs ;

  /*----------------------------
	Go Right
  -------------------------------*/
  for(int  i=0; i<break_d; i += dgemv_bs ){
    #pragma unroll 8 
    for(int j=0; j < dgemv_bs ; j+=4){
        la[tx][ty+j] = A[j*lda] ;
    }
    buff[tx]  = x[i*inx];
    __syncthreads();

    #pragma unroll 8 
    for(int j=0; j < 8 ; j++){
       res+=la[tx][j+ty*8]*buff[j+ty*8];
    }
    A+=lda* dgemv_bs ;
    __syncthreads(); 
  }

 
  /*------------------------------------
	Diagonal 
	Copy + Transpose lower triangle
  --------------------------------------*/
  #pragma unroll 8
  for(int j =0; j<dgemv_bs; j+=4)
         la[ty+j][tx] = A[ j * lda];


  A+= dgemv_bs ;
  x+= break_d * inx  ; 
  __syncthreads();
  //buff[tx]  = x[break_d*inx];
  buff[tx]  = x[0*inx];
  /*--------------------------------------------
	Mirror Upper Triangle to Lower triangle
  ---------------------------------------------*/
  #pragma unroll 8
  for(int  i=ty*8; i<(1+ty)* dgemv_bs/4 ; i++){
         if ( i < tx )   {
	        la[tx][i] = la[i][tx] ; 
         }
	 else 
	        la[tx][i] = la[tx][i]  ;
  
  }
  __syncthreads();
  /*--------------------------------
	Do diagonal Computation
  -----------------------------------*/
    #pragma unroll 8
    for(int j=0; j < dgemv_bs/4 ; j++){
     res+=la[tx][j+ty*8]*buff[j+ty*8];
    }
  x-= break_d  * inx ; 
  break_d  += dgemv_bs ; 
  __syncthreads();


  n -= m_mod_32 ;  // @ 
  /*-----------------------------
	Go Down 
  -------------------------------*/
  for(int i=break_d; i<n; i += dgemv_bs ){
    buff[tx]  = x[i*inx];
   #pragma unroll 8
    for(int j=0; j<dgemv_bs; j+=4)
       la[ty+j][tx] = A[ j * lda];
    A+= dgemv_bs ;
      __syncthreads();
    #pragma unroll 8
    for(int j=0; j < dgemv_bs/4;j++){
       res+=la[tx][j+ty*8]*buff[j+ty*8];
    }
      __syncthreads();
  }

  
  /*---------------------------------------------
	doing m_mod_32 stuffs here.
	Symmetric is giving us benefit .. true
  -----------------------------------------------*/
    x-=tx * inx;
    x+=n*inx;
    A-=tx;
    if( tx < m_mod_32){
        buff[tx]  = x[tx*inx];
	A+=tx;
    }
    else{
        buff[tx]  = 0*x[m_mod_32-1]; /*This will confirm valid memory reference*/
	A+=(m_mod_32-1); /* Same as above*/
    }

   #pragma unroll 8
    for(int j=0; j<dgemv_bs; j+=4){
       if( tx < m_mod_32 ) 
       la[ty+j][tx] = 1.0 * A[ j * lda];
       else
       la[ty+j][tx] = 0.0 * A[ j * lda];
       
    }
    __syncthreads();

    /*----------------------------------------
	What about doing some Zeroing here?
	instead of zeroing before?
    -----------------------------------------*/	
    #pragma unroll 8
    for(int j=0; j < dgemv_bs/4;j++){
       //if( ( j+ty*8) < m_mod_32 )
       res+=la[tx][j+ty*8]*buff[j+ty*8];
    }
    __syncthreads();
   

   la[tx][ty]= res ;
   __syncthreads();
   /*--------------------------------------------------------
	The leader accumulates all the results from his peer. 
   ----------------------------------------------------------*/
   if( ty == 0 ) {
     res = res + la[tx][1]+ la[tx][2]+ la[tx][3] ;
     y[ind * iny] = beta * y[ind * iny]  + alpha * res;
   }

  }

}

__global__ void
u_ssymv_generic (int n, float alpha ,  float* A, int lda, float *x, int inx , float beta ,  float *y , int iny , int m_full_block , int m_mod_32){

  
  int tx = threadIdx.x ; 
  int ty = threadIdx.y ; 

  int ind = blockIdx.x*  dgemv_bs + tx ;
  
  float res = 0.f;


  __shared__ float buff [dgemv_bs];
  __shared__ float la   [dgemv_bs][dgemv_bs+1];
  int blockIdxx =  blockIdx.x ;

  if( blockIdx.x == m_full_block ) {

  /************************************************************************
   -- Last block --
   -- We will do something unusual here 
   -- For sufficiently large matrix the overhead will be very low
  *************************************************************************/

  ind =  tx ;
  A+= lda*(n-1) ; 
  x+= (n-1)*inx;


       if  ( tx < m_mod_32 ){
		A+= (  tx ) ;
       } 	 	
       else{
		A+= (  m_mod_32 -1) ; 
       }
       x-=tx*inx;
       A-= ty * lda  ;  
       int break_d  =   (blockIdx.x)* dgemv_bs ;

	  /*----------------------------
		Go Right
	  -------------------------------*/

	  for(int  i=0; i<break_d; i += dgemv_bs ){
	    #pragma unroll 8 
	    for(int j=0; j < dgemv_bs ; j+=4){
	        la[tx][ty+j] = A[-j*lda] ;
	    }
	    buff[tx]  = x[-i*inx];
	    __syncthreads();

	    #pragma unroll 8 
	    for(int j=0; j < 8 ; j++){
	       res+=la[tx][j+ty*8]*buff[j+ty*8];
	    }
	    A-=lda* dgemv_bs ;
	    __syncthreads(); 
	  }
	  /*
           we don't need to make zero, as those computation will be discarded. 
          */
          if( ty==0  ) {
		x-= ( break_d -tx ) ; 
	        //buff[tx]  = x[i*inx];
		/*--------------------------------------------
			he will compute the triangular parts
			others will be waiting with values. 
                -----------------------------------------------*/
		int j ;
                int count = 1 ; 
		if( tx < m_mod_32 ) 
			count =m_mod_32- tx ; 
		else
			count = m_mod_32 ;
		for(j =0;j<count;j++){
			res+= A[-j*lda] * x[-j*inx];
                }
		A-=(count-1)*lda;
		count = 1 ; 
		for(;j<m_mod_32;j++){
			res+= A[-count] * x[-j*inx];
			count++;
		}
          }
          else{
          }
	  __syncthreads(); 
   	 la[tx][ty]= res ;
          __syncthreads();
         /*--------------------------------------------------------
	 The leader accumulates all the results from his peer. 
         ----------------------------------------------------------*/
         if( ty == 0 ) {
             res = res + la[tx][1]+ la[tx][2]+ la[tx][3] ;
	     if( tx < m_mod_32)
                 y[ind *iny] = beta * y[ind *iny]  + alpha * res;
         }
	 
  }

  else{ 
  /***************************************
    -----------------------------------
  -- All the blocks but the last one --
  -- By the way this code can be optimized more. 
  ****************************************
  -------------------------------------*/
  ind = blockIdx.x *  dgemv_bs + tx + m_mod_32 ;
  float *A1 = A ; 
  float *x1 = x ; 
  A+= lda*(n-1)  ; 
  x+= (n-1)*inx;

  A += ind;
  x -= tx * inx ;
  A-= ty * lda  ;  

  int break_d  = (n / dgemv_bs -   blockIdxx-1 )* dgemv_bs ;
  /*----------------------------
	Go Left
  -------------------------------*/
  for(int  i=0; i<break_d; i += dgemv_bs ){
    #pragma unroll 8 
    for(int j=0; j < dgemv_bs ; j+=4){
        la[tx][ty+j] = A[-j*lda] ;
    }
    buff[tx]  = x[-i*inx];
    __syncthreads();

    #pragma unroll 8 
    for(int j=0; j < 8 ; j++){
       res+=la[tx][j+ty*8]*buff[j+ty*8];
    }
    A-=lda* dgemv_bs ;
    __syncthreads(); 
  }

 
  /*------------------------------------
	Diagonal 
	Copy + Transpose lower triangle
  --------------------------------------*/
  #pragma unroll 8
  for(int j =0; j<dgemv_bs; j+=4){
         la[tx][31-ty-j] = A[ -j * lda];
  }

  A-= dgemv_bs ;
  x-= break_d *  inx ; 
  __syncthreads();
  buff[31-tx]  = x[0*inx];
  /*--------------------------------------------
	Mirror Upper Triangle to Lower triangle
  ---------------------------------------------*/
  #pragma unroll 8
  for(int  i=ty*8; i<(1+ty)* dgemv_bs/4 ; i++){
         if ( i <tx ){
	        la[tx][i] = la[i][tx]; 
         }
	 else{ 
	        la[tx][i] = la[tx][i]  ;
	 }
  }
  __syncthreads();
  /*--------------------------------
	Do diagonal Computation
  -----------------------------------*/
    #pragma unroll 8
    for(int j=0; j < dgemv_bs/4 ; j++){
     res+=la[tx][j+ty*8]*buff[j+ty*8];
    }
  x+=break_d * inx ; 
  break_d  += dgemv_bs ; 
  __syncthreads();


  n -= m_mod_32 ;  // @ 
  /*-----------------------------
	Go Up 
  -------------------------------*/
  int i ;
  for( i=break_d; i<n; i+= dgemv_bs ){
    buff[31-tx]  = x[-i*inx] ;
   #pragma unroll 8
    for(int j=0; j<dgemv_bs; j+=4){
       la[ty+j][tx] = A[- j * lda];
    }
    A-= dgemv_bs ;
      __syncthreads();
    #pragma unroll 8
    for(int j=0; j < dgemv_bs/4;j++){
       res+=la[31-tx][j+ty*8]*buff[j+ty*8];
    }
      __syncthreads();
  }
  /*---------------------------------------------
	doing m_mod_32 stuffs here.
	Symmetric is giving us benefit .. true
	Do the other way please......
  -----------------------------------------------*/
   A1 = A1 + m_mod_32 * lda + tx *lda ;  
   if( ty == 0  ) {
	for( int j = 0 ;  j < m_mod_32 ; j++){
		res+= x1[j*inx] * A1[ j + lda * (blockIdx.x) * 32 ];
	}
   }
    __syncthreads();

   la[tx][ty]= res ;
   __syncthreads();
   /*--------------------------------------------------------
	The leader accumulates all the results from his peer. 
   ----------------------------------------------------------*/
   if( ty == 0 ) {
     res = res + la[tx][1]+ la[tx][2]+ la[tx][3] ;
     y[ind *iny] = beta * y[ind * iny]  + alpha * res;
   }

  }

}






__global__ void
u_ssymv_special (int n, float alpha ,  float* A, int lda, float *x, int inx , float beta ,  float *y , int iny ){
  int tx = threadIdx.x ; 
  int ty = threadIdx.y ; 
  int ind = blockIdx.x*  dgemv_bs + tx ;
  float res = 0.f;

  /*
	Reverse Computation ... 
		- Left 
		- Triangle 
		- Up 
  */

  A+= lda*(n-1) ; 
  x+= (n-1) * inx ;
  __shared__ float buff [dgemv_bs];
  __shared__ float la   [dgemv_bs][dgemv_bs+1];


  A += ind;
  x -= tx * inx ;
  A-= ty * lda  ;  
  int break_d  = (n / dgemv_bs -   blockIdx.x-1 )* dgemv_bs ;

  for(int  i=0; i<break_d; i += dgemv_bs ){
    #pragma unroll 8 
    for(int j=0; j < dgemv_bs ; j+=4){
        la[tx][ty+j] = A[-j*lda] ;
    }
    buff[tx]  = x[-i*inx];
    __syncthreads();

    #pragma unroll 8 
    for(int j=0; j < 8 ; j++){
       res+=la[tx][j+ty*8]*buff[j+ty*8];
    }
    A-=lda* dgemv_bs ;
    __syncthreads(); 
  }




  #pragma unroll 8
  for(int j =0; j<dgemv_bs; j+=4)
         la[tx][31-ty-j] = A[ -j * lda];
  /*
	Look at the indexing changes
  */

  A-= dgemv_bs ;
  x-= break_d * inx ; 
  __syncthreads();
  buff[31-tx]  = x[0*inx];
  #pragma unroll 8
  for(int  i=ty*8; i<(1+ty)* dgemv_bs/4 ; i++){
         if ( i <tx ){
	        la[tx][i] = la[i][tx]; 
         }
	 else{ 
	        la[tx][i] = la[tx][i]  ;
	 }
  
  }
  __syncthreads();
    #pragma unroll 8
    for(int j=0; j < dgemv_bs/4 ; j++){
     res+=la[tx][j+ty*8]*buff[j+ty*8];
    }

 x+=break_d *inx ; 
  break_d  += dgemv_bs ; 
  __syncthreads();



  for(int i=break_d; i<n; i+= dgemv_bs ){
    buff[31-tx]  = x[-i*inx] ;
   #pragma unroll 8
    for(int j=0; j<dgemv_bs; j+=4)
       la[ty+j][tx] = A[ -j * lda];

    A-= dgemv_bs ;
      __syncthreads();
    #pragma unroll 8
    for(int j=0; j < dgemv_bs/4;j++){
       res+=la[31-tx][j+ty*8]*buff[j+ty*8];
    }
      __syncthreads();
  }


   la[tx][ty]= res ;

   __syncthreads();
   if( ty == 0 ) {
     res = res + la[tx][1]+ la[tx][2]+ la[tx][3] ;
     y[ind *iny] =  beta * y[ind *iny]  + alpha * res;
   }

}





extern "C" void mssymv (char side , char uplo , int m , float alpha ,  float *A , int lda , 
 float *X , int incx , float beta , float *Y , int incy )
{
/*
Note:
	The UPLO = 'U' Version can be optimized more.
        side is not needed........................... 
*/
    int blocks;
    if (m % dgemv_bs==0)
        blocks = m/ dgemv_bs;
    else
        blocks = m/ dgemv_bs + 1;

    dim3 grid(blocks, 1, 1);
    dim3 threads(32, 4, 1);

    if( m % dgemv_bs == 0 ) {
	    if( uplo == 'L' || uplo == 'l'){	
		    l_ssymv_special <<<grid, threads>>> (m, alpha , A, lda, X, incx ,beta,  Y , incy);
	    }
            else{
		    u_ssymv_special <<<grid, threads>>> (m, alpha , A, lda, X, incx ,beta,  Y , incy);
	    } 
		
    } 
    else{	
	    int  m_full_block = (m - m % 32 ) /32 ; 
	    int  m_mod_32 = m%32 ;  
	    if( uplo == 'L' || uplo == 'l'){
		    l_ssymv_generic <<<grid, threads>>> (m, alpha , A, lda, X, incx ,beta,  Y , incy, m_full_block , m_mod_32);
	    }	
	    else{
		    u_ssymv_generic <<<grid, threads>>> (m, alpha , A, lda, X, incx ,beta,  Y , incy, m_full_block , m_mod_32);
	    }	
    }
}


/*
Interface ..................................
*/

extern "C" void  magmablas_ssymv (char uplo , int m , float alpha ,  float *A , int lda ,  float *X , int incx , float beta , float *Y , int incy )
{
/*
  DSYMV  performs the matrix-vector  operation

     y := alpha*A*x + beta*y,

  where alpha and beta are scalars, x and y are n element vectors and
  A is an n by n symmetric matrix.

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

  ALPHA  - SINGLE PRECISION.
           On entry, ALPHA specifies the scalar alpha.
           Unchanged on exit.

  A      - SINGLE PRECISION array of DIMENSION ( LDA, n ).
           Before entry with  UPLO = 'U' or 'u', the leading n by n
           upper triangular part of the array A must contain the upper
           triangular part of the symmetric matrix and the strictly
           lower triangular part of A is not referenced.
           Before entry with UPLO = 'L' or 'l', the leading n by n
           lower triangular part of the array A must contain the lower
           triangular part of the symmetric matrix and the strictly
           upper triangular part of A is not referenced.
           Unchanged on exit.

  LDA    - INTEGER.
           On entry, LDA specifies the first dimension of A as declared
           in the calling (sub) program. LDA must be at least
           max( 1, n ).
           Unchanged on exit.

  X      - SINGLE PRECISION array of dimension at least
           ( 1 + ( n - 1 )*abs( INCX ) ).
           Before entry, the incremented array X must contain the n
           element vector x.
           Unchanged on exit.

  INCX   - INTEGER.
           On entry, INCX specifies the increment for the elements of
           X. INCX must not be zero.
           Unchanged on exit.

  BETA   - SINGLE PRECISION.
           On entry, BETA specifies the scalar beta. When BETA is
           supplied as zero then Y need not be set on input.
           Unchanged on exit.

  Y      - SINGLE PRECISION array of dimension at least
           ( 1 + ( n - 1 )*abs( INCY ) ).
           Before entry, the incremented array Y must contain the n
           element vector y. On exit, Y is overwritten by the updated
           vector y.

  INCY   - INTEGER.
           On entry, INCY specifies the increment for the elements of
           Y. INCY must not be zero.
           Unchanged on exit.

*/
        char side = 'a' ;
	mssymv (side, uplo , m , alpha , A , lda , X , incx , beta , Y , incy );

}
