/*
    -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010
*/

#include <stdio.h>
#include "cuda.h"
#include "magma.h"
#include "magma_lapack.h"
#define BLOCK_SIZE 32

//#define num_threads 64
#define dgemv_bs 32



/*- ---------------------------------------------- UPLO = 'L' ----------------------------------*/

__global__ void
l_dlat2s_special (int n, double* A, int lda,  float *SA , int *INFO , double RMAX ,int ldsa ){

  double RMAX_ = -1.0 * RMAX ;  
  int tx = threadIdx.x ; 
  int ty = threadIdx.y ; 
  int ind = blockIdx.x*  dgemv_bs + tx ;

  __shared__ double la   [dgemv_bs][dgemv_bs+1];

  A += ind;            
  SA += ind;            
  A+= ty * lda; 
  SA+= ty * ldsa; 
   
  int break_d  =   blockIdx.x* dgemv_bs ;
  double flag = 0 ; 
  double temp ; 
  for(int  i=0; i<break_d; i += dgemv_bs ){
    #pragma unroll 8 
    for(int j=0; j < dgemv_bs ; j+=4){
        temp = A[j*lda] ;
        if( temp < RMAX_ || temp > RMAX )
            flag = 1 ;
         SA[j*ldsa] = temp;
    }
    A+=lda* dgemv_bs ;
    SA+=ldsa* dgemv_bs ;
  }


  #pragma unroll 8
  for(int j =0; j<dgemv_bs; j+=4)
         la[ty+j][tx] = A[ j * lda];

  __syncthreads();

  A+= dgemv_bs ;
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
  for(int j =0; j<dgemv_bs; j+=4){
        temp =  la[ty+j][tx] ;
        if( temp < RMAX_ || temp > RMAX )
            flag = 1 ;
         SA[j*ldsa] = temp;
  }

  __syncthreads();
   la[tx] [ty] = flag ; 
  __syncthreads();
  if( ty == 0 ) {
    //  INFO[0] = flag+ la[tx] [1] +  la[tx] [2] + la[tx] [3] ;
  } 

}

__global__ void
l_dlat2s_generic(int n, double* A, int lda, float *SA , int m_full_block, 
                 int m_mod_32 , int *INFO , double RMAX , int ldsa)
{ 
  int tx = threadIdx.x ; 
  int ty = threadIdx.y ; 

  int ind = blockIdx.x*  dgemv_bs + tx ;
  

  __shared__ double la   [dgemv_bs][dgemv_bs+1];

    double RMAX_ = -1.0 * RMAX ;
    double temp ; 
    double flag = 0.0 ; 


  if( blockIdx.x == m_full_block ) {
  /************************************************************************
   -- Last block --
   -- We will do something unusual here 
   -- For sufficiently large matrix the overhead will be very low
  *************************************************************************/
       if  ( tx < m_mod_32 ){
		A+= ( blockIdx.x * dgemv_bs + tx ) ;
		SA+= ( blockIdx.x * dgemv_bs + tx ) ;
       } 	 	
       else{
		A+= ( blockIdx.x * dgemv_bs + m_mod_32 -1) ; 
		SA+= ( blockIdx.x * dgemv_bs + m_mod_32 -1) ; 
       }
       A+= ty * lda  ;  
       SA+= ty * ldsa  ;  
       int break_d  =   blockIdx.x* dgemv_bs ;

	  /*----------------------------
		Go Right
	  -------------------------------*/

	  for(int  i=0; i<break_d; i += dgemv_bs ){
	    #pragma unroll 8 
	    for(int j=0; j < dgemv_bs ; j+=4){
	        temp = A[j*lda] ;
	        if( temp < RMAX_ || temp > RMAX )
	            flag = 1 ;
	         SA[j*ldsa] = temp;
	    }
	    A+=lda* dgemv_bs ;
	    SA+=ldsa* dgemv_bs ;
	  }
	  /*
           we don't need to make zero, as those computation will be discarded. 
          */
          if( ty==0  ) {
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
			temp =  A[j*lda] ;
		        if( temp < RMAX_ || temp > RMAX )
	                   flag = 1 ;
	                SA[j*ldsa] = temp;
                }
		A+=(tx)*lda;
		SA+=(tx)*ldsa;
		count = 1 ; 
		for(;j<m_mod_32;j++){
			temp= A[count] ;
		        if( temp < RMAX_ || temp > RMAX )
	                   flag = 1 ;
	                SA[count] = temp;
			count++;
		}
          }
          else{
          }

	 la[tx][ty] = flag ; 

	  __syncthreads(); 
         /*--------------------------------------------------------
	 The leader accumulates all the results from his peer. 
         ----------------------------------------------------------*/
         if( ty == 0 ) {
               //  INFO[ind] = ld[tx][0] +  ld[tx][1] + ld[tx][2] + ld[tx][3] ;
         }
	 
  }

  else{ 
  /***************************************
    -----------------------------------
  -- All the blocks but the last one --
  ****************************************
  -------------------------------------*/
  A += ind;
  SA += ind;
  A+= ty * lda  ;  
  SA+= ty * ldsa  ;  
  int break_d  =   blockIdx.x* dgemv_bs ;

  /*----------------------------
	Go Right
  -------------------------------*/
  for(int  i=0; i<break_d; i += dgemv_bs ){
    #pragma unroll 8 
    for(int j=0; j < dgemv_bs ; j+=4){
        temp = A[j*lda] ;
	if( temp < RMAX_ || temp > RMAX )
            flag = 1 ;
        SA[j*ldsa] = temp;
    }
    A+=lda* dgemv_bs ;
    SA+=ldsa* dgemv_bs ;
  }

 
  /*------------------------------------
	Diagonal 
	Copy + Transpose lower triangle
  --------------------------------------*/
  #pragma unroll 8
  for(int j =0; j<dgemv_bs; j+=4)
         la[ty+j][tx] = A[ j * lda];


  A+= dgemv_bs ;
  __syncthreads();

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
  for(int j =0; j<dgemv_bs; j+=4){
        temp =  la[ty+j][tx] ;
	if( temp < RMAX_ || temp > RMAX )
            flag = 1 ;
        SA[j*ldsa] = temp -0.00;
  }
  __syncthreads();
    la[tx] [ty ] = flag ;
   __syncthreads();
   /*--------------------------------------------------------
	The leader accumulates all the results from his peer. 
   ----------------------------------------------------------*/
    if( ty == 0 )
    {
   //  INFO [ind] =  flag + la[tx][1]+ la[tx][2]+ la[tx][3] ;
    }
  }
}


/*- ---------------------------------------------- UPLO = 'U' ----------------------------------*/
/* Generic Case*/

__global__ void
u_dlat2s_generic(int n, double* A, int lda, float *SA, int m_full_block, 
                 int m_mod_32 , int *INFO , double RMAX , int ldsa)
{ 
  int tx = threadIdx.x ; 
  int ty = threadIdx.y ; 

  int ind = blockIdx.x*  dgemv_bs + tx ;
  

  double RMAX_ = -1.0 * RMAX ;
  double flag = 0 ;

  __shared__ double la   [dgemv_bs][dgemv_bs+1];
  int blockIdxx =  blockIdx.x ;
  double temp ; 
  if( blockIdx.x == m_full_block ) {

  /************************************************************************
   -- Last block --
   -- We will do something unusual here 
   -- For sufficiently large matrix the overhead will be very low
  *************************************************************************/

  ind =  tx ;
  A+= lda*(n-1) ; 
  SA+= ldsa*(n-1) ; 


       if  ( tx < m_mod_32 ){
		A+= (  tx ) ;
		SA+= (  tx ) ;
       } 	 	
       else{
		A+= (  m_mod_32 -1) ; 
		SA+= (  m_mod_32 -1) ; 
       }
       A-= ty * lda  ;  
       SA-= ty * ldsa  ;  
       int break_d  =   (blockIdx.x)* dgemv_bs ;

	  /*----------------------------
		Go Right
	  -------------------------------*/

	  for(int  i=0; i<break_d; i += dgemv_bs ){
	    #pragma unroll 8 
	    for(int j=0; j < dgemv_bs ; j+=4){
	       temp  = A[-j*lda] ;
	       if( temp < RMAX_ || temp > RMAX )
	            flag = 1 ;
	       SA[-j*ldsa] = temp ;
	    }
	    A-=lda* dgemv_bs ;
	    SA-=ldsa* dgemv_bs ;
	  }
	  /*
           we don't need to make zero, as those computation will be discarded. 
          */
          if( ty==0  ) {
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
			temp =  A[-j*lda] ;
	       		if( temp < RMAX_ || temp > RMAX )
		            flag = 1 ;
		        SA[-j*ldsa] = temp ;
                }
		A-=(count-1)*lda;
		SA-=(count-1)*ldsa;
		count = 1 ; 
		for(;j<m_mod_32;j++){
			temp =  A[-count] ;
	       		if( temp < RMAX_ || temp > RMAX )
		            flag = 1 ;
		        SA[-count] = temp ;
			count++;
		}
          }
          else{
          }

         /*--------------------------------------------------------
	 The leader accumulates all the results from his peer. 
         ----------------------------------------------------------*/
        la[tx][ty] = flag ; 
        __syncthreads();
	if( ty == 0 ) { 
		// INFO [ind] = flag  + la[tx][1] + la[tx][2] + la[tx][3]  ; 
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
  A+= lda*(n-1)  ; 
  SA+= ldsa*(n-1)  ; 
  

  A += ind;
  SA += ind;
  A-= ty * lda  ;  
  SA-= ty * ldsa  ;  

  int break_d  = (n / dgemv_bs -   blockIdxx-1 )* dgemv_bs ;
  /*----------------------------
	Go Left
  -------------------------------*/
  for(int  i=0; i<break_d; i += dgemv_bs ){
    #pragma unroll 8 
    for(int j=0; j < dgemv_bs ; j+=4){
        temp  = A[-j*lda] ;
        if( temp < RMAX_ || temp > RMAX )
            flag = 1 ;
         SA[-j*ldsa] = temp ;
    }
    A-=lda* dgemv_bs ;
    SA-=ldsa* dgemv_bs ;
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
  __syncthreads();
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
  for(int j =0; j<dgemv_bs; j+=4){
         temp =   la[tx][31-ty-j];
         // temp =  la[ty+j][tx] ;
         if( temp < RMAX_ || temp > RMAX )
            flag = 1 ;
         SA[- j*ldsa] = temp  ;
  }

   /*--------------------------------------------------------
	The leader accumulates all the results from his peer. 
   ----------------------------------------------------------*/
   la[tx] [ty] = flag ; 
    __syncthreads(); 
  
   if( ty == 0 ) {
    //  INFO[ind] = flag +  la[tx] [1] +  la[tx] [2] + la[tx] [3]  ;
   }

  }

}








/*- ---------------------------------------------- UPLO = 'U' ----------------------------------*/
/*Good Dimension*/
__global__ void
u_dlat2s_special (int n, double* A, int lda, float  *SA , int * INFO , double RMAX , int ldsa ){
  int tx = threadIdx.x ; 
  int ty = threadIdx.y ; 
  int ind = blockIdx.x*  dgemv_bs + tx ;
  double RMAX_ = -1.0 * RMAX ;
  double flag = 0 ; 


  /*
	Reverse Computation ... 
		- Left 
		- Triangle 
		- Up 
  */


  A+= lda*(n-1) ; 
  SA+= ldsa*(n-1) ; 

  __shared__ double la   [dgemv_bs][dgemv_bs+1];

  A += ind;
  SA += ind;

  A-= ty * lda  ;  
  SA-= ty * ldsa  ; 

  double temp ;   
  int break_d  = (n / dgemv_bs -   blockIdx.x-1 )* dgemv_bs ;

  for(int  i=0; i<break_d; i += dgemv_bs ){
    #pragma unroll 8 
    for(int j=0; j < dgemv_bs ; j+=4){
        //   la[tx][ty+j] = A[-j*lda] ;
        temp =  A[-j*lda] ;
        if( temp < RMAX_ || temp > RMAX )
            flag = 1 ;
         SA[-j*ldsa] = temp ;
    }
    A-=lda* dgemv_bs ;
    SA-=ldsa* dgemv_bs ;
  }


  #pragma unroll 8
  for(int j =0; j<dgemv_bs; j+=4)
         la[tx][31-ty-j] = A[ -j * lda];
  /*
	Look at the indexing changes
  */

  A-= dgemv_bs ;
  __syncthreads();
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
  for(int j =0; j<dgemv_bs; j+=4){
         temp =   la[tx][31-ty-j];
         if( temp < RMAX_ || temp > RMAX )
            flag = 1 ;
         SA[- j*ldsa] = temp  ;
  }

  la[tx][ty] = flag ; 

  __syncthreads();

  if( ty == 0 ) {
    // INFO[0] = flag + la[tx][1] + la[tx][2] + la[tx][3] ; 	
  }
}


extern "C" void 
mdlat2s(char uplo, int m, double *A, int lda, float *Y, int LDSA, int *INFO)  
{
/*
Note:
	The UPLO = 'U' Version can be optimized more.
*/
    double RMAX = lapackf77_slamch("O"); 
    int blocks;
    if (m % dgemv_bs==0)
        blocks = m/ dgemv_bs;
    else
        blocks = m/ dgemv_bs + 1;

    dim3 grid(blocks, 1, 1);
    dim3 threads(32, 4, 1);

    if( m % dgemv_bs == 0 ) {
	    if( uplo == 'L' || uplo == 'l'){
                  l_dlat2s_special <<<grid, threads>>> (m, A, lda, Y ,INFO ,  RMAX , LDSA  );
	    }
            else{
	          u_dlat2s_special <<<grid, threads>>> (m, A, lda,  Y , INFO , RMAX , LDSA  );
	    } 
		
    } 
    else{	
	    int  m_full_block = (m - m % 32 ) /32 ; 
	    int  m_mod_32 = m%32 ;  
	    if( uplo == 'L' || uplo == 'l'){
		    l_dlat2s_generic <<<grid, threads>>> (m, A, lda, Y , m_full_block , m_mod_32 , INFO , RMAX , LDSA );
	    }	
	    else{
		    u_dlat2s_generic <<<grid, threads>>> (m, A, lda, Y , m_full_block , m_mod_32 , INFO , RMAX , LDSA );
	    }	
    }
}







/*
Interface ..................................
Reproduced from dlansy routines... 
How to deliver the info. 
*/


#include "cublas.h"
#include "magma.h"
#include <stdio.h>


extern "C" void 
magmablas_dlat2s(char uplo, int n, double *A, int lda,  
                 float *SA, int LDSA, int *INFO)
{
/*
     The routine converts a double-precision triangular 
     matrix A to a single-precision triangular matrix SA. 
*/
		mdlat2s ( uplo , n , A , lda , SA , LDSA ,  INFO );
/*
		int val = cublasIdamax(n,WORK,1);
	        double retVal[1];
		cublasGetMatrix( 1, 1, sizeof( double ), WORK+val-1, 1, retVal, 1 ) ;
		return retVal[0];
*/
}
