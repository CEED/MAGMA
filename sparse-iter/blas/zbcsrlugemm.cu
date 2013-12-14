/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> c d s

*/

#include <cuda_runtime_api.h>
#include <cublas_v2.h>  // include before magma.h

#include "magma.h"

#if (GPUSHMEM < 200)
   #define BLOCK_SIZE 128
#else
   #define BLOCK_SIZE 512
#endif

#define PRECISION_z

#define  Ablockinfo(i,j)  Ablockinfo[(i)*c_blocks   + (j)]
#define  Bblockinfo(i,j)  Bblockinfo[(i)*c_blocks   + (j)]
#define A(i,j) ((Ablockinfo(i,j)-1)*size_b*size_b)
#define B(i,j) ((Bblockinfo(i,j)-1)*size_b*size_b)

//============================================================

#define ldb m
#define lda m
#define ldc m

//texture<int2,1>  tex_x_double_A;
//texture<int2,1>  tex_x_double_B;

/*
#if defined(PRECISION_d)
static __inline__ __device__ double fetch_x_A(const int& i)
{
  register int2  v = tex1Dfetch(tex_x_double_A, i);
  return __hiloint2double(v.y, v.x);
}

static __inline__ __device__ double fetch_x_B(const int& i)
{
  register int2  v = tex1Dfetch(tex_x_double_B, i);
  return __hiloint2double(v.y, v.x);
}
#endif

#if defined(PRECISION_z)
static __inline__ __device__ double fetch_x_A(const int& i)
{

}

static __inline__ __device__ double fetch_x_B(const int& i)
{

}
#endif


#if defined(PRECISION_c)
static __inline__ __device__ double fetch_x_A(const int& i)
{

}

static __inline__ __device__ double fetch_x_B(const int& i)
{

}
#endif


#if defined(PRECISION_s)
static __inline__ __device__ double fetch_x_A(const int& i)
{

}

static __inline__ __device__ double fetch_x_B(const int& i)
{

}
#endif
*/
#define fetch_x_A(i) (((i)<m*m)?Aval[i]:0)
#define fetch_x_B(i) (((i)<m*m)?B[i]:0)


//============================================================

// every multiprocessor handles one BCSR-block
__global__ void 
zbcsr_gemm_kernel( 
                  int m,
                  int n,
                  int kblocks,   
                  double **Avals, 
                  double **Bval,
                  double **Cval,
                  int I, int K )
{
    if (blockIdx.z!=I)
       return;


#if defined(PRECISION_d)
    const  int tx = threadIdx.x;
    const  int ty = threadIdx.y;

    const int iby = 0;//blockIdx.y * 64;
    const int ibx = 0;//blockIdx.x * 64;
    const int idt = ty * 64 + tx;

    const int tx2 = idt%16;
    const int ty2 = idt/16;

    double xxB[4];
 

 // Prepare where to write the result
        double *C = Cval[blockIdx.z * kblocks + K];
/*
        if (tx == 0 && ty == 0){
              printf("\nPrinting C from the kernel %d\n", C);
              for(int i =0; i<m; i++){
                  for(int j =0; j<m; j++)
                      printf("  %7.4f", C[j*m+i]);
                      printf("\n");
              }
             printf("\n\n");
        }
*/

    double *B;

    int trackA = ibx +__mul24( ty2, lda) + tx2 ;
    double *Aval = Avals[blockIdx.z];// + trackA;

    __shared__ double Abs[64][65];
    __shared__ double  Bb[16][65];

/*
    #pragma unroll
    for(int j=0; j<4; j++){
        for(int y=0; y<4; y++)
           Abs[tx2+ y*16][ty2+16*j] = fetch_x_A(trackA + y*16) ;
    }
*/
    for(int j=ty2; j<64; j+=16){
        for(int y=tx2; y<64; y+=16){
           Abs[y][j] = fetch_x_A(trackA + y-tx2) ;
                //printf("Abs[%d][%d] = A[%d]\n", tx2+ y, ty2+j, trackA+y);
            }
        trackA += __mul24( 16, m);
    }

       // __syncthreads(); (3)
/*
        if (tx == 0 && ty == 0){
           printf("m = %d\n", m);
           for(int i =0; i<m; i++){
              for(int j =0; j<m; j++)
                 printf("  %7.4f", Abs[i][j]);
              printf(" [%d]\n",m);
           }
           printf("\n\n");
        }
    
*/
    //for(int k=0; k<kblocks; k++){
    int k =K; {

        B = Bval[k];
        int trackB = tx2+ __mul24(iby + ty2 * 4, ldb );

        // Prefetch part of B
/*
        #pragma unroll
        for(int y=0; y<4; y++)
           Bb[tx2][ty2*4+y] = fetch_x_B( trackB + y * ldb) ;
*/
        //if (tx2<m) {
          #pragma unroll
          for(int y=0; y<4; y++){
             //if (ty2*4+y < n)
                 Bb[tx2][ty2*4+y] = fetch_x_B( trackB + y * ldb) ;
               // printf("%d  %d  %d  -> %d\n", tx2, ty2, y, trackB + (y/16) * ldb);
          }
        //}
        __syncthreads();    // this is necessary!!!
/*
        if (tx == 0 && ty == 0 && k ==0){
           printf("\n");
           for(int i =0; i<16; i++){
              for(int j =0; j<m; j++)
                 printf("  %7.4f", Bb[i][j]);
              printf("\n");
           }
        }
*/
        double Axs[4];
        double Bxp[4];

        double Cb[16] = {0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0};

        int k1;
        for(k1=0; k1<m-16; k1+=16)
        {
//printf("enterrrrrrrrrrrrrrrrr\n");
                trackB += 16;

                #pragma unroll
                for( int y=0; y<4; y++)
                        xxB[y] = fetch_x_B( trackB + y*ldb);

                #pragma unroll
                for( int j1=0;j1<16;j1++)
                {
                        #pragma unroll
                        for( int y=0; y<4; y++){
                                Axs[y] =  Abs[tx2+y*16][j1+k1] ;
                                //printf("----Axs[%d]  = Abs[%d][%d]\n", y, tx2+y*16+k1*64, j1);
                        }

                        #pragma unroll
                        for( int y=0; y<4; y++){
                                Bxp[y]= Bb[j1][ty2+y*16];
                                //printf("Bxp[%d]  = Bb[%d][%d]\n", y, j1, ty2+y*16);
                        }

                        #pragma unroll
                        for( int x=0; x<4; x++)
                        {
                                #pragma unroll
                                for( int y=0; y<4; y++)
                                {
                                        Cb[x*4+y]  += Axs[x]*Bxp[y];
                                }
                        }
/*
                        if (tx == 0 && ty == 0){
                              printf("\n");
                              for(int i =0; i<4; i++){
                                  for(int j =0; j<4; j++)
                                      printf("  %7.4f", Cb[i*m+j]);
                                      printf("\n");
                              }
                             printf("\n\n");
                        }
*/

                }


                //__syncthreads();(2)

                #pragma unroll
                for(int y=0; y<4; y++)
                        Bb[tx2][ty2*4 + y] = xxB[y];

                __syncthreads();     // this is necessary!!!
/*
                if (tx == 0 && ty == 0 && k ==0){
                   printf("\n");
                   for(int i =0; i<16; i++){
                      for(int j =0; j<m; j++)
                        printf("  %7.4f", Bb[i][j]);
                      printf("\n");
                   }
                   printf("\n\n");
                }
*/
        }

        // Prepare where to write the result
        //double *C = Cval[blockIdx.z * kblocks + k];
/*
        if (tx == 0 && ty == 0){
              printf("\nPrinting C from the kernel %d\n", C);
              for(int i =0; i<m; i++){
                  for(int j =0; j<m; j++)
                      printf("  %7.4f", C[j*m+i]);
                      printf("\n");
              }
             printf("\n\n");
        }
*/
 //__syncthreads();(1)

        C += tx2 + ibx  + __mul24 (ty2 +  iby ,ldc);

        #pragma unroll
        for(int j1=0;j1<16;j1++)
        {

                #pragma unroll
                for( int y=0; y<4; y++)
                        Axs[y] =  Abs[tx2 + y*16][j1+k1] ;

                #pragma unroll
                for( int y=0; y<4; y++)
                        Bxp[y]= Bb[j1][ty2 + y*16];

//printf("Axs: %2d %2d: %f  %f   %f   %f\n", tx2, ty2, Axs[0], Axs[1], Axs[2], Axs[3]);
//printf("Bxp: %2d %2d: %f  %f   %f   %f\n", tx2, ty2, Bxp[0], Bxp[1], Bxp[2], Bxp[3]);

                #pragma unroll
                for( int x=0; x<4; x++)
                {
                        #pragma unroll
                        for( int y=0;y<4; y++)
                        {
                                Cb[x*4 + y]  += Axs[x]*Bxp[y];
                        }
                }
        }
/*
        if (tx == 0 && ty == 0){
              printf("\n");
              for(int i =0; i<4; i++){
                  for(int j =0; j<4; j++)
                      printf("  %7.4f", Cb[i*m+j]);
                      printf("\n");
              }
             printf("\n\n");
        }
*/
/*
    const  int tx = threadIdx.x;
    const  int ty = threadIdx.y;

    const int iby = 0;// blockIdx.y * 64;
    const int ibx = 0;// blockIdx.x * 64;
    const int idt = ty * 64 + tx;

    const int tx2 = idt%16;
    const int ty2 = idt/16;

    const int iby2 =  blockIdx.y * 64;
    const int ibx2 =  blockIdx.x * 64;  
*/
      


        int gy = iby + ty2;
        #pragma unroll
        for( int y=0;y<4;y++, gy+=16)
        {
                int gx = ibx + tx2;
        #pragma unroll
                for(int x=0;x<4;x++, gx+=16)
                {
                        if (gx < m && gy < n){
                           //printf("block %d  i=%2d j = %2d  val = %f %f\n",
                                 //  blockIdx.z, gx, gy, C[x*16],  Cb[y+x*4]);
                              C[x*16] -= Cb[y+x*4];

                           //printf("block %d  i=%2d j = %2d  val = %f %f\n",
                           //        blockIdx.z, gx, gy, C[x*16],  Cb[y+x*4]);
                       }
                }

                C += ldc*16;
        }

      }
#endif

}

// every multiprocessor handles one BCSR-block
__global__ void 
zbcsr_gemm_kernel32( 
                  int m,
                  int n,
                  int kblocks,   
                  double **Avals, 
                  double **Bval,
                  double **Cval)
{
#if defined(PRECISION_d)
    const  int tx = threadIdx.x;
    const  int ty = threadIdx.y;
  
    const int idt = ty * 64 + tx;

    const int tx2 = idt%16;
    const int ty2 = idt/16;

    double xxB[4];
    double *B;

    int trackA = __mul24( ty2, lda) + tx2 ;
    double *Aval = Avals[blockIdx.z];

    __shared__ double Abs[64][65];
    __shared__ double  Bb[16][65];


    for(int j=ty2; j<64; j+=16){
        for(int y=tx2; y<64; y+=16){
           Abs[y][j] = fetch_x_A(trackA + y-tx2) ;
            }
        trackA += __mul24( 16, m);
    }

    for(int k=0; k<kblocks; k++){
        B = Bval[k];
        int trackB = tx2+ __mul24( ty2 * 16, ldb );

        // Prefetch part of B
          #pragma unroll
          for(int y=0; y<4; y++){
                 Bb[tx2][ty2*4+y] = fetch_x_B( trackB + y * ldb) ;
          }
        __syncthreads();    // this is necessary!!!

        double Axs[4];
        double Bxp[4];
        double Cb[16] = {0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0};

        int k1;
        for(k1=0; k1<m-16; k1+=16)
        {
                trackB += 16;

                #pragma unroll
                for( int y=0; y<4; y++)
                        xxB[y] = fetch_x_B( trackB + y*ldb);
                #pragma unroll
                for( int j1=0;j1<16;j1++)
                {
                        #pragma unroll
                        for( int y=0; y<4; y++){
                                Axs[y] =  Abs[tx2+y*16][j1+k1] ;
                        }

                        #pragma unroll
                        for( int y=0; y<4; y++){
                                Bxp[y]= Bb[j1][ty2+y*16];
                        }

                        #pragma unroll
                        for( int x=0; x<4; x++)
                        {
                                #pragma unroll
                                for( int y=0; y<4; y++)
                                {
                                        Cb[x*4+y]  += Axs[x]*Bxp[y];
                                }
                        }

                }
                #pragma unroll
                for(int y=0; y<4; y++)
                        Bb[tx2][ty2*4 + y] = xxB[y];

                __syncthreads();     // this is necessary!!!
        }
        // Prepare where to write the result
        double *C = Cval[blockIdx.z * kblocks + k];
        C += tx2 + __mul24 (ty2 ,ldc);

        #pragma unroll
        for(int j1=0;j1<16;j1++)
        {

                #pragma unroll
                for( int y=0; y<4; y++)
                        Axs[y] =  Abs[tx2 + y*16][j1+k1] ;

                #pragma unroll
                for( int y=0; y<4; y++)
                        Bxp[y]= Bb[j1][ty2 + y*16];

                #pragma unroll
                for( int x=0; x<4; x++)
                {
                        #pragma unroll
                        for( int y=0;y<4; y++)
                        {
                                Cb[x*4 + y]  += Axs[x]*Bxp[y];
                        }
                }
        }   
        int gy = ty2;
        #pragma unroll
        for( int y=0;y<4;y++, gy+=16)
        {
                int gx = tx2;
        #pragma unroll
                for(int x=0;x<4;x++, gx+=16)
                {
                        if (gx < m && gy < n){
                              C[x*16] -= Cb[y+x*4];
                       }
                }
                C += ldc*16;
        }
      }
#endif
}

// every multiprocessor handles one BCSR-block
__global__ void 
zbcsr_gemm_kernel64( 
                  int m,
                  int n,
                  int kblocks,   
                  double **Avals, 
                  double **Bval,
                  double **Cval)
{
#if defined(PRECISION_d)
    const  int tx = threadIdx.x;
    const  int ty = threadIdx.y;
  
    const int idt = ty * 64 + tx;

    const int tx2 = idt%16;
    const int ty2 = idt/16;

    double xxB[4];

    double *B;

    int trackA = __mul24( ty2, lda) + tx2 ;
    double *Aval = Avals[blockIdx.z];

    __shared__ double Abs[64][65];
    __shared__ double  Bb[16][65];


    for(int j=ty2; j<64; j+=16){
        for(int y=tx2; y<64; y+=16){
           Abs[y][j] = fetch_x_A(trackA + y-tx2) ;
            }
        trackA += __mul24( 16, m);
    }


    for(int k=0; k<kblocks; k++){

        B = Bval[k];
        int trackB = tx2+ __mul24( ty2 * 4, ldb );

        // Prefetch part of B
          #pragma unroll
          for(int y=0; y<4; y++){
                 Bb[tx2][ty2*4+y] = fetch_x_B( trackB + y * ldb) ;
          }

        __syncthreads();    // this is necessary!!!

        double Axs[4];
        double Bxp[4];

        double Cb[16] = {0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0};

        int k1;
        for(k1=0; k1<m-16; k1+=16)
        {
                trackB += 16;

                #pragma unroll
                for( int y=0; y<4; y++)
                        xxB[y] = fetch_x_B( trackB + y*ldb);

                #pragma unroll
                for( int j1=0;j1<16;j1++)
                {
                        #pragma unroll
                        for( int y=0; y<4; y++){
                                Axs[y] =  Abs[tx2+y*16][j1+k1] ;
                        }

                        #pragma unroll
                        for( int y=0; y<4; y++){
                                Bxp[y]= Bb[j1][ty2+y*16];
                        }

                        #pragma unroll
                        for( int x=0; x<4; x++)
                        {
                                #pragma unroll
                                for( int y=0; y<4; y++)
                                {
                                        Cb[x*4+y]  += Axs[x]*Bxp[y];
                                }
                        }

                }

                __syncthreads();
                #pragma unroll
                for(int y=0; y<4; y++)
                        Bb[tx2][ty2*4 + y] = xxB[y];

                __syncthreads();     // this is necessary!!!

        }
        // Prepare where to write the result
        double *C = Cval[blockIdx.z * kblocks + k];
        C += tx2 + __mul24 (ty2 ,ldc);

        #pragma unroll
        for(int j1=0;j1<16;j1++)
        {

                #pragma unroll
                for( int y=0; y<4; y++)
                        Axs[y] =  Abs[tx2 + y*16][j1+k1] ;

                #pragma unroll
                for( int y=0; y<4; y++)
                        Bxp[y]= Bb[j1][ty2 + y*16];

                #pragma unroll
                for( int x=0; x<4; x++)
                {
                        #pragma unroll
                        for( int y=0;y<4; y++)
                        {
                                Cb[x*4 + y]  += Axs[x]*Bxp[y];
                        }
                }
        }   

        int gy = ty2;
        #pragma unroll
        for( int y=0;y<4;y++, gy+=16)
        {
                int gx = tx2;
        #pragma unroll
                for(int x=0;x<4;x++, gx+=16)
                {
                        if (gx < m && gy < n){
                              C[x*16] -= Cb[y+x*4];
                       }
                }

                C += ldc*16;
        }

      }
#endif


}





/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======
    
    For a Block-CSR ILU factorization, this routine updates all blocks in
    the trailing matrix.
    
    Arguments
    =========

    magma_int_t r_blocks            number of blocks
    magma_int_t size_b              blocksize in BCSR
    magma_int_t *ipiv               array containing pivots
    double *x           input/output vector x

    =====================================================================    */

extern "C" magma_int_t
magma_zbcsrluegemm( magma_int_t size_b, 
                    magma_int_t num_block_rows,
                    magma_int_t kblocks,
                    magmaDoubleComplex **dA,  
                    magmaDoubleComplex **dB,  
                    magmaDoubleComplex **dC ){

#if defined(PRECISION_d)
    int i, j, k;
    dim3 threads( 64, 4 );

    dim3 grid(1, 1, num_block_rows);
    zbcsr_gemm_kernel64<<< grid, threads, 0, magma_stream >>>( 
                  size_b, size_b, kblocks, dA, dB, dC );

#endif


    return MAGMA_SUCCESS;
}



