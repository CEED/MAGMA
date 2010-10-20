/*
    -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010
*/

__global__ void stranspose3_32( float *B, int ldb, float *A, int lda,
                                int m, int m32, int n, int n32)
{
 	__shared__ float a[32][33];

        int inx = threadIdx.x;
        int iny = threadIdx.y;
        int ibx = blockIdx.x*32;
        int iby = blockIdx.y*32;

        A += ibx + inx + __mul24( iby + iny, lda );

        a[iny+0][inx] = A[0*lda];
        a[iny+8][inx] = A[8*lda];
        a[iny+16][inx] = A[16*lda];
        a[iny+24][inx] = A[24*lda];

        __syncthreads();

	B += iby + inx + __mul24( ibx + iny, ldb );

	if (iby + inx < n){
          if (ibx + iny <m){
             B[0*ldb] = a[inx][iny+0];
             if (ibx + iny + 8 <m){
                B[8*ldb] = a[inx][iny+8];
                if (ibx + iny +16 <m){
                   B[16*ldb] = a[inx][iny+16];
                   if (ibx + iny +24 <m)
                      B[24*ldb] = a[inx][iny+24];
                }
             }
          }
        }
}



__global__ void stranspose2_32( float *B, int ldb, float *A, int lda, 
                                int m, int m32, int n, int n32)
{	
	__shared__ float a[32][33];
	
	int inx = threadIdx.x;
	int iny = threadIdx.y;
	int ibx = blockIdx.x*32;
	int iby = blockIdx.y*32;
	
	int dx, dy;
	if (ibx+32<m)
	   dx = 0;
	else
	   dx = m32;

        if (iby+32<n)
	   dy = 0;
        else
           dy = n32;

	A += ibx + inx -dx + __mul24( iby + iny - dy, lda );
	B += iby + inx -dy + __mul24( ibx + iny - dx, ldb );
	
	a[iny+0][inx] = A[0*lda];
	a[iny+8][inx] = A[8*lda];
	a[iny+16][inx] = A[16*lda];
	a[iny+24][inx] = A[24*lda];
	
	__syncthreads();
	
	B[0*ldb] = a[inx][iny+0];
	B[8*ldb] = a[inx][iny+8];
	B[16*ldb] = a[inx][iny+16];
	B[24*ldb] = a[inx][iny+24];
} 

//
//	m, n - dimensions in the source (input) matrix
//             This version transposes for general m, n .
//             Note that ldi >= m and ldo >= n.
//
extern "C" void 
magmablas_stranspose2(float *odata, int ldo, 
                      float *idata, int ldi, 
                      int m, int n )
{
	dim3 threads( 32, 8, 1 );
	dim3 grid( (m+31)/32, (n+31)/32, 1 );
	stranspose3_32<<< grid, threads >>>( odata, ldo, idata, ldi, 
                  //                           m, m%32, n, n%32);
                 m, (32-m%32)%32, n, (32-n%32)%32);
}

