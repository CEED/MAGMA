/*
    -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010
*/

__global__ void stranspose_32( float *B, int ldb, float *A, int lda )
{	
	__shared__ float a[32][33];
	
	int inx = threadIdx.x;
	int iny = threadIdx.y;
	int ibx = blockIdx.x*32;
	int iby = blockIdx.y*32;
	
	A += ibx + inx + __mul24( iby + iny, lda );
	B += iby + inx + __mul24( ibx + iny, ldb );
	
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
//	m, n - dimensions in the source matrix
//             This version works when m and n are divisible by 32.
//
extern "C" void 
magmablas_stranspose(float *odata, int ldo, 
                     float *idata, int ldi, 
                     int m, int n )
{
	//assert( (m%32) == 0 && (n%32) == 0, "misaligned transpose" );
	dim3 threads( 32, 8, 1 );
	dim3 grid( m/32, n/32, 1 );
	stranspose_32<<< grid, threads >>>( odata, ldo, idata, ldi );
}
