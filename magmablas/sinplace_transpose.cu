/*
    -- MAGMA (version 0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
	Univ. of Colorado, Denver
       November 2009
*/

__global__ void sinplace_T_even( float *matrix, int lda, int half )
{	
	__shared__ float a[32][33], b[32][33];
	
	int inx = threadIdx.x;
	int iny = threadIdx.y;

	bool bottom = ( blockIdx.x > blockIdx.y );
	int ibx = bottom ? (blockIdx.x - 1) : (blockIdx.y + half);
	int iby = bottom ? blockIdx.y       : (blockIdx.x + half);

	ibx *= 32;
	iby *= 32;

	float *A = matrix + ibx + inx + __mul24( iby + iny, lda );
	a[iny][inx] = A[0];
	a[iny+16][inx] = A[16*lda];
	
	if( ibx == iby )
	{
		__syncthreads();
		A[0] = a[inx][iny];
		A[16*lda] = a[inx][iny+16];
	}
	else
	{
		float *B = matrix + iby + inx + __mul24( ibx + iny, lda );

		b[iny][inx] = B[0];
		b[iny+16][inx] = B[16*lda];
		__syncthreads();
		A[0] = b[inx][iny];
		A[16*lda] = b[inx][iny+16];
		B[0] = a[inx][iny];
		B[16*lda] = a[inx][iny+16];
	}
} 

__global__ void sinplace_T_odd( float *matrix, int lda, int half )
{	
	__shared__ float a[32][33], b[32][33];
	
	int inx = threadIdx.x;
	int iny = threadIdx.y;

	bool bottom = ( blockIdx.x >= blockIdx.y );
	int ibx = bottom ? blockIdx.x  : (blockIdx.y + half - 1);
	int iby = bottom ? blockIdx.y  : (blockIdx.x + half);

	ibx *= 32;
	iby *= 32;

	float *A = matrix + ibx + inx + __mul24( iby + iny, lda );
	a[iny][inx] = A[0];
	a[iny+16][inx] = A[16*lda];
	
	if( ibx == iby )
	{
		__syncthreads();
		A[0] = a[inx][iny];
		A[16*lda] = a[inx][iny+16];
	}
	else
	{
		float *B = matrix + iby + inx + __mul24( ibx + iny, lda );

		b[iny][inx] = B[0];
		b[iny+16][inx] = B[16*lda];
		__syncthreads();
		A[0] = b[inx][iny];
		A[16*lda] = b[inx][iny+16];
		B[0] = a[inx][iny];
		B[16*lda] = a[inx][iny+16];
	}
} 

extern "C" void 
magmablas_sinplace_transpose( float *A, int lda, int n )
{
	dim3 threads( 32, 16 );
	int in = n / 32;
	if( in&1 )
	{
		dim3 grid( in, in/2+1 );
		sinplace_T_odd<<< grid, threads >>>( A, lda, in/2+1 );
	}
	else
	{
		dim3 grid( in+1, in/2 );
		sinplace_T_even<<< grid, threads >>>( A, lda, in/2 );
	}
}
