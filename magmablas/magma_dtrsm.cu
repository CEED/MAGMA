/*
    -- MAGMA (version 0.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       October 2009
*/

#include "cublas.h"
#include "cuda.h"

#define BLOCK_SIZE 32 

__global__ void
dgemv32_kernel(int n, double alpha, double* A, int lda, double *x, double *y)
{
/*  -- MAGMA (version 0.1) --

    Purpose
    =======

    This routine computes y = alpha A x where A is double precision
    array of dimension (N, 32).
*/

    int ind = blockIdx.x*32 + threadIdx.x;

    A += ind;
    x += threadIdx.x;

    double res = 0.f;

    __shared__ double buff[32];
    buff[threadIdx.x]  = x[0];

    __syncthreads();
    #pragma unroll
    for(int j=0; j < 32; j++){
       res+=A[0]*buff[j];
       A+=lda;
    }

    if (ind<n)
      y[ind] = alpha*res;
}

__global__ void 
dgemvT32_kernel(int m, double alpha, double* A, int lda, double *x, double *y)
{
/*  -- MAGMA (version 0.1) --

    Purpose
    =======

    This routine computes y = alpha A^T x where A is double precision
    array of dimension (32, M).
*/
 
    const int inx = threadIdx.x;
    const int iny = threadIdx.y;

    int ind  = iny + __mul24(blockIdx.x,32);
    ind = inx + __mul24(ind,lda);
    int ind2 = inx + __mul24(iny,32);

    A += ind;
    x += inx;

    double res = 0.f;

    __shared__ double buff[64];
    __shared__ double la[32][33];

    buff[ind2]  = x[0];
    #pragma unroll
    for(int j=0; j<16; j++)
      la[iny+__mul24(2, j)][inx] = A[j*__mul24(2,lda)];

    __syncthreads();

    #pragma unroll
    for(int j=0; j < 16; j++)
      res += la[inx][j+iny*16]*buff[j+iny*16];

    ind = inx + __mul24(blockIdx.x,32);
    la[inx][iny]= res;

    __syncthreads();

    if (ind<m){
      res = la[inx][0] + la[inx][1];
      y[ind] = alpha*res;
    }
}

void magmablas_dgemv32(char tran, int n, double alpha, double *A, int lda,
		       double *x, double *y)
{
/*  -- MAGMA (version 0.1) --

    Purpose
    =======

    This routine computes
       y = alpha A^T x 	      	 for tran = 'T' / 't' or
      	y = alpha A x
    where A is double precision array of dimension (32, N) for
    tran = 'T' / 't', or of dimension (N, 32) otherwise.
*/

    int blocks;
    if (n % 32==0)
      blocks = n/32;
    else
      blocks = n/32 + 1;
    dim3 grid(blocks, 1, 1);

    if (tran == 'T' || tran == 't'){
      dim3 threads(32, 2, 1);
      dgemvT32_kernel<<<grid, threads>>>(n, alpha, A, lda, x, y);
    }
    else
    {
      dim3 threads(32, 1, 1);
      dgemv32_kernel<<<grid, threads>>>(n, alpha, A, lda, x, y);
    }
}

__global__ void
inplace_dgemm_kernel_T(int M, double alpha, double *A, int lda, double *B, int ldb)
{
        int i;
        double myvalue1=0, myvalue2= 0 ;
        double med;
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        const int bx = blockIdx.x;
        __shared__ double As[BLOCK_SIZE][BLOCK_SIZE+1];
        __shared__ double Bs[BLOCK_SIZE/2][BLOCK_SIZE];

        A+= bx*32 + __mul24(lda,ty) + tx ;
        B+=      __mul24(ldb,tx) + ty ;

        As[tx][ty]=A[0];
        As[tx][ty+16]= A[16*lda];
        Bs[ty][tx]= B[0];
        __syncthreads();

        med  = As[tx][0];
        double py1 = Bs[ty][0] ;
        #pragma unroll
        for (i=0; i<31; i++){
                myvalue1 +=  med*py1;
                py1 = Bs[ty][i+1] ;
                med = As[tx][i+1];
        }
        myvalue1 += med*py1;

        Bs[ty][tx]= B[16];
        __syncthreads();

        med  = As[tx][0];
        double py2 = Bs[ty][0] ;
        #pragma unroll
        for (i=0; i<31; i++){
                myvalue2 +=  med*py2;
                py2 = Bs[ty][i+1] ;
                med = As[tx][i+1];
        }
        myvalue2 += med*py2;
        
		A[0] = alpha*myvalue1 ;
        A[lda*16] = alpha*myvalue2;
}

__global__ void
inplace_dgemm_kernel_N(int M, double alpha, double *A, int lda, double *B, int ldb)
{
        int i;
        double myvalue1=0, myvalue2= 0 ;
        double med;
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        const int bx = blockIdx.x;
        __shared__ double As[BLOCK_SIZE][BLOCK_SIZE+1];
        __shared__ double Bs[BLOCK_SIZE/2][BLOCK_SIZE];

        A+= bx*32 + __mul24(lda,ty) + tx ;
        B+=      __mul24(ldb,ty) + tx ;

        As[tx][ ty]=A[0];
        As[tx][ty+16]= A[16*lda];
        Bs[ty][tx]= B[0];

        __syncthreads();
        med  = As[tx][0];
        double py1 = Bs[ty][0] ;

        #pragma unroll
        for (i=0; i<31; i++){
                myvalue1 +=  med*py1;
                py1 = Bs[ty][i+1] ;
                med  = As[tx][i+1];
        }
        myvalue1 +=  med*py1;

        Bs[ty][tx]= B[16*ldb];
        __syncthreads();

        med  = As[tx][0];
        double py2 = Bs[ty][0] ;
        #pragma unroll
        for (i=0; i<31; i++){
                myvalue2 +=  med*py2;
                py2 = Bs[ty][i+1] ;
                med  = As[tx][i+1];
        }
        myvalue2 +=  med*py2;

        A[0] = alpha*myvalue1 ;
        A[lda*16] = alpha*myvalue2;
}

__global__ void
diag_dtrtri_kernel (char uplo, char diag, double *A, double *d_dinvA, int lda)
{
	int i,j;
	double Ystx=0;
	double *Bw=NULL, *x=NULL, *y=NULL, *Aoff=NULL;
	double *my_d_dinvA;
	int switcher=0;

	// Thread index
	int tx = threadIdx.x;
	int txw;

	// Block index
	int bx = blockIdx.x;
		
	Aoff = A+bx*lda*BLOCK_SIZE+bx*BLOCK_SIZE;
	my_d_dinvA = d_dinvA+bx*BLOCK_SIZE*BLOCK_SIZE;

	__shared__ double Bs[BLOCK_SIZE*BLOCK_SIZE];
	__shared__ double workspace[BLOCK_SIZE]; // workspace used to store the current working column

	// load A
	#pragma unroll
	for (i=0; i<BLOCK_SIZE; i++)
		Bs[i*BLOCK_SIZE+tx] = *(Aoff+i*lda+tx);	// read in the whole square block of my A
												// not the upper or lower diagonal

	// Synchronize to make sure the matrices are loaded
	__syncthreads();

	Bs[tx*BLOCK_SIZE+tx] = ((diag=='u' || diag=='U')?1:(1/Bs[tx*BLOCK_SIZE+tx]));	// solve the diagonals

	if (uplo == 'l' || uplo == 'L')
	{
		/*
		 * the lower case
		 */
		if (tx < BLOCK_SIZE-1)
			Bs[(BLOCK_SIZE-1)*BLOCK_SIZE+tx] = 0;	//zero out the last column, except the diagonal element

		for (i=BLOCK_SIZE-2; i>=0; i--)
		{
			Ystx = 0;
			switcher = (tx>i);
			
			//dtrmv
			Bw = Bs+(i+1)*BLOCK_SIZE+i+1;
			workspace[tx] = *(Bs+i*BLOCK_SIZE+tx);
			x = workspace+i+1;
			y = Bs+i*BLOCK_SIZE;

			txw = (tx-i-1);

			#pragma unroll
			for (j=0; j<txw+1; j++)
				Ystx += (double)switcher*(*(Bw+j*BLOCK_SIZE+txw)*x[j]);

			//sscal
			switcher = (tx != i); 
			//if (tx !=i ) y[tx]=switcher*Ystx*(-Bs[i*BLOCK_SIZE+i]);
			y[tx] = (double)switcher*Ystx*(-Bs[i*BLOCK_SIZE+i])+(double)(!switcher)*y[tx];

			__syncthreads();
		}

	}
	else
	{
		 /* the upper case */
		for (i=0; i<BLOCK_SIZE; i++)
		{
			Ystx = 0;
			switcher = (double)(tx<i);
			
			//dtrmv
			workspace[tx] = *(Bs+i*BLOCK_SIZE+tx);
			y = Bs+i*BLOCK_SIZE;

			#pragma unroll
			for (j=tx; j<i; j++)
				Ystx += switcher*(*(Bs+j*BLOCK_SIZE+tx)*workspace[j]);

			//sscal
			switcher = (tx != i); // if (tx !=i ) y[tx]=switcher*Ystx*(-Bs[i*BLOCK_SIZE+i]);
			y[tx] = switcher*Ystx*(-Bs[i*BLOCK_SIZE+i])+!switcher*y[tx];

			__syncthreads();
		}


	}
		
	// write back A
	#pragma unroll
	for (i=0; i<BLOCK_SIZE; i++)
		*(my_d_dinvA+i*BLOCK_SIZE+tx) = Bs[i*BLOCK_SIZE+tx];
}

extern "C" void
inplace_dgemm (char tran, int M, double alpha, double *A, int lda, double *B, int ldb)
{
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE/2);

	if (tran == 'n' || tran == 'N')
		inplace_dgemm_kernel_N<<<M/BLOCK_SIZE,dimBlock>>>(M, alpha, A, lda, B, ldb); 
	else
		inplace_dgemm_kernel_T<<<M/BLOCK_SIZE,dimBlock>>>(M, alpha, A, lda, B, ldb); 
}

/*
 * magmablas_dtrsmx
 * the expert interface
 */

void magmablas_dtrsmx (char side, char uplo, char tran, char diag, int M, int N, double alpha, double* A, int lda, double* b, int ldb, double * d_dinvA)
{
/*  -- MAGMA (version 0.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       October 2009

	   Purpose
	   =======
	   
	   DTRSM  solves one of the matrix equations on GPU
	   
	      op( A )*X = alpha*B,   or   X*op( A ) = alpha*B,
	   
	   where alpha is a scalar, X and B are m by n matrices, A is a unit, or
	   non-unit,  upper or lower triangular matrix  and  op( A )  is one  of
	   
	      op( A ) = A   or   op( A ) = A'.
	   
	   The matrix X is overwritten on B.
	   
	   When M or N is not a multiple of blocking size, which is 32 for now, cublasDtrsm will
	   be called instead. There soon will not be this limitation both for arbitrary problem 
	   size and blocking size.
	   
	   
	   Arguments
	   ==========
	   
	   side   - CHARACTER*1.
	            On entry, side specifies whether op( A ) appears on the left
	            or right of X as follows:
	   
	               side = 'L' or 'l'   op( A )*X = alpha*B.
	   
	               side = 'R' or 'r'   X*op( A ) = alpha*B.
	   
	            Unchanged on exit.
	   
	   uplo   - CHARACTER*1.
	            On entry, uplo specifies whether the matrix A is an upper or
	            lower triangular matrix as follows:
	   
	               uplo = 'U' or 'u'   A is an upper triangular matrix.
	   
	               uplo = 'L' or 'l'   A is a lower triangular matrix.
	   
	            Unchanged on exit.
	   
	   tran - CHARACTER*1.
	            On entry, tran specifies the form of op( A ) to be used in
	            the matrix multiplication as follows:
	   
	               tran = 'N' or 'n'   op( A ) = A.
	   
	               tran = 'T' or 't'   op( A ) = A'.
	   
	               tran = 'C' or 'c'   op( A ) = A'.
	   
	            Unchanged on exit.
	   
	   diag   - CHARACTER*1.
	            On entry, diag specifies whether or not A is unit triangular
	            as follows:
	   
	               diag = 'U' or 'u'   A is assumed to be unit triangular.
	   
	               diag = 'N' or 'n'   A is not assumed to be unit
	                                   triangular.
	   
	            Unchanged on exit.
	   
	   m      - INTEGER.
	            On entry, m specifies the number of rows of B. m must be at
	            least zero.
	            Unchanged on exit.
	   
	    n      - INTEGER.
	             On entry, n specifies the number of columns of B.  n must be
	             at least zero.
	             Unchanged on exit.
	   
	    alpha  - DOUBLE PRECISION.
	             On entry,  alpha specifies the scalar  alpha. When  alpha is
	             zero then  A is not referenced and  B need not be set before
	             entry.
	             Unchanged on exit.
	   
	    A      - DOUBLE PRECISION             array of DIMENSION ( lda, k ), where k is m
	             when  side = 'L' or 'l'  and is  n  when  side = 'R' or 'r'.
	             Before entry  with  uplo = 'U' or 'u',  the  leading  k by k
	             upper triangular part of the array  A must contain the upper
	             triangular matrix  and the dtrictly lower triangular part of
	             A is not referenced.
	             Before entry  with  uplo = 'L' or 'l',  the  leading  k by k
	             lower triangular part of the array  A must contain the lower
	             triangular matrix  and the dtrictly upper triangular part of
	             A is not referenced.
	             Note that when  diag = 'U' or 'u',  the diagonal elements of
	             A  are not referenced either,  but are assumed to be  unity.
	             Unchanged on exit.
	   
	    lda    - INTEGER.
	             On entry, lda specifies the first dimension of A as declared
	             in the calling (sub) program.  When  side = 'L' or 'l'  then
	             lda  must be at least  max( 1, m ),  when  side = 'R' or 'r'
	             then lda must be at least max( 1, n ).
	             Unchanged on exit.
	   
	    b      - DOUBLE PRECISION array of DIMENSION ( ldb, n ).
	             Before entry,  the leading  m by n part of the array  B must
	             contain  the  right-hand  side  matrix  B,  and  on exit  is
	             overwritten by the solution matrix  X.
	   
	    ldb    - INTEGER.
	             On entry, ldb specifies the first dimension of B as declared
	             in  the  calling  (sub)  program.   ldb  must  be  at  least
	             max( 1, m ).
	             Unchanged on exit.

		d_dinvA -DOUBLE PRECISION array of DIMENSION (BLOCKSIZE, M) when side='L', 
				 (BLOCKSIZE, N) when side='R'. On exit this space is filled
			     with the inverse of blocks on the diagonal, each inverse is
				 of size BLOCKSIZE x BLOCKSIZE, and the leading dimension of
				 d_dinvA is BLOCKSIZE;
	   
	   
	    Level 3 Blas routine.
		*
    ===================================================================== */

	int i, nblocks;

	/* 
	 * call cublasDtrsm when size of the problem is not a multiple of blocksize which is 32
	 * subject to change soon
	 */
	if ((M%BLOCK_SIZE) != 0 || (N%BLOCK_SIZE) != 0 )
	{
		cublasDtrsm (side, uplo, tran, diag, M, N, alpha, A, lda, b, ldb);
		return;
	}

	if (d_dinvA == NULL)
		return;

	if (side == 'l' || side == 'L')
	{
		/* inverse the diagonals
		 * Allocate device memory for the inversed diagonal blocks, size=m*BLOCK_SIZE 
		 */
		nblocks = M/BLOCK_SIZE;
		diag_dtrtri_kernel<<<nblocks, BLOCK_SIZE>>>(uplo, diag, A, d_dinvA, lda);

		if (tran == 'N' || tran == 'n')
		/* the non-transpose case */
		{
			if (uplo == 'L' || uplo == 'l')
			{
			/* the lower case */
				
				/* handle the first block seperately with alpha */
				if (N == 1)
					magmablas_dgemv32 ('N', BLOCK_SIZE, alpha, d_dinvA, BLOCK_SIZE, b, b);
				else
					cublasDgemm ('N', 'N', BLOCK_SIZE, N, BLOCK_SIZE, alpha, d_dinvA, BLOCK_SIZE, b, ldb, 0, b, ldb);  

				if (BLOCK_SIZE>=M)
					return;

				cublasDgemm ('N', 'N', M-BLOCK_SIZE, N, BLOCK_SIZE, -1.0, A+BLOCK_SIZE, lda, b, ldb, alpha, b+BLOCK_SIZE, ldb);

				/* the rest blocks */
				for (i=BLOCK_SIZE; i<M; i+=BLOCK_SIZE)
				{
					if (N == 1)
						magmablas_dgemv32 ('N', BLOCK_SIZE, 1.0, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE, b+i, b+i);
					else
						cublasDgemm ('N', 'N', BLOCK_SIZE, N, BLOCK_SIZE, 1.0, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE, b+i, ldb, 0, b+i, ldb);  

					if (i+BLOCK_SIZE>=M)
						break;

					cublasDgemm ('N', 'N', M-i-BLOCK_SIZE, N, BLOCK_SIZE, -1.0, A+i*lda+i+BLOCK_SIZE, lda, b+i, ldb, 1.0, b+i+BLOCK_SIZE, ldb);
				}
			}
			else
			{
			/* the upper case */

				/* handle the first block seperately with alpha */
				i = M-BLOCK_SIZE;
				if (N == 1)
					magmablas_dgemv32 ('N', BLOCK_SIZE, alpha, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE, b+i, b+i);
				else
					cublasDgemm ('N', 'N', BLOCK_SIZE, N, BLOCK_SIZE, alpha, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE, b+i, ldb, 0.0, b+i, ldb); 
					
				if (i-BLOCK_SIZE<0)
					return;

				cublasDgemm ('N', 'N', i, N, BLOCK_SIZE, -1.0, A+i*lda, lda, b+i, ldb, alpha, b, ldb);

				/* the rest blocks */
				for (i=M-2*BLOCK_SIZE; i>=0; i-=BLOCK_SIZE)
				{
					if (N == 1)
						magmablas_dgemv32 ('N', BLOCK_SIZE, 1.0, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE, b+i, b+i);
					else
						cublasDgemm ('N', 'N', BLOCK_SIZE, N, BLOCK_SIZE, 1.0, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE, b+i, ldb, 0.0, b+i, ldb); 

					if (i-BLOCK_SIZE<0)
						break;

					cublasDgemm ('N', 'N', i, N, BLOCK_SIZE, -1.0, A+i*lda, lda, b+i, ldb, 1.0, b, ldb);
				}
			}
		}
		else
		/* the transpose case */
		{
			if (uplo == 'L' || uplo == 'l')
			{
			/* the lower case */
				
				/* handle the first block seperately with alpha */
				i=M-BLOCK_SIZE; 
				if (N == 1)
					magmablas_dgemv32 ('T', BLOCK_SIZE, alpha, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE, b+i, b+i);
				else
					cublasDgemm ('T', 'N', BLOCK_SIZE, N, BLOCK_SIZE, alpha, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE, b+i, ldb, 0, b+i, ldb);  

				if (i-BLOCK_SIZE<0)
					return;

				cublasDgemm ('T', 'N', i, N, BLOCK_SIZE, -1.0, A+i, lda, b+i, ldb, alpha, b, ldb);

				/* the rest blocks */
				for (i=M-2*BLOCK_SIZE; i>=0; i-=BLOCK_SIZE)
				{
					if (N == 1)
						magmablas_dgemv32 ('T', BLOCK_SIZE, 1.0, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE, b+i, b+i);
					else
						cublasDgemm ('T', 'N', BLOCK_SIZE, N, BLOCK_SIZE, 1.0, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE, b+i, ldb, 0, b+i, ldb);  

					if (i-BLOCK_SIZE<0)
						break;

					cublasDgemm ('T', 'N', i, N, BLOCK_SIZE, -1.0, A+i, lda, b+i, ldb, 1.0, b, ldb);
				}
			}
			else
			{
			/* the upper case */
					
				/* handle the first block seperately with alpha */
				if (N == 1)
					magmablas_dgemv32 ('T', BLOCK_SIZE, alpha, d_dinvA, BLOCK_SIZE, b, b);
				else
					cublasDgemm ('T', 'N', BLOCK_SIZE, N, BLOCK_SIZE, alpha, d_dinvA, BLOCK_SIZE, b, ldb, 0, b, ldb);  

				if (BLOCK_SIZE>=M)
					return;

				cublasDgemm ('T', 'N', M-BLOCK_SIZE, N, BLOCK_SIZE, -1.0, A+(BLOCK_SIZE)*lda, lda, b, ldb, alpha, b+BLOCK_SIZE, ldb);

				/* the rest blocks */
				for (i=BLOCK_SIZE; i<M; i+=BLOCK_SIZE)
				{
					if (N == 1)
						magmablas_dgemv32 ('T', BLOCK_SIZE, 1.0, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE, b+i, b+i);
					else
						cublasDgemm ('T', 'N', BLOCK_SIZE, N, BLOCK_SIZE, 1.0, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE, b+i, ldb, 0, b+i, ldb);  
					
					if (i+BLOCK_SIZE>=M)
						break;

					cublasDgemm ('T', 'N', M-i-BLOCK_SIZE, N, BLOCK_SIZE, -1.0, A+(i+BLOCK_SIZE)*lda+i, lda, b+i, ldb, 1.0, b+i+BLOCK_SIZE, ldb);
				}
			}
		}
	}
	else
	{	// side=R

		/* inverse the diagonals
		 * Allocate device memory for the inversed diagonal blocks, size=N*BLOCK_SIZE 
		 */
		nblocks = N/BLOCK_SIZE;
		diag_dtrtri_kernel<<<nblocks, BLOCK_SIZE>>>(uplo, diag, A, d_dinvA, lda);
		
		if (tran == 'N' || tran == 'n')
		/* the non-transpose case */
		{
			if (uplo == 'L' || uplo == 'l')
			{
			/* the lower case */
				
				/* handle the first block seperately with alpha */
				i=N-BLOCK_SIZE;
				inplace_dgemm ('N', M, alpha, b+ldb*i, ldb, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE);

				if (i-BLOCK_SIZE<0)
					return;

				cublasDgemm ('N', 'N', M, i, BLOCK_SIZE, -1.0, b+ldb*i, ldb, A+i, lda, alpha, b, ldb);

				/* the rest blocks */
				for (i=N-2*BLOCK_SIZE; i>=0; i-=BLOCK_SIZE)
				{
					inplace_dgemm ('N', M, 1.0, b+ldb*i, ldb, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE);
					
					if (i-BLOCK_SIZE<0)
						break;

					cublasDgemm ('N', 'N', M, i, BLOCK_SIZE, -1.0, b+ldb*i, ldb, A+i, lda, 1.0, b, ldb);
				}
			}
			else
			{
			/* the upper case */
				
				/* handle the first block seperately with alpha */
				inplace_dgemm ('N', M, alpha, b, ldb, d_dinvA, BLOCK_SIZE);

				if (BLOCK_SIZE>=N)
					return;

				cublasDgemm ('N', 'N', M, N-BLOCK_SIZE, BLOCK_SIZE, -1.0, b, ldb, A+(BLOCK_SIZE)*lda, lda, alpha, b+(BLOCK_SIZE)*ldb, ldb);
				
				
				/* the rest blocks */
				for (i=BLOCK_SIZE; i<N; i+=BLOCK_SIZE)
				{
					inplace_dgemm ('N', M, 1.0, b+ldb*i, ldb, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE);

					if (i+BLOCK_SIZE>=N)
						break;

					cublasDgemm ('N', 'N', M, N-i-BLOCK_SIZE, BLOCK_SIZE, -1.0, b+i*ldb, ldb, A+(i+BLOCK_SIZE)*lda+i, lda, 1.0, b+(i+BLOCK_SIZE)*ldb, ldb);
				}
			}
		}
		else
		/* the transpose case */
		{
			if (uplo == 'L' || uplo == 'l')
			{
			/* the lower case */
				
				/* handle the first block seperately with alpha */
				inplace_dgemm ('T', M, alpha, b, ldb, d_dinvA, BLOCK_SIZE);

				if (BLOCK_SIZE>=N)
					return;

				cublasDgemm ('N', 'T', M, N-BLOCK_SIZE, BLOCK_SIZE, -1.0, b, ldb, A+BLOCK_SIZE, lda, alpha, b+(BLOCK_SIZE)*ldb, ldb);

				/* the rest blocks */
				for (i=BLOCK_SIZE; i<N; i+=BLOCK_SIZE)
				{
					inplace_dgemm ('T', M, 1.0, b+ldb*i, ldb, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE);

					if (i+BLOCK_SIZE>=N)
						break;

					cublasDgemm ('N', 'T', M, N-i-BLOCK_SIZE, BLOCK_SIZE, -1.0, b+ldb*i, ldb, A+i*lda+BLOCK_SIZE+i, lda, 1.0, b+(i+BLOCK_SIZE)*ldb, ldb);
				}
			}
			else
			{
			/* the upper case */
				
				/* handle the first block seperately with alpha */
				i=N-BLOCK_SIZE;
				inplace_dgemm ('T', M, alpha, b+ldb*i, ldb, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE);

				if (i-BLOCK_SIZE<0)
					return;

				cublasDgemm ('N', 'T', M, i, BLOCK_SIZE, -1.0, b+i*ldb, ldb, A+i*lda, lda, alpha, b, ldb);
				
				/* the rest blocks */
				for (i=N-2*BLOCK_SIZE; i>=0; i-=BLOCK_SIZE)
				{
					inplace_dgemm ('T', M, 1.0, b+ldb*i, ldb, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE);

					if (i-BLOCK_SIZE<0)
						break;

					cublasDgemm ('N', 'T', M, i, BLOCK_SIZE, -1.0, b+i*ldb, ldb, A+i*lda, lda, 1.0, b, ldb);
				}
			}
		}
	}
}

/*
 * magmablas_dtrsm
 */

extern "C"
void magmablas_dtrsm (char side, char uplo, char tran, char diag, int M, int N, double alpha, double* A, int lda, double* b, int ldb)
{
/*  -- MAGMA (version 0.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       October 2009

	   Purpose
	   =======
	   
	   DTRSM  solves one of the matrix equations on GPU
	   
	      op( A )*X = alpha*B,   or   X*op( A ) = alpha*B,
	   
	   where alpha is a scalar, X and B are m by n matrices, A is a unit, or
	   non-unit,  upper or lower triangular matrix  and  op( A )  is one  of
	   
	      op( A ) = A   or   op( A ) = A'.
	   
	   The matrix X is overwritten on B.
	   
	   When M or N is not a multiple of blocking size, which is 32 for now, cublasDtrsm will
	   be called instead. There soon will not be this limitation both for arbitrary problem 
	   size and blocking size.
	   
	   Arguments
	   ==========
	   
	   side   - CHARACTER*1.
	            On entry, side specifies whether op( A ) appears on the left
	            or right of X as follows:
	   
	               side = 'L' or 'l'   op( A )*X = alpha*B.
	   
	               side = 'R' or 'r'   X*op( A ) = alpha*B.
	   
	            Unchanged on exit.
	   
	   uplo   - CHARACTER*1.
	            On entry, uplo specifies whether the matrix A is an upper or
	            lower triangular matrix as follows:
	   
	               uplo = 'U' or 'u'   A is an upper triangular matrix.
	   
	               uplo = 'L' or 'l'   A is a lower triangular matrix.
	   
	            Unchanged on exit.
	   
	   tran - CHARACTER*1.
	            On entry, tran specifies the form of op( A ) to be used in
	            the matrix multiplication as follows:
	   
	               tran = 'N' or 'n'   op( A ) = A.
	   
	               tran = 'T' or 't'   op( A ) = A'.
	   
	               tran = 'C' or 'c'   op( A ) = A'.
	   
	            Unchanged on exit.
	   
	   diag   - CHARACTER*1.
	            On entry, diag specifies whether or not A is unit triangular
	            as follows:
	   
	               diag = 'U' or 'u'   A is assumed to be unit triangular.
	   
	               diag = 'N' or 'n'   A is not assumed to be unit
	                                   triangular.
	   
	            Unchanged on exit.
	   
	   m      - INTEGER.
	            On entry, m specifies the number of rows of B. m must be at
	            least zero.
	            Unchanged on exit.
	   
	    n      - INTEGER.
	             On entry, n specifies the number of columns of B.  n must be
	             at least zero.
	             Unchanged on exit.
	   
	    alpha  - DOUBLE PRECISION.
	             On entry,  alpha specifies the scalar  alpha. When  alpha is
	             zero then  A is not referenced and  B need not be set before
	             entry.
	             Unchanged on exit.
	   
	    A      - DOUBLE PRECISION             array of DIMENSION ( lda, k ), where k is m
	             when  side = 'L' or 'l'  and is  n  when  side = 'R' or 'r'.
	             Before entry  with  uplo = 'U' or 'u',  the  leading  k by k
	             upper triangular part of the array  A must contain the upper
	             triangular matrix  and the dtrictly lower triangular part of
	             A is not referenced.
	             Before entry  with  uplo = 'L' or 'l',  the  leading  k by k
	             lower triangular part of the array  A must contain the lower
	             triangular matrix  and the dtrictly upper triangular part of
	             A is not referenced.
	             Note that when  diag = 'U' or 'u',  the diagonal elements of
	             A  are not referenced either,  but are assumed to be  unity.
	             Unchanged on exit.
	   
	    lda    - INTEGER.
	             On entry, lda specifies the first dimension of A as declared
	             in the calling (sub) program.  When  side = 'L' or 'l'  then
	             lda  must be at least  max( 1, m ),  when  side = 'R' or 'r'
	             then lda must be at least max( 1, n ).
	             Unchanged on exit.
	   
	    b      - DOUBLE PRECISION             array of DIMENSION ( ldb, n ).
	             Before entry,  the leading  m by n part of the array  B must
	             contain  the  right-hand  side  matrix  B,  and  on exit  is
	             overwritten by the solution matrix  X.
	   
	    ldb    - INTEGER.
	             On entry, ldb specifies the first dimension of B as declared
	             in  the  calling  (sub)  program.   ldb  must  be  at  least
	             max( 1, m ).
	             Unchanged on exit.
	   
	   
	    Level 3 Blas routine.
		*
    ===================================================================== */

	int i, nblocks;
	double *d_dinvA;
	
	/* 
	 * call cublasDtrsm when size of the problem is not a multiple of blocksize which is 32
	 * subject to change soon
	 */
	if ((M%BLOCK_SIZE) != 0 || (N%BLOCK_SIZE) != 0 )
	{
		cublasDtrsm (side, uplo, tran, diag, M, N, alpha, A, lda, b, ldb);
		return;
	}

	if (side == 'l' || side == 'L')
	{
		/* inverse the diagonals
		 * Allocate device memory for the inversed diagonal blocks, size=m*BLOCK_SIZE 
		 */
		cudaMalloc((void**)&d_dinvA, BLOCK_SIZE*M*sizeof(double));
		nblocks = M/BLOCK_SIZE;

		diag_dtrtri_kernel<<<nblocks, BLOCK_SIZE>>>(uplo, diag, A, d_dinvA, lda);

		if (tran == 'N' || tran == 'n')
		/* the non-transpose case */
		{
			if (uplo == 'L' || uplo == 'l')
			{
			/* the lower case */
				
				/* handle the first block seperately with alpha */
				if (N == 1)
					magmablas_dgemv32 ('N', BLOCK_SIZE, alpha, d_dinvA, BLOCK_SIZE, b, b);
				else
					cublasDgemm ('N', 'N', BLOCK_SIZE, N, BLOCK_SIZE, alpha, d_dinvA, BLOCK_SIZE, b, ldb, 0, b, ldb);  

				if (BLOCK_SIZE>=M)
				{
					cudaFree(d_dinvA);
					return;
				}

				cublasDgemm ('N', 'N', M-BLOCK_SIZE, N, BLOCK_SIZE, -1.0, A+BLOCK_SIZE, lda, b, ldb, alpha, b+BLOCK_SIZE, ldb);

				/* the rest blocks */
				for (i=BLOCK_SIZE; i<M; i+=BLOCK_SIZE)
				{
					if (N == 1)
						magmablas_dgemv32 ('N', BLOCK_SIZE, 1.0, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE, b+i, b+i);
					else
						cublasDgemm ('N', 'N', BLOCK_SIZE, N, BLOCK_SIZE, 1.0, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE, b+i, ldb, 0, b+i, ldb);  

					if (i+BLOCK_SIZE>=M)
						break;

					cublasDgemm ('N', 'N', M-i-BLOCK_SIZE, N, BLOCK_SIZE, -1.0, A+i*lda+i+BLOCK_SIZE, lda, b+i, ldb, 1.0, b+i+BLOCK_SIZE, ldb);
				}
			}
			else
			{
			/* the upper case */

				/* handle the first block seperately with alpha */
				i = M-BLOCK_SIZE;
				if (N == 1)
					magmablas_dgemv32 ('N', BLOCK_SIZE, alpha, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE, b+i, b+i);
				else
					cublasDgemm ('N', 'N', BLOCK_SIZE, N, BLOCK_SIZE, alpha, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE, b+i, ldb, 0.0, b+i, ldb); 
					
				if (i-BLOCK_SIZE<0)
				{
					cudaFree(d_dinvA);
					return;
				}

				cublasDgemm ('N', 'N', i, N, BLOCK_SIZE, -1.0, A+i*lda, lda, b+i, ldb, alpha, b, ldb);

				/* the rest blocks */
				for (i=M-2*BLOCK_SIZE; i>=0; i-=BLOCK_SIZE)
				{
					if (N == 1)
						magmablas_dgemv32 ('N', BLOCK_SIZE, 1.0, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE, b+i, b+i);
					else
						cublasDgemm ('N', 'N', BLOCK_SIZE, N, BLOCK_SIZE, 1.0, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE, b+i, ldb, 0.0, b+i, ldb); 

					if (i-BLOCK_SIZE<0)
						break;

					cublasDgemm ('N', 'N', i, N, BLOCK_SIZE, -1.0, A+i*lda, lda, b+i, ldb, 1.0, b, ldb);
				}
			}
		}
		else
		/* the transpose case */
		{
			if (uplo == 'L' || uplo == 'l')
			{
			/* the lower case */
				
				/* handle the first block seperately with alpha */
				i=M-BLOCK_SIZE; 
				if (N == 1)
					magmablas_dgemv32 ('T', BLOCK_SIZE, alpha, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE, b+i, b+i);
				else
					cublasDgemm ('T', 'N', BLOCK_SIZE, N, BLOCK_SIZE, alpha, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE, b+i, ldb, 0, b+i, ldb);  

				if (i-BLOCK_SIZE<0)
				{
					cudaFree(d_dinvA);
					return;
				}

				cublasDgemm ('T', 'N', i, N, BLOCK_SIZE, -1.0, A+i, lda, b+i, ldb, alpha, b, ldb);

				/* the rest blocks */
				for (i=M-2*BLOCK_SIZE; i>=0; i-=BLOCK_SIZE)
				{
					if (N == 1)
						magmablas_dgemv32 ('T', BLOCK_SIZE, 1.0, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE, b+i, b+i);
					else
						cublasDgemm ('T', 'N', BLOCK_SIZE, N, BLOCK_SIZE, 1.0, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE, b+i, ldb, 0, b+i, ldb);  

					if (i-BLOCK_SIZE<0)
						break;

					cublasDgemm ('T', 'N', i, N, BLOCK_SIZE, -1.0, A+i, lda, b+i, ldb, 1.0, b, ldb);
				}
			}
			else
			{
			/* the upper case */
					
				/* handle the first block seperately with alpha */
				if (N == 1)
					magmablas_dgemv32 ('T', BLOCK_SIZE, alpha, d_dinvA, BLOCK_SIZE, b, b);
				else
					cublasDgemm ('T', 'N', BLOCK_SIZE, N, BLOCK_SIZE, alpha, d_dinvA, BLOCK_SIZE, b, ldb, 0, b, ldb);  

				if (BLOCK_SIZE>=M)
				{
					cudaFree(d_dinvA);
					return;
				}

				cublasDgemm ('T', 'N', M-BLOCK_SIZE, N, BLOCK_SIZE, -1.0, A+(BLOCK_SIZE)*lda, lda, b, ldb, alpha, b+BLOCK_SIZE, ldb);

				/* the rest blocks */
				for (i=BLOCK_SIZE; i<M; i+=BLOCK_SIZE)
				{
					if (N == 1)
						magmablas_dgemv32 ('T', BLOCK_SIZE, 1.0, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE, b+i, b+i);
					else
						cublasDgemm ('T', 'N', BLOCK_SIZE, N, BLOCK_SIZE, 1.0, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE, b+i, ldb, 0, b+i, ldb);  
					
					if (i+BLOCK_SIZE>=M)
						break;

					cublasDgemm ('T', 'N', M-i-BLOCK_SIZE, N, BLOCK_SIZE, -1.0, A+(i+BLOCK_SIZE)*lda+i, lda, b+i, ldb, 1.0, b+i+BLOCK_SIZE, ldb);
				}
			}
		}
	}
	else
	{	// side=R

		/* inverse the diagonals
		 * Allocate device memory for the inversed diagonal blocks, size=N*BLOCK_SIZE 
		 */
		cudaMalloc((void**)&d_dinvA, BLOCK_SIZE*N*sizeof(double));
		nblocks = N/BLOCK_SIZE;
		diag_dtrtri_kernel<<<nblocks, BLOCK_SIZE>>>(uplo, diag, A, d_dinvA, lda);
		
		if (tran == 'N' || tran == 'n')
		/* the non-transpose case */
		{
			if (uplo == 'L' || uplo == 'l')
			{
			/* the lower case */
				
				/* handle the first block seperately with alpha */
				i=N-BLOCK_SIZE;
				inplace_dgemm ('N', M, alpha, b+ldb*i, ldb, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE);

				if (i-BLOCK_SIZE<0)
				{
					cudaFree(d_dinvA);
					return;
				}

				cublasDgemm ('N', 'N', M, i, BLOCK_SIZE, -1.0, b+ldb*i, ldb, A+i, lda, alpha, b, ldb);

				/* the rest blocks */
				for (i=N-2*BLOCK_SIZE; i>=0; i-=BLOCK_SIZE)
				{
					inplace_dgemm ('N', M, 1.0, b+ldb*i, ldb, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE);
					
					if (i-BLOCK_SIZE<0)
						break;

					cublasDgemm ('N', 'N', M, i, BLOCK_SIZE, -1.0, b+ldb*i, ldb, A+i, lda, 1.0, b, ldb);
				}
			}
			else
			{
			/* the upper case */
				
				/* handle the first block seperately with alpha */
				inplace_dgemm ('N', M, alpha, b, ldb, d_dinvA, BLOCK_SIZE);

				if (BLOCK_SIZE>=N)
				{
					cudaFree(d_dinvA);
					return;
				}

				cublasDgemm ('N', 'N', M, N-BLOCK_SIZE, BLOCK_SIZE, -1.0, b, ldb, A+(BLOCK_SIZE)*lda, lda, alpha, b+(BLOCK_SIZE)*ldb, ldb);
				
				
				/* the rest blocks */
				for (i=BLOCK_SIZE; i<N; i+=BLOCK_SIZE)
				{
					inplace_dgemm ('N', M, 1.0, b+ldb*i, ldb, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE);

					if (i+BLOCK_SIZE>=N)
						break;

					cublasDgemm ('N', 'N', M, N-i-BLOCK_SIZE, BLOCK_SIZE, -1.0, b+i*ldb, ldb, A+(i+BLOCK_SIZE)*lda+i, lda, 1.0, b+(i+BLOCK_SIZE)*ldb, ldb);
				}
			}
		}
		else
		/* the transpose case */
		{
			if (uplo == 'L' || uplo == 'l')
			{
			/* the lower case */
				
				/* handle the first block seperately with alpha */
				inplace_dgemm ('T', M, alpha, b, ldb, d_dinvA, BLOCK_SIZE);

				if (BLOCK_SIZE>=N)
				{
					cudaFree(d_dinvA);
					return;
				}

				cublasDgemm ('N', 'T', M, N-BLOCK_SIZE, BLOCK_SIZE, -1.0, b, ldb, A+BLOCK_SIZE, lda, alpha, b+(BLOCK_SIZE)*ldb, ldb);

				/* the rest blocks */
				for (i=BLOCK_SIZE; i<N; i+=BLOCK_SIZE)
				{
					inplace_dgemm ('T', M, 1.0, b+ldb*i, ldb, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE);

					if (i+BLOCK_SIZE>=N)
						break;

					cublasDgemm ('N', 'T', M, N-i-BLOCK_SIZE, BLOCK_SIZE, -1.0, b+ldb*i, ldb, A+i*lda+BLOCK_SIZE+i, lda, 1.0, b+(i+BLOCK_SIZE)*ldb, ldb);
				}
			}
			else
			{
			/* the upper case */
				
				/* handle the first block seperately with alpha */
				i=N-BLOCK_SIZE;
				inplace_dgemm ('T', M, alpha, b+ldb*i, ldb, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE);

				if (i-BLOCK_SIZE<0)
				{
					cudaFree(d_dinvA);
					return;
				}

				cublasDgemm ('N', 'T', M, i, BLOCK_SIZE, -1.0, b+i*ldb, ldb, A+i*lda, lda, alpha, b, ldb);
				
				/* the rest blocks */
				for (i=N-2*BLOCK_SIZE; i>=0; i-=BLOCK_SIZE)
				{
					inplace_dgemm ('T', M, 1.0, b+ldb*i, ldb, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE);

					if (i-BLOCK_SIZE<0)
						break;

					cublasDgemm ('N', 'T', M, i, BLOCK_SIZE, -1.0, b+i*ldb, ldb, A+i*lda, lda, 1.0, b, ldb);
				}
			}
		}
	}

	cudaFree(d_dinvA);
}

