/*
    -- MAGMA (version 0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       October 2009
*/

#include "cublas.h"
#include "cuda.h"
#include "magmablas.h"

#define BLOCK_SIZE 32 

__global__ void
inplace_sgemm_kernel_T(int M, float alpha, float *A, int lda, float *B, int ldb)
{
        int i;
        float myvalue1=0, myvalue2= 0 ;
        float med;
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        const int bx = blockIdx.x;
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE+1];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        A+= bx*32 + __mul24(lda,ty) + tx ;
        B+=      __mul24(ldb,ty) + tx ;


        As[tx][ ty]=A[0];
        As[tx][ty+16]= A[16*lda];
        Bs[tx][ty]= B[0];
        Bs[tx][ty+16]= B[16*ldb];

        __syncthreads();
        med  = As[tx][0];
        float py1 = Bs[ty][0] ;
        float py2 = Bs[ty+16][0] ;
        #pragma unroll
        for (i=0; i<31; i++){
                myvalue1 +=  med*py1;
                py1 = Bs[ty][i+1] ;
                myvalue2 +=  med*py2;
                py2 = Bs[ty+16][i+1] ;
                med  = As[tx][i+1];
        }
        myvalue1 +=  med*py1;
        myvalue2 +=  med*py2;
        A[0] = alpha*myvalue1 ;
        A[lda*16] = alpha*myvalue2;
}

__global__ void
inplace_sgemm_kernel_N(int M, float alpha, float *A, int lda, float *B, int ldb)
{
        int i;
        float myvalue1=0, myvalue2= 0 ;
        float med;
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        const int bx = blockIdx.x;
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE+1];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        A+= bx*32 + __mul24(lda,ty) + tx ;
        B+=      __mul24(ldb,ty) + tx ;


        As[tx][ ty]=A[0];
        As[tx][ty+16]= A[16*lda];
        Bs[ty][tx]= B[0];
        Bs[ty+16][tx]= B[16*ldb];

        __syncthreads();
        med  = As[tx][0];
        float py1 = Bs[ty][0] ;
        float py2 = Bs[ty+16][0] ;
        #pragma unroll
        for (i=0; i<31; i++){
                myvalue1 +=  med*py1;
                py1 = Bs[ty][i+1] ;
                myvalue2 +=  med*py2;
                py2 = Bs[ty+16][i+1] ;
                med  = As[tx][i+1];
        }
        myvalue1 +=  med*py1;
        myvalue2 +=  med*py2;
        A[0] = alpha*myvalue1 ;
        A[lda*16] = alpha*myvalue2;
}

__global__ void
diag_strtri_kernel (char uplo, char diag, float *A, float *d_dinvA, int lda)
{
	int i,j;
	float Ystx=0;
	float *Bw=NULL, *x=NULL, *y=NULL, *Aoff=NULL;
	float *my_d_dinvA;
	int switcher=0;

	// Thread index
	int tx = threadIdx.x;
	int txw;

	// Block index
	int bx = blockIdx.x;
		
	Aoff = A+bx*lda*BLOCK_SIZE+bx*BLOCK_SIZE;
	my_d_dinvA = d_dinvA+bx*BLOCK_SIZE*BLOCK_SIZE;

	__shared__ float Bs[BLOCK_SIZE*BLOCK_SIZE];
	__shared__ float workspace[BLOCK_SIZE]; // workspace used to store the current working column

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
			
			//strmv
			Bw = Bs+(i+1)*BLOCK_SIZE+i+1;
			workspace[tx] = *(Bs+i*BLOCK_SIZE+tx);
			x = workspace+i+1;
			y = Bs+i*BLOCK_SIZE;

			txw = (tx-i-1);

			#pragma unroll
			for (j=0; j<txw+1; j++)
				Ystx += (float)switcher*(*(Bw+j*BLOCK_SIZE+txw)*x[j]);

			//sscal
			switcher = (tx != i); 
			//if (tx !=i ) y[tx]=switcher*Ystx*(-Bs[i*BLOCK_SIZE+i]);
			y[tx] = (float)switcher*Ystx*(-Bs[i*BLOCK_SIZE+i])+(float)(!switcher)*y[tx];

			__syncthreads();
		}

	}
	else
	{
		 /* the upper case */
		for (i=0; i<BLOCK_SIZE; i++)
		{
			Ystx = 0;
			switcher = (float)(tx<i);
			
			//strmv
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
inplace_sgemm (char tran, int M, float alpha, float *A, int lda, float *B, int ldb)
{
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE/2);

	if (tran == 'n' || tran == 'N')
		inplace_sgemm_kernel_N<<<M/BLOCK_SIZE,dimBlock>>>(M, alpha, A, lda, B, ldb); 
	else
		inplace_sgemm_kernel_T<<<M/BLOCK_SIZE,dimBlock>>>(M, alpha, A, lda, B, ldb); 
}

/*
 * magmablas_strsmx
 * the expert interface
 */
void magmablas_strsmx ( char side, char uplo, char tran, char diag, int M, int N, float alpha, float* A, int lda, float* b, int ldb, float *d_dinvA)
{
/*  -- MAGMA (version 0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       October 2009

	   Purpose
	   =======
	   
	   STRSM  solves one of the matrix equations on GPU
	   
	      op( A )*X = alpha*B,   or   X*op( A ) = alpha*B,
	   
	   where alpha is a scalar, X and B are m by n matrices, A is a unit, or
	   non-unit,  upper or lower triangular matrix  and  op( A )  is one  of
	   
	      op( A ) = A   or   op( A ) = A'.
	   
	   The matrix X is overwritten on B.

	   When M or N is not a multiple of blocking size, which is 32 for now, cublasStrsm will
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
	   
	    alpha  - REAL.
	             On entry,  alpha specifies the scalar  alpha. When  alpha is
	             zero then  A is not referenced and  B need not be set before
	             entry.
	             Unchanged on exit.
	   
	    A      - REAL             array of DIMENSION ( lda, k ), where k is m
	             when  side = 'L' or 'l'  and is  n  when  side = 'R' or 'r'.
	             Before entry  with  uplo = 'U' or 'u',  the  leading  k by k
	             upper triangular part of the array  A must contain the upper
	             triangular matrix  and the strictly lower triangular part of
	             A is not referenced.
	             Before entry  with  uplo = 'L' or 'l',  the  leading  k by k
	             lower triangular part of the array  A must contain the lower
	             triangular matrix  and the strictly upper triangular part of
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
	   
	    b      - REAL             array of DIMENSION ( ldb, n ).
	             Before entry,  the leading  m by n part of the array  B must
	             contain  the  right-hand  side  matrix  B,  and  on exit  is
	             overwritten by the solution matrix  X.
	   
	    ldb    - INTEGER.
	             On entry, ldb specifies the first dimension of B as declared
	             in  the  calling  (sub)  program.   ldb  must  be  at  least
	             max( 1, m ).
	             Unchanged on exit.

		d_dinvA  REAL array of DIMENSION (BLOCKSIZE, M) when side='L', 
				 (BLOCKSIZE, N) when side='R'. On exit this space is filled
			     with the inverse of blocks on the diagonal, each inverse is
				 of size BLOCKSIZE x BLOCKSIZE, and the leading dimension of
				 d_dinvA is BLOCKSIZE;
	   
	   
	    Level 3 Blas routine.
		*
    ===================================================================== */

	int i, nblocks;
	
	/* quick return on wrong size */
	if (M<=0 || N<=0 || d_dinvA == NULL)
		return;

	/* 
	 * call cublasStrsm when size of the problem is not a multiple of blocksize which is 32
	 * subject to change soon
	 */

	if ((M%BLOCK_SIZE)!=0 || (N>1 && (N%BLOCK_SIZE)!=0))
	{
		cublasStrsm (side, uplo, tran, diag, M, N, alpha, A, lda, b, ldb);
		return;
	}

	if (side == 'l' || side == 'L')
	{
		/* inverse the diagonals
		 * Allocate device memory for the inversed diagonal blocks, size=m*BLOCK_SIZE 
		 */
		nblocks = M/BLOCK_SIZE;
		diag_strtri_kernel<<<nblocks, BLOCK_SIZE>>>(uplo, diag, A, d_dinvA, lda);

		if (tran == 'N' || tran == 'n')
		/* the non-transpose case */
		{
			if (uplo == 'L' || uplo == 'l')
			{
			/* the lower case */
				
				/* handle the first block seperately with alpha */
				if (N == 1)
					magmablas_sgemv32 ('N', BLOCK_SIZE, alpha, d_dinvA, BLOCK_SIZE, b, b);
				else
					cublasSgemm ('N', 'N', BLOCK_SIZE, N, BLOCK_SIZE, alpha, d_dinvA, BLOCK_SIZE, b, ldb, 0, b, ldb);  

				if (BLOCK_SIZE>=M)
					return;

				cublasSgemm ('N', 'N', M-BLOCK_SIZE, N, BLOCK_SIZE, -1.0, A+BLOCK_SIZE, lda, b, ldb, alpha, b+BLOCK_SIZE, ldb);

				/* the rest blocks */
				for (i=BLOCK_SIZE; i<M; i+=BLOCK_SIZE)
				{
					if (N == 1)
						magmablas_sgemv32 ('N', BLOCK_SIZE, 1.0, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE, b+i, b+i);
					else
						cublasSgemm ('N', 'N', BLOCK_SIZE, N, BLOCK_SIZE, 1.0, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE, b+i, ldb, 0, b+i, ldb);  

					if (i+BLOCK_SIZE>=M)
						break;

					cublasSgemm ('N', 'N', M-i-BLOCK_SIZE, N, BLOCK_SIZE, -1.0, A+i*lda+i+BLOCK_SIZE, lda, b+i, ldb, 1.0, b+i+BLOCK_SIZE, ldb);
				}
			}
			else
			{
			/* the upper case */

				/* handle the first block seperately with alpha */
				i = M-BLOCK_SIZE;
				if (N == 1)
					magmablas_sgemv32 ('N', BLOCK_SIZE, alpha, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE, b+i, b+i);
				else
					cublasSgemm ('N', 'N', BLOCK_SIZE, N, BLOCK_SIZE, alpha, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE, b+i, ldb, 0.0, b+i, ldb); 
					
				if (i-BLOCK_SIZE<0)
					return;

				cublasSgemm ('N', 'N', i, N, BLOCK_SIZE, -1.0, A+i*lda, lda, b+i, ldb, alpha, b, ldb);

				/* the rest blocks */
				for (i=M-2*BLOCK_SIZE; i>=0; i-=BLOCK_SIZE)
				{
					if (N == 1)
						magmablas_sgemv32 ('N', BLOCK_SIZE, 1.0, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE, b+i, b+i);
					else
						cublasSgemm ('N', 'N', BLOCK_SIZE, N, BLOCK_SIZE, 1.0, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE, b+i, ldb, 0.0, b+i, ldb); 

					if (i-BLOCK_SIZE<0)
						break;

					cublasSgemm ('N', 'N', i, N, BLOCK_SIZE, -1.0, A+i*lda, lda, b+i, ldb, 1.0, b, ldb);
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
					magmablas_sgemv32 ('T', BLOCK_SIZE, alpha, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE, b+i, b+i);
				else
					cublasSgemm ('T', 'N', BLOCK_SIZE, N, BLOCK_SIZE, alpha, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE, b+i, ldb, 0, b+i, ldb);  

				if (i-BLOCK_SIZE<0)
					return;

				cublasSgemm ('T', 'N', i, N, BLOCK_SIZE, -1.0, A+i, lda, b+i, ldb, alpha, b, ldb);

				/* the rest blocks */
				for (i=M-2*BLOCK_SIZE; i>=0; i-=BLOCK_SIZE)
				{
					if (N == 1)
						magmablas_sgemv32 ('T', BLOCK_SIZE, 1.0, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE, b+i, b+i);
					else
						cublasSgemm ('T', 'N', BLOCK_SIZE, N, BLOCK_SIZE, 1.0, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE, b+i, ldb, 0, b+i, ldb);  

					if (i-BLOCK_SIZE<0)
						break;

					cublasSgemm ('T', 'N', i, N, BLOCK_SIZE, -1.0, A+i, lda, b+i, ldb, 1.0, b, ldb);
				}
			}
			else
			{
			/* the upper case */
					
				/* handle the first block seperately with alpha */
				if (N == 1)
					magmablas_sgemv32 ('T', BLOCK_SIZE, alpha, d_dinvA, BLOCK_SIZE, b, b);
				else
					cublasSgemm ('T', 'N', BLOCK_SIZE, N, BLOCK_SIZE, alpha, d_dinvA, BLOCK_SIZE, b, ldb, 0, b, ldb);  

				if (BLOCK_SIZE>=M)
					return;

				cublasSgemm ('T', 'N', M-BLOCK_SIZE, N, BLOCK_SIZE, -1.0, A+(BLOCK_SIZE)*lda, lda, b, ldb, alpha, b+BLOCK_SIZE, ldb);

				/* the rest blocks */
				for (i=BLOCK_SIZE; i<M; i+=BLOCK_SIZE)
				{
					if (N == 1)
						magmablas_sgemv32 ('T', BLOCK_SIZE, 1.0, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE, b+i, b+i);
					else
						cublasSgemm ('T', 'N', BLOCK_SIZE, N, BLOCK_SIZE, 1.0, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE, b+i, ldb, 0, b+i, ldb);  
					
					if (i+BLOCK_SIZE>=M)
						break;

					cublasSgemm ('T', 'N', M-i-BLOCK_SIZE, N, BLOCK_SIZE, -1.0, A+(i+BLOCK_SIZE)*lda+i, lda, b+i, ldb, 1.0, b+i+BLOCK_SIZE, ldb);
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
		diag_strtri_kernel<<<nblocks, BLOCK_SIZE>>>(uplo, diag, A, d_dinvA, lda);
		
		if (tran == 'N' || tran == 'n')
		/* the non-transpose case */
		{
			if (uplo == 'L' || uplo == 'l')
			{
			/* the lower case */
				
				/* handle the first block seperately with alpha */
				i=N-BLOCK_SIZE;
				inplace_sgemm ('N', M, alpha, b+ldb*i, ldb, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE);

				if (i-BLOCK_SIZE<0)
					return;

				cublasSgemm ('N', 'N', M, i, BLOCK_SIZE, -1.0, b+ldb*i, ldb, A+i, lda, alpha, b, ldb);

				/* the rest blocks */
				for (i=N-2*BLOCK_SIZE; i>=0; i-=BLOCK_SIZE)
				{
					inplace_sgemm ('N', M, 1.0, b+ldb*i, ldb, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE);
					
					if (i-BLOCK_SIZE<0)
						break;

					cublasSgemm ('N', 'N', M, i, BLOCK_SIZE, -1.0, b+ldb*i, ldb, A+i, lda, 1.0, b, ldb);
				}
			}
			else
			{
			/* the upper case */
				
				/* handle the first block seperately with alpha */
				inplace_sgemm ('N', M, alpha, b, ldb, d_dinvA, BLOCK_SIZE);

				if (BLOCK_SIZE>=N)
					return;

				cublasSgemm ('N', 'N', M, N-BLOCK_SIZE, BLOCK_SIZE, -1.0, b, ldb, A+(BLOCK_SIZE)*lda, lda, alpha, b+(BLOCK_SIZE)*ldb, ldb);
				
				
				/* the rest blocks */
				for (i=BLOCK_SIZE; i<N; i+=BLOCK_SIZE)
				{
					inplace_sgemm ('N', M, 1.0, b+ldb*i, ldb, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE);

					if (i+BLOCK_SIZE>=N)
						break;

					cublasSgemm ('N', 'N', M, N-i-BLOCK_SIZE, BLOCK_SIZE, -1.0, b+i*ldb, ldb, A+(i+BLOCK_SIZE)*lda+i, lda, 1.0, b+(i+BLOCK_SIZE)*ldb, ldb);
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
				inplace_sgemm ('T', M, alpha, b, ldb, d_dinvA, BLOCK_SIZE);

				if (BLOCK_SIZE>=N)
					return;

				cublasSgemm ('N', 'T', M, N-BLOCK_SIZE, BLOCK_SIZE, -1.0, b, ldb, A+BLOCK_SIZE, lda, alpha, b+(BLOCK_SIZE)*ldb, ldb);

				/* the rest blocks */
				for (i=BLOCK_SIZE; i<N; i+=BLOCK_SIZE)
				{
					inplace_sgemm ('T', M, 1.0, b+ldb*i, ldb, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE);

					if (i+BLOCK_SIZE>=N)
						break;

					cublasSgemm ('N', 'T', M, N-i-BLOCK_SIZE, BLOCK_SIZE, -1.0, b+ldb*i, ldb, A+i*lda+BLOCK_SIZE+i, lda, 1.0, b+(i+BLOCK_SIZE)*ldb, ldb);
				}
			}
			else
			{
			/* the upper case */
				
				/* handle the first block seperately with alpha */
				i=N-BLOCK_SIZE;
				inplace_sgemm ('T', M, alpha, b+ldb*i, ldb, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE);

				if (i-BLOCK_SIZE<0)
					return;

				cublasSgemm ('N', 'T', M, i, BLOCK_SIZE, -1.0, b+i*ldb, ldb, A+i*lda, lda, alpha, b, ldb);
				
				/* the rest blocks */
				for (i=N-2*BLOCK_SIZE; i>=0; i-=BLOCK_SIZE)
				{
					inplace_sgemm ('T', M, 1.0, b+ldb*i, ldb, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE);

					if (i-BLOCK_SIZE<0)
						break;

					cublasSgemm ('N', 'T', M, i, BLOCK_SIZE, -1.0, b+i*ldb, ldb, A+i*lda, lda, 1.0, b, ldb);
				}
			}
		}
	}
}

/*
 * magmablas_strsm
 */
extern "C"
void magmablas_strsm ( char side, char uplo, char tran, char diag, int M, int N, float alpha, float* A, int lda, float* b, int ldb)
{
/*  -- MAGMA (version 0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       October 2009

	   Purpose
	   =======
	   
	   STRSM  solves one of the matrix equations on GPU
	   
	      op( A )*X = alpha*B,   or   X*op( A ) = alpha*B,
	   
	   where alpha is a scalar, X and B are m by n matrices, A is a unit, or
	   non-unit,  upper or lower triangular matrix  and  op( A )  is one  of
	   
	      op( A ) = A   or   op( A ) = A'.
	   
	   The matrix X is overwritten on B.
	   
	   When M or N is not a multiple of blocking size, which is 32 for now, cublasStrsm will
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
	   
	    alpha  - REAL.
	             On entry,  alpha specifies the scalar  alpha. When  alpha is
	             zero then  A is not referenced and  B need not be set before
	             entry.
	             Unchanged on exit.
	   
	    A      - REAL             array of DIMENSION ( lda, k ), where k is m
	             when  side = 'L' or 'l'  and is  n  when  side = 'R' or 'r'.
	             Before entry  with  uplo = 'U' or 'u',  the  leading  k by k
	             upper triangular part of the array  A must contain the upper
	             triangular matrix  and the strictly lower triangular part of
	             A is not referenced.
	             Before entry  with  uplo = 'L' or 'l',  the  leading  k by k
	             lower triangular part of the array  A must contain the lower
	             triangular matrix  and the strictly upper triangular part of
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
	   
	    b      - REAL             array of DIMENSION ( ldb, n ).
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
	float *d_dinvA;

	/* quick return on wrong size */
	if (M<=0 || N<=0)
		return;

	/* 
	 * call cublasStrsm when size of the problem is not a multiple of blocksize which is 32
	 * subject to change soon
	 */
	if ((M%BLOCK_SIZE)!=0 || (N>1 && (N%BLOCK_SIZE)!=0))
	{
		cublasStrsm (side, uplo, tran, diag, M, N, alpha, A, lda, b, ldb);
		return;
	}

	if (side == 'l' || side == 'L')
	{
		/* inverse the diagonals
		 * Allocate device memory for the inversed diagonal blocks, size=m*BLOCK_SIZE 
		 */
		cudaMalloc((void**)&d_dinvA, BLOCK_SIZE*M*sizeof(float));
		nblocks = M/BLOCK_SIZE;

		diag_strtri_kernel<<<nblocks, BLOCK_SIZE>>>(uplo, diag, A, d_dinvA, lda);

		if (tran == 'N' || tran == 'n')
		/* the non-transpose case */
		{
			if (uplo == 'L' || uplo == 'l')
			{
			/* the lower case */
				
				/* handle the first block seperately with alpha */
				if (N == 1)
					magmablas_sgemv32 ('N', BLOCK_SIZE, alpha, d_dinvA, BLOCK_SIZE, b, b);
				else
					cublasSgemm ('N', 'N', BLOCK_SIZE, N, BLOCK_SIZE, alpha, d_dinvA, BLOCK_SIZE, b, ldb, 0, b, ldb);  

				if (BLOCK_SIZE>=M)
				{
					cudaFree(d_dinvA);
					return;
				}

				cublasSgemm ('N', 'N', M-BLOCK_SIZE, N, BLOCK_SIZE, -1.0, A+BLOCK_SIZE, lda, b, ldb, alpha, b+BLOCK_SIZE, ldb);

				/* the rest blocks */
				for (i=BLOCK_SIZE; i<M; i+=BLOCK_SIZE)
				{
					if (N == 1)
						magmablas_sgemv32 ('N', BLOCK_SIZE, 1.0, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE, b+i, b+i);
					else
						cublasSgemm ('N', 'N', BLOCK_SIZE, N, BLOCK_SIZE, 1.0, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE, b+i, ldb, 0, b+i, ldb);  

					if (i+BLOCK_SIZE>=M)
						break;

					cublasSgemm ('N', 'N', M-i-BLOCK_SIZE, N, BLOCK_SIZE, -1.0, A+i*lda+i+BLOCK_SIZE, lda, b+i, ldb, 1.0, b+i+BLOCK_SIZE, ldb);
				}
			}
			else
			{
			/* the upper case */

				/* handle the first block seperately with alpha */
				i = M-BLOCK_SIZE;
				if (N == 1)
					magmablas_sgemv32 ('N', BLOCK_SIZE, alpha, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE, b+i, b+i);
				else
					cublasSgemm ('N', 'N', BLOCK_SIZE, N, BLOCK_SIZE, alpha, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE, b+i, ldb, 0.0, b+i, ldb); 
					
				if (i-BLOCK_SIZE<0)
				{
					cudaFree(d_dinvA);
					return;
				}

				cublasSgemm ('N', 'N', i, N, BLOCK_SIZE, -1.0, A+i*lda, lda, b+i, ldb, alpha, b, ldb);

				/* the rest blocks */
				for (i=M-2*BLOCK_SIZE; i>=0; i-=BLOCK_SIZE)
				{
					if (N == 1)
						magmablas_sgemv32 ('N', BLOCK_SIZE, 1.0, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE, b+i, b+i);
					else
						cublasSgemm ('N', 'N', BLOCK_SIZE, N, BLOCK_SIZE, 1.0, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE, b+i, ldb, 0.0, b+i, ldb); 

					if (i-BLOCK_SIZE<0)
						break;

					cublasSgemm ('N', 'N', i, N, BLOCK_SIZE, -1.0, A+i*lda, lda, b+i, ldb, 1.0, b, ldb);
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
					magmablas_sgemv32 ('T', BLOCK_SIZE, alpha, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE, b+i, b+i);
				else
					cublasSgemm ('T', 'N', BLOCK_SIZE, N, BLOCK_SIZE, alpha, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE, b+i, ldb, 0, b+i, ldb);  

				if (i-BLOCK_SIZE<0)
				{
					cudaFree(d_dinvA);
					return;
				}

				cublasSgemm ('T', 'N', i, N, BLOCK_SIZE, -1.0, A+i, lda, b+i, ldb, alpha, b, ldb);

				/* the rest blocks */
				for (i=M-2*BLOCK_SIZE; i>=0; i-=BLOCK_SIZE)
				{
					if (N == 1)
						magmablas_sgemv32 ('T', BLOCK_SIZE, 1.0, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE, b+i, b+i);
					else
						cublasSgemm ('T', 'N', BLOCK_SIZE, N, BLOCK_SIZE, 1.0, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE, b+i, ldb, 0, b+i, ldb);  

					if (i-BLOCK_SIZE<0)
						break;

					cublasSgemm ('T', 'N', i, N, BLOCK_SIZE, -1.0, A+i, lda, b+i, ldb, 1.0, b, ldb);
				}
			}
			else
			{
			/* the upper case */
					
				/* handle the first block seperately with alpha */
				if (N == 1)
					magmablas_sgemv32 ('T', BLOCK_SIZE, alpha, d_dinvA, BLOCK_SIZE, b, b);
				else
					cublasSgemm ('T', 'N', BLOCK_SIZE, N, BLOCK_SIZE, alpha, d_dinvA, BLOCK_SIZE, b, ldb, 0, b, ldb);  

				if (BLOCK_SIZE>=M)
				{
					cudaFree(d_dinvA);
					return;
				}

				cublasSgemm ('T', 'N', M-BLOCK_SIZE, N, BLOCK_SIZE, -1.0, A+(BLOCK_SIZE)*lda, lda, b, ldb, alpha, b+BLOCK_SIZE, ldb);

				/* the rest blocks */
				for (i=BLOCK_SIZE; i<M; i+=BLOCK_SIZE)
				{
					if (N == 1)
						magmablas_sgemv32 ('T', BLOCK_SIZE, 1.0, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE, b+i, b+i);
					else
						cublasSgemm ('T', 'N', BLOCK_SIZE, N, BLOCK_SIZE, 1.0, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE, b+i, ldb, 0, b+i, ldb);  
					
					if (i+BLOCK_SIZE>=M)
						break;

					cublasSgemm ('T', 'N', M-i-BLOCK_SIZE, N, BLOCK_SIZE, -1.0, A+(i+BLOCK_SIZE)*lda+i, lda, b+i, ldb, 1.0, b+i+BLOCK_SIZE, ldb);
				}
			}
		}
	}
	else
	{	// side=R

		/* inverse the diagonals
		 * Allocate device memory for the inversed diagonal blocks, size=N*BLOCK_SIZE 
		 */
		cudaMalloc((void**)&d_dinvA, BLOCK_SIZE*N*sizeof(float));
		nblocks = N/BLOCK_SIZE;
		diag_strtri_kernel<<<nblocks, BLOCK_SIZE>>>(uplo, diag, A, d_dinvA, lda);
		
		if (tran == 'N' || tran == 'n')
		/* the non-transpose case */
		{
			if (uplo == 'L' || uplo == 'l')
			{
			/* the lower case */
				
				/* handle the first block seperately with alpha */
				i=N-BLOCK_SIZE;
				inplace_sgemm ('N', M, alpha, b+ldb*i, ldb, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE);

				if (i-BLOCK_SIZE<0)
				{
					cudaFree(d_dinvA);
					return;
				}

				cublasSgemm ('N', 'N', M, i, BLOCK_SIZE, -1.0, b+ldb*i, ldb, A+i, lda, alpha, b, ldb);

				/* the rest blocks */
				for (i=N-2*BLOCK_SIZE; i>=0; i-=BLOCK_SIZE)
				{
					inplace_sgemm ('N', M, 1.0, b+ldb*i, ldb, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE);
					
					if (i-BLOCK_SIZE<0)
						break;

					cublasSgemm ('N', 'N', M, i, BLOCK_SIZE, -1.0, b+ldb*i, ldb, A+i, lda, 1.0, b, ldb);
				}
			}
			else
			{
			/* the upper case */
				
				/* handle the first block seperately with alpha */
				inplace_sgemm ('N', M, alpha, b, ldb, d_dinvA, BLOCK_SIZE);

				if (BLOCK_SIZE>=N)
				{
					cudaFree(d_dinvA);
					return;
				}

				cublasSgemm ('N', 'N', M, N-BLOCK_SIZE, BLOCK_SIZE, -1.0, b, ldb, A+(BLOCK_SIZE)*lda, lda, alpha, b+(BLOCK_SIZE)*ldb, ldb);
				
				
				/* the rest blocks */
				for (i=BLOCK_SIZE; i<N; i+=BLOCK_SIZE)
				{
					inplace_sgemm ('N', M, 1.0, b+ldb*i, ldb, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE);

					if (i+BLOCK_SIZE>=N)
						break;

					cublasSgemm ('N', 'N', M, N-i-BLOCK_SIZE, BLOCK_SIZE, -1.0, b+i*ldb, ldb, A+(i+BLOCK_SIZE)*lda+i, lda, 1.0, b+(i+BLOCK_SIZE)*ldb, ldb);
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
				inplace_sgemm ('T', M, alpha, b, ldb, d_dinvA, BLOCK_SIZE);

				if (BLOCK_SIZE>=N)
				{
					cudaFree(d_dinvA);
					return;
				}

				cublasSgemm ('N', 'T', M, N-BLOCK_SIZE, BLOCK_SIZE, -1.0, b, ldb, A+BLOCK_SIZE, lda, alpha, b+(BLOCK_SIZE)*ldb, ldb);

				/* the rest blocks */
				for (i=BLOCK_SIZE; i<N; i+=BLOCK_SIZE)
				{
					inplace_sgemm ('T', M, 1.0, b+ldb*i, ldb, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE);

					if (i+BLOCK_SIZE>=N)
						break;

					cublasSgemm ('N', 'T', M, N-i-BLOCK_SIZE, BLOCK_SIZE, -1.0, b+ldb*i, ldb, A+i*lda+BLOCK_SIZE+i, lda, 1.0, b+(i+BLOCK_SIZE)*ldb, ldb);
				}
			}
			else
			{
			/* the upper case */
				
				/* handle the first block seperately with alpha */
				i=N-BLOCK_SIZE;
				inplace_sgemm ('T', M, alpha, b+ldb*i, ldb, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE);

				if (i-BLOCK_SIZE<0)
				{
					cudaFree(d_dinvA);
					return;
				}

				cublasSgemm ('N', 'T', M, i, BLOCK_SIZE, -1.0, b+i*ldb, ldb, A+i*lda, lda, alpha, b, ldb);
				
				/* the rest blocks */
				for (i=N-2*BLOCK_SIZE; i>=0; i-=BLOCK_SIZE)
				{
					inplace_sgemm ('T', M, 1.0, b+ldb*i, ldb, d_dinvA+i*BLOCK_SIZE, BLOCK_SIZE);

					if (i-BLOCK_SIZE<0)
						break;

					cublasSgemm ('N', 'T', M, i, BLOCK_SIZE, -1.0, b+i*ldb, ldb, A+i*lda, lda, 1.0, b, ldb);
				}
			}
		}
	}
		
	cudaFree(d_dinvA);
}

