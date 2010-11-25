/*
    -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

*/
#include <stdio.h>
#include "cuda.h"
#include "cublas.h"
#include "magma.h"

#define magmablas_zhemv_tesla magmablas_zhemv 

inline __host__ __device__ double2 make_double2(double s)
{
	return make_double2(s, s);
}
inline __host__ __device__ double2 make_double2(int2 a)
{
	return make_double2(double(a.x), double(a.y));
}

// negate
inline __host__ __device__ double2 operator-(double2 &a)
{
	return make_double2(-a.x, -a.y);
}
// addition
inline __host__ __device__ double2 operator+(double2 a, double2 b)
{
	return make_double2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(double2 &a, double2 b)
{
	a.x += b.x; a.y += b.y;
}

// subtract
inline __host__ __device__ double2 operator-(double2 a, double2 b)
{
	return make_double2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(double2 &a, double2 b)
{
	a.x -= b.x; a.y -= b.y;
}

// multiply
inline __host__ __device__ double2 operator*(double2 a, double2 b)
{
    return make_double2(a.x * b.x - a.y * b.y, a.y * b.x + a.x * b.y);
}
inline __host__ __device__ double2 operator*(double2 a, double s)
{
	return make_double2(a.x * s, a.y * s);
}
inline __host__ __device__ double2 operator*(double s, double2 a)
{
	return make_double2(a.x * s, a.y * s);
}
inline __host__ __device__ void operator*=(double2 &a, double s)
{
	a.x *= s; a.y *= s;
}

inline __host__ __device__ double2 conjugate(double2 a)
{
   double2 b;
   b.x = a.x;
   b.y = 0.0f-a.y;
   return b;
}

#define thread_seg 128  // used in zhemv_tesla1_kernel 
#define threadSize 128  // used in zhemv_tesla2_kernel

__global__ void zhemv_tesla1_kernel( int m , double2 alpha ,  double2 *A , int lda ,  double2 *x, int incx , double2 beta ,
double2 *y ,  int incy, int n1 )
{
	int tid = blockIdx.x * thread_seg  + threadIdx.x;
    
	double2 res;
    MAGMA_Z_SET2REAL(res, 0.0f);
	 
	if(tid < m)
	{
        #pragma unroll
		for (int i=0; i< tid; i++)
		{
			res +=  A[tid + i * lda] * x[i];
			
		}

		y[tid] = beta * y[tid] + alpha * (res);
	}
}	


__global__ void zhemv_tesla2_kernel( int m , double2 alpha ,  double2 *A , int lda ,  double2 *x, int incx , double2 beta ,
double2 *y ,  int incy, int n1 )

{
	unsigned int tx = threadIdx.x;

	__shared__ double2 sdata[threadSize];
	

	double2 res;
    MAGMA_Z_SET2REAL(res, 0.0f);
	double2 zero;
    MAGMA_Z_SET2REAL(zero, 0.0f);
    
	int m1 = ((m - blockIdx.y)/threadSize) * threadSize;

	for(int i=blockIdx.y; i< (m1 + blockIdx.y); i+= threadSize)
	{
		res +=  conjugate(A[tx + i + lda * blockIdx.y])  * x[tx + i];
	}

	
	if(m > (m1 + blockIdx.y))
	{
		if( (tx + m1 + blockIdx.y) <  m )
		{
			res  += conjugate (A[tx + m1 + blockIdx.y + lda *blockIdx.y]) * x[tx + m1 + blockIdx.y];
		}
		else 
		{
			res  += zero;
		}
	}	

    sdata[tx] = res;
	__syncthreads();
    

	if(tx < 64) 
	{
		sdata[tx] += sdata[tx + 64];
	}
    __syncthreads();
	
    if(tx < 32) 
	{
		sdata[tx] += sdata[tx + 32];
        sdata[tx] += sdata[tx + 16];
        sdata[tx] += sdata[tx +  8];
        sdata[tx] += sdata[tx +  4];
        sdata[tx] += sdata[tx +  2];
        sdata[tx] += sdata[tx +  1];
    } 
	if( tx == 0 ) 
	{
		y[blockIdx.y] = alpha * sdata[0] + y[blockIdx.y]; 		

	}
}


extern "C" void mzhemv_tesla(char side , char uplo , int m , double2 alpha ,  double2 *A , int lda , 
 double2 *X , int incx , double2 beta , double2 *Y , int incy )

{

    int blocks = (m-1)/thread_seg + 1;
	int n1 = (m / thread_seg ) * thread_seg;


	dim3 grid1( blocks, 1, 1);
	dim3 threads1(thread_seg, 1, 1);

	zhemv_tesla1_kernel <<< grid1, threads1 >>> (m , alpha ,  A , lda , X , incx, beta , Y, incy, n1);


	n1 = (m/ threadSize) * threadSize;
	dim3 grid2(  1, m,  1);
	dim3 threads2(threadSize, 1, 1);

	zhemv_tesla2_kernel <<< grid2, threads2 >>> (m , alpha ,  A , lda , X , incx, beta , Y, incy, n1);

  
}


/*
Interface ..................................
*/

extern "C" void  magmablas_zhemv_tesla (char uplo , int m , double2 alpha ,  double2 *A , int lda ,  double2 *X , int incx , double2 beta , double2 *Y , int incy )
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
	mzhemv_tesla (side, uplo , m , alpha , A , lda , X , incx , beta , Y , incy );

}
