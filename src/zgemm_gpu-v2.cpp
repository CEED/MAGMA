/*
    -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

       @precisions normal z -> s d c

*/

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include "magma.h"
#include "magmablas.h"

extern "C" void 
magmablas_zgemm(char TRANSA, char TRANSB, int m , int n , int k , 
		double2 alpha, const double2 *A, int lda, const double2 *B, 
		int ldb, double2 beta, double2 *C, int ldc)
{
/*  -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       August 2009

    Purpose
    =======

    ZGEMM  performs one of the matrix-matrix operations
      C := alpha*op( A )*op( B ) + beta*C,
    where  op( X ) is one of
      op( X ) = X   or   op( X ) = X',

    alpha and beta are scalars, and A, B and C are matrices, with op( A )
    an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.

    Parameters
    ==========

    TRANSA   CHARACTER*1.
             On entry, TRANSA specifies the form of op( A ) to be used in
             the matrix multiplication as follows:
                TRANSA = 'N' or 'n',  op( A ) = A.
                TRANSA = 'T' or 't',  op( A ) = A'.
                TRANSA = 'C' or 'c',  op( A ) = A'.
             Unchanged on exit.

    TRANSB   CHARACTER*1.
             On entry, TRANSB specifies the form of op( B ) to be used in
             the matrix multiplication as follows:
                TRANSB = 'N' or 'n',  op( B ) = B.
                TRANSB = 'T' or 't',  op( B ) = B'.
                TRANSB = 'C' or 'c',  op( B ) = B'.
             Unchanged on exit.

    M        INTEGER.
             On entry,  M  specifies  the number  of rows  of the  matrix
             op( A )  and of the  matrix  C.  M  must  be at least  zero.
             Unchanged on exit.

    N        INTEGER.
             On entry,  N  specifies the number  of columns of the matrix
             op( B ) and the number of columns of the matrix C. N must be
             at least zero.
             Unchanged on exit.

    K        INTEGER.
             On entry,  K  specifies  the number of columns of the matrix
             op( A ) and the number of rows of the matrix op( B ). K must
             be at least  zero.
             Unchanged on exit.

    ALPHA    SINGLE PRECISION.
             On entry, ALPHA specifies the scalar alpha.
             Unchanged on exit.

    A        SINGLE PRECISION array of DIMENSION ( LDA, ka ), where ka is
             k  when  TRANSA = 'N' or 'n',  and is  m  otherwise.
             Before entry with  TRANSA = 'N' or 'n',  the leading  m by k
             part of the array  A  must contain the matrix  A,  otherwise
             the leading  k by m  part of the array  A  must contain  the
             matrix A.
             Unchanged on exit.

    LDA      INTEGER.
             On entry, LDA specifies the first dimension of A as declared
             in the calling (sub) program. When  TRANSA = 'N' or 'n' then
             LDA must be at least  max( 1, m ), otherwise  LDA must be at
             least  max( 1, k ).
             Unchanged on exit.

    B        SINGLE PRECISION array of DIMENSION ( LDB, kb ), where kb is
             n  when  TRANSB = 'N' or 'n',  and is  k  otherwise.
             Before entry with  TRANSB = 'N' or 'n',  the leading  k by n
             part of the array  B  must contain the matrix  B,  otherwise
             the leading  n by k  part of the array  B  must contain  the
             matrix B.
             Unchanged on exit.

    LDB      INTEGER.
             On entry, LDB specifies the first dimension of B as declared
             in the calling (sub) program. When  TRANSB = 'N' or 'n' then
             LDB must be at least  max( 1, k ), otherwise  LDB must be at
             least  max( 1, n ).
             Unchanged on exit.

    BETA     SINGLE PRECISION.
             On entry,  BETA  specifies the scalar  beta.  When  BETA  is
             supplied as zero then C need not be set on input.
             Unchanged on exit.

    C        SINGLE PRECISION array of DIMENSION ( LDC, n ).
             Before entry, the leading  m by n  part of the array  C must
             contain the matrix  C,  except when  beta  is zero, in which
             case C need not be set on entry.
             On exit, the array  C  is overwritten by the  m by n  matrix
             ( alpha*op( A )*op( B ) + beta*C ).

    LDC      INTEGER.
             On entry, LDC specifies the first dimension of C as declared
             in  the  calling  (sub)  program.   LDC  must  be  at  least
             max( 1, m ).
             Unchanged on exit.

    Level 3 Blas routine.
    =====================================================================    */
 
    if(m==0 || n==0  || ( ( alpha==0 || k==0 ) && beta ==1 ) ){
      return ;
    }
    
    int cutoff = 512 ;

    /* Allocate memory for the result */
    double2 *Cc, gpu_perf1, gpu_perf2;
    TimeStruct start, end;

    cublasAlloc(m*n, sizeof(double2), (void**)&Cc);
    cudaMemcpy2D(Cc, m*sizeof(double2),
		 C, ldc*sizeof(double2),
		 sizeof(double2)*m, n,
		 cudaMemcpyDeviceToDevice);
    start = get_current_time();

    if( alpha == 0.0){
      if( beta == 0.0){
	magmablas_zgemm_kernel_ab_0( C,A,B, m, n,k,lda,ldb, ldc, alpha, beta);
	goto L40; //return ;
      }	
      else{
	magmablas_zgemm_kernel_a_0( C,A,B, m, n,k,lda,ldb, ldc, alpha, beta);
	goto L40; //return ;
      }		
    }

    if(ldc < m ) return ;
    TRANSA = toupper( TRANSA) ; 
    TRANSB = toupper( TRANSB) ; 
    if(TRANSA=='N' ){
      if(TRANSB=='N')
	{ 
	  if(lda < m ) return ;
	  if(ldb < k ) return ;
	  /*==================================================================
	    ============  C = alpha * A * B + beta * C =======================
	    =================================================================*/
	  if( m > cutoff && n > cutoff ){
	    if( m % 64 == 0 && n%16 == 0 && k%16 == 0 ) 
	      magmablas_zgemm_kernel_N_N_64_16_16_16_4_special(C,A,B, m, n,k,
							       lda,ldb,ldc, 
							       alpha, beta);
	    else
	      magmablas_zgemm_kernel_N_N_64_16_16_16_4(C,A,B, m, n, k,
						       lda, ldb, ldc, 
						       alpha, beta);
	  }
	  else{
	    if( m % 64 == 0 && n%16 == 0 && k%16 == 0 ) 
	      cublasZgemm(TRANSA, TRANSB, m, n, k, alpha, A, lda, 
			  B, ldb, beta, C, ldc );
	    else
	      magmablas_zgemm_kernel_N_N_64_16_16_16_4(C,A,B, m, n, k, lda,
						       ldb, ldc, alpha, beta);
	    
	  }
	}
      
      else{ 
	if(lda < m ) return ;
	if(ldb < n ) return ;
	/*====================================================================
	  ==============  C = alpha * A * B^T + beta * C =====================
	  ===================================================================*/
	if( m > cutoff && n > cutoff ){
	  if( m%64 == 0 && n %16 ==0 && k%4==0) 
	    magmablas_zgemm_kernel_N_T_64_16_4_16_4( C,A,B, m, n,k,lda,
						     ldb, ldc, alpha, beta);
	  else 
	    magmablas_zgemm_kernel_N_T_64_16_4_16_4( C,A,B, m, n,k,lda,
						     ldb, ldc, alpha, beta);
	}
	else{
	  if( m%64 == 0 && n %16 ==0 && k%4==0) 
	    cublasZgemm(TRANSA, TRANSB, m, n, k, alpha, A, lda, B, 
			ldb, beta, C, ldc );
	  else 
	    magmablas_zgemm_kernel_N_T_64_16_4_16_4(C,A,B, m, n,k,lda,
						    ldb, ldc, alpha, beta);
	}
      }
    }
    else{
      if(TRANSB=='N'){
	if(lda < k ) return ;
	if(ldb < k ) return ;
	/*====================================================================
	  ==============  C = alpha * A^T * B + beta * C =====================
	  ==================================================================*/
	if(m>cutoff && n > cutoff){
	  if( m%32 == 0 && n %32 ==0 && k%8==0) 
	    magmablas_zgemm_kernel_T_N_32_32_8_8_8(C, A, B, m, n, k, lda,
						   ldb, ldc, alpha, beta);
	  
	  else
	    magmablas_zgemm_kernel_T_N_32_32_8_8_8(C, A, B, m, n, k, lda,
						   ldb, ldc, alpha, beta);
	}
	else{
	  if( m%32 == 0 && n %32 ==0 && k%8==0) 
	    cublasZgemm(TRANSA, TRANSB, m, n, k, alpha, A, lda, B, ldb, 
			beta, C, ldc );
	  else
	    magmablas_zgemm_kernel_T_N_32_32_8_8_8(C, A, B, m, n, k, lda,
						   ldb, ldc, alpha, beta);
	}	
      }
      else{
	if(lda < k) return ;
	if(ldb < n ) return ;
	/*=====================================================================
	  ===============  C = alpha * A^T* B^T + beta * C ====================
	  ===================================================================*/
	if( m > cutoff && n > cutoff ){
	  if( m%64 == 0 && n %16 ==0 && k%16==0) 
	    magmablas_zgemm_kernel_T_T_64_16_16_16_4_v2(C, B, A, n, m, k, ldb,
							lda, ldc, alpha, beta);
	  else 
	    magmablas_zgemm_kernel_T_T_64_16_16_16_4(C, B, A, n, m, k, ldb,
						     lda, ldc, alpha, beta);
	}
	else{
	  if( m%64 == 0 && n %16 ==0 && k%16==0) 
	    cublasZgemm(TRANSA, TRANSB, m, n, k, alpha, A, lda, B, ldb, 
			beta, C, ldc );
	  else 
	    magmablas_zgemm_kernel_T_T_64_16_16_16_4(C, B, A, n, m, k, ldb,
						      lda, ldc, alpha, beta);
	}
	
      }
    }
    
 L40:

    /* 1. Current implementation */
    end = get_current_time();
    gpu_perf1 = 2.*m*n*k/(1000000*GetTimerValue(start,end));
    printf("%5d  %5d  %5d  %6.2f ", m, n, k, gpu_perf1);

    /* 2. CUBLAS */
    cudaMemcpy2D(C, ldc*sizeof(double2),
                 Cc,  m*sizeof(double2),
                 sizeof(double2)*m, n,
                 cudaMemcpyDeviceToDevice);
    
    start = get_current_time();
    cublasZgemm(TRANSA, TRANSB, m, n, k, alpha, A, lda,
		B, ldb, beta, C, ldc );
    end = get_current_time();
    gpu_perf2 = 2.*m*n*k/(1000000*GetTimerValue(start,end));
    printf("%6.2f ", gpu_perf2);
    if (gpu_perf1 > gpu_perf2)
      printf(" +%6.2f\n", gpu_perf1-gpu_perf2);
    else
      printf("        -%6.2f\n", gpu_perf2-gpu_perf1);

    /* 3. */
    /*
    magmablas_zgemm_kernel_N_N_64_16_16_16_4_special(C,A,B, m, n,k,
						     lda,ldb,ldc,
						     alpha, beta);
    magmablas_zgemm_kernel_N_N_64_16_16_16_4(C,A,B, m, n, k,
					     lda, ldb, ldc,
					     alpha, beta);

    magmablas_zgemm_kernel_N_T_64_16_4_16_4( C,A,B, m, n,k,lda,
					     ldb, ldc, alpha, beta);
    magmablas_zgemm_kernel_T_N_32_32_8_8_8(C, A, B, m, n, k, lda,
					   ldb, ldc, alpha, beta);

    magmablas_zgemm_kernel_T_T_64_16_16_16_4_v2(C, B, A, n, m, k, ldb,
						lda, ldc, alpha, beta);
    magmablas_zgemm_kernel_T_T_64_16_16_16_4(C, B, A, n, m, k, ldb,
					     lda, ldc, alpha, beta);

    */
    cublasFree(Cc);
} 
