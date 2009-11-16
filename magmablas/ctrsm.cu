/*
    -- MAGMA (version 0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2009
*/

#include "cublas.h"
#include "magma.h"

extern "C" void
magmablas_ctrsm(char side, char uplo, char transa, char diag, 
		int m, int n, float2 alpha, 
		float2 *A, int lda,
		float2 *B, int ldb) {
/*  -- MAGMA (version 0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2009

    Purpose   
    =======   

    CTRSM  solves one of the matrix equations   

       op( A )*X = alpha*B,   or   X*op( A ) = alpha*B,   

    where alpha is a scalar, X and B are m by n matrices, A is a unit, or   
    non-unit,  upper or lower triangular matrix  and  op( A )  is one  of 
       op( A ) = A   or   op( A ) = A'   or   op( A ) = conjg( A' ).   
    The matrix X is overwritten on B.   

    Parameters   
    ==========   

    SIDE   - CHARACTER   
             On entry, SIDE specifies whether op( A ) appears on the left
             or right of X as follows:
                SIDE = 'L' or 'l'   op( A )*X = alpha*B.   
                SIDE = 'R' or 'r'   X*op( A ) = alpha*B.   

             Unchanged on exit.   

    UPLO   - CHARACTER
             On entry, UPLO specifies whether the matrix A is an upper or 
             lower triangular matrix as follows:   
                UPLO = 'U' or 'u'   A is an upper triangular matrix.   
                UPLO = 'L' or 'l'   A is a lower triangular matrix.   

             Unchanged on exit.   

    TRANSA - CHARACTER
             On entry, TRANSA specifies the form of op( A ) to be used in 
             the matrix multiplication as follows:   
                TRANSA = 'N' or 'n'   op( A ) = A.   
                TRANSA = 'T' or 't'   op( A ) = A'.   
                TRANSA = 'C' or 'c'   op( A ) = conjg( A' ).   

             Unchanged on exit.   

    DIAG   - CHARACTER
             On entry, DIAG specifies whether or not A is unit triangular 
             as follows:   

                DIAG = 'U' or 'u'   A is assumed to be unit triangular.   
                DIAG = 'N' or 'n'   A is not assumed to be unit   
                                    triangular.   

             Unchanged on exit.   

    M      - INTEGER.   
             On entry, M specifies the number of rows of B. M must be at 
             least zero.   
             Unchanged on exit.   

    N      - INTEGER.   
             On entry, N specifies the number of columns of B.  N must be 
             at least zero.   
             Unchanged on exit.   

    ALPHA  - COMPLEX         .   
             On entry,  ALPHA specifies the scalar  alpha. When  alpha is 
             zero then  A is not referenced and  B need not be set before 
             entry.   
             Unchanged on exit.   

    A      - COMPLEX array of DIMENSION (LDA, k), where k is m 
             when  SIDE = 'L' or 'l'  and is  n  when  SIDE = 'R' or 'r'.   
             Before entry  with  UPLO = 'U' or 'u',  the  leading  k by k 
             upper triangular part of the array  A must contain the upper 
             triangular matrix  and the strictly lower triangular part of 
             A is not referenced.   
             Before entry  with  UPLO = 'L' or 'l',  the  leading  k by k 
             lower triangular part of the array  A must contain the lower 
             triangular matrix  and the strictly upper triangular part of 
             A is not referenced.   
             Note that when  DIAG = 'U' or 'u',  the diagonal elements of 
             A  are not referenced either,  but are assumed to be  unity. 
 
             Unchanged on exit.   

    LDA    - INTEGER.   
             On entry, LDA specifies the first dimension of A as declared  
             in the calling (sub) program.  When  SIDE = 'L' or 'l'  then 
             LDA  must be at least  max( 1, m ),  when  SIDE = 'R' or 'r' 
             then LDA must be at least max( 1, n ).   
             Unchanged on exit.   

    B      - COMPLEX array of DIMENSION ( LDB, n ).   
             Before entry,  the leading  m by n part of the array  B must 
             contain  the  right-hand  side  matrix  B,  and  on exit  is 
             overwritten by the solution matrix  X.   

    LDB    - INTEGER.   
             On entry, LDB specifies the first dimension of B as declared 
             in  the  calling  (sub)  program.   LDB  must  be  at  least 
             max( 1, m ).   
             Unchanged on exit.   

    Level 3 Blas routine.   

    ===================================================================== */

    int k;
    if (side == 'L' || side == 'l')
       k = m;
    else 
       k = n;

    float2 *a = (float2*)malloc(k*k * sizeof(float2));
    float2 *b = (float2*)malloc(m*n * sizeof(float2));

    cublasGetMatrix(k, k, sizeof(float2), A, lda, a, k);
    cublasGetMatrix(m, n, sizeof(float2), B, ldb, b, m);

    if (m>0)
    ctrsm_(&side, &uplo, &transa, &diag,
           &m, &n, &alpha, a, &k, b, &m);     

    cublasSetMatrix(m, n, sizeof(float2), b, m, B, ldb);
   
    free(a);
    free(b);
}
