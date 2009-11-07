/*
    -- MAGMA (version 0.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       June 2009
*/

#include "cublas.h"
#include "magma.h"

extern "C" void
magmablas_ztrmm(char side, char uplo, char transa, char diag, 
		int m, int n, double2 alpha, 
		double2 *A, int lda,
		double2 *B, int ldb) {
/*  -- MAGMA (version 0.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       June 2009

    Purpose   
    =======   

    ZTRMM  performs one of the matrix-matrix operations   

       B := alpha*op( A )*B,   or   B := alpha*B*op( A )   

    where  alpha  is a scalar,  B  is an m by n matrix,  A  is a unit, or   
    non-unit,  upper or lower triangular matrix  and  op( A )  is one  of 

       op( A ) = A   or   op( A ) = A'   or   op( A ) = conjg( A' ).   

    Parameters   
    ==========   

    SIDE   - CHARACTER   
             On entry,  SIDE specifies whether  op( A ) multiplies B from 
             the left or right as follows:   
                SIDE = 'L' or 'l'   B := alpha*op( A )*B.   
                SIDE = 'R' or 'r'   B := alpha*B*op( A ).   
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

    M      - INTEGER
             On entry, M specifies the number of rows of B. M must be at 
             least zero.   
             Unchanged on exit.   

    N      - INTEGER   
             On entry, N specifies the number of columns of B.  N must be 
             at least zero.   
             Unchanged on exit.   

    ALPHA  - DOUBLE COMPLEX  
             On entry,  ALPHA specifies the scalar  alpha. When  alpha is 
             zero then  A is not referenced and  B need not be set before 
             entry.   
             Unchanged on exit.   

    A      - DOUBLE COMPLEX array of DIMENSION ( LDA, k ), where k is m 
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

    LDA    - INTEGER
             On entry, LDA specifies the first dimension of A as declared 
             in the calling (sub) program.  When  SIDE = 'L' or 'l'  then 
             LDA  must be at least  max( 1, m ),  when  SIDE = 'R' or 'r' 
             then LDA must be at least max( 1, n ).   
             Unchanged on exit.   

    B      - DOUBLE COMPLEX array of DIMENSION ( LDB, n ).   
             Before entry,  the leading  m by n part of the array  B must 
             contain the matrix  B,  and  on exit  is overwritten  by the 
             transformed matrix.   

    LDB    - INTEGER
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

    double2 *a = (double2*)malloc(k*k * sizeof(double2));
    double2 *b = (double2*)malloc(m*n * sizeof(double2));

    cublasGetMatrix(k, k, sizeof(double2), A, lda, a, k);
    cublasGetMatrix(m, n, sizeof(double2), B, ldb, b, m);

    ztrmm_(&side, &uplo, &transa, &diag,
           &m, &n, &alpha, a, &k, b, &m);     

    cublasSetMatrix(m, n, sizeof(double2), b, m, B, ldb);
   
    free(a);
    free(b);
}
