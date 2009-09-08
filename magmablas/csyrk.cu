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
magmablas_csyrk(char uplo, char trans, int n, int k, float2 alpha, 
            	float2 *A, int lda, float2 beta, float2 *C, int ldc){
/*  -- MAGMA (version 0.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       June 2009

    Purpose   
    =======   

    CSYRK  performs one of the symmetric rank k operations   
       C := alpha*A*A' + beta*C,   
    or   
       C := alpha*A'*A + beta*C,   
    where  alpha and beta  are scalars,  C is an  n by n symmetric matrix 
    and  A  is an  n by k  matrix in the first case and a  k by n  matrix 
    in the second case.   

    Parameters   
    ==========   

    UPLO   - CHARACTER  
             On  entry,   UPLO  specifies  whether  the  upper  or  lower 
             triangular  part  of the  array  C  is to be  referenced  as 
             follows:   
                UPLO = 'U' or 'u'   Only the  upper triangular part of  C 
                                    is to be referenced.   
                UPLO = 'L' or 'l'   Only the  lower triangular part of  C 
                                    is to be referenced.   

             Unchanged on exit.   

    TRANS  - CHARACTER   
             On entry,  TRANS  specifies the operation to be performed as 
             follows:   
                TRANS = 'N' or 'n'   C := alpha*A*A' + beta*C.   
                TRANS = 'T' or 't'   C := alpha*A'*A + beta*C.   

             Unchanged on exit.   

    N      - INTEGER.   
             On entry,  N specifies the order of the matrix C.  N must be   
             at least zero.   
             Unchanged on exit.   

    K      - INTEGER.   
             On entry with  TRANS = 'N' or 'n',  K  specifies  the number 
             of  columns   of  the   matrix   A,   and  on   entry   with 
             TRANS = 'T' or 't',  K  specifies  the number of rows of the 
             matrix A.  K must be at least zero.   
             Unchanged on exit.   

    ALPHA  - COMPLEX.
             On entry, ALPHA specifies the scalar alpha.   
             Unchanged on exit.   

    A      - COMPLEX array of DIMENSION ( LDA, ka ), where ka is 
             k  when  TRANS = 'N' or 'n',  and is  n  otherwise.   
             Before entry with  TRANS = 'N' or 'n',  the  leading  n by k 
             part of the array  A  must contain the matrix  A,  otherwise 
             the leading  k by n  part of the array  A  must contain  the 
             matrix A.   
             Unchanged on exit.   

    LDA    - INTEGER.   
             On entry, LDA specifies the first dimension of A as declared 
             in  the  calling  (sub)  program.   When  TRANS = 'N' or 'n' 
             then  LDA must be at least  max( 1, n ), otherwise  LDA must 
             be at least  max( 1, k ).   
             Unchanged on exit.   

    BETA   - COMPLEX.   
             On entry, BETA specifies the scalar beta.   
             Unchanged on exit.   

    C      - COMPLEX array of DIMENSION ( LDC, n ).   
             Before entry  with  UPLO = 'U' or 'u',  the leading  n by n 
             upper triangular part of the array C must contain the upper 
             triangular part  of the  symmetric matrix  and the strictly 
             lower triangular part of C is not referenced.  On exit, the 
             upper triangular part of the array  C is overwritten by the 
             upper triangular part of the updated matrix.   
             Before entry  with  UPLO = 'L' or 'l',  the leading  n by n 
             lower triangular part of the array C must contain the lower 
             triangular part  of the  symmetric matrix  and the strictly 
             upper triangular part of C is not referenced.  On exit, the 
             lower triangular part of the array  C is overwritten by the 
             lower triangular part of the updated matrix.   

    LDC    - INTEGER.   
             On entry, LDC specifies the first dimension of C as declared 
             in  the  calling  (sub)  program.   LDC  must  be  at  least 
             max( 1, n ).   
             Unchanged on exit.   

    Level 3 Blas routine.   

    ===================================================================== */

    int ka, ldamin;
    if (trans == 'N' || trans == 'n')
       ka = k, ldamin = n;
    else
       ka = n, ldamin = k;


    float2 *a = (float2*)malloc(ka*ldamin * sizeof(float2));
    float2 *c = (float2*)malloc(n*n * sizeof(float2));

    cublasGetMatrix(ldamin, ka, sizeof(float2), A, lda, a, ldamin);
    cublasGetMatrix(n, n, sizeof(float2), C, ldc, c, n);

    csyrk_(&uplo, &trans, &n, &k, &alpha, a, &ldamin, &beta, c, &n);

    cublasSetMatrix(n, n, sizeof(float2), c, n, C, ldc);

    free(a);
    free(c);
}

