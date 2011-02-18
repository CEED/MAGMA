/*
    -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

       @precisions normal z -> z
       
*/
#include "common_magma.h"

#define magmablas_zgemv_tesla magmablas_zgemv

extern "C" void
magmablas_zgemv_tesla(char trans, int m, int n, double2 alpha, double2 *A, int lda, double2 *x, int incx, double2 beta, double2 *y, int incy) 
{
    cublasZgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}
