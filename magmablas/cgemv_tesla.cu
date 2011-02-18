/*
    -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

*/
#include "common_magma.h"

#define magmablas_cgemv_tesla magmablas_cgemv

extern "C" void
magmablas_cgemv_tesla(char trans, int m, int n, float2 alpha, float2 *A, int lda, float2 *x, int incx, float2 beta, float2 *y, int incy) 
{
    cublasCgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}
