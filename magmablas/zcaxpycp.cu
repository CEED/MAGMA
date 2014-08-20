/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions mixed zc -> ds

*/
#include "common_magma.h"

#define NB 64

// adds   x += r (including conversion to double)  --and--
// copies w = b
// each thread does one index, x[i] and w[i]
__global__ void
zcaxpycp_kernel(
    int m, magmaFloatComplex *r, magmaDoubleComplex *x,
    const magmaDoubleComplex *b, magmaDoubleComplex *w )
{
    const int i = threadIdx.x + blockIdx.x*NB;
    if ( i < m ) {
        x[i] = MAGMA_Z_ADD( x[i], cuComplexFloatToDouble( r[i] ) );
        w[i] = b[i];
    }
}


// adds   x += r  --and--
// copies r = b
// each thread does one index, x[i] and r[i]
__global__ void
zaxpycp_kernel(
    int m, magmaDoubleComplex *r, magmaDoubleComplex *x,
    const magmaDoubleComplex *b)
{
    const int i = threadIdx.x + blockIdx.x*NB;
    if ( i < m ) {
        x[i] = MAGMA_Z_ADD( x[i], r[i] );
        r[i] = b[i];
    }
}


// adds   x += r (including conversion to double)  --and--
// copies w = b
extern "C" void
magmablas_zcaxpycp(
    magma_int_t m, magmaFloatComplex *r, magmaDoubleComplex *x,
    const magmaDoubleComplex *b, magmaDoubleComplex *w)
{
    dim3 threads( NB );
    dim3 grid( (m + NB - 1)/NB );
    zcaxpycp_kernel <<< grid, threads, 0, magma_stream >>> ( m, r, x, b, w );
}


// adds   x += r  --and--
// copies r = b
extern "C" void
magmablas_zaxpycp(
    magma_int_t m, magmaDoubleComplex *r, magmaDoubleComplex *x,
    const magmaDoubleComplex *b)
{
    dim3 threads( NB );
    dim3 grid( (m + NB - 1)/NB );
    zaxpycp_kernel <<< grid, threads, 0, magma_stream >>> ( m, r, x, b );
}
