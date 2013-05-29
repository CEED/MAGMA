/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions mixed zc -> ds

*/
#include "common_magma.h"

extern "C" __global__ void
zcaxpycp_special(magmaFloatComplex *R, magmaDoubleComplex *X, magma_int_t M, magmaDoubleComplex *B,magmaDoubleComplex *W )
{
    const magma_int_t ibx = blockIdx.x * 64;
    const magma_int_t idt = threadIdx.x;
    X += ibx+idt;
    R += ibx+idt;
    B += ibx+idt;
    W += ibx+idt;
    X[0] = MAGMA_Z_ADD( X[0], cuComplexFloatToDouble(R[0]) );
    W[0] = B[0];
}

extern "C" __global__ void
zaxpycp_special(magmaDoubleComplex *R, magmaDoubleComplex *X, magma_int_t M, magmaDoubleComplex *B)
{
    const magma_int_t ibx = blockIdx.x * 64;
    const magma_int_t idt = threadIdx.x;
    X += ibx+idt;
    R += ibx+idt;
    B += ibx+idt;
    X[0] = MAGMA_Z_ADD( X[0], R[0] );
    R[0] = B[0];
}

extern "C" __global__ void
zcaxpycp_generic(magmaFloatComplex *R, magmaDoubleComplex *X, magma_int_t M, magmaDoubleComplex *B,magmaDoubleComplex *W )
{
    const magma_int_t ibx = blockIdx.x * 64;
    const magma_int_t idt = threadIdx.x;
    if( ( ibx + idt ) < M ) {
        X += ibx+idt;
        R += ibx+idt;
        B += ibx+idt;
        W += ibx+idt;
    }
    else{
        X +=(M-1);
        R +=(M-1);
        B +=(M-1);
        W +=(M-1);
    }
    X[0] = MAGMA_Z_ADD( X[0], cuComplexFloatToDouble( R[0] ) );
    W[0] = B[0];
}

extern "C" __global__ void
zaxpycp_generic(magmaDoubleComplex *R, magmaDoubleComplex *X, magma_int_t M, magmaDoubleComplex *B)
{
    const magma_int_t ibx = blockIdx.x * 64;
    const magma_int_t idt = threadIdx.x;
    if( ( ibx + idt ) < M ) {
        X += ibx+idt;
        R += ibx+idt;
        B += ibx+idt;
    }
    else{
        X +=(M-1);
        R +=(M-1);
        B +=(M-1);
    }
    X[0] = MAGMA_Z_ADD( X[0], R[0] );
    R[0] = B[0];
}


extern "C" void
magmablas_zcaxpycp(magmaFloatComplex *R, magmaDoubleComplex *X, magma_int_t M, magmaDoubleComplex *B, magmaDoubleComplex *W)
{
    dim3 threads( 64, 1 );
    dim3 grid(M/64+(M%64!=0),1);
    if( M %64 == 0 ) {
        zcaxpycp_special <<< grid, threads, 0, magma_stream >>> ( R, X, M, B, W) ;
    }
    else{
        zcaxpycp_generic <<< grid, threads, 0, magma_stream >>> ( R, X, M, B, W) ;
    }
}

extern "C" void
magmablas_zaxpycp(magmaDoubleComplex *R, magmaDoubleComplex *X, magma_int_t M, magmaDoubleComplex *B)
{
    dim3 threads( 64, 1 );
    dim3 grid(M/64+(M%64!=0),1);
    if( M %64 == 0 ) {
        zaxpycp_special <<< grid, threads, 0, magma_stream >>> ( R, X, M, B) ;
    }
    else{
        zaxpycp_generic <<< grid, threads, 0, magma_stream >>> ( R, X, M, B) ;
    }
}
