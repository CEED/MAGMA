/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> s d c
       @author Mark Gates
*/

#ifndef MAGMA_ZGEHRD_H
#define MAGMA_ZGEHRD_H

#include "magma.h"

#ifdef __cplusplus
extern "C" {
#endif

struct zgehrd_data
{
    int ngpu;
    
    magma_int_t ldda;
    magma_int_t ldv;
    magma_int_t ldvd;
    
    cuDoubleComplex *A    [ MagmaMaxGPUs ];  // ldda*nlocal
    cuDoubleComplex *V    [ MagmaMaxGPUs ];  // ldv *nb, whole panel
    cuDoubleComplex *Vd   [ MagmaMaxGPUs ];  // ldvd*nb, block-cyclic
    cuDoubleComplex *Y    [ MagmaMaxGPUs ];  // ldda*nb
    cuDoubleComplex *W    [ MagmaMaxGPUs ];  // ldda*nb
    cuDoubleComplex *Ti   [ MagmaMaxGPUs ];  // nb*nb
    
    magma_stream_t streams[ MagmaMaxGPUs ];
};

#ifdef __cplusplus
}
#endif

#endif        //  #ifndef MAGMA_ZGEHRD_H
