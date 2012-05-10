/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @author Stan Tomov
       @precisions normal z -> s d c
*/
#include "common_magma.h"
#define PRECISION_z
#include "commonblas.h"

//===========================================================================
//  Set a matrix from CPU to multi-GPUs is 1D block cyclic distribution.
//  The dA arrays are pointers to the matrix data for the corresponding GPUs.
//===========================================================================
extern "C" void
magmablas_zsetmatrix_1D_bcyclic( int m, int n,
                                 cuDoubleComplex  *hA,   int lda,
                                 cuDoubleComplex  *dA[], int ldda,
                                 int num_gpus, int nb )
{
    int i, d, nk, cdevice;

    magma_getdevice( &cdevice );

    for( i = 0; i < n; i += nb ) {
        d = (i/nb) % num_gpus;
        magma_setdevice( d );
        nk = min(nb, n-i);
        magma_zsetmatrix_async( m, nk,
                                hA + i*lda, lda,
                                dA[d] + i/(nb*num_gpus)*nb*ldda, ldda, NULL );
    }

    magma_setdevice( cdevice );
}


//===========================================================================
//  Get a matrix with 1D block cyclic distribution on multiGPUs to the CPU.
//  The dA arrays are pointers to the matrix data for the corresponding GPUs.
//===========================================================================
extern "C" void
magmablas_zgetmatrix_1D_bcyclic( int m, int n,
                                 cuDoubleComplex  *dA[], int ldda,
                                 cuDoubleComplex  *hA,   int lda,
                                 int num_gpus, int nb )
{
    int i, d, nk, cdevice;

    magma_getdevice( &cdevice );

    for( i = 0; i < n; i += nb ) {
        d = (i/nb) % num_gpus;
        magma_setdevice( d );
        nk = min(nb, n-i);
        magma_zgetmatrix_async( m, nk,
                                dA[d] + i/(nb*num_gpus)*nb*ldda, ldda,
                                hA + i*lda, lda, NULL );
    }

    magma_setdevice( cdevice );
}
