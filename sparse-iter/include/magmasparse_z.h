/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> s d c
*/

#ifndef MAGMA_Z_H
#define MAGMA_Z_H

#include "magma_types.h"

#define PRECISION_z

#ifdef __cplusplus
extern "C" {
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA_SPARSE Auxiliary functions
*/


/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA_SPARSE function definitions / Data on CPU
*/


/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA_SPARSE function definitions / Data on CPU / Multi-GPU
*/

/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA_SPARSE function definitions / Data on GPU
*/
magma_int_t magma_zcg( magma_int_t dofs, magma_int_t & num_of_iter,
                       magmaDoubleComplex *x, magmaDoubleComplex *b,
                       magmaDoubleComplex *d_A, magma_int_t *d_I, magma_int_t *d_J,
                       magmaDoubleComplex *dwork,
                       double rtol = RTOLERANCE );

/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE utility function definitions
*/

#ifdef __cplusplus
}
#endif

#undef PRECISION_z
#endif /* MAGMA_Z_H */
