/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> s d c
*/

#ifndef MAGMASPARSE_Z_H
#define MAGMASPARSE_Z_H

#include "magma_types.h"
#include "magmasparse_types.h"

#define PRECISION_z


#ifdef __cplusplus
extern "C" {
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA_SPARSE Auxiliary functions
*/
magma_int_t read_z_csr_from_binary( magma_int_t* n_row, magma_int_t* n_col, 
                                    magma_int_t* nnz, magmaDoubleComplex **val, 
                                    magma_int_t **col, magma_int_t **row, 
                                    const char * filename);

magma_int_t read_z_csr_from_mtx( magma_int_t* n_row, magma_int_t* n_col, 
                                 magma_int_t* nnz, magmaDoubleComplex **val, 
                                 magma_int_t **col, magma_int_t **row, 
                                 const char *filename);

magma_int_t write_z_csr_mtx(magma_int_t n_row, magma_int_t n_col, magma_int_t nnz, 
                            magmaDoubleComplex **val, magma_int_t **col, 
                            magma_int_t **row, const char *filename);

magma_int_t cout_z_csr_mtx(magma_int_t n_row, magma_int_t n_col, magma_int_t nnz, 
                           magmaDoubleComplex **val, magma_int_t **col, 
                           magma_int_t **row);



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
                       double rtol );

/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA_SPARSE utility function definitions
*/

#ifdef __cplusplus
}
#endif

#undef PRECISION_z
#endif /* MAGMASPARSE_Z_H */
