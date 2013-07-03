/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> s d c
       @author Hartwig Anzt
*/

#ifndef MAGMASPARSE_Z_H
#define MAGMASPARSE_Z_H

#include "magma_types.h"
#include "magmasparse_types.h"

#define PRECISION_z


/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA_SPARSE Matrix Descriptors
*/
/* CSR Matrix descriptor */
typedef struct {
    int type;

    magma_int_t   m;
    magma_int_t   n;
    magma_int_t nnz;

    magmaDoubleComplex *d_val;
    magma_int_t *d_rowptr;
    magma_int_t *d_colind;

} magma_zmatrix_t;


/* BCSR Matrix descriptor */
typedef struct {
    int type;

    magma_int_t   rows_block;
    magma_int_t   cols_block;

    magma_int_t nrow_blocks;
    magma_int_t  nnz_blocks;

    magmaDoubleComplex *d_val;
    magma_int_t *d_rowptr;
    magma_int_t *d_colind;

} magma_zbcsr_t;


#ifdef __cplusplus
extern "C" {
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA_SPARSE Auxiliary functions
*/
magma_int_t 
read_z_csr_from_binary( magma_int_t* n_row, magma_int_t* n_col, 
                        magma_int_t* nnz, magmaDoubleComplex **val, 
                        magma_int_t **row, magma_int_t **col,
                        const char * filename);

magma_int_t 
read_z_csr_from_mtx(    magma_storage_t *type, magma_storage_t *location,
                        magma_int_t* n_row, magma_int_t* n_col, 
                        magma_int_t* nnz, magmaDoubleComplex **val, 
                        magma_int_t **row, magma_int_t **col, 
                        const char *filename);

magma_int_t 
magma_z_csr_mtx(        magma_z_sparse_matrix *A, const char *filename );

magma_int_t 
write_z_csr_mtx(        magma_int_t n_row, magma_int_t n_col, magma_int_t nnz, 
                        magmaDoubleComplex **val, magma_int_t **row, 
                        magma_int_t **col, magma_major_t MajorType,
                        const char *filename );

magma_int_t 
print_z_csr(            magma_int_t n_row, magma_int_t n_col, magma_int_t nnz, 
                        magmaDoubleComplex **val, magma_int_t **row, 
                        magma_int_t **col );

magma_int_t 
print_z_csr_mtx(        magma_int_t n_row, magma_int_t n_col, magma_int_t nnz, 
                        magmaDoubleComplex **val, magma_int_t **row, 
                        magma_int_t **col, magma_major_t MajorType );

magma_int_t 
print_z_csr_matrix(    magma_int_t n_row, magma_int_t n_col, magma_int_t nnz, 
                       magmaDoubleComplex **val, magma_int_t **row, 
                       magma_int_t **col );


magma_int_t 
z_transpose_csr(        magma_int_t n_rows, magma_int_t n_cols, 
                        magma_int_t nnz, magmaDoubleComplex *val, 
                        magma_int_t *row, magma_int_t *col, 
                        magma_int_t *new_n_rows, magma_int_t *new_n_cols, 
                        magma_int_t *new_nnz, magmaDoubleComplex **new_val, 
                        magma_int_t **new_row, magma_int_t **new_col );

magma_int_t 
magma_z_mtransfer(      magma_z_sparse_matrix A, magma_z_sparse_matrix *B, 
                        magma_location_t src, magma_location_t dst );

magma_int_t 
magma_z_vtransfer(      magma_z_vector x, magma_z_vector *y, 
                        magma_location_t src, magma_location_t dst );

magma_int_t 
magma_z_mconvert(       magma_z_sparse_matrix A, magma_z_sparse_matrix *B, 
                        magma_storage_t old_format, magma_storage_t new_format );

magma_int_t
magma_z_vinit(          magma_z_vector *x, 
                        magma_location_t memory_location,
                        magma_int_t num_rows, 
                        magmaDoubleComplex values );

magma_int_t
magma_z_vvisu(          magma_z_vector x, 
                        magma_int_t offset, 
                        magma_int_t displaylength );

magma_int_t
magma_z_vread(          magma_z_vector *x, 
                        magma_int_t length,
                        char * filename );

magma_int_t
magma_z_mvisu(          magma_z_sparse_matrix );

/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA_SPARSE function definitions / Data on CPU
*/


/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA_SPARSE function definitions / Data on CPU / Multi-GPU
*/

/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA_SPARSE function definitions / Data on GPU
*/
/*magma_int_t magma_zcg( magma_int_t dofs, magma_int_t & num_of_iter,
                       magmaDoubleComplex *x, magmaDoubleComplex *b,
                       magmaDoubleComplex *d_A, magma_int_t *d_I, magma_int_t *d_J,
                       magmaDoubleComplex *dwork,
                       double rtol );*/
magma_int_t magma_zcg( magma_z_sparse_matrix A, magma_z_vector b, magma_z_vector *x,  
                       magma_solver_parameters *solver_par );

magma_int_t magma_zgmres(
                       magma_z_sparse_matrix A, magma_z_vector b, magma_z_vector *x,  
                       magma_solver_parameters *solver_par );

/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA_SPARSE utility function definitions
*/

magma_int_t
magma_z_spmv(     magmaDoubleComplex alpha, magma_z_sparse_matrix A, 
                  magma_z_vector x, magmaDoubleComplex beta, magma_z_vector y );

/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA_SPARSE BLAS function definitions
*/
magma_int_t 
magma_zgecsrmv(        const char *transA,
                       magma_int_t m, magma_int_t n,
                       magmaDoubleComplex alpha,
                       magmaDoubleComplex *d_val,
                       magma_int_t *d_rowptr,
                       magma_int_t *d_colind,
                       magmaDoubleComplex *d_x,
                       magmaDoubleComplex beta,
                       magmaDoubleComplex *d_y );

magma_int_t 
magma_zgeellmv(        const char *transA,
                       magma_int_t m, magma_int_t n,
                       magma_int_t nnz_per_row,
                       magmaDoubleComplex alpha,
                       magmaDoubleComplex *d_val,
                       magma_int_t *d_colind,
                       magmaDoubleComplex *d_x,
                       magmaDoubleComplex beta,
                       magmaDoubleComplex *d_y );


magma_int_t 
magma_zgeelltmv(       const char *transA,
                       magma_int_t m, magma_int_t n,
                       magma_int_t nnz_per_row,
                       magmaDoubleComplex alpha,
                       magmaDoubleComplex *d_val,
                       magma_int_t *d_colind,
                       magmaDoubleComplex *d_x,
                       magmaDoubleComplex beta,
                       magmaDoubleComplex *d_y );


#ifdef __cplusplus
}
#endif

#undef PRECISION_z
#endif /* MAGMASPARSE_Z_H */
