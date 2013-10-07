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
magma_z_csr_compressor( magmaDoubleComplex ** val, magma_int_t ** row, 
                        magma_int_t ** col, magmaDoubleComplex ** valn, 
                        magma_int_t ** rown, magma_int_t ** coln, magma_int_t *n);

/*
magma_int_t 
magma_zlra(             magma_z_sparse_matrix A, 
                        magma_z_sparse_matrix *B, 
                        int icompression );
*/

magma_int_t 
magma_z_mpkinfo_one(    magma_z_sparse_matrix A, 
                        magma_int_t offset, 
                        magma_int_t blocksize, 
                        magma_int_t s,    
                        magma_int_t **num_add_rows,
                        magma_int_t **add_rows,
                        magma_int_t *num_add_vecs,
                        magma_int_t **add_vecs );

magma_int_t 
magma_z_mpkback(        magma_z_sparse_matrix A, 
                        magma_int_t num_procs,
                        magma_int_t *offset, 
                        magma_int_t *blocksize, 
                        magma_int_t s,
                        magma_int_t *num_add_vecs,
                        magma_int_t **add_vecs,
                        magma_int_t *num_vecs_back,
                        magma_int_t **vecs_back );

magma_int_t 
magma_z_mpkinfo(        magma_z_sparse_matrix A, 
                        magma_int_t num_procs,
                        magma_int_t *offset, 
                        magma_int_t *blocksize, 
                        magma_int_t s,
                        magma_int_t **num_add_rows,
                        magma_int_t **add_rows,
                        magma_int_t *num_add_vecs,
                        magma_int_t **add_vecs,
                        magma_int_t *num_vecs_back,
                        magma_int_t **vecs_back );

magma_int_t 
magma_z_mpksetup_one(   magma_z_sparse_matrix A, 
                        magma_z_sparse_matrix *B, 
                        magma_int_t offset, 
                        magma_int_t blocksize, 
                        magma_int_t s );

magma_int_t 
magma_z_mpksetup(       magma_z_sparse_matrix A, 
                        magma_z_sparse_matrix B[MagmaMaxGPUs], 
                        magma_int_t num_procs,
                        magma_int_t *offset, 
                        magma_int_t *blocksize, 
                        magma_int_t s );

magma_int_t 
magma_z_mpk_compress(   magma_int_t num_add_rows,
                        magma_int_t *add_rows,
                        magmaDoubleComplex *x,
                        magmaDoubleComplex *y );

magma_int_t 
magma_z_mpk_uncompress( magma_int_t num_add_rows,
                        magma_int_t *add_rows,
                        magmaDoubleComplex *x,
                        magmaDoubleComplex *y );

magma_int_t 
magma_z_mpk_uncompress_sel( magma_int_t num_add_rows,
                        magma_int_t *add_rows,
                        magma_int_t offset,
                        magma_int_t blocksize,
                        magmaDoubleComplex *x,
                        magmaDoubleComplex *y );

magma_int_t 
magma_z_mpk_compress_gpu( magma_int_t num_add_rows,
                        magma_int_t *add_rows,
                        magmaDoubleComplex *x,
                        magmaDoubleComplex *y );

magma_int_t 
magma_z_mpk_uncompress_gpu( magma_int_t num_add_rows,
                        magma_int_t *add_rows,
                        magmaDoubleComplex *x,
                        magmaDoubleComplex *y );

magma_int_t 
magma_z_mpk_uncompspmv(  magma_int_t offset,
                         magma_int_t blocksize,
                         magma_int_t num_add_rows,
                         magma_int_t *add_rows,
                         magmaDoubleComplex *x,
                         magmaDoubleComplex *y );

magma_int_t
magma_z_mpk_mcompresso(  magma_z_sparse_matrix A,
                         magma_z_sparse_matrix *B,
                         magma_int_t offset,
                         magma_int_t blocksize,
                         magma_int_t num_add_rows,
                         magma_int_t *add_rows );

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
z_transpose_csr(        magma_int_t n_rows, magma_int_t n_cols, 
                        magma_int_t nnz, magmaDoubleComplex *val, 
                        magma_int_t *row, magma_int_t *col, 
                        magma_int_t *new_n_rows, magma_int_t *new_n_cols, 
                        magma_int_t *new_nnz, magmaDoubleComplex **new_val, 
                        magma_int_t **new_row, magma_int_t **new_col );

magma_int_t 
magma_z_mtranspose( magma_z_sparse_matrix A, magma_z_sparse_matrix *B );


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
magma_z_mvisu(          magma_z_sparse_matrix A );

magma_int_t 
magma_zdiameter( magma_z_sparse_matrix *A );

magma_int_t 
magma_zrowentries( magma_z_sparse_matrix *A );

magma_int_t
magma_z_mfree(          magma_z_sparse_matrix *A );

magma_int_t
magma_z_vfree(          magma_z_vector *x );

magma_int_t
magma_zjacobisetup(     magma_z_sparse_matrix A, magma_z_vector b, 
                        magma_z_sparse_matrix *M, magma_z_vector *c );
magma_int_t
magma_zjacobisetup_matrix( magma_z_sparse_matrix A, magma_z_vector b, 
                        magma_z_sparse_matrix *M, magma_z_vector *d );
magma_int_t
magma_zjacobisetup_vector( magma_z_vector b,  magma_z_vector d, 
                           magma_z_vector *c );

magma_int_t
magma_zjacobiiter(      magma_z_sparse_matrix M, magma_z_vector c, magma_z_vector *x,  
                        magma_solver_parameters *solver_par );

/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA_SPARSE function definitions / Data on CPU
*/

magma_int_t
magma_zilusetup( magma_z_sparse_matrix A, magma_z_sparse_matrix *M );


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
magma_int_t 
magma_zcg(             magma_z_sparse_matrix A, magma_z_vector b, magma_z_vector *x,  
                       magma_solver_parameters *solver_par );

magma_int_t 
magma_zgmres(          magma_z_sparse_matrix A, magma_z_vector b, magma_z_vector *x,  
                       magma_solver_parameters *solver_par );

magma_int_t
magma_zbicgstab(       magma_z_sparse_matrix A, magma_z_vector b, magma_z_vector *x,  
                       magma_solver_parameters *solver_par );

magma_int_t
magma_zbicgstab_merge( magma_z_sparse_matrix A, magma_z_vector b, magma_z_vector *x,  
                       magma_solver_parameters *solver_par );

magma_int_t
magma_zbicgstab_merge2( magma_z_sparse_matrix A, magma_z_vector b, magma_z_vector *x,  
                       magma_solver_parameters *solver_par );

magma_int_t
magma_zpcg(            magma_z_sparse_matrix A, magma_z_vector b, magma_z_vector *x,  
                       magma_solver_parameters *solver_par, 
                       magma_precond_parameters *precond_par );

magma_int_t
magma_zpbicgstab(      magma_z_sparse_matrix A, magma_z_vector b, magma_z_vector *x,  
                       magma_solver_parameters *solver_par, 
                       magma_precond_parameters *precond_par );

magma_int_t
magma_zpgmres(         magma_z_sparse_matrix A, magma_z_vector b, magma_z_vector *x,  
                       magma_solver_parameters *solver_par, 
                       magma_precond_parameters *precond_par );
magma_int_t
magma_zjacobi(         magma_z_sparse_matrix A, magma_z_vector b, magma_z_vector *x,  
                       magma_solver_parameters *solver_par );

magma_int_t
magma_zir(             magma_z_sparse_matrix A, magma_z_vector b, magma_z_vector *x,  
                       magma_solver_parameters *solver_par, 
                       magma_precond_parameters *precond_par );

magma_int_t
magma_zp1gmres(        magma_z_sparse_matrix A, magma_z_vector b, magma_z_vector *x,  
                       magma_solver_parameters *solver_par );

magma_int_t
magma_zgmres_pipe( magma_z_sparse_matrix A, magma_z_vector b, magma_z_vector *x,  
                   magma_solver_parameters *solver_par );


/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA_SPARSE utility function definitions
*/

magma_int_t
magma_z_spmv(     magmaDoubleComplex alpha, magma_z_sparse_matrix A, 
                  magma_z_vector x, magmaDoubleComplex beta, magma_z_vector y );

magma_int_t
magma_z_spmv_shift(     magmaDoubleComplex alpha, magma_z_sparse_matrix A, magmaDoubleComplex lambda,
                        magma_z_vector x, magmaDoubleComplex beta, magma_int_t offset, magma_int_t blocksize,
                        magma_int_t *add_vecs, magma_z_vector y );
/*
magma_int_t
magma_z_mpk(      magmaDoubleComplex alpha, magma_z_sparse_matrix A, 
                  magma_z_vector x, magmaDoubleComplex beta, magma_z_vector y, 
                  magma_int_t k );
*/
magma_int_t
magma_z_precond( magma_z_sparse_matrix A, magma_z_vector b, 
                 magma_z_vector *x, magma_precond_parameters precond );



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
magma_zgecsrmv_shift(  const char *transA,
                       magma_int_t m, magma_int_t n,
                       magmaDoubleComplex alpha,
                       magmaDoubleComplex lambda,
                       magmaDoubleComplex *d_val,
                       magma_int_t *d_rowptr,
                       magma_int_t *d_colind,
                       magmaDoubleComplex *d_x,
                       magmaDoubleComplex beta,
                       int offset,
                       int blocksize,
                       int *add_rows,
                       magmaDoubleComplex *d_y );

magma_int_t 
magma_zmgecsrmv(        const char *transA,
                       magma_int_t m, magma_int_t n,
                       magma_int_t num_vecs,
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
magma_zgeellmv_shift(  const char *transA,
                       magma_int_t m, magma_int_t n,
                       magma_int_t nnz_per_row,
                       magmaDoubleComplex alpha,
                       magmaDoubleComplex lambda,
                       magmaDoubleComplex *d_val,
                       magma_int_t *d_colind,
                       magmaDoubleComplex *d_x,
                       magmaDoubleComplex beta,
                       int offset,
                       int blocksize,
                       int *add_rows,
                       magmaDoubleComplex *d_y );


magma_int_t 
magma_zmgeellmv(       const char *transA,
                       magma_int_t m, magma_int_t n,
                       magma_int_t num_vecs,
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

magma_int_t 
magma_zgeelltmv_shift( const char *transA,
                       magma_int_t m, magma_int_t n,
                       magma_int_t nnz_per_row,
                       magmaDoubleComplex alpha,
                       magmaDoubleComplex lambda,
                       magmaDoubleComplex *d_val,
                       magma_int_t *d_colind,
                       magmaDoubleComplex *d_x,
                       magmaDoubleComplex beta,
                       int offset,
                       int blocksize,
                       int *add_rows,
                       magmaDoubleComplex *d_y );

magma_int_t 
magma_zmgeelltmv(      const char *transA,
                       magma_int_t m, magma_int_t n,
                       magma_int_t num_vecs,
                       magma_int_t nnz_per_row,
                       magmaDoubleComplex alpha,
                       magmaDoubleComplex *d_val,
                       magma_int_t *d_colind,
                       magmaDoubleComplex *d_x,
                       magmaDoubleComplex beta,
                       magmaDoubleComplex *d_y );
/*
magma_int_t 
magma_zmpkgeelltmv(    const char *transA,
                       magma_int_t m, magma_int_t n,
                       magma_int_t num_vecs,
                       magma_int_t nnz_per_row,
                       magmaDoubleComplex alpha,
                       magmaDoubleComplex *d_val,
                       magma_int_t *d_colind,
                       magmaDoubleComplex *d_x,
                       magmaDoubleComplex beta,
                       magmaDoubleComplex *d_y );

magma_int_t
magma_zmpkgeelltmv_2( const char *transA,
                    magma_int_t m, magma_int_t n,
                    magma_int_t diameter,
                    magma_int_t num_vecs,
                    magma_int_t nnz_per_row,
                    magmaDoubleComplex alpha,
                    magmaDoubleComplex *d_val,
                    magma_int_t *d_colind,
                    magmaDoubleComplex *d_x,
                    magmaDoubleComplex beta,
                    magmaDoubleComplex *d_y );

magma_int_t
magma_zmpkgeelltmv_3( const char *transA,
                    magma_int_t m, magma_int_t n,
                    magma_int_t num_vecs,
                    magma_int_t nnz_per_row,
                    magmaDoubleComplex alpha,
                    magmaDoubleComplex *d_val,
                    magma_int_t *d_colind,
                    magmaDoubleComplex *d_x,
                    magmaDoubleComplex beta,
                    magmaDoubleComplex *d_y );
*/

magma_int_t 
magma_zp1gmres_mgs(    magma_int_t  n, 
                       magma_int_t  k, 
                       magmaDoubleComplex *skp, 
                       magmaDoubleComplex *v, 
                       magmaDoubleComplex *z );



magma_int_t
magma_zmergedgs(    magma_int_t n, 
                    magma_int_t ldh,
                    magma_int_t k, 
                    magmaDoubleComplex *v, 
                    magmaDoubleComplex *r,
                    magmaDoubleComplex *skp );

magma_int_t
magma_zcopyscale(   int n, 
                    int k,
                    magmaDoubleComplex *r, 
                    magmaDoubleComplex *v,
                    magmaDoubleComplex *skp );



magma_int_t
magma_zjacobisetup_vector_gpu(int num_rows, magmaDoubleComplex *b, magmaDoubleComplex *d, magmaDoubleComplex *c);

magma_int_t
magma_zbicgmerge1(  int n, 
                    magmaDoubleComplex *skp,
                    magmaDoubleComplex *v, 
                    magmaDoubleComplex *r, 
                    magmaDoubleComplex *p );


magma_int_t
magma_zbicgmerge2(  int n, 
                    magmaDoubleComplex *skp, 
                    magmaDoubleComplex *r,
                    magmaDoubleComplex *v, 
                    magmaDoubleComplex *s );

magma_int_t
magma_zbicgmerge3(  int n, 
                    magmaDoubleComplex *skp, 
                    magmaDoubleComplex *p,
                    magmaDoubleComplex *s,
                    magmaDoubleComplex *t,
                    magmaDoubleComplex *x, 
                    magmaDoubleComplex *r );
magma_int_t
magma_zbicgmerge4(  int type, 
                    magmaDoubleComplex *skp );

magma_int_t
magma_zmdotc(       magma_int_t n, 
                    magma_int_t k, 
                    magmaDoubleComplex *v, 
                    magmaDoubleComplex *r,
                    magmaDoubleComplex *d1,
                    magmaDoubleComplex *d2,
                    magmaDoubleComplex *skp );

magma_int_t
magma_zgemvmdot(    int n, 
                    int k, 
                    magmaDoubleComplex *v, 
                    magmaDoubleComplex *r,
                    magmaDoubleComplex *d1,
                    magmaDoubleComplex *d2,
                    magmaDoubleComplex *skp );

magma_int_t
magma_zbicgmerge_spmv1(  
                 int n,
                 magmaDoubleComplex *d1,
                 magmaDoubleComplex *d2,
                 magmaDoubleComplex *d_val, 
                 int *d_rowptr, 
                 int *d_colind,
                 magmaDoubleComplex *d_p,
                 magmaDoubleComplex *d_r,
                 magmaDoubleComplex *d_v,
                 magmaDoubleComplex *skp );

magma_int_t
magma_zbicgmerge_spmv2(  
                 int n,
                 magmaDoubleComplex *d1,
                 magmaDoubleComplex *d2,
                 magmaDoubleComplex *d_val, 
                 int *d_rowptr, 
                 int *d_colind,
                 magmaDoubleComplex *d_s,
                 magmaDoubleComplex *d_t,
                 magmaDoubleComplex *skp );

magma_int_t
magma_zbicgmerge_xrbeta(  
                 int n,
                 magmaDoubleComplex *d1,
                 magmaDoubleComplex *d2,
                 magmaDoubleComplex *rr,
                 magmaDoubleComplex *r,
                 magmaDoubleComplex *p,
                 magmaDoubleComplex *s,
                 magmaDoubleComplex *t,
                 magmaDoubleComplex *x, 
                 magmaDoubleComplex *skp );

#ifdef __cplusplus
}
#endif

#undef PRECISION_z
#endif /* MAGMASPARSE_Z_H */
