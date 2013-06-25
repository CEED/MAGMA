/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011
*/

#ifndef MAGMASPARSE_TYPES_H
#define MAGMASPARSE_TYPES_H

// ----------------------------------------
// Descriptors for sparse matrices and solvers

typedef struct {
    int type;

    magma_int_t   m;
    magma_int_t   n;
    magma_int_t nnz;

    magmaDoubleComplex *d_val;
    magma_int_t *d_rowptr;
    magma_int_t *d_colind;
   
} magma_zmatrix_t;

typedef struct {
  int type;

  magma_int_t   m;
  magma_int_t   n;
  magma_int_t nnz;

  magmaFloatComplex *d_val;
  magma_int_t *d_rowptr;
  magma_int_t *d_colind;

} magma_cmatrix_t;

typedef struct {
  int type;

  magma_int_t   m;
  magma_int_t   n;
  magma_int_t nnz;

  double *d_val;
  magma_int_t *d_rowptr;
  magma_int_t *d_colind;

} magma_dmatrix_t;

typedef struct {
  int type;

  magma_int_t   m;
  magma_int_t   n;
  magma_int_t nnz;

  float *d_val;
  magma_int_t *d_rowptr;
  magma_int_t *d_colind;

} magma_smatrix_t;



#ifdef __cplusplus
extern "C" {
#endif



#ifdef __cplusplus
}
#endif

#endif        //  #ifndef MAGMASPARSE_TYPES_H
