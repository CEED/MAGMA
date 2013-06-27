/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011
*/

#ifndef MAGMASPARSE_TYPES_H
#define MAGMASPARSE_TYPES_H


#ifdef __cplusplus
extern "C" {
#endif




struct magma_z_sparse_matrix{

    magma_storage_t    storage_type;
    magma_location_t   memory_location;
    magma_int_t        num_rows;
    magma_int_t        num_cols; 
    magma_int_t        nnz; 
    magma_int_t        max_nnz_row;
    magmaDoubleComplex *val;
    magma_int_t        *row; 
    magma_int_t        *col;


};

struct magma_c_sparse_matrix{

    magma_storage_t    storage_type;
    magma_location_t   memory_location;
    magma_int_t        num_rows;
    magma_int_t        num_cols; 
    magma_int_t        nnz; 
    magma_int_t        max_nnz_row;
    magmaFloatComplex  *val;
    magma_int_t        *row; 
    magma_int_t        *col;


};


struct magma_d_sparse_matrix{

    magma_storage_t    storage_type;
    magma_location_t   memory_location;
    magma_int_t        num_rows;
    magma_int_t        num_cols; 
    magma_int_t        nnz; 
    magma_int_t        max_nnz_row;
    double             *val;
    magma_int_t        *row; 
    magma_int_t        *col;


};


struct magma_s_sparse_matrix{

    magma_storage_t    storage_type;
    magma_location_t   memory_location;
    magma_int_t        num_rows;
    magma_int_t        num_cols; 
    magma_int_t        nnz; 
    magma_int_t        max_nnz_row;
    float              *val;
    magma_int_t        *row; 
    magma_int_t        *col;


};



#ifdef __cplusplus
}
#endif

#endif        //  #ifndef MAGMASPARSE_TYPES_H
