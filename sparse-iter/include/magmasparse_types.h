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




typedef struct magma_z_sparse_matrix{

    magma_storage_t    storage_type;
    magma_location_t   memory_location;
    magma_int_t        num_rows;
    magma_int_t        num_cols; 
    magma_int_t        nnz; 
    magma_int_t        max_nnz_row;
    magma_int_t        diameter;
    magmaDoubleComplex *val;
    magma_int_t        *row; 
    magma_int_t        *col;
    magma_int_t        *blockinfo;
    magma_int_t        blocksize;
    magma_int_t        numblocks;

}magma_z_sparse_matrix;

typedef struct magma_c_sparse_matrix{

    magma_storage_t    storage_type;
    magma_location_t   memory_location;
    magma_int_t        num_rows;
    magma_int_t        num_cols; 
    magma_int_t        nnz; 
    magma_int_t        max_nnz_row;
    magma_int_t        diameter;
    magmaFloatComplex  *val;
    magma_int_t        *row; 
    magma_int_t        *col;
    magma_int_t        *blockinfo;
    magma_int_t        blocksize;
    magma_int_t        numblocks;

}magma_c_sparse_matrix;


typedef struct magma_d_sparse_matrix{

    magma_storage_t    storage_type;
    magma_location_t   memory_location;
    magma_int_t        num_rows;
    magma_int_t        num_cols; 
    magma_int_t        nnz; 
    magma_int_t        max_nnz_row;
    magma_int_t        diameter;
    double             *val;
    magma_int_t        *row; 
    magma_int_t        *col;
    magma_int_t        *blockinfo;
    magma_int_t        blocksize;
    magma_int_t        numblocks;

}magma_d_sparse_matrix;


typedef struct magma_s_sparse_matrix{

    magma_storage_t    storage_type;
    magma_location_t   memory_location;
    magma_int_t        num_rows;
    magma_int_t        num_cols; 
    magma_int_t        nnz; 
    magma_int_t        max_nnz_row;
    magma_int_t        diameter;
    float              *val;
    magma_int_t        *row; 
    magma_int_t        *col;
    magma_int_t        *blockinfo;
    magma_int_t        blocksize;
    magma_int_t        numblocks;

}magma_s_sparse_matrix;



typedef struct magma_z_vector{

    magma_location_t   memory_location;
    magma_int_t        num_rows;
    magma_int_t        nnz; 
    magmaDoubleComplex *val;

}magma_z_vector;

typedef struct magma_c_vector{

    magma_location_t   memory_location;
    magma_int_t        num_rows;
    magma_int_t        nnz; 
    magmaFloatComplex  *val;

}magma_c_vector;


typedef struct magma_d_vector{

    magma_location_t   memory_location;
    magma_int_t        num_rows;
    magma_int_t        nnz; 
    double             *val;

}magma_d_vector;


typedef struct magma_s_vector{

    magma_location_t   memory_location;
    magma_int_t        num_rows;
    magma_int_t        nnz; 
    float              *val;

}magma_s_vector;



typedef enum {MAGMA_MGS, MAGMA_CGS, MAGMA_FUSED_CGS} magma_ortho_t;

typedef struct magma_solver_parameters{

    double             epsilon;  
    magma_int_t        maxiter;
    magma_int_t        restart; 
    magma_int_t        numiter;
    double             residual;

    // orthogonalization scheme form GMRES
    magma_ortho_t      ortho;
}magma_solver_parameters;


typedef struct magma_precond_parameters{

    magma_precond_type precond;
    magma_precision    format;
    double             epsilon;  
    magma_int_t        maxiter;
    magma_int_t        restart; 
    magma_int_t        numiter;
    double             residual;

}magma_precond_parameters;



#ifdef __cplusplus
}
#endif

#endif        //  #ifndef MAGMASPARSE_TYPES_H
