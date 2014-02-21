/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
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
    magma_index_t      *row; 
    magma_index_t      *col;
    magma_index_t      *blockinfo;
    magma_int_t        blocksize;
    magma_int_t        numblocks;
    magma_int_t        alignment;

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
    magma_index_t      *row; 
    magma_index_t      *col;
    magma_index_t      *blockinfo;
    magma_int_t        blocksize;
    magma_int_t        numblocks;
    magma_int_t        alignment;

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
    magma_index_t      *row; 
    magma_index_t      *col;
    magma_index_t      *blockinfo;
    magma_int_t        blocksize;
    magma_int_t        numblocks;
    magma_int_t        alignment;

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
    magma_index_t      *row; 
    magma_index_t      *col;
    magma_index_t      *blockinfo;
    magma_int_t        blocksize;
    magma_int_t        numblocks;
    magma_int_t        alignment;

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



typedef struct magma_solver_parameters{

    magma_solver_type  solver;
    magma_int_t        version;
    double             epsilon;  
    magma_int_t        maxiter;
    magma_int_t        restart; 
    magma_ortho_t      ortho;
    magma_int_t        numiter;
    double             init_res;
    double             final_res;
    real_Double_t      runtime;
    real_Double_t      *res_vec;
    real_Double_t      *timing;
    magma_int_t        verbose;
    magma_int_t        info;

//---------------------------------
// the input for verbose is:
// 0 = production mode
// k>0 = convergence and timing is monitored in *res_vec and *timeing every  
// k-th iteration 
//
// the output of info is:
//  0 = convergence (stopping criterion met)
// -1 = no convergence
// -2 = convergence but stopping criterion not met within maxiter
//--------------------------------

}magma_solver_parameters;


typedef struct magma_z_preconditioner{

    magma_solver_type       solver;
    magma_precision         format;
    double                  epsilon;  
    magma_int_t             maxiter;
    magma_int_t             restart; 
    magma_int_t             numiter;
    double                  init_res;
    double                  final_res;
    magma_z_sparse_matrix   M;
    magma_z_vector          d;

}magma_z_preconditioner;

typedef struct magma_c_preconditioner{

    magma_solver_type       solver;
    magma_precision         format;
    double                  epsilon;  
    magma_int_t             maxiter;
    magma_int_t             restart; 
    magma_int_t             numiter;
    double                  init_res;
    double                  final_res;
    magma_c_sparse_matrix   M;
    magma_c_vector          d;

}magma_c_preconditioner;


typedef struct magma_d_preconditioner{

    magma_solver_type       solver;
    magma_precision         format;
    double                  epsilon;  
    magma_int_t             maxiter;
    magma_int_t             restart; 
    magma_int_t             numiter;
    double                  init_res;
    double                  final_res;
    magma_d_sparse_matrix   M;
    magma_d_vector          d;

}magma_d_preconditioner;


typedef struct magma_s_preconditioner{

    magma_solver_type       solver;
    magma_precision         format;
    double                  epsilon;  
    magma_int_t             maxiter;
    magma_int_t             restart; 
    magma_int_t             numiter;
    double                  init_res;
    double                  final_res;
    magma_s_sparse_matrix   M;
    magma_s_vector          d;

}magma_s_preconditioner;






#ifdef __cplusplus
}
#endif

#endif        //  #ifndef MAGMASPARSE_TYPES_H
