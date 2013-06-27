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




struct magma_sparse_matrix{

    magma_storage_t    storage_type,
    magma_location_t   memory_location,
    magma_int_t        num_row, 
    magma_int_t        num_col, 
    magmaDoubleComplex *val, 
    magma_int_t        *rowptr, 
    magma_int_t        *colind,


};



#ifdef __cplusplus
}
#endif

#endif        //  #ifndef MAGMASPARSE_TYPES_H
