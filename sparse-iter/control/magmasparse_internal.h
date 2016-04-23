/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
*/

#ifndef MAGMASPARSE_INTERNAL_H
#define MAGMASPARSE_INTERNAL_H

#include "magma_internal.h"
#include "magmasparse.h"

#ifdef __cplusplus
extern "C" {
#endif


magma_int_t cusparse2magma_error( cusparseStatus_t status );


/**
    Macro checks the return code of a function;
    if non-zero, sets info to err, then does goto cleanup.
    err is evaluated only once.
    Assumes variable info and label cleanup exist.
    Usually, all paths (successful and error) exit through the cleanup code.
    Example:
    
        magma_int_t function()
        {
            magma_int_t info = 0;
            double *A=NULL, *B=NULL;
            CHECK( magma_malloc( &A, sizeA ));
            CHECK( magma_malloc( &B, sizeB ));
            ...
        cleanup:
            magma_free( A );
            magma_free( B );
            return info;
        }
    
    @ingroup internal
    ********************************************************************/
#define CHECK( err )             \
    do {                         \
        magma_int_t e_ = (err);  \
        if ( e_ != 0 ) {         \
            info = e_;           \
            goto cleanup;        \
        }                        \
    } while(0)


/**
    Macro checks the return code of a cusparse function;
    if non-zero, maps the cusparse error to a magma error and sets info,
    then does goto cleanup.
    
    @see CHECK
    @ingroup internal
    ********************************************************************/
#define CHECK_CUSPARSE( err )                   \
    do {                                        \
        cusparseStatus_t e_ = (err);            \
        if ( e_ != 0 ) {                        \
            info = cusparse2magma_error( e_ );  \
            goto cleanup;                       \
        }                                       \
    } while(0)


#ifdef __cplusplus
} // extern C
#endif

#endif        //  #ifndef MAGMASPARSE_INTERNAL_H
