/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Hartwig Anzt
*/

// includes, project
#include "common_magmasparse.h"
#include "magmasparse_z.h"
#include "magma.h"
#include "mmio.h"








/**
    Purpose
    -------

    Allocates memory for magma_z_matrix and initializes it
    with the passed value.


    Arguments
    ---------

    @param[out]
    x           magma_z_matrix*
                vector to initialize

    @param[in]
    mem_loc     magma_location_t
                memory for vector

    @param[in]
    num_rows    magma_int_t
                desired length of vector

    @param[in]
    values      magmaDoubleComplex
                entries in vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zvinit(
    magma_z_matrix *x,
    magma_location_t mem_loc,
    magma_int_t num_rows,
    magma_int_t num_cols,
    magmaDoubleComplex values,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    // set queue for old dense routines
    magma_queue_t orig_queue=NULL;
    magmablasGetKernelStream( &orig_queue );

    x->memory_location = Magma_CPU;
    x->num_rows = num_rows;
    x->storage_type = Magma_DENSE;
    x->ld = num_rows;
    x->num_cols = num_cols;
    x->nnz = num_rows*num_cols;
    x->major = MagmaColMajor;
    if ( mem_loc == Magma_CPU ) {
        x->memory_location = Magma_CPU;
        CHECK( magma_zmalloc_cpu( &x->val, x->nnz ));
        for( magma_int_t i=0; i<x->nnz; i++)
             x->val[i] = values;
    }
    else if ( mem_loc == Magma_DEV ) {
        x->memory_location = Magma_DEV;
        CHECK( magma_zmalloc( &x->val, x->nnz ));
        magmablas_zlaset(MagmaFull, x->num_rows, x->num_cols, values, values, x->val, x->num_rows);
    }
    
cleanup:
    magmablasSetKernelStream( orig_queue );
    return MAGMA_SUCCESS;
}



   


