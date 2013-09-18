/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> s d c
       @author Hartwig Anzt
*/

#include <fstream>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <iostream>
#include <ostream>
#include <assert.h>
#include <stdio.h>
#include "../include/magmasparse_z.h"
#include "../../include/magma.h"
#include "../include/mmio.h"



using namespace std;


/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======

    Determines for a matrix A, a number of processors num_procs, 
    the processor id proc_idi, and a number of matrix powers s, the number and
    which matrix rows are needed by this processor for the matrix power kernel.

    Arguments
    =========

    magma_sparse_matrix A                input matrix A
    magma_int_t num_procs                number of processors for matrix power kernel
    magma_int_t procs_id                 processor id
    magma_int_t s                        matrix powers
    magma_int_t *num_add_rows            number of additional rows
    magma_int_t **add_rows               inices of additional rows

    =====================================================================  */


magma_int_t 
magma_z_mpksetup(  magma_z_sparse_matrix A, 
                   magma_int_t num_procs, 
                   magma_int_t procs_id, 
                   magma_int_t s,    
                   magma_int_t *num_add_rows,
                   magma_int_t *add_rows ){
        
    magmaDoubleComplex one  = MAGMA_Z_MAKE(1.0, 0.0);
    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);

    magma_int_t blocksize = ( A.num_rows+num_procs-1) / num_procs ;
    magma_int_t local_blocksize = min( blocksize, A.num_cols - procs_id * blocksize);

    printf("local block size: %d\n", local_blocksize);

    magma_z_vector a, da, db;
    magma_z_sparse_matrix AT, dAT;
    magma_z_vinit( &a, Magma_CPU, A.num_cols, zero ); 
    magma_z_vinit( &da, Magma_DEV, A.num_cols, zero ); 
    magma_z_vinit( &db, Magma_DEV, A.num_cols, zero ); 

    for(magma_int_t i=procs_id*(blocksize); i<(procs_id)*(blocksize)+ local_blocksize; i++)
        a.val[i] = one;
    magma_z_vtransfer( a, &da, Magma_CPU, Magma_DEV);
    magma_z_vtransfer( da, &a, Magma_DEV, Magma_CPU);

    magma_z_mtranspose( A, &AT );
    magma_z_mtransfer( AT, &dAT, Magma_CPU, Magma_DEV);
    for(magma_int_t i=0; i<s; i++){
        magma_z_spmv( one, dAT, da, zero, db);
        magma_zcopy( dAT.num_rows, db.val, 1, da.val, 1 );
    }
    magma_z_vtransfer( da, &a, Magma_DEV, Magma_CPU);
    (*num_add_rows) = 0;
    for(magma_int_t i=0; i<procs_id*blocksize; i++){
        if( MAGMA_Z_REAL(a.val[i]) != MAGMA_Z_REAL(zero) )
            (*num_add_rows)++;
    }
    for(magma_int_t i=(procs_id+1)*blocksize; i<A.num_cols; i++){
        if( MAGMA_Z_REAL(a.val[i]) != MAGMA_Z_REAL(zero) )
            (*num_add_rows)++;
    }
    printf("additional rows: %d\n", (*num_add_rows));
    magma_imalloc_cpu( &(add_rows), (*num_add_rows) );
    (*num_add_rows) = 0;
    for(magma_int_t i=0; i<procs_id*blocksize; i++){
         if( MAGMA_Z_REAL(a.val[i]) != MAGMA_Z_REAL(zero) ){
            (add_rows)[(*num_add_rows)] = i;
            (*num_add_rows)++;
        }
    }
    for(magma_int_t i=(procs_id+1)*blocksize; i<A.num_cols; i++){
         if( MAGMA_Z_REAL(a.val[i]) != MAGMA_Z_REAL(zero) ){
            (add_rows)[(*num_add_rows)] = i;
            (*num_add_rows)++;
        }
    }
    for(int i=0; i<(*num_add_rows); i++){
        printf("%d  ", (add_rows)[i]);
    }
    printf("\n");

    magma_z_mfree(&AT);
    magma_z_mfree(&dAT);
    magma_free_cpu(add_rows);

}
